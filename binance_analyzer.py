import os
import datetime
import requests
import pandas as pd
import numpy as np

BINANCE_URL = "https://fapi.binance.com"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: 텔레그램 설정 누락")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        chunk = text[i:i + 3500]
        try:
            res = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
                timeout=15
            )
            print(f"Telegram status: {res.status_code}")
        except Exception as e:
            print(f"Telegram error: {e}")


def fmt_price(val):
    if val >= 1000:
        return f"{val:.2f}"
    elif val >= 10:
        return f"{val:.3f}"
    elif val >= 1:
        return f"{val:.4f}"
    elif val >= 0.01:
        return f"{val:.5f}"
    else:
        return f"{val:.6f}"


def get_top5_binance_gainers():
    try:
        res = requests.get(
            f"{BINANCE_URL}/fapi/v1/ticker/24hr",
            headers=HEADERS,
            timeout=10
        )
        data = res.json()
        valid = []
        for x in data:
            sym = x.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            chg = float(x.get("priceChangePercent", 0))
            last_p = float(x.get("lastPrice", 0))
            if last_p <= 0:
                continue
            valid.append({
                "symbol": sym,
                "price": last_p,
                "chg": chg,
                "volume_usdt": float(x.get("quoteVolume", 0))
            })
        # 거래대금 조건 없이 오직 상승률(chg) 기준 Top 5 추출
        valid.sort(key=lambda k: k["chg"], reverse=True)
        return valid[:5]
    except Exception as e:
        print(f"Top5 fetch error: {e}")
        return []


def get_binance_klines(symbol, interval="5m", limit=300):
    try:
        url = (
            f"{BINANCE_URL}/fapi/v1/klines?symbol={symbol}"
            f"&interval={interval}&limit={limit}"
        )
        res = requests.get(url, headers=HEADERS, timeout=10)
        rows = res.json()
        if not rows or not isinstance(rows, list):
            return pd.DataFrame()
        cols = [
            "ts", "open", "high", "low", "close", "volume",
            "close_ts", "turnover", "trades", "taker_buy_vol",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
            df[c] = df[c].astype(float)
        df["ts"] = df["ts"].astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def get_funding_and_oi(symbol):
    funding = 0.0
    oi_chg_24h = 0.0
    try:
        r1 = requests.get(
            f"{BINANCE_URL}/fapi/v1/premiumIndex?symbol={symbol}",
            headers=HEADERS,
            timeout=10
        ).json()
        funding = float(r1.get("lastFundingRate", 0)) * 100
    except Exception:
        pass

    try:
        r2 = requests.get(
            f"{BINANCE_URL}/futures/data/openInterestHist"
            f"?symbol={symbol}&period=1h&limit=25",
            headers=HEADERS,
            timeout=10
        ).json()
        if isinstance(r2, list) and len(r2) >= 2:
            oi_start = float(r2[0].get("sumOpenInterest", 0))
            oi_now = float(r2[-1].get("sumOpenInterest", 0))
            if oi_start > 0:
                oi_chg_24h = ((oi_now - oi_start) / oi_start) * 100
    except Exception:
        pass
    return funding, oi_chg_24h


def analyze_pre_surge(item):
    sym = item["symbol"]
    df = get_binance_klines(sym, "5m", 300)
    if len(df) < 150:
        return None

    for m in [5, 20, 50]:
        df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma20"] = df["volume"].rolling(20).mean()
    df["vma5"] = df["volume"].rolling(5).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()

    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    search_zone = df.iloc[-200:].copy()
    ig_idx = (search_zone["ret"] * search_zone["vol_ratio"]).idxmax()

    ig_candle = df.loc[ig_idx]
    pre_idx = max(50, ig_idx - 1)
    pre_candle = df.loc[pre_idx]

    kst = datetime.timezone(datetime.timedelta(hours=9))
    ig_time = datetime.datetime.fromtimestamp(
        ig_candle["ts"] / 1000, kst
    ).strftime("%m/%d %H:%M")

    ig_open = float(ig_candle["open"])
    post_high = float(df.loc[ig_idx:, "high"].max())
    surge_after_ig = ((post_high - ig_open) / (ig_open + 1e-9)) * 100

    ma_short_ok = pre_candle["ma5"] > pre_candle["ma20"] > pre_candle["ma50"]
    box_zone = df.loc[max(0, pre_idx - 24): pre_idx]
    box_high = float(box_zone["high"].max())
    box_low = float(box_zone["low"].min())
    box_range_pct = ((box_high - box_low) / (box_low + 1e-9)) * 100

    vol_mult = float(ig_candle["vol_ratio"])
    taker_ratio = (
        float(ig_candle["taker_buy_vol"]) / (float(ig_candle["volume"]) + 1e-9)
    ) * 100

    funding, oi_chg_24h = get_funding_and_oi(sym)

    return {
        "symbol": sym,
        "price": item["price"],
        "chg": item["chg"],
        "vol_usdt_m": item["volume_usdt"] / 1_000_000,
        "ig_time": ig_time,
        "ig_price": ig_open,
        "surge_after_ig": surge_after_ig,
        "ma_short_ok": bool(ma_short_ok),
        "box_range_pct": box_range_pct,
        "vol_mult": vol_mult,
        "taker_ratio": taker_ratio,
        "funding": funding,
        "oi_chg_24h": oi_chg_24h
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5 = get_top5_binance_gainers()
    if not top5:
        print("Top 5 데이터를 가져오지 못했습니다.")
        send_telegram("⚠️ [바이낸스 분석] 상승률 Top 5 데이터를 수집하지 못했습니다.")
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item)
            if r:
                results.append(r)
        except Exception as e:
            print(f"Analyze error: {e}")

    if not results:
        return

    lines = [
        f"📊 [바이낸스 상승률 Top 5 발화점(Pre-Surge) 정밀 해부 리포트]\n"
        f"({now_str} 기준)\n"
    ]

    for idx, r in enumerate(results, 1):
        s_tag = "✅정배열 완성" if r["ma_short_ok"] else "❌혼조 돌파"
        lines.append("━━━━━━━━━━━━━━")
        lines.append(
            f"{idx}️⃣ ${r['symbol']} (+{r['chg']:.1f}% | 현재가 {fmt_price(r['price'])} USDT)"
        )
        lines.append(
            f"• 24h 거래대금: {r['vol_usdt_m']:,.1f}M USDT | "
            f"펀딩비: {r['funding']:+.4f}% | 24h OI: {r['oi_chg_24h']:+.1f}%"
        )
        lines.append(
            f"• [초동 발화 시점]: {r['ig_time']} KST "
            f"(발화가 {fmt_price(r['ig_price'])} → 고점까지 +{r['surge_after_ig']:.1f}% 폭발)"
        )
        lines.append("• [급등 직전 컨펌 내역]:")
        lines.append(f"  - 5분봉 이평선(5>20>50): [{s_tag}]")
        lines.append(f"  - 직전 2시간 박스권 변동폭: {r['box_range_pct']:.1f}% 응축")
        lines.append(
            f"  - 발화 캔들 수급: 평소 대비 [{r['vol_mult']:.1f}배 폭발] / "
            f"시장가 매수 비중 [{r['taker_ratio']:.1f}%]"
        )

    lines.append("━━━━━━━━━━━━━━\n")
    lines.append("💡 [전략 피드백]")
    lines.append("• 시각 기준 순수 상승률 Top 5 종목의 발화 전 거래량 폭발 패턴과 이평선 정배열 상태 역산 완료.")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
