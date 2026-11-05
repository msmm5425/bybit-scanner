import os
import datetime
import requests
import pandas as pd
import numpy as np

BYBIT_URL = "https://api.bybit.com"
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


def fmt_price(val, ref):
    if ref >= 1000:
        return f"{val:.2f}"
    elif ref >= 10:
        return f"{val:.3f}"
    elif ref >= 1:
        return f"{val:.4f}"
    elif ref >= 0.01:
        return f"{val:.5f}"
    else:
        return f"{val:.6f}"


def get_top5_bybit_gainers():
    try:
        url = f"{BYBIT_URL}/v5/market/tickers?category=linear"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        raw_list = data.get("result", {}).get("list", [])
        valid = []
        for x in raw_list:
            sym = x.get("symbol", "")
            if not sym.endswith("USDT") or sym.startswith("1000000"):
                continue
            try:
                chg = float(x.get("price24hPcnt", 0)) * 100
                last_p = float(x.get("lastPrice", 0))
                turnover = float(x.get("turnover24h", 0))
                funding = float(x.get("fundingRate", 0)) * 100 if x.get("fundingRate") else 0.0
                oi_val = float(x.get("openInterestValue", 0)) if x.get("openInterestValue") else 0.0
            except Exception:
                continue

            if last_p <= 0:
                continue
            valid.append({
                "symbol": sym,
                "price": last_p,
                "chg": chg,
                "turnover": turnover,
                "funding": funding,
                "oi_val": oi_val
            })
        valid.sort(key=lambda k: k["chg"], reverse=True)
        return valid[:5]
    except Exception as e:
        print(f"Top5 fetch error: {e}")
        return []


def get_bybit_klines(symbol):
    url = f"{BYBIT_URL}/v5/market/kline?category=linear&symbol={symbol}&interval=5&limit=200"
    res = requests.get(url, headers=HEADERS, timeout=10)
    data = res.json()
    rows = data.get("result", {}).get("list", [])
    if not rows:
        return pd.DataFrame()
    cols = ["ts", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.iloc[::-1].reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume", "turnover"]:
        df[c] = df[c].astype(float)
    df["ts"] = df["ts"].astype(int)
    return df


def analyze_pre_surge(item):
    sym = item["symbol"]
    try:
        df = get_bybit_klines(sym)
    except Exception:
        return None

    if len(df) < 30:
        return None

    for m in [5, 20]:
        if len(df) >= m:
            df[f"ma{m}"] = df["close"].rolling(m).mean()
        else:
            df[f"ma{m}"] = df["close"]

    df["vma20"] = df["volume"].rolling(20).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()

    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    limit_cnt = min(len(df), 120)
    search_zone = df.iloc[-limit_cnt:].copy()
    if search_zone.empty:
        return None

    ig_idx = (search_zone["ret"] * search_zone["vol_ratio"]).idxmax()
    ig_candle = df.loc[ig_idx]
    pre_idx = max(0, ig_idx - 1)
    pre_candle = df.loc[pre_idx]

    kst = datetime.timezone(datetime.timedelta(hours=9))
    ig_time = datetime.datetime.fromtimestamp(
        ig_candle["ts"] / 1000, kst
    ).strftime("%m/%d %H:%M")

    ig_open = float(ig_candle["open"])
    post_high = float(df.loc[ig_idx:, "high"].max())
    surge_after_ig = ((post_high - ig_open) / (ig_open + 1e-9)) * 100

    ma_short_ok = bool(pre_candle.get("ma5", 0) > pre_candle.get("ma20", 0))
    box_start = max(0, pre_idx - 12)
    box_zone = df.loc[box_start:pre_idx]
    box_high = float(box_zone["high"].max())
    box_low = float(box_zone["low"].min())
    box_range_pct = ((box_high - box_low) / (box_low + 1e-9)) * 100

    vol_mult = float(ig_candle["vol_ratio"])
    if np.isnan(vol_mult):
        vol_mult = 1.0

    return {
        "symbol": sym,
        "ticker": sym.replace("USDT", ""),
        "price": item["price"],
        "chg": item["chg"],
        "turnover_m": item["turnover"] / 1_000_000,
        "ig_time": ig_time,
        "ig_price": ig_open,
        "surge_after_ig": surge_after_ig,
        "ma_short_ok": ma_short_ok,
        "box_range_pct": box_range_pct,
        "vol_mult": vol_mult,
        "funding": item["funding"],
        "oi_val_m": item["oi_val"] / 1_000_000
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5 = get_top5_bybit_gainers()
    if not top5:
        print("Top 5 데이터를 가져오지 못했습니다.")
        send_telegram("⚠️ [전략 분석] 바이비트 상승률 Top 5 데이터를 수집하지 못했습니다.")
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item)
            if r:
                results.append(r)
        except Exception as e:
            print(f"Analyze error for {item['symbol']}: {e}")

    if not results:
        send_telegram("⚠️ [전략 분석] 분석 가능한 종목 결과가 없습니다.")
        return

    lines = [
        f"📊 [바이비트 실시간 상승률 Top 5 발화점 정밀 해부 리포트]\n"
        f"({now_str} 기준)\n"
    ]

    for idx, r in enumerate(results, 1):
        s_tag = "✅정배열 완성" if r["ma_short_ok"] else "❌혼조 돌파"
        p = r["price"]
        lines.append("━━━━━━━━━━━━━━")
        lines.append(
            f"{idx}️⃣ ${r['ticker']} ({r['symbol']} | +{r['chg']:.1f}% | 현재가 {fmt_price(p, p)} USDT)"
        )
        lines.append(
            f"• 24h 거래대금: {r['turnover_m']:,.1f}M USDT | 미결제약정: {r['oi_val_m']:,.1f}M USDT | 펀딩비: {r['funding']:+.4f}%"
        )
        lines.append(
            f"• [초동 발화 시점]: {r['ig_time']} KST "
            f"(발화가 {fmt_price(r['ig_price'], p)} → 고점까지 +{r['surge_after_ig']:.1f}% 폭발)"
        )
        lines.append("• [급등 직전 컨펌 내역]:")
        lines.append(f"  - 5분봉 단기이평(5>20): [{s_tag}]")
        lines.append(f"  - 직전 1시간 박스권 변동폭: {r['box_range_pct']:.1f}% 응축")
        lines.append(f"  - 발화 캔들 수급: 평소 대비 [{r['vol_mult']:.1f}배 폭발]")

    lines.append("━━━━━━━━━━━━━━\n")
    lines.append("💡 [전략 피드백]")
    lines.append("• 바이비트 실시간 상승률 Top 5 종목의 발화 전 5분봉 단기 이평선 및 거래량 폭발 패턴 역산 완료.")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
