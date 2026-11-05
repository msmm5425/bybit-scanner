import os
import datetime
import requests
import pandas as pd
import numpy as np

BYBIT_URL = "https://api.bybit.com"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEADERS = {"User-Agent": "Mozilla/5.0"}

SKIP_PREFIXES = ("1000000", "100000", "10000", "1000")


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("토큰 설정 누락")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i+3500]}, timeout=15)
        except Exception as e:
            print(f"텔레그램 전송 오류: {e}")


def to_float(val, default=0.0):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default


def fmt_price(val, ref):
    if ref >= 1000: return f"{val:.2f}"
    elif ref >= 10: return f"{val:.3f}"
    elif ref >= 1: return f"{val:.4f}"
    elif ref >= 0.01: return f"{val:.5f}"
    else: return f"{val:.6f}"


def get_top5_bybit_gainers():
    try:
        url = f"{BYBIT_URL}/v5/market/tickers?category=linear"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()

        # retCode 안전 검사 (None이거나 0이 아니어도 result 리스트가 있으면 허용)
        lst = data.get("result", {}).get("list", [])
        if not lst:
            print(f"API 응답 데이터 없음: {data}")
            return []

        valid = []
        for x in lst:
            sym = x.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if any(sym.startswith(p) for p in SKIP_PREFIXES):
                continue

            last_p = to_float(x.get("lastPrice"))
            chg = to_float(x.get("price24hPcnt")) * 100
            turnover = to_float(x.get("turnover24h"))
            funding = to_float(x.get("fundingRate")) * 100
            oi_val = to_float(x.get("openInterestValue"))

            if last_p <= 0:
                continue

            valid.append({
                "symbol": sym, "price": last_p, "chg": chg,
                "turnover": turnover, "funding": funding, "oi_val": oi_val
            })

        if not valid:
            return []

        valid.sort(key=lambda k: k["chg"], reverse=True)
        return valid[:5]
    except Exception as e:
        print(f"상승률 Top 5 수집 중 예외 발생: {e}")
        return []


def get_bybit_klines(symbol):
    url = f"{BYBIT_URL}/v5/market/kline?category=linear&symbol={symbol}&interval=5&limit=200"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        rows = data.get("result", {}).get("list", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
        df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"{symbol} 캔들 수집 오류: {e}")
        return pd.DataFrame()


def analyze_pre_surge(item):
    sym = item["symbol"]
    df = get_bybit_klines(sym)
    if len(df) < 30:
        return None

    for m in [5, 20]:
        df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma20"] = df["volume"].rolling(20).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()

    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    search_zone = df.iloc[-min(len(df), 120):].copy()
    if search_zone.empty:
        return None

    score = search_zone["ret"] * search_zone["vol_ratio"]
    if score.notna().sum() == 0:
        return None

    ig_idx = score.idxmax()
    ig_candle = df.loc[ig_idx]
    pre_idx = max(0, ig_idx - 1)
    pre_candle = df.loc[pre_idx]

    kst = datetime.timezone(datetime.timedelta(hours=9))
    ig_time = datetime.datetime.fromtimestamp(ig_candle["ts"] / 1000, kst).strftime("%m/%d %H:%M")

    ig_open = float(ig_candle["open"])
    post_high = float(df.loc[ig_idx:, "high"].max())
    surge_after_ig = ((post_high - ig_open) / (ig_open + 1e-9)) * 100

    ma_short_ok = bool(
        pd.notna(pre_candle.get("ma5")) and
        pd.notna(pre_candle.get("ma20")) and
        pre_candle["ma5"] > pre_candle["ma20"]
    )

    box_start = max(0, pre_idx - 12)
    box_zone = df.loc[box_start:pre_idx]
    box_low = float(box_zone["low"].min())
    box_high = float(box_zone["high"].max())
    box_range_pct = 0.0 if box_low <= 0 else ((box_high - box_low) / box_low) * 100

    vol_mult = to_float(ig_candle["vol_ratio"], default=1.0)
    if np.isnan(vol_mult):
        vol_mult = 1.0

    return {
        "symbol": sym, "ticker": sym.replace("USDT", ""),
        "price": item["price"], "chg": item["chg"],
        "turnover_m": item["turnover"] / 1_000_000,
        "ig_time": ig_time, "ig_price": ig_open,
        "surge_after_ig": surge_after_ig, "ma_short_ok": ma_short_ok,
        "box_range_pct": box_range_pct, "vol_mult": vol_mult,
        "funding": item["funding"], "oi_val_m": item["oi_val"] / 1_000_000
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5 = get_top5_bybit_gainers()
    if not top5:
        send_telegram("⚠️ 바이비트 상승률 Top 5 데이터 수집에 실패했습니다.")
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item)
            if r:
                results.append(r)
        except Exception as e:
            print(f"분석 예외: {e}")

    if not results:
        send_telegram("⚠️ 분석 가능한 종목 결과가 없습니다.")
        return

    lines = [f"📊 [바이비트 상승률 Top 5 발화점 정밀 해부 리포트] ({now_str})\n"]
    for idx, r in enumerate(results, 1):
        s_tag = "✅ 정배열 완성" if r["ma_short_ok"] else "❌ 혼조 돌파"
        p = r["price"]
        lines.append("━━━━━━━━━━━━━━")
        lines.append(f"{idx}️⃣ ${r['ticker']} ({r['symbol']} | +{r['chg']:.1f}% | 현재가 {fmt_price(p, p)} USDT)")
        lines.append(f"• 거래대금: {r['turnover_m']:,.1f}M | 미결제약정: {r['oi_val_m']:,.1f}M | 펀딩비: {r['funding']:+.4f}%")
        lines.append(f"• [초동 발화 시점]: {r['ig_time']} KST (발화가 {fmt_price(r['ig_price'], p)} → 고점 +{r['surge_after_ig']:.1f}%)")
        lines.append(f"• [컨펌 내역]: 단기이평(5>20) [{s_tag}] | 박스권 응축 {r['box_range_pct']:.1f}% | 거래량 {r['vol_mult']:.1f}배 폭발")

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
