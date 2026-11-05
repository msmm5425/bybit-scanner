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
                raw_chg = x.get("price24hPcnt")
                chg = float(raw_chg) * 100 if raw_chg not in [None, ""] else 0.0

                raw_price = x.get("lastPrice")
                last_p = float(raw_price) if raw_price not in [None, ""] else 0.0

                raw_turnover = x.get("turnover24h")
                turnover = float(raw_turnover) if raw_turnover not in [None, ""] else 0.0

                raw_funding = x.get("fundingRate")
                funding = float(raw_funding) * 100 if raw_funding not in [None, ""] else 0.0

                raw_oi = x.get("openInterestValue")
                oi_val = float(raw_oi) if raw_oi not in [None, ""] else 0.0
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
    try:
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
    except Exception as e:
        print(f"Kline error for {symbol}: {e}")
        return pd.DataFrame()


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
        print("Top 5 데이터를 가져오지
