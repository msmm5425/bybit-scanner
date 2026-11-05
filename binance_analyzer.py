import os
import datetime
import requests
import pandas as pd
import numpy as np

BYBIT_URL = "https://api.bybit.com"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEADERS = {"User-Agent": "Mozilla/5.0"}


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i+3500]}, timeout=15)
        except Exception:
            pass


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
        lst = res.json().get("result", {}).get("list", [])
        valid = []
        for x in lst:
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
            valid.append({"symbol": sym, "price": last_p, "chg": chg, "turnover": turnover, "funding": funding, "oi_val": oi_val})
        valid.sort(key=lambda k: k["chg"], reverse=True)
        return valid[:5]
    except Exception:
        return []


def get_bybit_klines(symbol):
    url = f"{BYBIT_URL}/v5/market/kline?category=linear&symbol={symbol}&interval=5&limit=200"
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        rows = res.json().get("result", {}).get("list", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = df[c].astype(float)
        df["ts"] = df["ts"].astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def analyze_pre_surge(item):
    sym = item["symbol"]
    df = get_bybit_klines(sym)
    if len(df) < 30: return None

    for m in [5, 20]:
        df[f"ma{m}"] = df["close"].rolling(m).mean() if len(df) >= m else df["close"]
    df["vma20"] = df["volume"].rolling(20).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()

    df["ret"] = (df["close"] - df["open"]) / (
