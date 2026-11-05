import os
import json
import datetime
import requests
import pandas as pd
import numpy as np

BYBIT_URL = "https://api.bybit.com"
BINANCE_URL = "https://fapi.binance.com"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
STATE_FILE = "bot_state.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:
        pass

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: Telegram Token or Chat ID missing.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        chunk = text[i:i + 3500]
        try:
            res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=15)
            print(f"Telegram status: {res.status_code}")
        except Exception as e:
            print(f"Telegram error: {e}")

def fmt_price(val, ref):
    if ref >= 1000: return f"{val:.2f}"
    elif ref >= 10: return f"{val:.3f}"
    elif ref >= 1: return f"{val:.4f}"
    elif ref >= 0.01: return f"{val:.5f}"
    else: return f"{val:.6f}"

def get_binance_map():
    try:
        res = requests.get(f"{BINANCE_URL}/fapi/v1/ticker/24hr", headers=HEADERS, timeout=10)
        data = res.json()
        if isinstance(data, list):
            return {x["symbol"]: float(x.get("quoteVolume", 0)) for x in data if isinstance(x, dict) and "symbol" in x}
    except Exception:
        pass
    return {}

def get_bybit_tickers():
    try:
        url = f"{BYBIT_URL}/v5/market/tickers?category=linear"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        lst = []
        for x in data.get("result", {}).get("list", []):
            sym = x["symbol"]
            if not sym.endswith("USDT") or sym.startswith("1000000"): continue
            turnover = float(x.get("turnover24h", 0))
            price = float(x.get("lastPrice", 0))
            low = float(x.get("lowPrice24h", 0))
            if turnover < 5_000_000 or price <= 0 or low <= 0: continue
            if ((price - low) / low) * 100 < 1.5: continue
            lst.append({
                "symbol": sym, "lastPrice": price,
                "price24hPcnt": float(x.get("price24hPcnt", 0)) * 100,
                "fundingRate": float(x.get("fundingRate", 0)) * 100,
                "highPrice24h": float(x.get("highPrice24h", 0)),
                "lowPrice24h": low, "turnover24h": turnover
            })
        lst.sort(key=lambda k: k["turnover24h"], reverse=True)
        return lst[:65]
    except Exception:
        return []

def get_klines(symbol):
    try:
        url = f"{BYBIT_URL}/v5/market/kline?category=linear&symbol={symbol}&interval=15&limit=300"
        res = requests.get(url, headers=HEADERS, timeout=10)
        rows = res.json().get("result", {}).get("list", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]: df[c] = df[c].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def get_oi_chg(symbol):
    try:
        url = f"{BYBIT_URL}/v5/market/open-interest?category=linear&symbol={symbol}&intervalTime=1h&limit=25"
        res = requests.get(url, headers=HEADERS, timeout=10)
        lst = res.json().get("result", {}).get("list", [])
        if len(lst) >= 24:
            now_oi = float(lst[0]["openInterest"])
            old_oi = float(lst[-1]["openInterest"])
            if old_oi > 0: return ((now_oi - old_oi) / old_oi) * 100
    except Exception:
        pass
    return 0.0

def analyze(t):
    sym = t["symbol"]
    df = get_klines(sym)
    if len(df) < 200: return None

    for m in [5, 20, 50, 120, 200]: df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma10"] = df["volume"].rolling(10).mean()
    
    last = df.iloc[-1]
    price = t["lastPrice"]
    oi = get_oi_chg(sym)

    score = 65
    if last["ma5"] > last["ma20"] > last["ma50"]: score += 10
    if oi >= 5.0: score += 8
    elif oi > 0: score += 3
    if t["fundingRate"] <= 0.01: score += 5
    if last["volume"] < last["vma5"]: score += 7

    score = min(int(round(score)), 98)
    if score < 90: return None

    ma20 = last["ma20"]
    status = f"눌림목 대기중 (15분봉 MA 20선: {fmt_price(ma20, price)} USDT)"
    if abs(price - ma20) / ma20 <= 0.015:
        status = f"즉시 분할 진입 구간 (MA 20선 지지 및 거래량 건조)"

    sl_price = price * 0.97
    tp1 = price * 1.06
    tp2 = price * 1.09
    tp3 = price * 1.12
    tp4 = price * 1.16

    return {
        "symbol": sym, "ticker": sym.replace("USDT", ""), "score": score,
        "price": price, "chg": t["price24hPcnt"], "funding": t["fundingRate"],
        "oi_chg": oi, "status": status, "stop_price": sl_price,
        "tps
