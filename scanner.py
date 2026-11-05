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
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    chunk_size = 3500
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=15)
        except Exception:
            pass

def fmt_price(val, ref_price):
    if ref_price >= 1000: return f"{val:.2f}"
    elif ref_price >= 10: return f"{val:.3f}"
    elif ref_price >= 1: return f"{val:.4f}"
    elif ref_price >= 0.01: return f"{val:.5f}"
    else: return f"{val:.6f}"

def get_binance_futures_map():
    try:
        res = requests.get(f"{BINANCE_URL}/fapi/v1/ticker/24hr", headers=HEADERS, timeout=10)
        data = res.json()
        if isinstance(data, list):
            return {item["symbol"]: float(item.get("quoteVolume", 0)) for item in data if isinstance(item, dict) and "symbol" in item}
    except Exception:
        pass
    return {}

def get_active_bybit_tickers(binance_map):
    try:
        url = f"{BYBIT_URL}/v5/market/tickers?category=linear"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        tickers = []
        for item in data.get("result", {}).get("list", []):
            symbol = item["symbol"]
            if not symbol.endswith("USDT") or symbol.startswith("1000000"): continue
            turnover = float(item.get("turnover24h", 0))
            last_price = float(item.get("lastPrice", 0))
            low_24h = float(item.get("lowPrice24h", 0))
            high_24h = float(item.get("highPrice24h", 0))
            if turnover < 5_000_000 or last_price <= 0 or low_24h <= 0: continue
            if ((last_price - low_24h) / low_24h) * 100 < 1.5: continue
            tickers.append({
                "symbol": symbol, "lastPrice": last_price,
                "price24hPcnt": float(item.get("price24hPcnt", 0)) * 100,
                "fundingRate": float(item.get("fundingRate", 0)) * 100,
                "openInterestValue": float(item.get("openInterestValue", 0)),
                "highPrice24h": high_24h, "lowPrice24h": low_24h, "turnover24h": turnover
            })
        tickers.sort(key=lambda x: x["turnover24h"], reverse=True)
        return tickers[:65]
    except Exception:
        return []

def get_klines(symbol, interval="15", limit=300):
    try:
        url = f"{BYBIT_URL}/v5/market/kline?category=linear&symbol={symbol}&interval={interval}&limit={limit}"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        rows = data.get("result", {}).get("list", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for col in ["open", "high", "low", "close", "volume"]: df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def get_oi_change_24h(symbol):
    try:
        url = f"{BYBIT_URL}/v5/market/open-interest?category=linear&symbol={symbol}&intervalTime=1h&limit=25"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        lst = data.get("result", {}).get("list", [])
        if len(lst) >= 24:
            oi_now = float(lst[0]["openInterest"])
            oi_24h_ago = float(lst[-1]["openInterest"])
            if oi_24h_ago > 0: return ((oi_now - oi_24h_ago) / oi_24h_ago) * 100
    except Exception: pass
    return 0.0

def get_orderbook_imbalance(symbol):
    try:
        url = f"{BYBIT_URL}/v5/market/orderbook?category=linear&symbol={symbol}&limit=25"
        res = requests.get(url, headers=HEADERS, timeout=10)
        data = res.json()
        bids = sum(float(b[1]) for b in data["result"]["b"])
        asks = sum(float(a[1]) for a in data["result"]["a"])
        return bids / (asks + 1e-9)
    except Exception: return 1.0

def calc_indicators(df):
    for m in [5, 20, 50, 120, 200, 300]: df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma10"] = df["volume"].rolling(10).mean()
    df["vma20"] = df["volume"].rolling(20).mean()
    direction = np.where(df["close"] > df["close"].shift(1), 1, np.where(df["close"] < df["close"].shift(1), -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    for r in [6, 12, 24]:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(r).mean()
        loss = (-delta.clip(upper=0)).rolling(r).mean()
        df[f"rsi{r}"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df

def analyze_symbol(t):
    symbol = t["symbol"]
    df15 = get_klines(symbol, "15", 300)
    df5 = get_klines(symbol, "5", 300)
    if len(df15) < 300 or len(df5) < 300: return None

    df15 = calc_indicators(df15)
    df5 = calc_indicators(df5)
    last15 = df15.iloc[-1]
    last5 = df5.iloc[-1]
    price = t["lastPrice"]

    oi_chg = get_oi_change_24h(symbol)
    ob_ratio = get_orderbook_imbalance(symbol)

    score = 60
    if last15["ma5"] > last15["ma20"] > last15["ma50"]: score += 8
    if last15["ma50"] > last15["ma120"] > last15["ma200"]: score += 5
    if last5["ma5"] > last5["ma20"] > last5["ma50"]: score += 5
    if last15["obv"] > last15["obv_ma10"]: score += 4
    if oi_chg >= 5.0: score += 6
    elif oi_chg > 0: score += 2
    if t["fundingRate"] <= 0.01: score += 5
    if t["fundingRate"] < 0: score += 3
    if (df5["volume"].iloc[-15:] > (df5["vma20"].iloc[-15:] * 3.5)).any(): score += 4
    if (last5["volume"] < last5["vma5"]) or (last15["volume"] < last15["vma10"]): score += 5
    if last15["macd"] > 0 and last15["rsi12"] >= 50: score += 4
    if ob_ratio >= 1.12: score += 5

    score = min(int(round(score)), 98)
    if score < 90: return None

    ma20_15 = last15["ma20"]
    support_ref = ma20_15
    support_str = fmt_price(support_ref, price)

    near_support = abs(price - support_ref) / support_ref <= 0.015
    vol_dry = (last5["volume"] < last5["vma5"]) or (last15["volume"] < last15["vma10"])

    if near_support and vol_dry and price >= support_ref * 0.995:
        status = f"즉시 분할 진입 구간 (15분봉 MA 20선인 {support_str} USDT 지지·거래량 건조 완료)"
    else:
        status = f"눌림목 컨펌대기중 (15분봉 MA 20선인 {support_str} USDT 대기 중)"

    entry_low = min(price, support_ref) * 0.996
    entry_high = max(price, support_ref) * 1.001
    avg_entry = (entry_low + entry_high) / 2.0
    atr15 = last15["atr14"]
    sl_pct = max(2, min(6, int(round((1.5 * atr15 / avg_entry) * 100))))
    stop_price = avg_entry * (1.0 - sl_pct / 100.0)

    min_tp1_pct = sl_pct * 2
    swing_range = max(t["highPrice24h"] - t["lowPrice24h"], avg_entry * (min_tp1_pct / 100.0))
    fib_levels = [(1.272, "전고점 돌파 및 피보나치 1.272 확장"), (1.618, "피보나치 1.618 확장"), (2.000, "피보나치 2.0 확장"), (2.618, "피보나치 2.618 확장")]
    tps = []
    prev_pct = min_tp1_pct - 1
    for idx, (fib, reason) in enumerate(fib_levels):
        raw_target = avg_entry + swing_range * (fib - 0.5)
        pct = int(round(((raw_target - avg_entry) / avg_entry) * 100))
        pct = max(pct, min_tp1_pct if idx == 0 else prev_pct + max(3, sl_pct))
        prev_pct = pct
        tps.append((avg_entry * (1.0 + pct / 100.0), pct, reason))

    return {
        "symbol": symbol, "ticker": symbol.replace("USDT", ""), "score": score,
        "price": price, "chg": t["price24hPcnt"], "funding": t["fundingRate"],
        "oi_chg": oi_chg, "status": status, "entry_low": entry_low, "entry_high": entry_high,
        "avg_entry": avg_entry, "stop_price": stop_price, "sl_pct": sl_pct, "tps": tps
    }

def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_dt
