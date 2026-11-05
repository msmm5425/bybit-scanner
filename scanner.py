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

def get_bybit_tickers():
    try:
        url = f"{BYBIT_URL}/v5/market/tickers?category=linear"
        res = requests.get(url, headers=HEADERS, timeout=10)
        lst = []
        for x in res.json().get("result", {}).get("list", []):
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
                "turnover24h": turnover
            })
        lst.sort(key=lambda k: k["turnover24h"], reverse=True)
        return lst[:50]
    except Exception:
        return []

def get_klines(symbol):
    try:
        url = f"{BYBIT_URL}/v5/market/kline?category=linear&symbol={symbol}&interval=15&limit=200"
        res = requests.get(url, headers=HEADERS, timeout=10)
        rows = res.json().get("result", {}).get("list", [])
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume"]: df[c] = df[c].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

def analyze(t):
    sym = t["symbol"]
    df = get_klines(sym)
    if len(df) < 100: return None

    for m in [5, 20, 50]: df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma5"] = df["volume"].rolling(5).mean()
    
    last = df.iloc[-1]
    price = t["lastPrice"]

    score = 65
    if last["ma5"] > last["ma20"] > last["ma50"]: score += 15
    if t["fundingRate"] <= 0.01: score += 10
    if last["volume"] < last["vma5"]: score += 10

    score = min(int(round(score)), 98)
    if score < 90: return None

    ma20 = last["ma20"]
    status = f"눌림목 대기중 (15분봉 MA 20선: {fmt_price(ma20, price)} USDT)"
    if abs(price - ma20) / ma20 <= 0.015:
        status = f"즉시 분할 진입 구간 (MA 20선 지지 및 거래량 건조)"

    return {
        "symbol": sym, "ticker": sym.replace("USDT", ""), "score": score,
        "price": price, "chg": t["price24hPcnt"], "funding": t["fundingRate"], "status": status
    }

def main():
    send_telegram("🔔 [테스트] 바이비트 30분 자동 감시 스캐너가 성공적으로 가동되었습니다!")
    
    kst = datetime.timezone(datetime.timedelta(hours=9))
    time_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    tickers = get_bybit_tickers()
    qualified = []
    for t in tickers:
        try:
            r = analyze(t)
            if r: qualified.append(r)
        except Exception:
            continue

    if not qualified:
        print("90점 이상 포착된 종목 없음. 대기 중.")
        return

    lines = [f"🚨 [바이비트 90점 이상 실시간 포착 브리핑] ({time_str})\n"]
    for r in qualified:
        lines.append(f"■ ${r['ticker']} — [전략 점수: {r['score']}점]")
        lines.append(f"- 현황 진단: {r['status']}")
        lines.append(f"- 현재가: {fmt_price(r['price'], r['price'])} USDT ({r['chg']:+.2f}%) | 펀딩비: {r['funding']:+.4f}%")
        lines.append("")

    send_telegram("\n".join(lines))

if __name__ == "__main__":
    main()
