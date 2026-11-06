import os
import json
import time
import base64
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta, timezone

TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
    or ""
).strip()

TELEGRAM_CHAT_ID = (
    os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("CHAT_ID")
    or ""
).strip()

KST = timezone(timedelta(hours=9))
STATE_FILE = "bot_state.json"

# 오늘 발굴된 신규 전략 기반 튜닝 파라미터
SCORE_THRESHOLD = 85
VOL_MULTIPLIER_MIN = 3.5
PRE_BOX_RANGE_MAX_PCT = 8.7
PULLBACK_DRYUP_MAX_RATIO = 0.15
PRE_VOL_POCKET_MAX_RATIO = 0.35
EMA_RIBBON_MAX_SPREAD_PCT = 1.8
FUNDING_NEUTRAL_MAX = 0.0100

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

TRADFI_BLACKLIST = {
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "TLT",
    "GLD", "SLV", "USO", "UNG", "UVXY", "VIX", "ARKK", "SMH", "XLF",
    "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "GDX", "HYG",
    "EEM", "FXI", "KWEB", "IBIT", "ETHA", "BITO", "MSTX", "MSTU", "NVDL",
    "NVDU", "TSLL", "TSLS", "CONL", "AMDL",
    "AAPL", "GOOGL", "GOOG", "MSFT", "NVDA", "AMZN", "META", "TSLA",
    "NFLX", "AMD", "INTC", "AVGO", "TSM", "ASML", "ARM", "SMCI", "MU",
    "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "SNPS", "CDNS",
    "PLTR", "COIN", "MSTR", "HOOD", "MARA", "RIOT", "CLSK", "IREN",
    "CORZ", "HUT", "WULF", "BITF", "CRCL", "UNITREE", "BABA", "JD",
    "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME", "TCEHY", "BYDDY",
    "XIAOMI", "MEITUAN", "BRK", "JPM", "BAC", "WFC", "C", "GS", "MS",
    "V", "MA", "AXP", "PYPL", "SQ", "SHOP", "WMT", "COST", "TGT", "HD",
    "LOW", "DIS", "CMCSA", "BA", "LMT", "RTX", "GE", "CAT", "DE", "XOM",
    "CVX", "COP", "OXY", "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV",
    "AMGN", "GILD", "ISRG", "PG", "KO", "PEP", "MCD", "SBUX", "NKE",
    "LULU", "UBER", "LYFT", "ABNB", "BKNG", "DASH", "SNOW", "CRM",
    "ORCL", "IBM", "CSCO", "ADBE", "NOW", "INTU", "PANW", "CRWD", "FTNT",
    "NET", "DDOG", "ZS", "MDB", "TEAM", "U", "RBLX", "RDDT",
    "SPOT", "SNAP", "PINS", "DJT", "GME", "AMC", "CVNA", "RIVN", "LCID",
    "XAU", "XAG", "XPT", "XPD", "WTI", "BRENT", "NATGGAS", "COPPER",
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNH", "HKD", "SGD"
}


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def fmt_price(price):
    p = to_float(price)
    if p <= 0:
        return "0"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 10:
        return f"{p:.3f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    return f"{p:.6f}"


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {"tracked": {}}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[상태 저장 오류] {e}")


def parse_bybit_payload(resp, mode):
    text = resp.text.strip()
    if not text:
        return None
    data = json.loads(text)
    if mode == "allorigins_get":
        contents = data.get("contents", "")
        if not contents:
            return None
        if isinstance(contents, str) and contents.startswith("data:") and "base64," in contents:
            b64_part = contents.split("base64,", 1)[1]
            contents = base64.b64decode(b64_part).decode("utf-8", errors="ignore")
        data = json.loads(contents) if isinstance(contents, str) else contents
    elif isinstance(data, str):
        data = json.loads(data)

    if isinstance(data, dict) and data.get("retCode") == 0:
        result = data.get("result", {})
        if isinstance(result, dict) and result.get("list"):
            return result
    return None


def fetch_bybit(path, params=None):
    query_str = f"?{urlencode(params)}" if params else ""
    target_bybit = f"https://api.bybit.com{path}{query_str}"
    encoded_bybit = quote(target_bybit, safe="")

    routes = [
        ("raw", f"https://api.codetabs.com/v1/proxy?quest={encoded_bybit}"),
        ("raw", f"https://api.codetabs.com/v1/proxy?quest={target_bybit}"),
        ("allorigins_get", f"https://api.allorigins.win/get?url={encoded_bybit}"),
        ("raw", f"https://api.allorigins.win/raw?url={encoded_bybit}"),
        ("raw", f"https://thingproxy.freeboard.io/fetch/{target_bybit}"),
        ("raw", f"https://api.bytick.com{path}{query_str}"),
        ("raw", target_bybit),
    ]

    for mode, url in routes:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
            if resp.status_code == 200:
                result = parse_bybit_payload(resp, mode)
                if result is not None:
                    time.sleep(1.0)
                    return result
        except Exception:
            pass
        time.sleep(0.3)

    return {}


def get_scanner_candidates():
    res = fetch_bybit("/v5/market/tickers", {"category": "linear"})
    tickers = res.get("list", [])
    candidates = []

    for t in tickers:
        symbol = str(t.get("symbol", "")).upper()
        if not symbol.endswith("USDT") or "-" in symbol:
            continue
        if symbol[:-4] in TRADFI_BLACKLIST:
            continue

        last_price = to_float(t.get("lastPrice"))
        turnover_24h = to_float(t.get("turnover24h"))
        change_24h = to_float(t.get("price24hPcnt")) * 100.0
        oi_val = to_float(t.get("openInterestValue"))
        funding_rate = to_float(t.get("fundingRate")) * 100.0

        if last_price > 0 and turnover_24h >= 3_000_000 and (1.5 <= change_24h <= 42.0):
            candidates.append({
                "symbol": symbol,
                "change_24h": change_24h,
                "turnover_24h": turnover_24h,
                "oi_value": oi_val,
                "funding_rate": funding_rate,
                "last_price": last_price
            })

    if candidates:
        candidates.sort(key=lambda x: x["change_24h"] * np.log10(max(10.0, x["turnover_24h"])), reverse=True)
        return candidates[:15]

    url = "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=unexpired"
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
        if resp.status_code == 200:
            for t in resp.json().get("tickers", []):
                if str(t.get("contract_type", "")).lower() != "perpetual":
                    continue
                symbol = str(t.get("symbol", "")).upper()
                if not symbol.endswith("USDT") or "-" in symbol or "_" in symbol:
                    continue
                if symbol[:-4] in TRADFI_BLACKLIST:
                    continue
                last_price = to_float(t.get("last"))
                change_24h = to_float(
                    t.get("h24_percentage_change")
                    if t.get("h24_percentage_change") is not None
                    else t.get("price_percentage_change_24h")
                )
                turnover_24h = to_float((t.get("converted_volume") or {}).get("usd"))
                if last_price > 0 and 1.5 <= change_24h <= 42.0 and turnover_24h >= 2_000_000:
                    candidates.append({
                        "symbol": symbol,
                        "change_24h": change_24h,
                        "turnover_24h": turnover_24h,
                        "oi_value": to_float(t.get("open_interest_usd")),
                        "funding_rate": to_float(t.get("funding_rate")),
                        "last_price": last_price
                    })
            candidates.sort(key=lambda x: x["change_24h"], reverse=True)
            return candidates[:15]
    except Exception:
        pass

    return []


def fetch_raw_5m_rows_with_fallback(symbol):
    res = fetch_bybit("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": "5",
        "limit": 200
    })
    rows = res.get("list", [])
    if rows and len(rows) >= 50:
        return [r[:7] for r in rows]

    base_coin = symbol
