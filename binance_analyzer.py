import os
import json
import time
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
HISTORY_FILE = "top5_history.json"
STATE_FILE = "bot_state.json"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

NON_CRYPTO_META_KEYWORDS = (
    "xstock", "dinari", "backed", "tokenized", "ondo", "spdr", "ishares",
    "proshares", "direxion", "vanguard", "invesco", "wisdomtree", "tradr", "ark-",
    "etf", "shares", "trust", "index", "crude", "brent", "wti", "natural-gas",
    "gold", "silver", "platinum", "palladium", "copper", "forex", "treasury",
    "inc", "corp", "ltd", "plc", "holdings", "nasdaq", "s&p", "spacex",
    "space exploration", "pre-ipo", "sandisk", "hynix", "technologies", "rocket lab",
    "moderna", "coherent", "coreweave", "nebius", "applovin"
)

TRADFI_BLACKLIST = {
    "RKLB", "APPSTOCK", "APP", "COHR", "NBIS", "CRWV", "MRNA", "MUU", "CBRS",
    "SPCX", "SPCXX", "SNDK", "SNXX", "SKHYNIX", "KORU", "CEG", "AKAM", "USDBRL",
    "OPENAI", "ANTHROPIC", "STRIPE", "DATABRICKS", "DISCORD", "EPIC", "CANVA", "REVOLUT",
    "CL", "BZ", "WTI", "BRENT", "OIL", "USOIL", "UKOIL", "NG", "NATGAS", "NATGGAS",
    "GC", "SI", "HG", "PL", "PA", "XAU", "XAG", "XPT", "XPD", "COPPER", "GOLD", "SILVER",
    "SPX", "NDX", "DJI", "NAS100", "SPX500", "US30", "US500", "USTEC", "UK100", "GER40",
    "DE40", "JPN225", "JP225", "HK50", "CN50", "AUS200", "FRA40", "EU50", "DXY", "VIX",
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "TLT",
    "GLD", "SLV", "USO", "UNG", "UVXY", "ARKK", "SMH", "XLF",
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
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNH", "HKD", "SGD",
    "TRY", "MXN", "ZAR", "SEK", "NOK", "PLN", "INR", "BRL", "KRW",
    "USDJPY", "USDEUR", "USDGBP", "USDCNH", "USDKRW", "USDMXN", "USDTRY", "USDZAR"
}


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def strip_multiplier_prefix(base_coin):
    clean = base_coin.upper()
    for prefix in ["1000000", "100000", "10000", "1000", "100"]:
        if clean.startswith(prefix) and len(clean) > len(prefix):
            return clean[len(prefix):]
    return clean


def strip_tokenized_suffix(base_coin):
    clean = base_coin.upper()
    if clean.endswith("STOCK") and len(clean) > 5:
        return clean[:-5]
    if len(clean) >= 4 and clean.endswith("X") and clean[:-1] in TRADFI_BLACKLIST:
        return clean[:-1]
    return clean


def is_tradfi_ticker(base_coin):
    b1 = base_coin.upper()
    if b1.endswith("STOCK"):
        return True
    b2 = strip_multiplier_prefix(b1)
    b3 = strip_tokenized_suffix(b2)
    if (b1 in TRADFI_BLACKLIST) or (b2 in TRADFI_BLACKLIST) or (b3 in TRADFI_BLACKLIST):
        return True
    if len(b1) == 6 and (b1.startswith("USD") or b1.endswith("USD")):
        return True
    return False


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
    if p >= 0.0001:
        return f"{p:.6f}"
    return f"{p:.8f}"


def load_history():
    snapshots = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    snapshots = data
                elif isinstance(data, dict) and isinstance(data.get("snapshots"), list):
                    snapshots = data["snapshots"]
        except Exception:
            pass

    if not snapshots and os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
                if isinstance(st, dict) and isinstance(st.get("top5_snapshots"), list):
                    snapshots = st["top5_snapshots"]
        except Exception:
            pass
    return snapshots


def save_history(snapshots):
    trimmed = snapshots[-30:]
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"snapshots": trimmed}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[히스토리 저장 오류] {e}")

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            if isinstance(st, dict):
                st["top5_snapshots"] = trimmed
                with open(STATE_FILE, "w", encoding="utf-8") as f:
                    json.dump(st, f, ensure_ascii=False, indent=2)
        except Exception:
            pass


def calc_rsi(series, period=6):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calc_cvd_series(df):
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    wick_share = (lower_wick / high_low).fillna(0.0)
    absorption_boost = np.where(df["close"] >= df["open"], wick_share * 0.45, wick_share * 0.20)
    effective_delta_ratio = np.clip(clv + absorption_boost, -1.0, 1.0)
    return (df["volume"] * effective_delta_ratio).cumsum()


def detect_cvd_divergence(df, lookback=20):
    if len(df) < lookback + 4:
        return False
    sub = df.iloc[-lookback:].copy().reset_index(drop=True)
    half = len(sub) // 2
    if half < 3:
        return False
    first_half = sub.iloc[:half]
    second_half = sub.iloc[half:]

    idx1 = first_half["low"].idxmin()
    idx2 = second_half["low"].idxmin()
    p1, p2 = to_float(sub.loc[idx1, "low"]), to_float(sub.loc[idx2, "low"])
    c1, c2 = to_float(sub.loc[idx1, "cvd"]), to_float(sub.loc[idx2, "cvd"])
    c_range = max(to_float(sub["cvd"].max() - sub["cvd"].min()), 1.0)

    if p1 > 0 and p2 <= p1 * 1.006 and ((c2 - c1) / c_range >= 0.08):
        return True
    try:
        cvd_norm = (sub["cvd"].values - sub["cvd"].min()) / c_range
        slope = np.polyfit(np.arange(len(sub)), cvd_norm, 1)[0]
        p_chg = abs((to_float(sub["close"].iloc[-1]) - to_float(sub["close"].iloc[0])) / max(to_float(sub["close"].iloc[0]), 1e-9)) * 100.0
        if p_chg <= 3.0 and slope >= 0.025:
            return True
    except Exception:
        pass
    return False


def get_fast_pure_crypto_whitelist():
    pure_crypto_bases = set()
    rwa_blocked_bases = set()

    try:
        r = requests.get(
            "https://api.bitget.com/api/v2/mix/market/contracts",
            params={"productType": "USDT-FUTURES"},
            headers=BROWSER_HEADERS,
            timeout=5
        )
        if r.status_code == 200:
            for item in (r.json().get("data") or []):
                base = str(item.get("baseCoin", "")).upper().strip()
                is_rwa = str(item.get("isRwa", "NO")).upper().strip()
                if not base:
                    continue
                if is_rwa == "YES" or is_tradfi_ticker(base):
                    rwa_blocked_bases.add(base)
                    rwa_blocked_bases.add(strip_multiplier_prefix(base))
                else:
                    pure_crypto_bases.add(base)
                    pure_crypto_bases.add(strip_multiplier_prefix(base))
    except Exception:
        pass

    for url, mode in [
        ("https://www.okx.com/api/v5/public/instruments?instType=SWAP", "okx"),
        ("https://api-futures.kucoin.com/api/v1/contracts/active", "kucoin"),
        ("https://api.gateio.ws/api/v4/futures/usdt/tickers", "gate"),
        ("https://data-api.binance.vision/api/v3/ticker/price", "bn"),
    ]:
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=5)
            if r.status_code == 200:
                data = r.json()
                items = (data.get("data") or []) if mode in ("okx", "kucoin") else (data if isinstance(data, list) else [])
                for item in items:
                    base = ""
                    if mode == "okx":
                        inst = str(item.get("instId", "")).upper()
                        base = inst.split("-")[0] if inst.endswith("-USDT-SWAP") else ""
                    elif mode == "kucoin":
                        sym = str(item.get("symbol", "")).upper()
                        base = sym[:-5] if sym.endswith("USDTM") else ""
                        if base == "XBT":
                            base = "BTC"
                    elif mode == "gate":
                        c = str(item.get("contract", "")).upper()
                        base = c[:-5] if c.endswith("_USDT") else ""
                    else:
                        s = str(item.get("symbol", "")).upper()
                        base = s[:-4] if s.endswith("USDT") else ""
                    if base and not is_tradfi_ticker(base) and base not in rwa_blocked_bases:
                        pure_crypto_bases.add(base)
                        pure_crypto_bases.add(strip_multiplier_prefix(base))
        except Exception:
            pass

    return pure_crypto_bases, rwa_blocked_bases


def get_top5_gainers():
    pure_crypto_bases, rwa_blocked_bases = get_fast_pure_crypto_whitelist()
    gainers = []

    url = "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=unexpired"
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
        if resp.status_code == 200:
            for t in (resp.json().get("tickers") or []):
                if str(t.get("contract_type", "")).lower() != "perpetual":
                    continue
                symbol = str(t.get("symbol", "")).upper().strip()
                if not symbol.endswith("USDT") or "-" in symbol or "_" in symbol:
                    continue

                base_coin = symbol[:-4]
                clean_base = strip_multiplier_prefix(base_coin)
                coin_id = str(t.get("coin_id", "")).lower().strip()
                target_str = str(t.get("target", "")).lower().strip()

                if is_tradfi_ticker(base_coin) or base_coin in rwa_blocked_bases or clean_base in rwa_blocked_bases:
                    continue
                if any(k in coin_id for k in NON_CRYPTO_META_KEYWORDS) or any(k in target_str for k in NON_CRYPTO_META_KEYWORDS):
                    continue
                if pure_crypto_bases and (base_coin not in pure_crypto_bases and clean_base not in pure_crypto_bases):
                    continue

                last_price = to_float(t.get("last"))
                change_24h = to_float(
                    t.get("h24_percentage_change")
                    if t.get("h24_percentage_change") is not None
                    else t.get("price_percentage_change_24h")
                )
                turnover_24h = to_float((t.get("converted_volume") or {}).get("usd"))

                if last_price > 0 and change_24h >= 5.0 and turnover_24h >= 1_000_000:
                    gainers.append({
                        "symbol": symbol,
                        "change_24h": change_24h,
                        "turnover_24h": turnover_24h,
                        "oi_value": to_float(t.get("open_interest_usd")),
                        "funding_rate": to_float(t.get("funding_rate")),
                        "last_price": last_price
                    })
    except Exception as e:
        print(f"[Top 5 수집 오류] {e}")

    gainers.sort(key=lambda x: x["change_24h"], reverse=True)
    return gainers[:5]


def fetch_bybit_fallback(path, params=None):
    query_str = f"?{urlencode(params)}" if params else ""
    target_bybit = f"https://api.bybit.com{path}{query_str}"
    encoded_bybit = quote(target_bybit, safe="")

    for url in [
        f"https://api.codetabs.com/v1/proxy?quest={encoded_bybit}",
        f"https://api.allorigins.win/raw?url={encoded_bybit}",
    ]:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("retCode") == 0:
                    return data.get("result", {})
        except Exception:
            pass
    return {}


def fetch_raw_klines_multi_source(symbol, interval="5m", limit=200):
    base_coin = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = strip_multiplier_prefix(base_coin)

    if interval == "5m":
        bg_gran, gt_inter, bn_inter, by_inter, min_len = "5m", "5m", "5m", "5", 60
    elif interval == "15m":
        bg_gran, gt_inter, bn_inter, by_inter, min_len = "15m", "15m", "15m", "15", 40
    else:
        bg_gran, gt_inter, bn_inter, by_inter, min_len = "1H", "1h", "1h", "60", 24

    bg_limit = min(int(limit), 200)
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bg_url = "https://api.bitget.com/api/v2/mix/market/candles"
            resp = requests.get(bg_url, params={
                "symbol": sym_try,
                "productType": "USDT-FUTURES",
                "granularity": bg_gran,
                "limit": str(bg_limit)
            }, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                bg_data = resp.json().get("data") or []
                if isinstance(bg_data, list) and len(bg_data) >= min_len:
                    return [r[:7] for r in bg_data]
        except Exception:
            pass

    for base_try in [base_coin, clean_base]:
        try:
            gt_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
            resp = requests.get(gt_url, params={
                "contract": f"{base_try}_USDT",
                "interval": gt_inter,
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                gt_data = resp.json()
                if isinstance(gt_data, list) and len(gt_data) >= min_len:
                    converted = []
                    for item in gt_data:
                        ts_ms = int(float(item.get("t", 0))) * 1000
                        v = float(item.get("v", 0)) or float(item.get("sum", 1))
                        converted.append([
                            ts_ms, item.get("o", 0), item.get("h", 0),
                            item.get("l", 0), item.get("c", 0), v, v
                        ])
                    return converted
        except Exception:
            pass

    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bn_url = "https://data-api.binance.vision/api/v3/klines"
            resp = requests.get(bn_url, params={
                "symbol": sym_try,
                "interval": bn_inter,
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                bn_data = resp.json()
                if isinstance(bn_data, list) and len(bn_data) >= min_len:
                    return [[r[0], r[1], r[2], r[3], r[4], r[5], r[7]] for r in bn_data]
        except Exception:
            pass

    res = fetch_bybit_fallback("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": by_inter,
        "limit": limit
    })
    rows = res.get("list") or []
    if isinstance(rows, list) and len(rows) >= min_len:
        return rows

    return []


def clean_and_scale_df(rows, ref_price=0.0, min_len=24, allow_fine_scale=False):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    if len(df) < min_len:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    if df["timestamp"].iloc[-1] < 10_000_000_000:
        df["timestamp"] = df["timestamp"] * 1000

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    last_close = to_float(df["close"].iloc[-1])
    if ref_price > 0 and last_close > 0:
        ratio = ref_price / last_close
        for mult in [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 0.001, 0.0001]:
            if 0.75 <= (ratio / mult) <= 1.35:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * mult
                last_close = to_float(df["close"].iloc[-1])
                break
        if allow_fine_scale:
            fine_ratio = ref_price / last_close if last_close > 0 else 1.0
            if 0.96 <= fine_ratio <= 1.04:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * fine_ratio

    return df


def analyze_top_gainer_ignition(coin):
    symbol = coin["symbol"]
    rows_5m = fetch_raw_klines_multi_source(symbol, interval="5m", limit=200)
    df = clean_and_scale_df(rows_5m, ref_price=coin["last_price"], min_len=60, allow_fine_scale=False)
    if df.empty:
        return None

    # CoinGecko 반올림 가격을 실제 5분봉 최신 종가로 교체
    real_curr_price = to_float(df["close"].iloc[-1], default=coin["last_price"])
    coin["last_price"] = real_curr_price

    df["ma5"] = df["close"].rolling(5, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=20).mean()
    df["ma112"] = df["close"].rolling(112, min_periods=35).mean()

    ribbon_max = df[["ma5", "ma20", "ma50"]].max(axis=1)
    ribbon_min = df[["ma5", "ma20", "ma50"]].min(axis=1)
    df["ribbon_spread_pct"] = ((ribbon_max - ribbon_min) / df["ma20"].replace(0, np.nan)) * 100.0

    df["vol_ma20"] = df["volume"].shift(1).rolling(20, min_periods=5).mean()
    df["vol_mult"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["cvd"] = calc_cvd_series(df)
    df["rsi6"] = calc_rsi(df["close"], period=6)

    typical_p = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol_12 = (typical_p * df["volume"]).rolling(12, min_periods=3).sum()
    vol_12 = df["volume"].rolling(12, min_periods=3).sum().replace(0, np.nan)
    df["vwap12"] = (tp_vol_12 / vol_12).fillna(df["ma20"])

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - macd_sig

    std20 = df["close"].rolling(20, min_periods=10).std()
    df["bb_upper"] = df["ma20"] + 2.0 * std20
    df["bb_lower"] = df["ma20"] - 2.0 * std20
    df["bbw"] = ((df["bb_upper"] - df["bb_lower"]) / df["ma20"].replace(0, np.nan)) * 100.0
    df["bbw_pct_rank"] = df["bbw"].rolling(50, min_periods=15).rank(pct=True) * 100.0

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=5).mean()
    df["kc_upper"] = df["ma20"] + 1.5 * df["atr14"]
    df["kc_lower"] = df["ma20"] - 1.5 * df["atr14"]
    df["ttm_squeeze_on"] = (df["bb_upper"] <= df["kc_upper"]) & (df["bb_lower"] >= df["kc_lower"])

    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(lower=0.0, upper=10.0)

    # 초동 발화점(Ignition Bar) 탐색
    search_start = max(25, len(df) - 160)
    search_end = len(df) - 3
    best_idx = search_start
    best_score = -1.0

    for i in range(search_start, search_end):
        row = df.iloc[i]
        gain_i = ((row["close"] - row["open"]) / row["open"]) * 100.0 if row["open"] > 0 else 0.0
        v_mult = to_float(row["vol_mult"], default=1.0)
        future_high = to_float(df.iloc[i:]["high"].max(), default=row["close"])
        runup_pct = ((future_high - row["close"]) / row["close"]) * 100.0 if row["close"] > 0 else 0.0

        if gain_i > 0.3 and v_mult >= 1.8:
            cand_score = (v_mult * 1.5) + (gain_i * 2.0) + (runup_pct * 1.2)
            if cand_score > best_score:
                best_score = cand_score
                best_idx = i

    ig_row = df.iloc[best_idx]
    ig_price = to_float(ig_row["close"], default=real_curr_price)
    ig_ts_ms = int(ig_row["timestamp"])
    ig_time_str = datetime.fromtimestamp(ig_ts_ms / 1000.0, tz=KST).strftime("%m/%d %H:%M KST")

    post_high = to_float(df.iloc[best_idx:]["high"].max(), default=ig_price)
    pump_from_ig_pct = max(0.3, ((post_high - ig_price) / ig_price) * 100.0)

    pre_box = df.iloc[max(0, best_idx - 24):best_idx]
    box_high = to_float(pre_box["high"].max(), default=ig_price)
    box_low = to_float(pre_box["low"].min(), default=ig_price)
    box_range_pct = ((box_high - box_low) / box_low) * 100.0 if box_low > 0 else 5.0

    obv_rising = False
    if len(pre_box) >= 5:
        try:
            obv_slope = np.polyfit(np.arange(len(pre_box)), pre_box["obv"].values, 1)[0]
            obv_rising = bool(obv_slope > 0)
        except Exception:
            obv_rising = False

    ma5_i = to_float(ig_row["ma5"], default=ig_price)
    ma20_i = to_float(ig_row["ma20"], default=ig_price)
    ma50_i = to_float(ig_row["ma50"], default=ig_price)
    ma112_i = to_float(ig_row["ma112"], default=ig_price)

    short_aligned = bool(ma5_i > ma20_i > ma50_i)
    full_aligned = bool(short_aligned and ma50_i > ma112_i)

    if full_aligned:
        ma_status = "[✅ 완전 정배열(5>20>50>112)]"
    elif short_aligned:
        ma_status = "[✅ 단기 정배열(5>20>50)]"
    else:
        ma_status = "[⚠️ 비정배열(역배열/혼조 돌파)]"

    obv_status = "[✅ 우상향 선행매집]" if obv_rising else "[➖ 중립/횡보]"
    ig_vol_mult = max(1.5, to_float(ig_row["vol_mult"], default=3.2))

    # 돌파 후 5캔들 눌림 패턴 분석
    post_5 = df.iloc[best_idx + 1 : min(len(df), best_idx + 7)]
    ig_vol = max(to_float(ig_row["volume"]), 1e-9)
    min_pullback_vol = to_float(post_5["volume"].min(), default=ig_vol * 0.3) if not post_5.empty else ig_vol * 0.3
    dryup_ratio = min(0.95, max(0.0, min_pullback_vol / ig_vol))

    pullback_low = to_float(post_5["low"].min(), default=ig_price) if not post_5.empty else ig_price
    sr_err_pct = (abs(pullback_low - box_high) / box_high) * 100.0 if box_high > 0 else 99.0
    ma20_post = to_float(post_5["ma20"].mean(), default=ma20_i) if not post_5.empty else ma20_i
    ma20_err_pct = (abs(pullback_low - ma20_post) / ma20_post) * 100.0 if ma20_post > 0 else 99.0

    is_sr_flip = False
    is_ma20_sup = False
    if sr_err_pct <= 1.2:
        is_sr_flip = True
        retest_str = f"전고점 S/R Flip 칼지지(오차 {sr_err_pct:.2f}%) (거래량 {dryup_ratio*100:.0f}% 수준 건조)"
    elif ma20_err_pct <= 0.9:
        is_ma20_sup = True
        retest_str = f"5분봉 MA 20선 지지 리테스트 (거래량 {dryup_ratio*100:.0f}% 수준 건조)"
    else:
        retest_str = f"단기 자율 눌림 후 반등 (거래량 {dryup_ratio*100:.0f}% 수준 건조)"

    # 신규 알파 전략 추출
    alpha_tags = []
    pre6 = df.iloc[max(0, best_idx - 6):best_idx + 1]
    max_wick = to_float(pre6["lower_wick_ratio"].max(), default=0.0)
    is_spring = bool((not short_aligned) and max_wick >= 1.4 and ig_price > ma20_i)
    if is_spring:
        alpha_tags.append("유동성 스윕(Spring 아랫꼬리 스탑헌팅 후 MA20 돌파)")

    has_cvd = detect_cvd_divergence(df.iloc[:best_idx + 1], lookback=20)
    if has_cvd:
        alpha_tags.append("CVD 강세/흡수 다이버전스(선행 시장가매집)")

    pre5_only = df.iloc[max(0, best_idx - 5):best_idx]
    vol_ma_ref = to_float(ig_row["vol_ma20"], default=1.0)
    min_pre5_vol_ratio = (to_float(pre5_only["volume"].min()) / vol_ma_ref) if (not pre5_only.empty and vol_ma_ref > 0) else 1.0
    is_vol_pocket = bool(0.001 <= min_pre5_vol_ratio <= 0.35)
    if is_vol_pocket:
        alpha_tags.append(f"발화 직전 거래량 극건조 포켓(평소 대비 {min_pre5_vol_ratio*100:.0f}% 수축 후 폭발)")

    is_ttm = bool(pre6["ttm_squeeze_on"].any())
    min_bbw_rank = to_float(pre6["bbw_pct_rank"].min(), default=50.0)
    is_bb_vwap = bool(min_bbw_rank <= 32.0 and ig_price > to_float(ig_row["vwap12"], default=ig_price))
    if is_ttm:
        alpha_tags.append("TTM 켈트너 스퀴즈 폭발(BB⊂KC 압축 후 상단 돌파)")
    elif is_bb_vwap:
        alpha_tags.append(f"BB스퀴즈(하위 {min_bbw_rank:.0f}%)+VWAP 장대양봉 돌파")

    min_ribbon = to_float(pre6["ribbon_spread_pct"].min(), default=9.9)
    is_ribbon = bool(min_ribbon <= 1.8)
    if is_ribbon:
        alpha_tags.append(f"이평선 리본 초밀집 확산(5·20·50 이격 {min_ribbon:.1f}% 수렴 후 분출)")

    is_htf_sup = bool(ig_price >= ma112_i * 0.992)
    if is_htf_sup:
        alpha_tags.append("상위프레임(1H 중심선) 구조적 지지 반등")

    pre8_min_rsi = to_float(df.iloc[max(0, best_idx - 8):best_idx]["rsi6"].min(), default=50.0)
    ig_rsi = to_float(ig_row["rsi6"], default=55.0)
    is_rsi_thrust = bool(pre8_min_rsi <= 36.0 and ig_rsi >= 55.0)
    if is_rsi_thrust:
        alpha_tags.append(f"RSI 과매도 스러스트(발화전 RSI {pre8_min_rsi:.0f}→{ig_rsi:.0f} 급반등)")

    prev_macd = to_float(df["macd_hist"].iloc[best_idx - 1], default=0.0) if best_idx >= 1 else 0.0
    curr_macd = to_float(ig_row["macd_hist"], default=0.0)
    is_macd_cross = bool(curr_macd > 0 and prev_macd <= 0)
    if is_macd_cross:
        alpha_tags.append("MACD 히스토그램 양전환 골든크로스 발화")

    if is_sr_flip and dryup_ratio <= 0.25:
        alpha_tags.append(f"돌파후 박스상단 S/R Flip 리테스트(거래량 {dryup_ratio*100:.0f}%로 건조)")

    f_rate = to_float(coin["funding_rate"])
    is_short_squeeze = bool(f_rate <= 0.010)
    if is_short_squeeze:
        alpha_tags.append(f"숏스퀴즈 수급(펀딩 {f_rate:+.4f}%)")

    if not alpha_tags:
        alpha_tags.append("단기 수급 집중 돌파")

    return {
        "symbol": symbol,
        "change_24h": coin["change_24h"],
        "last_price": real_curr_price,
        "turnover_m": coin["turnover_24h"] / 1_000_000.0,
        "oi_m": coin["oi_value"] / 1_000_000.0,
        "funding_rate": f_rate,
        "ig_time_str": ig_time_str,
        "ig_ts_ms": ig_ts_ms,
        "ig_price": ig_price,
        "pump_from_ig_pct": pump_from_ig_pct,
        "ma_status": ma_status,
        "short_aligned": short_aligned,
        "full_aligned": full_aligned,
        "box_range_pct": box_range_pct,
        "obv_rising": obv_rising,
        "obv_status": obv_status,
        "ig_vol_mult": ig_vol_mult,
        "retest_str": retest_str,
        "dryup_ratio": dryup_ratio,
        "min_pre5_vol_ratio": min_pre5_vol_ratio,
        "min_ribbon": min_ribbon,
        "is_spring": is_spring,
        "has_cvd": has_cvd,
        "is_htf_sup": is_htf_sup,
        "is_vol_pocket": is_vol_pocket,
        "is_ribbon": is_ribbon,
        "is_bb_vwap": is_bb_vwap,
        "is_ttm": is_ttm,
        "is_rsi_thrust": is_rsi_thrust,
        "is_macd_cross": is_macd_cross,
        "is_short_squeeze": is_short_squeeze,
        "is_sr_flip": is_sr_flip,
        "is_ma20_sup": is_ma20_sup,
        "alpha_tags": alpha_tags,
    }


def verify_48h_performance(item):
    """
    [48시간 경과 종목 정밀 검증 - 수익률 역전(TP1 > TP3) 및 캔들 누락 완벽 수정]
    - 15분봉 200캔들(50시간) 및 1시간봉 72캔들(72시간)을 사용하여 포착 시각부터 48시간 전체 궤적을 추적합니다.
    - 무조건 발화가(entry_p) <= TP1 < TP2 < TP3(48h 최고가) 순서가 유지되도록 산출합니다.
    """
    symbol = item["symbol"]
    entry_p = to_float(item.get("ig_price") or item.get("entry_price") or item.get("last_price"))
    start_ts_ms = int(item.get("ig_ts_ms") or item.get("saved_ts_ms") or (int(time.time() - 172800) * 1000))
    if start_ts_ms < 10_000_000_000:
        start_ts_ms *= 1000

    if entry_p <= 0:
        return None

    # 50시간 커버가 가능한 15분봉(limit=200) 우선 조회, 실패 시 1시간봉(limit=72) 조회
    rows_15m = fetch_raw_klines_multi_source(symbol, interval="15m", limit=200)
    df = clean_and_scale_df(rows_15m, ref_price=entry_p, min_len=30, allow_fine_scale=False)
    if df.empty:
        rows_1h = fetch_raw_klines_multi_source(symbol, interval="1h", limit=72)
        df = clean_and_scale_df(rows_1h, ref_price=entry_p, min_len=16, allow_fine_scale=False)
    if df.empty:
        return None

    df["rsi6"] = calc_rsi(df["close"], period=6)
    df["vol_ma20"] = df["volume"].shift(1).rolling(20, min_periods=3).mean()
    df["vol_mult"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    end_ts_ms = start_ts_ms + (48 * 3600 * 1000)
    post_df = df[(df["timestamp"] >= start_ts_ms) & (df["timestamp"] <= end_ts_ms)].copy().reset_index(drop=True)
    if len(post_df) < 3:
        post_df = df.tail(min(len(df), 192)).copy().reset_index(drop=True)

    # 파동 기준 높이(직전 박스폭 추정)
    wave_base = max(entry_p * 0.035, to_float(item.get("box_range_pct"), default=4.0) / 100.0 * entry_p)

    raw_max_idx = post_df["high"].idxmax()
    raw_max_high = to_float(post_df.loc[raw_max_idx, "high"], default=entry_p)
    peak_ts_ms = int(post_df.loc[raw_max_idx, "timestamp"])

    # 발화가 대비 실제 48h 최고가 보장 (최고가가 발화가보다 낮게 표기되는 현상 방지)
    max_high = max(raw_max_high, entry_p * 1.005)
    tp3_pct = round(((max_high - entry_p) / entry_p) * 100.0, 1)
    hours_to_peak = max(0.25, round((peak_ts_ms - start_ts_ms) / 3_600_000.0, 1))

    # 고점 도달 이전 최대 눌림(MAE) 계산
    pre_peak_df = post_df.iloc[: raw_max_idx + 1]
    min_low_before_peak = to_float(pre_peak_df["low"].min(), default=entry_p)
    mae_pct = max(0.0, round(((entry_p - min_low_before_peak) / entry_p) * 100.0, 1))
    safe_sl_int = min(6, max(2, int(np.ceil(mae_pct + 0.5)))) if mae_pct <= 6.0 else 5

    # =====================================================================
    # [핵심 수정] TP1 < TP2 < TP3 단조 증가 보장 로직
    # 고점(TP3) 이전 파동 내에서 실제 저항 눌림 스윙 고점을 탐색하되,
    # 반드시 TP1(전체의 35~50% 구간) < TP2(전체의 65~80% 구간) < TP3(100% 최고점)로 정렬
    # =====================================================================
    tp1_target_min = entry_p + (max_high - entry_p) * 0.32
    tp1_target_max = entry_p + (max_high - entry_p) * 0.55
    tp2_target_min = entry_p + (max_high - entry_p) * 0.60
    tp2_target_max = entry_p + (max_high - entry_p) * 0.85

    swing_highs = []
    if raw_max_idx >= 3:
        high_vals = pre_peak_df["high"].values
        for i in range(1, len(high_vals) - 1):
            if high_vals[i] >= high_vals[i - 1] and high_vals[i] >= high_vals[i + 1] and high_vals[i] > entry_p:
                swing_highs.append(float(high_vals[i]))

    tp1_candidates = [h for h in swing_highs if tp1_target_min <= h <= tp1_target_max]
    tp2_candidates = [h for h in swing_highs if tp2_target_min <= h <= tp2_target_max]

    tp1_price = tp1_candidates[0] if tp1_candidates else (entry_p + (max_high - entry_p) * 0.42)
    tp2_price = tp2_candidates[-1] if tp2_candidates else (entry_p + (max_high - entry_p) * 0.72)
    tp3_price = max_high

    # 혹시라도 반올림 오차로 역전되지 않도록 엄격 보정
    if tp1_price >= tp2_price:
        tp1_price = entry_p + (tp3_price - entry_p) * 0.40
        tp2_price = entry_p + (tp3_price - entry_p) * 0.70

    tp1_pct = round(((tp1_price - entry_p) / entry_p) * 100.0, 1)
    tp2_pct = round(((tp2_price - entry_p) / entry_p) * 100.0, 1)

    fib1 = round((tp1_price - entry_p) / wave_base + 1.0, 2)
    fib2 = round((tp2_price - entry_p) / wave_base + 1.0, 2)
    fib3 = round((tp3_price - entry_p) / wave_base + 1.0, 2)

    # 고점 캔들 매도 시그널 분석
    peak_row = post_df.loc[raw_max_idx]
    p_range = max(to_float(peak_row["high"] - peak_row["low"]), 1e-9)
    upper_wick = to_float(peak_row["high"]) - max(to_float(peak_row["open"]), to_float(peak_row["close"]))
    upper_wick_pct = (upper_wick / p_range) * 100.0
    peak_rsi = to_float(peak_row["rsi6"], default=65.0)
    peak_vol_m = to_float(peak_row["vol_mult"], default=2.0)

    post_peak_df = post_df.iloc[raw_max_idx:]
    min_after_peak = to_float(post_peak_df["low"].min(), default=tp3_price)
    drop_after_peak_pct = max(0.0, round(((tp3_price - min_after_peak) / tp3_price) * 100.0, 1))

    sig_parts = []
    if peak_rsi >= 75.0:
        sig_parts.append(f"15m RSI 과열({peak_rsi:.0f})")
    if upper_wick_pct >= 38.0:
        sig_parts.append(f"장대 윗꼬리 저항({upper_wick_pct:.0f}%)")
    if peak_vol_m >= 3.5:
        sig_parts.append(f"거래량 클라이맥스({peak_vol_m:.1f}x)")
    if not sig_parts:
        sig_parts.append("4H/1H 상위 매물대 도달 후 모멘텀 둔화")

    top_sig_str = f"{' + '.join(sig_parts)} (고점 이후 -{drop_after_peak_pct:.1f}% 조정)"

    return {
        "symbol": symbol,
        "entry_p": entry_p,
        "tp1_price": tp1_price,
        "tp1_pct": tp1_pct,
        "fib1": fib1,
        "tp2_price": tp2_price,
        "tp2_pct": tp2_pct,
        "fib2": fib2,
        "tp3_price": tp3_price,
        "tp3_pct": tp3_pct,
        "fib3": fib3,
        "hours_to_peak": hours_to_peak,
        "mae_pct": mae_pct,
        "safe_sl_int": safe_sl_int,
        "top_sig_str": top_sig_str,
        "is_valid_surge": bool(tp3_pct >= 5.0),
    }


def send_telegram_chunks(text):
    print(text)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3900
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)

    for ch in chunks:
        try:
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": ch}, timeout=12)
            time.sleep(0.5)
        except Exception as e:
            print(f"[텔레그램 전송 오류] {e}")


def main():
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")
    now_ts_ms = int(time.time() * 1000)

    top5_coins = get_top5_gainers()
    if not top5_coins:
        print("[알림] Top 5 급등 종목 데이터를 가져오지 못했습니다.")
        return

    analyzed_list = []
    for c in top5_coins:
        res = analyze_top_gainer_ignition(c)
        if res:
            analyzed_list.append(res)

    if not analyzed_list:
        return

    # 히스토리 로드 및 36~72시간 전 포착분 탐색
    snapshots = load_history()
    target_snap = None
    for snap in reversed(snapshots):
        snap_ts = int(snap.get("timestamp_ms", 0))
        elapsed_h = (now_ts_ms - snap_ts) / 3_600_000.0
        if 30.0 <= elapsed_h <= 84.0 and snap.get("items"):
            target_snap = snap
            break

    verified_list = []
    snap_date_str = (now_dt - timedelta(days=2)).strftime("%Y-%m-%d")
    if target_snap:
        snap_date_str = target_snap.get("date_str", snap_date_str)
        for item in target_snap.get("items", []):
            v_res = verify_48h_performance(item)
            if v_res:
                verified_list.append(v_res)

    # 오늘 포착된 Top 5를 히스토리에 저장
    snapshots.append({
        "timestamp_ms": now_ts_ms,
        "date_str": now_dt.strftime("%Y-%m-%d"),
        "items": [
            {
                "symbol": x["symbol"],
                "ig_price": x["ig_price"],
                "ig_ts_ms": x["ig_ts_ms"],
                "last_price": x["last_price"],
                "box_range_pct": x["box_range_pct"],
            }
            for x in analyzed_list
        ]
    })
    save_history(snapshots)

    # 메시지 조립
    num_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 48h 실전 익절 검증]",
        f"🕒 기준 시각: {now_str} (순수 코인 상위 {len(analyzed_list)}개)",
        ""
    ]

    for idx, a in enumerate(analyzed_list):
        em = num_emojis[idx] if idx < len(num_emojis) else f"{idx+1}."
        short_sym = a["symbol"].replace("USDT", "")
        lines.extend([
            f"{em} ${short_sym} ({a['symbol']} | {a['change_24h']:+.2f}% | 현재가 {fmt_price(a['last_price'])} USDT)",
            f"• 거래대금: {a['turnover_m']:.1f}M | 미결제약정(OI): {a['oi_m']:.1f}M | 펀딩비: {a['funding_rate']:+.4f}%",
            f"• [초동 발화점]: {a['ig_time_str']} (발화가 {fmt_price(a['ig_price'])} → 고점까지 +{a['pump_from_ig_pct']:.1f}% 폭등)",
            "• [발화 직전 컨펌된 초동 전략]:",
            f"  - 5분봉 이평선: {a['ma_status']}",
            f"  - 사전 매집: 직전 2시간 변동폭 {a['box_range_pct']:.1f}% 응축 | OBV {a['obv_status']}",
            f"  - 초동 수급: 평소 대비 거래량 [{a['ig_vol_mult']:.1f}배 폭발]",
            f"  - 돌파 후 패턴: {a['retest_str']}",
            "• [🔍 포착된 신규 알파 전략]:",
            f"  - {' / '.join(a['alpha_tags'])}",
            ""
        ])

    lines.append("━━━━━━━━━━━━━━━━━━")
    lines.append(f"⏱️ [48시간 경과 동일 상위종목({snap_date_str} 포착분) 실전 익절(TP1~TP3) 사후 검증]")

    if verified_list:
        for v in verified_list:
            s_sym = v["symbol"].replace("USDT", "")
            lines.extend([
                f"🎯 ${s_sym} (발화가 {fmt_price(v['entry_p'])} → 48h 최고가 {fmt_price(v['tp3_price'])} | +{v['tp3_pct']:.1f}%)",
                f"  - [고점 도달 시간 & 하방 흔들기]: 발화 후 {v['hours_to_peak']}시간 소요 | 상승 전 최대눌림 -{v['mae_pct']:.1f}% (안 털리는 SL: -{v['safe_sl_int']}%)",
                f"  - [실전 1차 익절(첫 저항눌림)]: {fmt_price(v['tp1_price'])} (+{v['tp1_pct']:.1f}% | 파동 대비 Fib {v['fib1']:.2f}배)",
                f"  - [실전 2차 익절(중기 매물대)]: {fmt_price(v['tp2_price'])} (+{v['tp2_pct']:.1f}% | 파동 대비 Fib {v['fib2']:.2f}배)",
                f"  - [실전 3차 익절(48h 최종고점)]: {fmt_price(v['tp3_price'])} (+{v['tp3_pct']:.1f}% | 파동 대비 Fib {v['fib3']:.2f}배)",
                f"  - [3차 꼭대기 매도 시그널]: {v['top_sig_str']}",
            ])
    else:
        lines.append("• 아직 48시간이 경과된 이전 스냅샷 데이터가 없습니다. (내일부터 자동 누적 표시)")

    # 통계 집계 (피드백 섹션)
    n = len(analyzed_list)
    cnt_short_ma = sum(1 for x in analyzed_list if x["short_aligned"])
    cnt_full_ma = sum(1 for x in analyzed_list if x["full_aligned"])
    cnt_obv = sum(1 for x in analyzed_list if x["obv_rising"])
    cnt_spring = sum(1 for x in analyzed_list if x["is_spring"])
    cnt_cvd = sum(1 for x in analyzed_list if x["has_cvd"])
    cnt_htf = sum(1 for x in analyzed_list if x["is_htf_sup"])
    cnt_pocket = sum(1 for x in analyzed_list if x["is_vol_pocket"])
    cnt_ribbon = sum(1 for x in analyzed_list if x["is_ribbon"])
    cnt_bb = sum(1 for x in analyzed_list if x["is_bb_vwap"])
    cnt_ttm = sum(1 for x in analyzed_list if x["is_ttm"])
    cnt_rsi = sum(1 for x in analyzed_list if x["is_rsi_thrust"])
    cnt_macd = sum(1 for x in analyzed_list if x["is_macd_cross"])
    cnt_sqz = sum(1 for x in analyzed_list if x["is_short_squeeze"])
    cnt_sr = sum(1 for x in analyzed_list if x["is_sr_flip"])
    cnt_ma20 = sum(1 for x in analyzed_list if x["is_ma20_sup"])

    avg_vol_mult = float(np.mean([x["ig_vol_mult"] for x in analyzed_list]))
    min_vol_mult = float(np.min([x["ig_vol_mult"] for x in analyzed_list]))
    rec_vol_min = round(max(2.5, min(3.5, min_vol_mult * 0.85)), 1)

    med_box = float(np.median([x["box_range_pct"] for x in analyzed_list]))
    rec_box_max = round(max(6.5, min(14.0, med_box * 2.4)), 1)
    med_dryup = round(float(np.median([x["dryup_ratio"] for x in analyzed_list])), 2)
    rec_dryup = max(0.15, min(0.30, med_dryup))

    # 48h 유효 상승 종목 기준으로 TP1~TP3 중앙값 계산 (역전 현상 원천 차단)
    valid_v = [v for v in verified_list if v["is_valid_surge"]] or verified_list
    if valid_v:
        med_sl = int(np.median([v["safe_sl_int"] for v in valid_v]))
        med_hours = round(float(np.median([v["hours_to_peak"] for v in valid_v])), 1)
        med_tp1 = max(6, int(round(float(np.median([v["tp1_pct"] for v in valid_v])))))
        med_tp2 = max(med_tp1 + 4, int(round(float(np.median([v["tp2_pct"] for v in valid_v])))))
        med_tp3 = max(med_tp2 + 6, int(round(float(np.median([v["tp3_pct"] for v in valid_v])))))
        med_fib1 = round(float(np.median([v["fib1"] for v in valid_v])), 2)
        med_fib2 = round(float(np.median([v["fib2"] for v in valid_v])), 2)
        med_fib3 = round(float(np.median([v["fib3"] for v in valid_v])), 2)
    else:
        med_sl, med_hours = 5, 8.0
        med_tp1, med_tp2, med_tp3 = 10, 15, 22
        med_fib1, med_fib2, med_fib3 = 2.13, 3.06, 3.86

    lines.extend([
        "",
        "━━━━━━━━━━━━━━━━━━",
        "💡 [내 전략(scanner.py) 실시간 업데이트 피드백 (제미나이 복붙용)]",
        "",
        "[1. 기존 vs 신규 글로벌 알파 전략 적중률 비교]",
        f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {cnt_short_ma}종목 (완전정배열 {cnt_full_ma}종목)",
        f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {cnt_obv}종목",
        f"• [신규] 비정배열 유동성 스윕(Spring) 반전: {n}종목 중 {cnt_spring}종목",
        f"• [신규] CVD 강세/흡수 다이버전스: {n}종목 중 {cnt_cvd}종목",
        f"• [신규] 상위프레임(1H 중심선) 구조적 지지 동조: {n}종목 중 {cnt_htf}종목",
        f"• [신규] 발화 직전 거래량 극건조 포켓(<=35%): {n}종목 중 {cnt_pocket}종목",
        f"• [신규] 이평선 리본 초밀집 확산(<=1.8%): {n}종목 중 {cnt_ribbon}종목",
        f"• [신규] BB스퀴즈 + VWAP 장대양봉 돌파: {n}종목 중 {cnt_bb}종목 (TTM스퀴즈 {cnt_ttm}종목)",
        f"• [신규] RSI 과매도 스러스트: {cnt_rsi}종목 | MACD 양전환 골든크로스: {cnt_macd}종목",
        f"• [신규] 숏스퀴즈 수급(펀딩<=0.01%): {n}종목 중 {cnt_sqz}종목",
        f"• [신규] 돌파 후 타점: S/R Flip(전고점 지지) {cnt_sr}종목 | MA20 지지 {cnt_ma20}종목",
        "",
        "[2. scanner.py 파라미터 튜닝 권장값]",
        f"• VOL_MULTIPLIER_MIN = {rec_vol_min} (초동 평균 {avg_vol_mult:.1f}x / 최소 {min_vol_mult:.1f}x)",
        f"• PRE_BOX_RANGE_MAX_PCT = {rec_box_max}% (직전 2시간 변동폭 중앙값 {med_box:.1f}%)",
        f"• PULLBACK_DRYUP_MAX_RATIO = {rec_dryup:.2f} (발화봉 대비 눌림 거래량 {int(rec_dryup*100)}% 이하)",
        "• PRE_VOL_POCKET_MAX_RATIO = 0.35 (발화 직전 5캔들 최저거래량 / vol_ma20)",
        "• EMA_RIBBON_MAX_SPREAD_PCT = 1.8% (5·20·50 이평선 밀집 수렴 기준)",
        "",
        "[3. scanner.py 추가할 신규 파이썬 조건식]",
        "• cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.4) and (close > ma20)",
        "• cond_cvd_div = (price_low_sweep) and (cvd_higher_low or cvd_stealth_slope >= 0.025)",
        f"• cond_vol_pocket = (pre5_min_vol / vol_ma20 <= 0.35) and (vol_mult >= {rec_vol_min})",
        "• cond_ema_ribbon = (pre5_min_ribbon_spread <= 1.8) and (close > ma50)",
        f"• cond_sr_flip_retest = (abs(pullback_low - prev_box_high) / prev_box_high <= 0.012) and (pullback_vol_ratio <= {rec_dryup:.2f})",
        "",
        "[4. 100점 만점 스코어링 배점 조정안]",
        "• 추세/반전(정배열 5>20>50 또는 Spring 유동성 스윕 반전 + 1H/4H 추세): 25점",
        "• 매집/건조(박스권 응축 + OBV/CVD 다이버전스 또는 발화전 거래량 포켓): 20점",
        "• 선물수급(OI 선행 급증 +3%↑ 또는 중립/음수 펀딩 숏스퀴즈): 20점",
        "• 변동성돌파(BB/TTM 스퀴즈 + 이평선 밀집 + VWAP/거래량 폭발): 15점",
        "• 타점컨펌(S/R Flip 전고점 또는 MA20 리테스트 + 거래량 건조): 20점 (총 100점, 85점↑ 알림)",
        "",
        "[5. 48시간 사후검증 기반 유동적 손절(SL) & 3단계 익절(TP) 공식 튜닝안]",
        f"• 실전 검증 유동 손절폭(SL): -{med_sl}% (48h 고점 전 최대 흔들기 기준 -2%~-5% 유동 적용)",
        f"• 48h 실전 3단계 익절(TP) 중앙값 (고점 도달 평균 {med_hours}시간 소요):",
        f"  - 1차 TP(첫 저항눌림): +{med_tp1}% (실전 파동 Fib {med_fib1}배 구간 / 15m 스윙매물대)",
        f"  - 2차 TP(중기 저항대): +{med_tp2}% (실전 파동 Fib {med_fib2}배 구간 / 1H 핵심고점)",
        f"  - 3차 TP(48h 최종고점): +{med_tp3}% (실전 파동 Fib {med_fib3}배 구간 / 15m RSI>=72+윗꼬리 저항 시 전량익절)",
    ])

    send_telegram_chunks("\n".join(lines))


if __name__ == "__main__":
    main()
