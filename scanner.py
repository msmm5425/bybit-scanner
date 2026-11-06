import os
import json
import time
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta, timezone

# 1. 환경변수 (TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN 모두 호환)
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

# =====================================================================
# [차트 기반 2-Track 및 고점 컷오프 핵심 파라미터]
# =====================================================================
SCORE_THRESHOLD = 85               # 독립 채점 85점 이상 시 발송
PULLBACK_DRYUP_RATIO_MAX = 0.35    # 선취매 시 거래량 수축 기준 (35% 이하)
PRE_BOX_RANGE_MAX_PCT = 7.5        # 박스권 응축 허용 폭 (%)

EARLY_VOL_MULT_MIN = 1.8           # 돌파 초입 최소 거래량 배수
EARLY_VOL_MULT_MAX = 3.8           # 과도한 폭발(>3.8x)은 고점 진행형으로 차단
EARLY_BREAKOUT_MAX_PCT = 2.2       # 돌파 후 +2.2% 이내 초입만 허용

# [Hard Cutoff: 고점 추격 원천 차단 필터 (15분봉 기준)]
CUTOFF_RSI6_15M_MAX = 75.0         # 15분봉 RSI(6) 75 이상 진입 차단
CUTOFF_KDJ_J_15M_MAX = 88.0        # 15분봉 KDJ J값 88 이상 진입 차단
CUTOFF_BB_MID_EXT_MAX_PCT = 3.5    # 15분봉 BB 중심선 대비 +3.5% 이상 이격 시 차단

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

# 원유(CL), 천연가스(NG), 금(XAU/GC), 지수, 주식, ETF 등 비암호화폐 영구 배제 목록
TRADFI_BLACKLIST = {
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
    "TRY", "MXN", "ZAR", "SEK", "NOK", "PLN", "INR", "BRL", "KRW"
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


def fmt_price(price):
    """소수점 가격이 잘리지 않도록 정밀 포맷팅 (예: 0.16635, 0.008452 등)"""
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


def calc_rsi(series, period=6):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calc_kdj_j(df, period=9):
    low_min = df["low"].rolling(period, min_periods=3).min()
    high_max = df["high"].rolling(period, min_periods=3).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = (((df["close"] - low_min) / denom) * 100.0).fillna(50.0)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return j


def get_pure_crypto_whitelist():
    """
    [0단계: 순수 암호화폐 검증 필터]
    바이낸스(Binance) / 게이트아이오(Gate.io) / 비트겟(Bitget) / 코인게코 시총 상위 코인 목록에서
    순수 암호화폐 심볼 집합을 먼저 확보하여, CL(원유)·주식·외환 등 비암호화폐를 분석 전에 100% 제거
    """
    pure_bases = set()

    # 1) Binance Futures / Spot (미국 IP에서도 열리는 vision / public 미러 포함)
    for b_url in [
        "https://data-api.binance.vision/api/v3/ticker/price",
        "https://fapi.binance.com/fapi/v1/ticker/price"
    ]:
        try:
            r = requests.get(b_url, headers=BROWSER_HEADERS, timeout=6)
            if r.status_code == 200:
                for item in r.json():
                    sym = str(item.get("symbol", "")).upper()
                    if sym.endswith("USDT"):
                        base = sym[:-4]
                        pure_bases.add(base)
                        pure_bases.add(strip_multiplier_prefix(base))
        except Exception:
            pass

    # 2) Gate.io USDT Futures (순수 코인 선물 전용)
    try:
        r = requests.get(
            "https://api.gateio.ws/api/v4/futures/usdt/tickers",
            headers=BROWSER_HEADERS,
            timeout=8
        )
        if r.status_code == 200:
            for item in r.json():
                contract = str(item.get("contract", "")).upper()
                if contract.endswith("_USDT"):
                    base = contract[:-5]
                    pure_bases.add(base)
                    pure_bases.add(strip_multiplier_prefix(base))
    except Exception:
        pass

    # 3) CoinGecko Pure Crypto Market Cap List (상위 500개 순수 코인)
    for page in [1, 2]:
        try:
            r = requests.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "per_page": 250, "page": page},
                headers=BROWSER_HEADERS,
                timeout=8
            )
            if r.status_code == 200:
                for item in r.json():
                    sym = str(item.get("symbol", "")).upper()
                    if sym:
                        pure_bases.add(sym)
        except Exception:
            pass

    # 블랙리스트 티커는 화이트리스트에서 강제 제거
    pure_bases = {b for b in pure_bases if b not in TRADFI_BLACKLIST}
    return pure_bases


def get_scanner_candidates():
    """
    1단계: 순수 암호화폐 화이트리스트 확보
    2단계: 바이비트 Futures 종목 중 순수 암호화폐만 추려서 후보군 생성 (CL 등 비암호화폐 원천 차단)
    """
    pure_crypto_bases = get_pure_crypto_whitelist()
    candidates = []

    url = "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=unexpired"
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=15)
        if resp.status_code == 200:
            for t in resp.json().get("tickers", []):
                if str(t.get("contract_type", "")).lower() != "perpetual":
                    continue
                symbol = str(t.get("symbol", "")).upper().strip()
                if not symbol.endswith("USDT") or "-" in symbol or "_" in symbol:
                    continue

                base_coin = symbol[:-4]
                clean_base = strip_multiplier_prefix(base_coin)

                # 블랙리스트 차단
                if base_coin in TRADFI_BLACKLIST or clean_base in TRADFI_BLACKLIST:
                    continue

                # 순수 암호화폐 화이트리스트 교차 검증 (바이낸스/게이트/코인게코에 없는 원유 CL 등 즉시 배제)
                if pure_crypto_bases and (base_coin not in pure_crypto_bases and clean_base not in pure_crypto_bases):
                    continue

                last_price = to_float(t.get("last"))
                change_24h = to_float(
                    t.get("h24_percentage_change")
                    if t.get("h24_percentage_change") is not None
                    else t.get("price_percentage_change_24h")
                )
                turnover_24h = to_float((t.get("converted_volume") or {}).get("usd"))

                if last_price > 0 and (-3.0 <= change_24h <= 25.0) and turnover_24h >= 3_000_000:
                    candidates.append({
                        "symbol": symbol,
                        "change_24h": change_24h,
                        "turnover_24h": turnover_24h,
                        "oi_value": to_float(t.get("open_interest_usd")),
                        "funding_rate": to_float(t.get("funding_rate")),
                        "last_price": last_price
                    })

            if candidates:
                candidates.sort(
                    key=lambda x: np.log10(max(10.0, x["turnover_24h"])) * (1.0 + max(0.0, x["change_24h"]) * 0.1),
                    reverse=True
                )
                return candidates[:16]
    except Exception as e:
        print(f"[후보군 수집 예외] {e}")

    return []


def parse_bybit_payload(resp, mode):
    text = resp.text.strip()
    if not text:
        return None
    data = json.loads(text)
    if mode == "allorigins_get":
        contents = data.get("contents", "")
        if not contents:
            return None
        data = json.loads(contents)
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
        ("allorigins_get", f"https://api.allorigins.win/get?url={encoded_bybit}"),
        ("raw", f"https://api.allorigins.win/raw?url={encoded_bybit}"),
        ("raw", f"https://api.cors.lol/?url={encoded_bybit}"),
    ]

    for mode, url in routes:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=8)
            if resp.status_code == 200:
                result = parse_bybit_payload(resp, mode)
                if result is not None:
                    return result
        except Exception:
            pass
    return {}


def fetch_raw_5m_rows_multi_source(symbol, limit=150):
    base_coin = symbol[:-4].upper()
    clean_base = strip_multiplier_prefix(base_coin)

    # 1순위: Bitget USDT-FUTURES (초고속 직통)
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bg_url = "https://api.bitget.com/api/v2/mix/market/candles"
            resp = requests.get(bg_url, params={
                "symbol": sym_try,
                "productType": "USDT-FUTURES",
                "granularity": "5m",
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=7)
            if resp.status_code == 200:
                bg_data = resp.json().get("data", [])
                if bg_data and len(bg_data) >= 45:
                    return [r[:7] for r in bg_data]
        except Exception:
            pass

    # 2순위: Gate.io USDT Futures (초고속 직통)
    for base_try in [base_coin, clean_base]:
        try:
            gt_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
            resp = requests.get(gt_url, params={
                "contract": f"{base_try}_USDT",
                "interval": "5m",
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=7)
            if resp.status_code == 200:
                gt_data = resp.json()
                if isinstance(gt_data, list) and len(gt_data) >= 45:
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

    # 3순위: Binance Vision Public Data (초고속 직통)
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bn_url = "https://data-api.binance.vision/api/v3/klines"
            resp = requests.get(bn_url, params={
                "symbol": sym_try,
                "interval": "5m",
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=7)
            if resp.status_code == 200:
                bn_data = resp.json()
                if isinstance(bn_data, list) and len(bn_data) >= 45:
                    return [[r[0], r[1], r[2], r[3], r[4], r[5], r[7]] for r in bn_data]
        except Exception:
            pass

    # 4순위: Bybit V5 API (프록시)
    res = fetch_bybit("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": "5",
        "limit": limit
    })
    rows = res.get("list", [])
    if rows and len(rows) >= 45:
        return rows

    return []


def get_klines_5m_and_15m(symbol, ref_price=0.0, limit=150):
    rows = fetch_raw_5m_rows_multi_source(symbol, limit=limit)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    if len(df) < 45:
        return pd.DataFrame(), pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # [24/7 암호화폐 캔들 연속성 검증] 장 마감 공백(>15분)이 있는 비암호화폐(TradFi) 원천 배제
    max_gap = df["timestamp"].diff().max()
    if pd.notna(max_gap) and max_gap > 900_000:
        return pd.DataFrame(), pd.DataFrame()

    # [1000PEPE 등 단위 배수 자동 보정]
    last_close = to_float(df["close"].iloc[-1])
    if ref_price > 0 and last_close > 0:
        ratio = ref_price / last_close
        for mult in [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 0.001, 0.0001]:
            if 0.75 <= (ratio / mult) <= 1.35:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * mult
                break

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

    std20 = df["close"].rolling(20, min_periods=10).std()
    df["bb_upper"] = df["ma20"] + 2.0 * std20
    df["bb_lower"] = df["ma20"] - 2.0 * std20
    df["bbw"] = ((df["bb_upper"] - df["bb_lower"]) / df["ma20"].replace(0, np.nan)) * 100.0

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=5).mean()

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / candle_range).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(lower=0.0, upper=10.0)

    df["rsi6"] = calc_rsi(df["close"], period=6)
    df["kdj_j"] = calc_kdj_j(df, period=9)

    df_15m = df.groupby(df.index // 3).agg({
        "timestamp": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).reset_index(drop=True)

    df_15m["ma20"] = df_15m["close"].rolling(20, min_periods=8).mean()
    df_15m["ma50"] = df_15m["close"].rolling(50, min_periods=15).mean()
    df_15m["rsi6"] = calc_rsi(df_15m["close"], period=6)
    df_15m["kdj_j"] = calc_kdj_j(df_15m, period=9)
    df_15m["bb_mid_ext_pct"] = (
        (df_15m["close"] - df_15m["ma20"]) / df_15m["ma20"].replace(0, np.nan)
    ) * 100.0

    return df, df_15m


def analyze_symbol(coin):
    symbol = coin["symbol"]
    df, df_15m = get_klines_5m_and_15m(symbol, ref_price=coin["last_price"], limit=150)
    if df.empty or df_15m.empty or len(df) < 45:
        return None

    curr = df.iloc[-1]
    curr_15m = df_15m.iloc[-1]

    current_price = to_float(curr["close"], default=coin["last_price"])
    coin["last_price"] = current_price

    rsi6_15m = to_float(curr_15m["rsi6"], default=50.0)
    kdj_j_15m = to_float(curr_15m["kdj_j"], default=50.0)
    bb_mid_15m = to_float(curr_15m["ma20"], default=current_price)
    bb_mid_ext_15m = to_float(curr_15m["bb_mid_ext_pct"], default=0.0)

    curr_vol_mult = to_float(curr["vol_mult"], default=1.0)
    curr_candle_gain_pct = ((curr["close"] - curr["open"]) / curr["open"]) * 100.0 if curr["open"] > 0 else 0.0

    # [Hard Cutoff: 15분봉 과열 및 꼭대기 추격 원천 차단]
    if rsi6_15m >= CUTOFF_RSI6_15M_MAX:
        return None
    if kdj_j_15m >= CUTOFF_KDJ_J_15M_MAX:
        return None
    if bb_mid_ext_15m >= CUTOFF_BB_MID_EXT_MAX_PCT:
        return None
    if curr_candle_gain_pct >= 3.0 or curr_vol_mult >= 4.5:
        return None

    pre_24 = df.iloc[-26:-2]
    box_high = to_float(pre_24["high"].max(), default=current_price)
    box_low = to_float(pre_24["low"].min(), default=current_price)
    box_range_pct = ((box_high - box_low) / box_low) * 100.0 if box_low > 0 else 99.0

    obv_accum = False
    if len(pre_24) >= 6:
        try:
            obv_slope = np.polyfit(np.arange(len(pre_24)), pre_24["obv"].values, 1)[0]
            obv_accum = bool(obv_slope > 0)
        except Exception:
            obv_accum = False

    recent_low_6 = to_float(df.iloc[-6:]["low"].min(), default=current_price)
    prior_low_12 = to_float(df.iloc[-18:-6]["low"].min(), default=current_price)
    higher_low = bool(recent_low_6 > prior_low_12 * 1.001)

    ma20_5m = to_float(curr["ma20"], default=current_price)
    ma50_5m = to_float(curr["ma50"], default=current_price)
    ma112_5m = to_float(curr["ma112"], default=current_price)

    # --- Track A (1차 선취매 / 눌림 타점 채점) ---
    score_a = 0
    tags_a = []
    dist_ma50 = abs(current_price - ma50_5m) / ma50_5m * 100.0 if ma50_5m > 0 else 99.0
    dist_ma112 = abs(current_price - ma112_5m) / ma112_5m * 100.0 if ma112_5m > 0 else 99.0
    dist_bb_mid_15m = abs(current_price - bb_mid_15m) / bb_mid_15m * 100.0 if bb_mid_15m > 0 else 99.0

    if (dist_ma50 <= 1.2 or dist_ma112 <= 1.2 or dist_bb_mid_15m <= 1.0) and current_price >= min(ma50_5m, ma112_5m) * 0.995:
        score_a += 20
        tags_a.append("이평선 밀집 지지선 안착")
    if higher_low:
        score_a += 15
        tags_a.append("단기 저점 높임(Higher Low)")

    recent_3_vol_ratio = to_float(df.iloc[-3:]["volume"].min() / curr["vol_ma20"], default=1.0) if to_float(curr["vol_ma20"]) > 0 else 1.0
    if recent_3_vol_ratio <= 0.25:
        score_a += 30
        tags_a.append(f"거래량 극건조 수축({recent_3_vol_ratio*100:.0f}%)")
    elif recent_3_vol_ratio <= PULLBACK_DRYUP_RATIO_MAX:
        score_a += 24
        tags_a.append(f"거래량 수축({recent_3_vol_ratio*100:.0f}%)")
    elif recent_3_vol_ratio <= 0.50:
        score_a += 15

    max_wick_6 = to_float(df.iloc[-6:]["lower_wick_ratio"].max(), default=0.0)
    if box_range_pct <= PRE_BOX_RANGE_MAX_PCT and obv_accum:
        score_a += 20
        tags_a.append(f"박스권 응축({box_range_pct:.1f}%)+OBV 매집")
    elif max_wick_6 >= 1.5 and current_price >= ma20_5m * 0.998:
        score_a += 20
        tags_a.append("아랫꼬리 유동성 스윕(Spring 지지)")
    elif obv_accum:
        score_a += 12
        tags_a.append("OBV 우상향 매집")

    if 35.0 <= rsi6_15m <= 62.0 and -1.2 <= bb_mid_ext_15m <= 1.8:
        score_a += 15
        tags_a.append(f"15분봉 비과열 초입(RSI6 {rsi6_15m:.0f})")

    # --- Track B (2차 돌파 초입 타점 채점) ---
    score_b = 0
    tags_b = []
    swing_high_8 = to_float(df.iloc[-9:-1]["high"].max(), default=current_price)
    breakout_diff_pct = ((current_price - swing_high_8) / swing_high_8) * 100.0 if swing_high_8 > 0 else 99.0
    reclaimed_bb_mid = bool(current_price > bb_mid_15m and current_price > ma20_5m and curr["close"] > curr["open"])
    is_early_breakout = bool(-0.2 <= breakout_diff_pct <= EARLY_BREAKOUT_MAX_PCT)

    if reclaimed_bb_mid and is_early_breakout:
        score_b += 35
        tags_b.append(f"BB중심선+스윙고점({fmt_price(swing_high_8)}) 돌파 초입")
    elif reclaimed_bb_mid and 0.0 <= bb_mid_ext_15m <= 2.0:
        score_b += 24
        tags_b.append("15분봉 BB중심선(MA20) 상향 장악")

    max_vol_mult_2 = max(curr_vol_mult, to_float(df.iloc[-2]["vol_mult"], default=1.0))
    if EARLY_VOL_MULT_MIN <= max_vol_mult_2 <= EARLY_VOL_MULT_MAX and to_float(curr["body_ratio"]) >= 0.50:
        score_b += 25
        tags_b.append(f"초동 수급 유입({max_vol_mult_2:.1f}배)")

    pre5_min_vol_ratio = (
        to_float(df.iloc[-7:-2]["volume"].min() / curr["vol_ma20"], default=1.0)
        if to_float(curr["vol_ma20"]) > 0 else 1.0
    )
    if pre5_min_vol_ratio <= 0.35:
        score_b += 25
        tags_b.append(f"발화직전 거래량 건조 포켓({pre5_min_vol_ratio*100:.0f}%)")

    if rsi6_15m <= 68.0 and bb_mid_ext_15m <= 2.8:
        score_b += 15
        tags_b.append(f"추격 안전권(15m RSI6 {rsi6_15m:.0f})")

    if score_a < SCORE_THRESHOLD and score_b < SCORE_THRESHOLD:
        return None

    # 1차 지지선(현재가 아래)과 2차 돌파선(현재가 위 최소 +1.2% 이상 의미 있는 스윙 저항선) 명확히 분리
    support_line = min(recent_low_6, ma50_5m, current_price * 0.994)
    resistance_line = max(box_high, swing_high_8, current_price * 1.015)

    if score_a >= score_b:
        track_type = "TRACK_A"
        track_title = "🟢 [1차 선취매/눌림 타점 포착]"
        final_score = int(min(100, score_a))
        matched_tags = tags_a
        entry_low = current_price * 0.994
        entry_high = current_price * 1.002
    else:
        track_type = "TRACK_B"
        track_title = "🚀 [2차 돌파 초입 타점 포착]"
        final_score = int(min(100, score_b))
        matched_tags = tags_b
        entry_low = current_price * 0.995
        entry_high = current_price * 1.003

    avg_entry = (entry_low + entry_high) / 2.0

    atr5 = to_float(curr["atr14"], default=avg_entry * 0.012)
    atr15_approx = atr5 * 1.732
    raw_sl_pct = (1.4 * atr15_approx / avg_entry) * 100.0 if avg_entry > 0 else 3.0
    sl_pct = max(2, min(5, int(round(raw_sl_pct))))
    stop_price = avg_entry * (1.0 - sl_pct / 100.0)

    tp1_pct = max(sl_pct * 2, 5)
    tp2_pct = max(int(round(tp1_pct * 2.0)), 10)
    tp3_pct = max(int(round(tp1_pct * 3.2)), 16)
    tp4_pct = max(int(round(tp1_pct * 4.8)), 24)

    tps = [
        avg_entry * (1.0 + tp1_pct / 100.0),
        avg_entry * (1.0 + tp2_pct / 100.0),
        avg_entry * (1.0 + tp3_pct / 100.0),
        avg_entry * (1.0 + tp4_pct / 100.0),
    ]

    return {
        "symbol": symbol,
        "track_type": track_type,
        "track_title": track_title,
        "score": final_score,
        "change_24h": coin["change_24h"],
        "last_price": current_price,
        "turnover_m": coin["turnover_24h"] / 1_000_000.0,
        "oi_m": coin["oi_value"] / 1_000_000.0,
        "funding_rate": coin["funding_rate"],
        "rsi6_15m": rsi6_15m,
        "kdj_j_15m": kdj_j_15m,
        "bb_mid_ext_15m": bb_mid_ext_15m,
        "swing_high_line": resistance_line,
        "support_line": support_line,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_price": stop_price,
        "sl_pct": sl_pct,
        "tp_pcts": [tp1_pct, tp2_pct, tp3_pct, tp4_pct],
        "tps": tps,
        "matched_tags": matched_tags
    }


def build_briefing_message(res, now_str):
    """1차/2차 상세 브리핑 메시지 (단독 발송용)"""
    short_sym = res["symbol"].replace("USDT", "")
    return (
        f"{res['track_title']} ${short_sym} ({res['symbol']} | {res['score']}점)\n"
        f"🕒 포착 시각: {now_str}\n"
        f"• 현재가: {fmt_price(res['last_price'])} USDT (24h {res['change_24h']:+.2f}%)\n"
        f"• 거래대금: {res['turnover_m']:.1f}M | OI: {res['oi_m']:.1f}M | 펀딩비: {res['funding_rate']:+.4f}%\n"
        f"• [과열 필터 통과]: 15m RSI6 {res['rsi6_15m']:.1f} (<75) | J값 {res['kdj_j_15m']:.1f} (<88) | BB중심선 이격 {res['bb_mid_ext_15m']:+.2f}%\n"
        f"• [포착 근거]: {' / '.join(res['matched_tags'])}\n"
        f"• [핵심 기준선 & 타점 가이드]:\n"
        f"  - 1차 지지선(눌림 기준): {fmt_price(res['support_line'])} USDT\n"
        f"  - 2차 돌파선(스윙 고점): {fmt_price(res['swing_high_line'])} USDT\n"
        f"  - 권장 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n"
        f"  - 손절가(SL): {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%)\n"
        f"  - 4단계 익절(TP): 1차 {fmt_price(res['tps'][0])}(+{res['tp_pcts'][0]}%) / "
        f"2차 {fmt_price(res['tps'][1])}(+{res['tp_pcts'][1]}%) / "
        f"3차 {fmt_price(res['tps'][2])}(+{res['tp_pcts'][2]}%) / "
        f"4차 {fmt_price(res['tps'][3])}(+{res['tp_pcts'][3]}%)"
    )


def build_x_post_message(res):
    """요청하신 X(트위터) 양식 전용 메시지 (종목별 1개씩 별도 발송용)"""
    short_sym = res["symbol"].replace("USDT", "")
    return (
        f"${short_sym} ({res['symbol']})\n\n"
        f"현재가: {fmt_price(res['last_price'])} USDT\n\n"
        f"- 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n\n"
        f"- SL: {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%)\n\n"
        f"- 1차 {fmt_price(res['tps'][0])}(+{res['tp_pcts'][0]}%)\n"
        f"- 2차 {fmt_price(res['tps'][1])}(+{res['tp_pcts'][1]}%)\n"
        f"- 3차 {fmt_price(res['tps'][2])}(+{res['tp_pcts'][2]}%)\n"
        f"- 4차 {fmt_price(res['tps'][3])}(+{res['tp_pcts'][3]}%)"
    )


def send_telegram_single(text):
    """절대 합쳐서 보내지 않고 1건씩 독립된 텔레그램 메시지로 개별 전송"""
    print(text)
    print("--------------------------------------------------")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        time.sleep(0.6)
    except Exception as e:
        print(f"[Telegram 전송 오류] {e}")


def main():
    state = load_state()
    tracked = state.get("tracked", {})
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    # 18시간 경과 종목 정리 및 비암호화폐(CL 등) 기존 추적 기록 즉시 삭제
    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        if base_c in TRADFI_BLACKLIST or (now_ts - tracked[sym].get("created_at", 0) > 64800):
            del tracked[sym]

    # 1단계: 바이낸스/게이트/코인게코 교차검증으로 바이비트 Futures 순수 암호화폐만 추림
    candidates = get_scanner_candidates()
    if not candidates:
        print("[알림] 조건을 만족하는 순수 암호화폐 후보군 데이터를 가져오지 못했습니다.")
        save_state(state)
        return

    sent_count = 0

    # 2단계: 추려진 순수 암호화폐만 분석 및 개별 알림 전송
    for coin in candidates:
        symbol = coin["symbol"]
        curr_p = coin["last_price"]

        # 기존 포착 종목의 손절 이탈 / 2차 돌파선 장악 개별 알림 (합치지 않고 즉시 단독 발송)
        if symbol in tracked:
            t_info = tracked[symbol]
            if not t_info.get("sl_alerted") and curr_p <= t_info["stop_price"]:
                t_info["sl_alerted"] = True
                sl_msg = (
                    f"⚠️ [손절 기준가 이탈] ${symbol.replace('USDT', '')} ({symbol})\n"
                    f"• 현재가: {fmt_price(curr_p)} USDT (손절선 {fmt_price(t_info['stop_price'])} 이탈)\n"
                    f"• 리스크 관리 원칙 준수를 권장합니다."
                )
                send_telegram_single(sl_msg)
                sent_count += 1
                continue

            if (
                t_info.get("track_type") == "TRACK_A"
                and not t_info.get("breakout_alerted")
                and curr_p >= t_info.get("swing_high_line", curr_p * 1.02)
            ):
                t_info["breakout_alerted"] = True
                bo_msg = (
                    f"🔥 [선취매 종목 2차 돌파선 장악] ${symbol.replace('USDT', '')} ({symbol})\n"
                    f"• 현재가: {fmt_price(curr_p)} USDT (직전 저항 {fmt_price(t_info['swing_high_line'])} 돌파)\n"
                    f"• 1차 목표가(1:2 R:R): {fmt_price(t_info['tps'][0])} (+{t_info['tp_pcts'][0]}%)"
                )
                send_telegram_single(bo_msg)
                sent_count += 1
                continue

        try:
            res = analyze_symbol(coin)
            if not res:
                continue

            prev_track = tracked.get(symbol, {}).get("track_type")
            if symbol in tracked and prev_track == res["track_type"]:
                continue

            tracked[symbol] = {
                "created_at": now_ts,
                "track_type": res["track_type"],
                "swing_high_line": res["swing_high_line"],
                "entry_low": res["entry_low"],
                "entry_high": res["entry_high"],
                "stop_price": res["stop_price"],
                "sl_pct": res["sl_pct"],
                "tps": res["tps"],
                "tp_pcts": res["tp_pcts"],
                "breakout_alerted": (res["track_type"] == "TRACK_B"),
                "sl_alerted": False
            }

            # 1) 상세 브리핑 단독 전송
            briefing_msg = build_briefing_message(res, now_str)
            send_telegram_single(briefing_msg)

            # 2) X(트위터) 양식 메시지 별도 단독 전송 (종목별 1개씩 분리)
            x_msg = build_x_post_message(res)
            send_telegram_single(x_msg)

            sent_count += 1
        except Exception as e:
            print(f"[스캔 예외] {symbol}: {e}")

    state["tracked"] = tracked
    save_state(state)

    if sent_count == 0:
        print(f"[{now_str}] 고점 컷오프 통과 및 85점 이상 신규 초입 종목 없음.")


if __name__ == "__main__":
    main()
