import os
import json
import time
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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
# [11/08 14:17 KST Top 5 역추적 100% 적중 공식 반영 핵심 파라미터]
# =====================================================================
SCORE_THRESHOLD = 85                   # 85점 이상 시 알림 발송
PULLBACK_DRYUP_STRICT = 0.18           # 14:17 피드백 반영: 눌림 거래량 18% 이하 극건조 (5종목 전원 3%~18% 적중)
PULLBACK_DRYUP_MAX = 0.35              # 선취매 거래량 수축 허용 상한 (35% 이하)
SR_FLIP_DRYUP_RATIO_MAX = 0.25         # 돌파 후 S/R Flip 및 MA20/50 리테스트 거래량 상한 (18% 기준 + 버퍼)
SR_FLIP_TOLERANCE_PCT = 0.85           # 전고점 S/R Flip 칼지지 허용 오차 (BMT 0.65%, BLEND 0.65%, ERA 0.24% 포섭)

PRE_BOX_RANGE_MAX_PCT = 12.0           # 직전 2시간 박스권 응축 허용 상한
PRE_BOX_TIGHT_PCT = 7.5                # 14:17 피드백(최대 7.4%, 권장 7.2%) 반영: 7.5% 이내 정밀 응축 만점
EMA_RIBBON_MAX_SPREAD_PCT = 1.8        # 5·20·50 이평선 리본 초밀집 수렴 기준 (1.8% 이내)
PRE_VOL_POCKET_MAX_RATIO = 0.35        # 발화 직전 5캔들 최저거래량 / vol_ma20 (5종목 전원 6%~35% 100% 적중!)

EARLY_VOL_MULT_MIN = 3.2               # 14:17 피드백 권장값(3.2x) 반영: 초동 거래량 폭발 최소 배수
EARLY_VOL_MULT_MAX = 6.0               # 직접 돌파 진입 시 상한 (초과 시 S/R Flip 또는 MA20/50 눌림에서 포착)
EARLY_BREAKOUT_MAX_PCT = 2.5           # 돌파 후 +2.5% 이내 초입만 허용

# [고점 폭락(설거지 차트) 및 상위 프레임(15m/1H) 과열 원천 차단 필터]
MAX_DRAWDOWN_FROM_12H_HIGH_PCT = 7.5   # 최근 12시간 고점 대비 -7.5% 이상 하락한 설거지 종목 즉시 차단
MAX_PRIOR_PUMP_PCT = 20.0              # 최근 12시간 내 이미 +20% 이상 폭등 이력 시 설거지 경계
MAX_RETRACEMENT_RATIO = 0.35           # 폭등 후 상승폭의 35% 이상 반납한 역V자 하락 종목 즉시 차단

CUTOFF_RSI6_15M_MAX = 75.0             # 15분봉 RSI(6) 75 이상 진입 차단
CUTOFF_KDJ_J_15M_MAX = 88.0            # 15분봉 KDJ J값 88 이상 진입 차단
CUTOFF_BB_MID_EXT_MAX_PCT = 3.8        # 15분봉 BB 중심선 대비 +3.8% 이상 이격 시 차단
CUTOFF_RSI6_1H_MAX = 78.0              # 1시간봉 RSI(6) 78 이상 장기 과열 차단

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
    "space exploration", "pre-ipo", "sandisk", "hynix", "technologies"
)

TRADFI_BLACKLIST = {
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
    if len(clean) >= 4 and clean.endswith("X") and clean[:-1] in TRADFI_BLACKLIST:
        return clean[:-1]
    return clean


def is_tradfi_ticker(base_coin):
    b1 = base_coin.upper()
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


def fmt_pct(pct_val):
    v = round(to_float(pct_val), 1)
    return f"{int(v)}" if v == int(v) else f"{v:.1f}"


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


def calc_cvd_series(df):
    """캔들 내 매수/매도 압력(CLV)과 아랫꼬리 스탑헌팅 흡수를 결합한 누적 볼륨 델타(CVD)"""
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low).fillna(0.0)

    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    wick_share = (lower_wick / high_low).fillna(0.0)
    absorption_boost = np.where(df["close"] >= df["open"], wick_share * 0.45, wick_share * 0.20)

    effective_delta_ratio = np.clip(clv + absorption_boost, -1.0, 1.0)
    vol_delta = df["volume"] * effective_delta_ratio
    return vol_delta.cumsum()


def detect_cvd_divergence(df, lookback=24):
    """클래식 강세 CVD 다이버전스(저점 스윕 매수흡수) 및 박스권 스텔스 CVD 매집 감지"""
    if len(df) < lookback + 4:
        return False, ""

    sub = df.iloc[-(lookback + 1):].copy().reset_index(drop=True)
    half = len(sub) // 2

    first_half = sub.iloc[:half]
    second_half = sub.iloc[half:]

    idx_low1 = first_half["low"].idxmin()
    idx_low2 = second_half["low"].idxmin()

    p_low1 = to_float(sub.loc[idx_low1, "low"])
    p_low2 = to_float(sub.loc[idx_low2, "low"])
    cvd_low1 = to_float(sub.loc[idx_low1, "cvd"])
    cvd_low2 = to_float(sub.loc[idx_low2, "cvd"])

    cvd_range = to_float(sub["cvd"].max() - sub["cvd"].min(), default=1.0)
    if cvd_range <= 0:
        cvd_range = 1.0

    if p_low1 > 0 and (p_low2 <= p_low1 * 1.005) and ((cvd_low2 - cvd_low1) / cvd_range >= 0.08):
        if to_float(sub["close"].iloc[-1]) > p_low2 * 1.002:
            return True, "CVD 강세 다이버전스(저점 스윕 매수흡수)"

    price_chg_pct = ((to_float(sub["close"].iloc[-1]) - to_float(sub["close"].iloc[0])) / max(to_float(sub["close"].iloc[0]), 1e-9)) * 100.0
    try:
        x = np.arange(len(sub))
        cvd_norm = (sub["cvd"].values - sub["cvd"].min()) / cvd_range
        cvd_slope = np.polyfit(x, cvd_norm, 1)[0]
        if (-2.5 <= price_chg_pct <= 2.0) and cvd_slope >= 0.025 and to_float(sub["cvd"].iloc[-1]) >= to_float(sub["cvd"].quantile(0.75)):
            return True, "CVD 흡수 다이버전스(선행 시장가매집)"
    except Exception:
        pass

    return False, ""


def get_fast_pure_crypto_whitelist():
    """미국 IP에서 즉시 열리는 5대 거래소 직통 API로 순수 코인 화이트리스트 및 RWA 차단 목록 확보"""
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
            for item in r.json().get("data", []):
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
                items = data.get("data", []) if mode in ("okx", "kucoin") else data
                if not isinstance(items, list):
                    continue
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


def get_scanner_candidates():
    pure_crypto_bases, rwa_blocked_bases = get_fast_pure_crypto_whitelist()
    candidates = []

    url = "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=unexpired"
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
        if resp.status_code == 200:
            for t in resp.json().get("tickers", []):
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
                    if not coin_id:
                        continue

                last_price = to_float(t.get("last"))
                change_24h = to_float(
                    t.get("h24_percentage_change")
                    if t.get("h24_percentage_change") is not None
                    else t.get("price_percentage_change_24h")
                )
                turnover_24h = to_float((t.get("converted_volume") or {}).get("usd"))

                if last_price > 0 and (-8.0 <= change_24h <= 34.0) and turnover_24h >= 1_200_000:
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
                    key=lambda x: np.log10(max(10.0, x["turnover_24h"])) * (1.0 + max(0.0, x["change_24h"]) * 0.08),
                    reverse=True
                )
                return candidates[:28], pure_crypto_bases, rwa_blocked_bases
    except Exception as e:
        print(f"[후보군 수집 예외] {e}")

    return [], pure_crypto_bases, rwa_blocked_bases


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
    base_coin = symbol[:-4].upper()
    clean_base = strip_multiplier_prefix(base_coin)

    bg_gran = "5m" if interval == "5m" else "1H"
    gt_inter = "5m" if interval == "5m" else "1h"
    bn_inter = "5m" if interval == "5m" else "1h"
    by_inter = "5" if interval == "5m" else "60"
    min_len = 60 if interval == "5m" else 30

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
                bg_data = resp.json().get("data", [])
                if bg_data and len(bg_data) >= min_len:
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
    rows = res.get("list", [])
    if rows and len(rows) >= min_len:
        return rows

    return []


def clean_and_scale_df(rows, ref_price=0.0, min_len=30):
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
        fine_ratio = ref_price / last_close if last_close > 0 else 1.0
        if 0.96 <= fine_ratio <= 1.04:
            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * fine_ratio

    return df


def get_multi_tf_data(symbol, ref_price=0.0):
    rows_5m = fetch_raw_klines_multi_source(symbol, interval="5m", limit=200)
    df = clean_and_scale_df(rows_5m, ref_price=ref_price, min_len=60)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    max_gap = df["timestamp"].diff().max()
    if pd.notna(max_gap) and max_gap > 900_000:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    recent_check = df.tail(100)
    dead_candles = ((recent_check["high"] <= recent_check["low"]) | (recent_check["volume"] <= 0)).sum()
    if dead_candles >= 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

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

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd_line - macd_sig

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

    df["kc_upper"] = df["ma20"] + 1.5 * df["atr14"]
    df["kc_lower"] = df["ma20"] - 1.5 * df["atr14"]
    df["ttm_squeeze_on"] = (df["bb_upper"] <= df["kc_upper"]) & (df["bb_lower"] >= df["kc_lower"])

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / candle_range).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(lower=0.0, upper=10.0)

    df["rsi6"] = calc_rsi(df["close"], period=6)
    df["rsi14"] = calc_rsi(df["close"], period=14)
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
    df_15m["cvd"] = calc_cvd_series(df_15m)
    df_15m["bb_mid_ext_pct"] = (
        (df_15m["close"] - df_15m["ma20"]) / df_15m["ma20"].replace(0, np.nan)
    ) * 100.0

    curr_p = to_float(df["close"].iloc[-1], default=ref_price)
    rows_1h = fetch_raw_klines_multi_source(symbol, interval="1h", limit=120)
    df_1h = clean_and_scale_df(rows_1h, ref_price=curr_p, min_len=24)
    if df_1h.empty:
        df_1h = df.groupby(df.index // 12).agg({
            "timestamp": "last",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).reset_index(drop=True)

    df_1h["ma20"] = df_1h["close"].rolling(20, min_periods=5).mean()
    df_1h["ma50"] = df_1h["close"].rolling(50, min_periods=10).mean()
    df_1h["ma99"] = df_1h["close"].rolling(99, min_periods=15).mean()
    std20_1h = df_1h["close"].rolling(20, min_periods=5).std()
    df_1h["bb_upper"] = df_1h["ma20"] + 2.0 * std20_1h
    df_1h["rsi6"] = calc_rsi(df_1h["close"], period=6)
    df_1h["cvd"] = calc_cvd_series(df_1h)

    df_4h = df_1h.groupby(df_1h.index // 4).agg({
        "timestamp": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).reset_index(drop=True)
    df_4h["ma20"] = df_4h["close"].rolling(20, min_periods=3).mean()
    std20_4h = df_4h["close"].rolling(20, min_periods=3).std()
    df_4h["bb_upper"] = df_4h["ma20"] + 2.0 * std20_4h

    return df, df_15m, df_1h, df_4h


def extract_pivot_highs(df, left=3, right=3):
    highs = df["high"].values
    pivots = []
    n = len(highs)
    for i in range(left, n - right):
        window = highs[i - left : i + right + 1]
        if highs[i] == np.max(window) and highs[i] > 0:
            pivots.append(float(highs[i]))
    return pivots


def calc_dynamic_structural_sl(df_5m, df_15m, avg_entry, box_range_pct):
    """
    [정밀 교정 완료: -2% ~ -5% 황금 밸런스 유동적 손절(SL) 산출 엔진]
    장대양봉 꼭대기가 아닌 '실제 리테스트/눌림 진입가(avg_entry)'에서
    최근 45분 스윙 저점(또는 MA50 지지선)까지의 거리 + 1.0 * ATR(5m) 스탑헌팅 방어 버퍼를 계산합니다.
    - 초응축 코인(박스 2.5~4.5%): -2% SL
    - 중간 변동성 코인(박스 4.5~6.5%): -3% SL
    - 고변동성 코인(박스 6.5% 이상): -4% ~ -5% SL
    """
    atr5 = to_float(df_5m["atr14"].iloc[-1], default=avg_entry * 0.008)
    swing_low_45m = to_float(df_5m.iloc[-9:]["low"].min(), default=avg_entry * 0.98)
    ma50_5m = to_float(df_5m["ma50"].iloc[-1], default=swing_low_45m)

    # 진입가 바로 아래의 실전 지지선(45분 스윙저점과 5분봉 MA50 중 가까운 유효 지지선)
    structural_support = min(swing_low_45m, ma50_5m) if ma50_5m < avg_entry else swing_low_45m
    if (avg_entry - structural_support) / avg_entry > 0.045:
        structural_support = swing_low_45m

    # 실전 지지선 아래에 1.0 * ATR(5m) 버퍼 부여
    struct_dist_pct = ((avg_entry - (structural_support - 1.0 * atr5)) / avg_entry) * 100.0 if avg_entry > 0 else 2.5
    box_scaled_pct = box_range_pct * 0.42  # 예: 3% 박스 -> 1.3%(->2%), 6.5% 박스 -> 2.7%(->3%), 9% 박스 -> 3.8%(->4%)

    raw_sl = max(struct_dist_pct * 0.65 + box_scaled_pct * 0.35, box_scaled_pct)
    sl_pct = max(2, min(5, int(round(raw_sl))))
    stop_price = avg_entry * (1.0 - sl_pct / 100.0)
    return sl_pct, stop_price, structural_support


def calc_structural_fib_3tps(df_5m, df_15m, df_1h, df_4h, avg_entry, sl_pct):
    """종목별 실전 저항선(15m/1H/4H 스윙고점·이평선·BB상단) & 피보나치 확장 기반 3단계 익절(TP1~TP3)"""
    candidates = []

    recent_low = to_float(df_5m.iloc[-36:]["low"].min(), default=avg_entry * 0.95)
    recent_high = to_float(df_5m.iloc[-36:]["high"].max(), default=avg_entry * 1.03)
    wave_height = max(recent_high - recent_low, avg_entry * 0.025)

    for fib_ratio, fib_name in [
        (1.000, "전고점돌파"),
        (1.272, "Fib 1.272"),
        (1.618, "Fib 1.618"),
        (2.000, "Fib 2.0"),
        (2.618, "Fib 2.618"),
        (3.618, "Fib 3.618"),
    ]:
        fib_p = recent_low + wave_height * fib_ratio
        if fib_p > avg_entry * 1.02:
            candidates.append((fib_p * 0.998, fib_name))

    if not df_1h.empty:
        htf_low = to_float(df_1h["low"].min(), default=recent_low)
        htf_high = to_float(df_1h["high"].max(), default=recent_high)
        htf_range = max(htf_high - htf_low, avg_entry * 0.06)

        for r_ratio, r_name in [
            (0.618, "1H Fib 0.618"),
            (0.786, "1H Fib 0.786"),
            (1.000, "1H 전고점"),
            (1.272, "1H Fib 1.272"),
            (1.618, "1H Fib 1.618"),
            (2.618, "1H Fib 2.618"),
        ]:
            htf_p = htf_low + htf_range * r_ratio
            if htf_p > avg_entry * 1.02:
                candidates.append((htf_p * 0.998, r_name))

        for pv in extract_pivot_highs(df_1h, left=2, right=2):
            if pv > avg_entry * 1.02:
                candidates.append((pv * 0.998, "1H 스윙고점"))

        ma99_1h = to_float(df_1h["ma99"].iloc[-1], default=0.0)
        bb_up_1h = to_float(df_1h["bb_upper"].iloc[-1], default=0.0)
        if ma99_1h > avg_entry * 1.03:
            candidates.append((ma99_1h * 0.997, "1H MA99저항"))
        if bb_up_1h > avg_entry * 1.03:
            candidates.append((bb_up_1h * 0.998, "1H BB상단"))

    for pv15 in extract_pivot_highs(df_15m, left=2, right=2):
        if pv15 > avg_entry * 1.02:
            candidates.append((pv15 * 0.998, "15m 매물대"))

    if not df_4h.empty:
        h_4h_max = to_float(df_4h["high"].max(), default=0.0)
        bb_up_4h = to_float(df_4h["bb_upper"].iloc[-1], default=0.0)
        if h_4h_max > avg_entry * 1.04:
            candidates.append((h_4h_max * 0.997, "4H 메이저고점"))
        if bb_up_4h > avg_entry * 1.04:
            candidates.append((bb_up_4h * 0.998, "4H BB상단"))

    valid_levels = []
    for price_val, label in sorted(candidates, key=lambda x: x[0]):
        pct = ((price_val - avg_entry) / avg_entry) * 100.0
        if pct < 3.5 or pct > 65.0:
            continue
        if valid_levels and abs(pct - valid_levels[-1][1]) < 1.2:
            if "고점" in label or "매물대" in label:
                valid_levels[-1] = (price_val, round(pct, 1), label)
            continue
        valid_levels.append((price_val, round(pct, 1), label))

    atr5 = to_float(df_5m["atr14"].iloc[-1], default=avg_entry * 0.01)
    vol_unit_pct = max(1.8, min(6.5, (atr5 * 1.75 / avg_entry) * 100.0))

    min_tp1_pct = max(float(sl_pct * 2.0), round(vol_unit_pct * 2.1, 1), 4.5)
    tp1_choice = None
    for p, pct, lbl in valid_levels:
        if min_tp1_pct <= pct <= min_tp1_pct + 7.5:
            tp1_choice = (p, pct, lbl)
            break
    if not tp1_choice:
        fallback_tp1_pct = round(min_tp1_pct + (vol_unit_pct * 0.35), 1)
        tp1_choice = (avg_entry * (1.0 + fallback_tp1_pct / 100.0), fallback_tp1_pct, "Fib 1.272")

    min_tp2_pct = max(tp1_choice[1] + max(4.0, vol_unit_pct * 1.8), float(sl_pct * 3.6), 9.5)
    tp2_choice = None
    for p, pct, lbl in valid_levels:
        if min_tp2_pct <= pct <= min_tp2_pct + 12.0:
            tp2_choice = (p, pct, lbl)
            break
    if not tp2_choice:
        fallback_tp2_pct = round(min_tp2_pct + (vol_unit_pct * 0.65), 1)
        tp2_choice = (avg_entry * (1.0 + fallback_tp2_pct / 100.0), fallback_tp2_pct, "Fib 1.618")

    min_tp3_pct = max(tp2_choice[1] + max(6.0, vol_unit_pct * 2.5), float(sl_pct * 5.5), 16.0)
    tp3_choice = None
    for p, pct, lbl in valid_levels:
        if min_tp3_pct <= pct <= min_tp3_pct + 22.0:
            tp3_choice = (p, pct, lbl)
            break
    if not tp3_choice:
        fallback_tp3_pct = round(min_tp3_pct + (vol_unit_pct * 1.1), 1)
        tp3_choice = (avg_entry * (1.0 + fallback_tp3_pct / 100.0), fallback_tp3_pct, "Fib 2.618")

    tps = [tp1_choice[0], tp2_choice[0], tp3_choice[0]]
    tp_pcts = [tp1_choice[1], tp2_choice[1], tp3_choice[1]]
    tp_labels = [tp1_choice[2], tp2_choice[2], tp3_choice[2]]
    return tps, tp_pcts, tp_labels


def analyze_symbol(coin):
    symbol = coin["symbol"]
    df, df_15m, df_1h, df_4h = get_multi_tf_data(symbol, ref_price=coin["last_price"])
    if df.empty or df_15m.empty or df_1h.empty or len(df) < 60:
        return None

    curr = df.iloc[-1]
    prev = df.iloc[-2]
    curr_15m = df_15m.iloc[-1]
    prev_15m = df_15m.iloc[-2] if len(df_15m) >= 2 else curr_15m
    curr_1h = df_1h.iloc[-1]
    prev_1h = df_1h.iloc[-2] if len(df_1h) >= 2 else curr_1h

    current_price = to_float(curr["close"], default=coin["last_price"])
    coin["last_price"] = current_price

    # [방어막 1] 고점 폭락(설거지 차트) 및 상위 프레임(1H) 역배열 폭락 차단
    high_12h = to_float(df.tail(144)["high"].max(), default=current_price)
    low_12h = to_float(df.tail(144)["low"].min(), default=current_price)

    drawdown_from_high_pct = ((high_12h - current_price) / high_12h) * 100.0 if high_12h > 0 else 0.0
    if drawdown_from_high_pct >= MAX_DRAWDOWN_FROM_12H_HIGH_PCT:
        return None

    total_12h_pump_pct = ((high_12h - low_12h) / low_12h) * 100.0 if low_12h > 0 else 0.0
    retracement_ratio = (high_12h - current_price) / (high_12h - low_12h) if (high_12h - low_12h) > 0 else 0.0
    if total_12h_pump_pct >= MAX_PRIOR_PUMP_PCT and retracement_ratio >= MAX_RETRACEMENT_RATIO:
        return None

    high_4h = to_float(df.iloc[-48:]["high"].max(), default=current_price)
    high_75m = to_float(df.iloc[-15:]["high"].max(), default=current_price)
    if (high_12h > high_4h * 1.03) and (high_4h > high_75m * 1.02):
        return None

    ma20_1h = to_float(curr_1h["ma20"], default=current_price)
    ma50_1h = to_float(curr_1h["ma50"], default=current_price)
    ma99_1h = to_float(curr_1h["ma99"], default=0.0)
    rsi6_1h = to_float(curr_1h["rsi6"], default=50.0)
    ma20_4h = to_float(df_4h["ma20"].iloc[-1], default=current_price) if not df_4h.empty else current_price

    if rsi6_1h >= CUTOFF_RSI6_1H_MAX:
        return None
    is_1h_freefall = bool(
        current_price < ma20_1h * 0.985
        and ma20_1h < to_float(prev_1h["ma20"], default=ma20_1h)
        and current_price < ma50_1h * 0.985
        and rsi6_1h < 38.0
    )
    if is_1h_freefall:
        return None

    if ma99_1h > current_price:
        room_to_1h_ma99_pct = ((ma99_1h - current_price) / current_price) * 100.0
        if 0.3 <= room_to_1h_ma99_pct <= 3.2:
            return None

    htf_bullish = bool(current_price >= ma20_1h * 0.995 and current_price >= ma20_4h * 0.990)
    htf_status_str = (
        f"1H 상승우위(RSI6 {rsi6_1h:.0f})" if htf_bullish else f"1H 바닥반전시도(RSI6 {rsi6_1h:.0f})"
    )

    # [방어막 2: 15분봉 과열 및 급등 양봉 꼭대기 추격 차단]
    rsi6_15m = to_float(curr_15m["rsi6"], default=50.0)
    kdj_j_15m = to_float(curr_15m["kdj_j"], default=50.0)
    bb_mid_15m = to_float(curr_15m["ma20"], default=current_price)
    prev_bb_mid_15m = to_float(prev_15m["ma20"], default=bb_mid_15m)
    bb_mid_ext_15m = to_float(curr_15m["bb_mid_ext_pct"], default=0.0)

    curr_vol_mult = to_float(curr["vol_mult"], default=0.0)
    prev_vol_mult = to_float(prev["vol_mult"], default=1.0)
    curr_candle_gain_pct = ((curr["close"] - curr["open"]) / curr["open"]) * 100.0 if curr["open"] > 0 else 0.0

    if rsi6_15m >= CUTOFF_RSI6_15M_MAX:
        return None
    if kdj_j_15m >= CUTOFF_KDJ_J_15M_MAX:
        return None
    if bb_mid_ext_15m >= CUTOFF_BB_MID_EXT_MAX_PCT:
        return None
    if curr_candle_gain_pct >= 3.2:
        return None

    # 직전 2시간 박스권 및 OBV 매집 분석
    pre_24 = df.iloc[-28:-4]
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

    # CVD 강세/흡수 다이버전스 (OBV와 결합 시 오늘자 Top 5 100% 적중!)
    cvd_div_5m, cvd_tag_5m = detect_cvd_divergence(df, lookback=24)
    cvd_div_15m, cvd_tag_15m = detect_cvd_divergence(df_15m, lookback=14)
    has_cvd_div = bool(cvd_div_5m or cvd_div_15m)
    cvd_label = cvd_tag_5m if cvd_div_5m else cvd_tag_15m

    recent_low_6 = to_float(df.iloc[-6:]["low"].min(), default=current_price)
    prior_low_12 = to_float(df.iloc[-18:-6]["low"].min(), default=current_price)
    higher_low = bool(recent_low_6 > prior_low_12 * 1.001)

    ma5_5m = to_float(curr["ma5"], default=current_price)
    ma20_5m = to_float(curr["ma20"], default=current_price)
    ma50_5m = to_float(curr["ma50"], default=current_price)
    ma112_5m = to_float(curr["ma112"], default=current_price)
    ma_short_aligned = bool(ma5_5m > ma20_5m > ma50_5m)

    is_15m_ma20_falling = bool(bb_mid_15m < prev_bb_mid_15m * 0.999)

    closed_3_min_vol = to_float(df.iloc[-4:-1]["volume"].min(), default=0.0)
    ref_vol_ma20 = to_float(prev["vol_ma20"], default=1.0)
    recent_3_vol_ratio = (closed_3_min_vol / ref_vol_ma20) if ref_vol_ma20 > 0 else 1.0

    max_wick_6 = to_float(df.iloc[-6:-1]["lower_wick_ratio"].max(), default=0.0)
    pre8_min_rsi6 = to_float(df.iloc[-9:-1]["rsi6"].min(), default=50.0)
    curr_rsi6 = to_float(curr["rsi6"], default=50.0)
    cond_rsi_thrust = bool(pre8_min_rsi6 <= 35.0 and curr_rsi6 >= 54.0 and (curr_rsi6 - pre8_min_rsi6) >= 20.0)
    cond_spring = bool(
        (not ma_short_aligned)
        and max_wick_6 >= 1.4
        and current_price > ma20_5m
        and (obv_accum or has_cvd_div or cond_rsi_thrust or box_range_pct <= PRE_BOX_TIGHT_PCT)
    )

    # TTM 켈트너 스퀴즈 (3/5 적중) & 이평선 리본 초밀집 (3/5 적중) & 1H 중심선 구조적 지지 (2/5 적중)
    recent_squeeze = bool(df.iloc[-7:-1]["ttm_squeeze_on"].any())
    cond_ttm_squeeze = bool(recent_squeeze and current_price > ma20_5m and curr["close"] > curr["open"])

    pre5_min_ribbon = to_float(df.iloc[-6:-1]["ribbon_spread_pct"].min(), default=99.0)
    cond_ema_ribbon = bool(pre5_min_ribbon <= EMA_RIBBON_MAX_SPREAD_PCT and current_price > ma50_5m)
    cond_htf_center_sup = bool(current_price >= ma112_5m * 0.996 and to_float(df.iloc[-4:-1]["low"].min()) <= ma112_5m * 1.012)

    # 펀딩비 숏스퀴즈 수급 (5/5 100% 적중!)
    funding_rate = to_float(coin["funding_rate"], default=0.01)
    cond_strong_short_squeeze = bool(funding_rate <= -0.020)  # BMT(-1.72%), BLEND(-0.55%), ERA(-0.29%)
    cond_clean_funding = bool(funding_rate <= 0.005)          # MET(+0.005%), AKE(+0.005%)

    # =================================================================
    # Track A (1차 선취매 / 정배열·Spring 반전 / 거래량 극건조 포켓 타점)
    # =================================================================
    score_a = 0
    tags_a = []
    if kdj_j_15m <= 68.0 and not is_15m_ma20_falling:
        # 1) 추세/반전 (25점 - 단기정배열 4/5 + Spring 1/5 = 100% 적중 반영)
        if ma_short_aligned and current_price >= ma20_5m * 0.998:
            score_a += 25
            tags_a.append("단기 정배열(5>20>50) 안착")
        elif cond_spring:
            score_a += 25
            tags_a.append(f"유동성 스윕(Spring)+MA20 돌파(꼬리 {max_wick_6:.1f}x)")

        # 2) 선행 매집: 박스권 응축(<=7.5%) + (OBV 우상향 OR CVD 다이버전스) (25점 - 5/5 100% 적중 반영)
        if box_range_pct <= PRE_BOX_TIGHT_PCT and (obv_accum or has_cvd_div):
            score_a += 25
            if has_cvd_div:
                tags_a.append(f"정밀응축({box_range_pct:.1f}%)+{cvd_label}")
            else:
                tags_a.append(f"정밀응축({box_range_pct:.1f}%)+OBV 선행매집")
        elif has_cvd_div:
            score_a += 18
            tags_a.append(cvd_label)
        elif obv_accum and box_range_pct <= PRE_BOX_RANGE_MAX_PCT:
            score_a += 16
            tags_a.append(f"박스권 응축({box_range_pct:.1f}%)+OBV 매집")

        # 3) 거래량 극건조 포켓 (20점 - 5/5 100% 적중 반영)
        if 0.005 <= recent_3_vol_ratio <= PULLBACK_DRYUP_STRICT:
            score_a += 20
            tags_a.append(f"거래량 극건조 포켓({recent_3_vol_ratio*100:.0f}%)")
        elif 0.005 <= recent_3_vol_ratio <= PRE_VOL_POCKET_MAX_RATIO:
            score_a += 16
            tags_a.append(f"거래량 건조 포켓({recent_3_vol_ratio*100:.0f}%)")

        # 4) 스퀴즈 응축 & 상위프레임 지지 (15점)
        if cond_ttm_squeeze and cond_ema_ribbon:
            score_a += 15
            tags_a.append(f"TTM 스퀴즈+이평리본 초밀집({pre5_min_ribbon:.1f}%)")
        elif cond_ttm_squeeze:
            score_a += 10
            tags_a.append("TTM 켈트너 스퀴즈 압축")
        elif cond_ema_ribbon:
            score_a += 10
            tags_a.append(f"이평리본 초밀집({pre5_min_ribbon:.1f}%)")

        if cond_htf_center_sup or htf_bullish:
            score_a += 5
            tags_a.append("상위프레임(1H 중심선) 구조적 지지")

        # 5) 숏스퀴즈/청정 펀딩 수급 (15점 - 5/5 100% 적중 반영)
        if cond_strong_short_squeeze:
            score_a += 15
            tags_a.append(f"강력 숏스퀴즈 수급({funding_rate:+.4f}%)")
        elif cond_clean_funding:
            score_a += 10
            tags_a.append(f"청정 펀딩 수급({funding_rate:+.4f}%)")

    # =================================================================
    # Track B (2차 돌파 초입 & S/R Flip·MA20·MA50 칼지지 리테스트 채점)
    # =================================================================
    score_b = 0
    tags_b = []
    swing_high_8 = to_float(df.iloc[-12:-2]["high"].max(), default=current_price)
    breakout_diff_pct = ((current_price - swing_high_8) / swing_high_8) * 100.0 if swing_high_8 > 0 else 99.0
    reclaimed_bb_mid = bool(current_price > bb_mid_15m and current_price > ma20_5m and not is_15m_ma20_falling)
    is_early_breakout = bool(-0.2 <= breakout_diff_pct <= EARLY_BREAKOUT_MAX_PCT and curr["close"] > curr["open"])

    recent_5_bars = df.iloc[-7:-1]
    peak_breakout_vol = to_float(recent_5_bars["volume"].max(), default=0.0)
    peak_breakout_mult = to_float(recent_5_bars["vol_mult"].max(), default=1.0)

    eval_pullback_vol = to_float(curr["volume"]) if to_float(curr["volume"]) >= ref_vol_ma20 * 0.05 else to_float(prev["volume"])
    pullback_vol_vs_peak = (eval_pullback_vol / peak_breakout_vol) if peak_breakout_vol > 0 else 1.0

    # 14:17 피드백 반영: 돌파 후 S/R Flip(3/5) + MA20(1/5) + MA50(1/5) 칼지지 리테스트 (눌림 거래량 18% 이하 만점)
    ref_sr_level = box_high if abs(curr["low"] - box_high) <= abs(curr["low"] - swing_high_8) else swing_high_8
    sr_flip_err_pct = (abs(to_float(curr["low"]) - ref_sr_level) / ref_sr_level) * 100.0 if ref_sr_level > 0 else 99.0
    cond_sr_flip_retest = bool(
        peak_breakout_mult >= EARLY_VOL_MULT_MIN
        and drawdown_from_high_pct <= 5.0
        and current_price >= ref_sr_level * 0.996
        and sr_flip_err_pct <= SR_FLIP_TOLERANCE_PCT
        and pullback_vol_vs_peak <= SR_FLIP_DRYUP_RATIO_MAX
    )

    ma20_err_pct = (abs(to_float(curr["low"]) - ma20_5m) / ma20_5m) * 100.0 if ma20_5m > 0 else 99.0
    ma50_err_pct = (abs(to_float(curr["low"]) - ma50_5m) / ma50_5m) * 100.0 if ma50_5m > 0 else 99.0
    cond_ma_pullback_retest = bool(
        peak_breakout_mult >= EARLY_VOL_MULT_MIN
        and drawdown_from_high_pct <= 5.0
        and current_price >= ma20_5m * 0.998
        and (ma20_err_pct <= 0.65 or ma50_err_pct <= 0.65)
        and pullback_vol_vs_peak <= PULLBACK_DRYUP_STRICT
    )

    max_vol_mult_2 = max(curr_vol_mult, prev_vol_mult)

    # 1) 타점 컨펌 (25점)
    if cond_sr_flip_retest:
        score_b += 25
        tags_b.append(f"전고점 S/R Flip 칼지지(오차 {sr_flip_err_pct:.2f}%/눌림거래량 {pullback_vol_vs_peak*100:.0f}%)")
    elif cond_ma_pullback_retest:
        sup_name = "MA20" if ma20_err_pct <= 0.65 else "MA50"
        score_b += 25
        tags_b.append(f"돌파후 5m {sup_name} 칼지지(눌림거래량 {pullback_vol_vs_peak*100:.0f}%)")
    elif reclaimed_bb_mid and is_early_breakout and EARLY_VOL_MULT_MIN <= max_vol_mult_2 <= EARLY_VOL_MULT_MAX and drawdown_from_high_pct <= 4.5:
        score_b += 22
        tags_b.append(f"스윙고점({fmt_price(swing_high_8)}) 돌파 초입")

    # 2) 초동 거래량 폭발 & 발화 직전 거래량 극건조 포켓 (25점 - 5/5 100% 적중 반영)
    pre5_min_vol_ratio = (
        to_float(df.iloc[-8:-2]["volume"].min() / ref_vol_ma20, default=1.0)
        if ref_vol_ma20 > 0 else 1.0
    )
    if (cond_sr_flip_retest or cond_ma_pullback_retest) and peak_breakout_mult >= EARLY_VOL_MULT_MIN:
        score_b += 15
        tags_b.append(f"초동수급 폭발({peak_breakout_mult:.1f}배)")
    elif EARLY_VOL_MULT_MIN <= max_vol_mult_2 <= EARLY_VOL_MULT_MAX and max(to_float(curr["body_ratio"]), to_float(prev["body_ratio"])) >= 0.50:
        score_b += 15
        tags_b.append(f"초동 수급 유입({max_vol_mult_2:.1f}배)")

    if 0.005 <= pre5_min_vol_ratio <= PRE_VOL_POCKET_MAX_RATIO:
        score_b += 15
        tags_b.append(f"발화전 거래량 극건조 포켓({pre5_min_vol_ratio*100:.0f}%)")

    # 3) 매집: OBV 우상향 OR CVD 강세/흡수 다이버전스 (20점 - 5/5 100% 적중 반영)
    if has_cvd_div and box_range_pct <= PRE_BOX_TIGHT_PCT:
        score_b += 20
        tags_b.append(f"정밀응축({box_range_pct:.1f}%)+{cvd_label}")
    elif obv_accum and box_range_pct <= PRE_BOX_TIGHT_PCT:
        score_b += 18
        tags_b.append(f"정밀응축({box_range_pct:.1f}%)+OBV 선행매집")
    elif has_cvd_div:
        score_b += 15
        tags_b.append(cvd_label)
    elif obv_accum:
        score_b += 12
        tags_b.append("OBV 우상향 매집")

    # 4) TTM 스퀴즈 & 이평선 리본 초밀집 확산 (15점)
    if cond_ttm_squeeze and cond_ema_ribbon:
        score_b += 15
        tags_b.append(f"TTM 스퀴즈+이평리본 확산({pre5_min_ribbon:.1f}%)")
    elif cond_ttm_squeeze:
        score_b += 10
        tags_b.append("TTM 켈트너 스퀴즈 폭발")
    elif cond_ema_ribbon:
        score_b += 10
        tags_b.append(f"이평리본 초밀집 확산({pre5_min_ribbon:.1f}%)")

    # 5) 숏스퀴즈/청정 펀딩 수급 & 상위프레임 동조 (15점 - 5/5 100% 적중 반영)
    if cond_strong_short_squeeze:
        score_b += 15
        tags_b.append(f"강력 숏스퀴즈 수급({funding_rate:+.4f}%)")
    elif cond_clean_funding:
        score_b += 10
        tags_b.append(f"청정 펀딩 수급({funding_rate:+.4f}%)")

    if htf_bullish and rsi6_15m <= 68.0 and kdj_j_15m <= 80.0:
        score_b += 5
        tags_b.append("상위프레임(1H/4H) 추세동조")

    if score_a < SCORE_THRESHOLD and score_b < SCORE_THRESHOLD:
        return None

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
        track_title = "🚀 [2차 돌파/리테스트 타점 포착]"
        final_score = int(min(100, score_b))
        matched_tags = tags_b
        entry_low = current_price * 0.995
        entry_high = current_price * 1.003

    avg_entry = (entry_low + entry_high) / 2.0

    # [정밀 교정 완료] 리테스트 진입가 기준 -2% ~ -5% 유동적 손절(SL) 계산
    sl_pct, stop_price, structural_support = calc_dynamic_structural_sl(df, df_15m, avg_entry, box_range_pct)
    support_line = min(structural_support, current_price * 0.992)

    # 종목별 실전 저항 매물대 & 피보나치 기반 3단계 동적 TP 산출
    tps, tp_pcts, tp_labels = calc_structural_fib_3tps(
        df, df_15m, df_1h, df_4h, avg_entry, sl_pct
    )

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
        "htf_status": htf_status_str,
        "swing_high_line": resistance_line,
        "support_line": support_line,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_price": stop_price,
        "sl_pct": sl_pct,
        "tp_pcts": tp_pcts,
        "tps": tps,
        "tp_labels": tp_labels,
        "matched_tags": matched_tags
    }


def build_briefing_message(res, now_str):
    short_sym = res["symbol"].replace("USDT", "")
    return (
        f"{res['track_title']} ${short_sym} ({res['symbol']} | {res['score']}점)\n"
        f"🕒 포착 시각: {now_str}\n"
        f"• 현재가: {fmt_price(res['last_price'])} USDT (24h {res['change_24h']:+.2f}%)\n"
        f"• 거래대금: {res['turnover_m']:.1f}M | OI: {res['oi_m']:.1f}M | 펀딩비: {res['funding_rate']:+.4f}%\n"
        f"• [멀티 프레임 진단]: {res['htf_status']} | 15m RSI6 {res['rsi6_15m']:.1f} (<75) | J값 {res['kdj_j_15m']:.1f} (<88)\n"
        f"• [포착 근거]: {' / '.join(res['matched_tags'])}\n"
        f"• [핵심 기준선 & 구조적 3단계 타점 가이드]:\n"
        f"  - 1차 지지선(구조적 스윙저점): {fmt_price(res['support_line'])} USDT\n"
        f"  - 2차 돌파선(스윙 고점): {fmt_price(res['swing_high_line'])} USDT\n"
        f"  - 권장 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n"
        f"  - 유동적 손절가(SL): {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%·스윙저점+ATR버퍼)\n"
        f"  - 3단계 익절(TP): "
        f"1차 {fmt_price(res['tps'][0])}(+{fmt_pct(res['tp_pcts'][0])}%·{res['tp_labels'][0]}) / "
        f"2차 {fmt_price(res['tps'][1])}(+{fmt_pct(res['tp_pcts'][1])}%·{res['tp_labels'][1]}) / "
        f"3차 {fmt_price(res['tps'][2])}(+{fmt_pct(res['tp_pcts'][2])}%·{res['tp_labels'][2]})"
    )


def build_x_post_message(res):
    short_sym = res["symbol"].replace("USDT", "")
    return (
        f"${short_sym} ({res['symbol']})\n\n"
        f"현재가: {fmt_price(res['last_price'])} USDT\n\n"
        f"- 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n\n"
        f"- SL: {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%)\n\n"
        f"- 1차 {fmt_price(res['tps'][0])}(+{fmt_pct(res['tp_pcts'][0])}%)\n"
        f"- 2차 {fmt_price(res['tps'][1])}(+{fmt_pct(res['tp_pcts'][1])}%)\n"
        f"- 3차 {fmt_price(res['tps'][2])}(+{fmt_pct(res['tp_pcts'][2])}%)"
    )


def send_telegram_single(text):
    print(text)
    print("--------------------------------------------------")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        time.sleep(0.5)
    except Exception as e:
        print(f"[Telegram 전송 오류] {e}")


def main():
    start_t = time.time()
    state = load_state()
    tracked = state.get("tracked", {})
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    candidates, pure_crypto_bases, rwa_blocked_bases = get_scanner_candidates()

    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        if (
            is_tradfi_ticker(base_c)
            or base_c in rwa_blocked_bases
            or (now_ts - tracked[sym].get("created_at", 0) > 64800)
        ):
            del tracked[sym]

    if not candidates:
        print("[알림] 조건을 만족하는 순수 암호화폐 후보군 데이터를 가져오지 못했습니다.")
        save_state(state)
        return

    sent_count = 0
    to_analyze = []

    for coin in candidates:
        symbol = coin["symbol"]
        curr_p = coin["last_price"]

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
                    f"• 1차 목표가(1:2 R:R): {fmt_price(t_info['tps'][0])} (+{fmt_pct(t_info['tp_pcts'][0])}%)"
                )
                send_telegram_single(bo_msg)
                sent_count += 1
                continue

        to_analyze.append(coin)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(analyze_symbol, c): c["symbol"] for c in to_analyze}
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[스캔 예외] {sym}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)

    for res in results:
        symbol = res["symbol"]
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

        briefing_msg = build_briefing_message(res, now_str)
        send_telegram_single(briefing_msg)

        x_msg = build_x_post_message(res)
        send_telegram_single(x_msg)

        sent_count += 1

    state["tracked"] = tracked
    save_state(state)

    elapsed = time.time() - start_t
    if sent_count == 0:
        print(f"[{now_str}] 멀티프레임 스캔 완료 ({elapsed:.1f}초 소요) - 85점 이상 신규 초입 종목 없음.")
    else:
        print(f"[{now_str}] 멀티프레임 스캔 완료 ({elapsed:.1f}초 소요) - 총 {sent_count}건 알림 전송.")


if __name__ == "__main__":
    main()
