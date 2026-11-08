import os
import time
import requests
import numpy as np
import pandas as pd
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

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

# 토큰화 주식/Pre-IPO/ETF/원자재 식별용 메타데이터 키워드
NON_CRYPTO_META_KEYWORDS = (
    "xstock", "dinari", "backed", "tokenized", "ondo", "spdr", "ishares",
    "proshares", "direxion", "vanguard", "invesco", "wisdomtree", "tradr", "ark-",
    "etf", "shares", "trust", "index", "crude", "brent", "wti", "natural-gas",
    "gold", "silver", "platinum", "palladium", "copper", "forex", "treasury",
    "inc", "corp", "ltd", "plc", "holdings", "nasdaq", "s&p", "spacex",
    "space exploration", "pre-ipo", "sandisk", "hynix", "technologies"
)

# 비암호화폐(TradFi/주식/Pre-IPO/원자재/외환) 영구 블랙리스트
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
    return f"{p:.7f}".rstrip("0").rstrip(".")


def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def calc_cvd_series(df):
    """캔들 내 매수/매도 압력(CLV) 및 아랫꼬리 스탑헌팅 흡수를 결합한 누적 볼륨 델타(CVD)"""
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    wick_share = (lower_wick / high_low).fillna(0.0)
    absorption_boost = np.where(df["close"] >= df["open"], wick_share * 0.45, wick_share * 0.20)
    effective_delta_ratio = np.clip(clv + absorption_boost, -1.0, 1.0)
    return (df["volume"] * effective_delta_ratio).cumsum()


def detect_cvd_div_at_ignition(df, ign_idx, lookback=24):
    """발화 직전 구간에서 CVD 강세 다이버전스(저점 스윕 매수흡수) 또는 스텔스 매집 감지"""
    start_idx = max(0, ign_idx - lookback)
    sub = df.iloc[start_idx : ign_idx + 1].copy().reset_index(drop=True)
    if len(sub) < 12:
        return False

    half = len(sub) // 2
    idx_l1 = sub.iloc[:half]["low"].idxmin()
    idx_l2 = sub.iloc[half:]["low"].idxmin()

    p_l1, p_l2 = to_float(sub.loc[idx_l1, "low"]), to_float(sub.loc[idx_l2, "low"])
    c_l1, c_l2 = to_float(sub.loc[idx_l1, "cvd"]), to_float(sub.loc[idx_l2, "cvd"])
    c_rng = max(to_float(sub["cvd"].max() - sub["cvd"].min()), 1.0)

    # 클래식 강세 다이버전스: 가격 저점 낮아짐/유지 + CVD 저점 상승
    if p_l1 > 0 and p_l2 <= p_l1 * 1.005 and ((c_l2 - c_l1) / c_rng >= 0.08):
        return True

    # 박스권 흡수 다이버전스: 가격은 횡보인데 CVD만 가파르게 우상향
    try:
        cvd_norm = (sub["cvd"].values - sub["cvd"].min()) / c_rng
        slope = np.polyfit(np.arange(len(sub)), cvd_norm, 1)[0]
        p_chg = ((to_float(sub["close"].iloc[-1]) - to_float(sub["close"].iloc[0])) / max(to_float(sub["close"].iloc[0]), 1e-9)) * 100.0
        if -2.5 <= p_chg <= 2.0 and slope >= 0.025:
            return True
    except Exception:
        pass
    return False


def fetch_oi_change_before_ignition(symbol, ign_ts_ms):
    """발화 직전 1시간 동안의 실제 미결제약정(OI) 선행 증감률(%) 조회"""
    base_coin = symbol[:-4].upper()
    clean_base = strip_multiplier_prefix(base_coin)

    # 1순위: Gate.io Futures contract_stats (미국 IP 직통)
    for base_try in [base_coin, clean_base]:
        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/contract_stats",
                params={"contract": f"{base_try}_USDT", "interval": "15m", "limit": "30"},
                headers=BROWSER_HEADERS,
                timeout=4
            )
            if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) >= 6:
                stats = sorted(r.json(), key=lambda x: int(x.get("time", 0)))
                ign_sec = ign_ts_ms // 1000
                pre_stats = [s for s in stats if int(s.get("time", 0)) <= ign_sec]
                if len(pre_stats) >= 5:
                    oi_now = to_float(pre_stats[-1].get("open_interest_usd") or pre_stats[-1].get("open_interest"))
                    oi_1h_ago = to_float(pre_stats[-5].get("open_interest_usd") or pre_stats[-5].get("open_interest"))
                    if oi_1h_ago > 0:
                        return ((oi_now - oi_1h_ago) / oi_1h_ago) * 100.0
        except Exception:
            pass

    return 0.0


def get_fast_pure_crypto_whitelist():
    """5대 거래소 직통 API로 순수 코인 화이트리스트 및 RWA/주식 차단 목록 확보"""
    pure_crypto_bases = set()
    rwa_blocked_bases = set()

    try:
        r = requests.get(
            "https://api.bitget.com/api/v2/mix/market/contracts",
            params={"productType": "USDT-FUTURES"},
            headers=BROWSER_HEADERS,
            timeout=6
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
        ("https://api.gateio.ws/api/v4/futures/usdt/tickers", "gate"),
        ("https://data-api.binance.vision/api/v3/ticker/price", "bn"),
    ]:
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=6)
            if r.status_code == 200:
                data = r.json()
                items = data.get("data", []) if mode == "okx" else data
                for item in items:
                    if mode == "okx":
                        inst = str(item.get("instId", "")).upper()
                        base = inst.split("-")[0] if inst.endswith("-USDT-SWAP") else ""
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


def get_top5_pure_crypto_gainers():
    """바이비트 선물 중 주식/TradFi를 100% 제거한 순수 코인 급등 상위 5개 추출"""
    pure_crypto_bases, rwa_blocked_bases = get_fast_pure_crypto_whitelist()
    gainers = []

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

                if last_price > 0 and change_24h > 0 and turnover_24h >= 1_000_000:
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


def fetch_klines_5m(symbol, ref_price=0.0, limit=200):
    base_coin = symbol[:-4].upper()
    clean_base = strip_multiplier_prefix(base_coin)
    rows = []

    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            r = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": sym_try, "productType": "USDT-FUTURES", "granularity": "5m", "limit": str(min(limit, 200))},
                headers=BROWSER_HEADERS,
                timeout=5
            )
            if r.status_code == 200 and len(r.json().get("data", [])) >= 65:
                rows = [x[:7] for x in r.json()["data"]]
                break
        except Exception:
            pass

    if not rows:
        for base_try in [base_coin, clean_base]:
            try:
                r = requests.get(
                    "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                    params={"contract": f"{base_try}_USDT", "interval": "5m", "limit": str(limit)},
                    headers=BROWSER_HEADERS,
                    timeout=5
                )
                if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) >= 65:
                    rows = [
                        [int(float(i.get("t", 0))) * 1000, i.get("o", 0), i.get("h", 0), i.get("l", 0), i.get("c", 0), float(i.get("v", 0)) or 1.0, 0]
                        for i in r.json()
                    ]
                    break
            except Exception:
                pass

    if not rows:
        for sym_try in [symbol, f"{clean_base}USDT"]:
            try:
                r = requests.get(
                    "https://data-api.binance.vision/api/v3/klines",
                    params={"symbol": sym_try, "interval": "5m", "limit": str(limit)},
                    headers=BROWSER_HEADERS,
                    timeout=5
                )
                if r.status_code == 200 and isinstance(r.json(), list) and len(r.json()) >= 65:
                    rows = [[x[0], x[1], x[2], x[3], x[4], x[5], x[7]] for x in r.json()]
                    break
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    if len(df) < 65:
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


def reverse_engineer_coin(coin):
    """
    단일 급등 종목의 발화점을 역추적하고 12가지 기관급 알파 전략 후보군을 전수 테스트하여
    어떤 전략이 작동했는지 발굴합니다.
    """
    df = fetch_klines_5m(coin["symbol"], ref_price=coin["last_price"], limit=200)
    if df.empty or len(df) < 65:
        return None

    # 기본 이평선 및 거래량 지표
    df["ma5"] = df["close"].rolling(5, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=20).mean()
    df["ma112"] = df["close"].rolling(112, min_periods=35).mean()
    df["vol_ma20"] = df["volume"].shift(1).rolling(20, min_periods=5).mean()
    df["vol_mult"] = (df["volume"] / df["vol_ma20"].replace(0, np.nan)).fillna(1.0)

    # 이평선 리본 스프레드
    ribbon_max = df[["ma5", "ma20", "ma50"]].max(axis=1)
    ribbon_min = df[["ma5", "ma20", "ma50"]].min(axis=1)
    df["ribbon_spread"] = ((ribbon_max - ribbon_min) / df["ma20"].replace(0, np.nan)) * 100.0

    # OBV, CVD, RSI(14), VWAP(48캔들=4시간)
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()
    df["cvd"] = calc_cvd_series(df)
    df["rsi14"] = calc_rsi(df["close"], period=14)

    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol_sum = (typical_price * df["volume"]).rolling(48, min_periods=12).sum()
    vol_sum_48 = df["volume"].rolling(48, min_periods=12).sum().replace(0, np.nan)
    df["vwap48"] = (tp_vol_sum / vol_sum_48).fillna(df["ma20"])

    # MACD (12, 26, 9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - sig

    # 볼린저 밴드(BBW) 및 TTM 켈트너 채널(KC) 스퀴즈
    std20 = df["close"].rolling(20, min_periods=10).std()
    df["bb_upper"] = df["ma20"] + 2.0 * std20
    df["bb_lower"] = df["ma20"] - 2.0 * std20
    df["bbw"] = ((df["bb_upper"] - df["bb_lower"]) / df["ma20"].replace(0, np.nan)) * 100.0

    prev_c = df["close"].shift(1)
    tr = pd.concat([(df["high"] - df["low"]).abs(), (df["high"] - prev_c).abs(), (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=5).mean()
    df["kc_upper"] = df["ma20"] + 1.5 * df["atr14"]
    df["kc_lower"] = df["ma20"] - 1.5 * df["atr14"]
    df["ttm_sqz"] = (df["bb_upper"] <= df["kc_upper"]) & (df["bb_lower"] >= df["kc_lower"])

    # 캔들 몸통 비율 및 아랫꼬리 비율
    candle_rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / candle_rng).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_sz = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_sz).clip(0.0, 10.0)

    # 초동 발화점(Ignition Point) 탐색
    search_df = df.iloc[30:-3].copy()
    search_df["score_ign"] = np.where(
        (search_df["close"] > search_df["open"]) & (search_df["vol_mult"] >= 2.2),
        search_df["vol_mult"] * ((search_df["close"] - search_df["open"]) / search_df["open"] * 100.0),
        0.0
    )
    ign_idx = int(search_df["score_ign"].idxmax()) if search_df["score_ign"].max() > 0 else int(len(df) - 15)
    ign_bar = df.loc[ign_idx]
    pre_bar = df.loc[max(0, ign_idx - 1)]

    ign_time_str = datetime.fromtimestamp(int(ign_bar["timestamp"]) / 1000, tz=KST).strftime("%m/%d %H:%M KST")
    ign_price = to_float(ign_bar["close"])
    post_high = to_float(df.loc[ign_idx:]["high"].max(), default=ign_price)
    pump_from_ign_pct = max(0.5, ((post_high - ign_price) / ign_price) * 100.0) if ign_price > 0 else 0.5

    # 발화 직전 상태 분석
    ma_aligned = bool(pre_bar["ma5"] > pre_bar["ma20"] > pre_bar["ma50"])
    ma_full_aligned = bool(ma_aligned and (pre_bar["ma50"] > to_float(pre_bar["ma112"], default=0.0)))
    ma_str = "[✅ 단기 정배열(5>20>50)]" if ma_aligned else "[⚠️ 비정배열(역배열/혼조 돌파)]"

    pre_box = df.loc[max(0, ign_idx - 24) : max(1, ign_idx - 1)]
    box_high = to_float(pre_box["high"].max(), default=ign_price)
    box_low = to_float(pre_box["low"].min(), default=ign_price)
    box_range_pct = ((box_high - box_low) / box_low) * 100.0 if box_low > 0 else 5.0

    obv_up = False
    if len(pre_box) >= 5:
        try:
            obv_up = bool(np.polyfit(np.arange(len(pre_box)), pre_box["obv"].values, 1)[0] > 0)
        except Exception:
            obv_up = False
    obv_str = "[✅ 우상향 선행매집]" if obv_up else "[➖ 중립/횡보]"

    vol_mult_ign = max(2.0, to_float(ign_bar["vol_mult"], default=3.0))

    # 돌파 후 리테스트 패턴(S/R Flip vs MA20 vs MA50) 분석
    post_bars = df.loc[ign_idx + 1 : min(len(df) - 1, ign_idx + 8)]
    pullback_low = to_float(post_bars["low"].min(), default=ign_price * 0.985) if not post_bars.empty else ign_price * 0.985
    pullback_min_vol = to_float(post_bars["volume"].min(), default=0.0) if not post_bars.empty else 0.0
    dryup_ratio = min(0.95, pullback_min_vol / max(to_float(ign_bar["volume"]), 1e-9))

    sr_err_pct = (abs(pullback_low - box_high) / box_high) * 100.0 if box_high > 0 else 99.0
    ma20_post = to_float(post_bars["ma20"].mean(), default=ign_price) if not post_bars.empty else ign_price
    ma50_post = to_float(post_bars["ma50"].mean(), default=ign_price) if not post_bars.empty else ign_price
    ma20_err_pct = (abs(pullback_low - ma20_post) / ma20_post) * 100.0 if ma20_post > 0 else 99.0
    ma50_err_pct = (abs(pullback_low - ma50_post) / ma50_post) * 100.0 if ma50_post > 0 else 99.0

    is_sr_flip = bool(sr_err_pct <= 1.2)
    is_ma20_sup = bool(not is_sr_flip and ma20_err_pct <= 1.0)
    is_ma50_sup = bool(not is_sr_flip and not is_ma20_sup and ma50_err_pct <= 1.0)

    if is_sr_flip:
        post_pattern_str = f"전고점 S/R Flip 칼지지(오차 {sr_err_pct:.2f}%) (거래량 {dryup_ratio*100:.0f}% 수준 건조)"
    elif is_ma20_sup:
        post_pattern_str = f"5분봉 MA 20선 지지 리테스트 (거래량 {dryup_ratio*100:.0f}% 수준 건조)"
    elif is_ma50_sup:
        post_pattern_str = f"5분봉 MA 50선 눌림 후 반등 (거래량 {dryup_ratio*100:.0f}% 수준 건조)"
    else:
        post_pattern_str = f"단기 자율 눌림 후 반등 (거래량 {dryup_ratio*100:.0f}% 수준 건조)"

    # =================================================================
    # [신규 알파 전략 발굴 테스트 (총 10종 글로벌 퀀트 전략 검증)]
    # =================================================================
    alphas = []
    pre6 = df.loc[max(0, ign_idx - 6) : ign_idx]

    # 1) 유동성 스윕 (Spring)
    wick_max = to_float(pre6["lower_wick_ratio"].max(), default=0.0)
    is_spring = bool((not ma_aligned) and wick_max >= 1.4 and ign_price > to_float(ign_bar["ma20"]))
    if is_spring:
        alphas.append("유동성 스윕(Spring 아랫꼬리 스탑헌팅 후 MA20 돌파)")

    # 2) CVD 강세/흡수 다이버전스
    has_cvd_div = detect_cvd_div_at_ignition(df, ign_idx, lookback=24)
    if has_cvd_div:
        alphas.append("CVD 강세/흡수 다이버전스(선행 시장가매집)")

    # 3) 발화 직전 거래량 극건조 포켓
    pre5_min_vol = to_float(df.loc[max(0, ign_idx - 5) : max(0, ign_idx - 1)]["volume"].min(), default=0.0)
    pocket_ratio = pre5_min_vol / max(to_float(ign_bar["vol_ma20"]), 1e-9)
    is_vol_pocket = bool(pocket_ratio <= 0.35)
    if is_vol_pocket:
        alphas.append(f"발화 직전 거래량 극건조 포켓(평소 대비 {pocket_ratio*100:.0f}% 수축 후 폭발)")

    # 4) BB스퀴즈 + VWAP48 장대양봉 돌파 (실제 연산)
    bbw_window = df.loc[max(0, ign_idx - 48) : max(1, ign_idx - 1)]["bbw"].dropna()
    bbw_pct_4h = float((bbw_window < to_float(pre_bar["bbw"])).mean() * 100.0) if len(bbw_window) >= 10 else 50.0
    is_bb_vwap = bool(
        bbw_pct_4h <= 35.0
        and ign_price > to_float(ign_bar["vwap48"])
        and to_float(ign_bar["body_ratio"]) >= 0.60
    )
    if is_bb_vwap:
        alphas.append(f"BB스퀴즈(하위 {bbw_pct_4h:.0f}%)+VWAP 장대양봉 돌파")

    # 5) TTM 켈트너 스퀴즈 폭발
    is_ttm = bool(pre6["ttm_sqz"].any())
    if is_ttm:
        alphas.append("TTM 켈트너 스퀴즈 폭발(BB⊂KC 압축 후 상단 돌파)")

    # 6) 이평선 리본 초밀집 확산
    min_ribbon = to_float(pre6["ribbon_spread"].min(), default=99.0)
    is_ribbon = bool(min_ribbon <= 1.8)
    if is_ribbon:
        alphas.append(f"이평선 리본 초밀집 확산(5·20·50 이격 {min_ribbon:.1f}% 수렴 후 분출)")

    # 7) 상위프레임(1H) 이평선(MA20=5분봉 MA240 근사 / MA112) 구조적 지지 동조
    ma112_val = to_float(ign_bar["ma112"], default=ign_price)
    is_htf_support = bool(ign_price >= ma112_val * 0.995 and to_float(pre_bar["low"]) <= ma112_val * 1.015)
    if is_htf_support:
        alphas.append("상위프레임(1H 중심선) 구조적 지지 반등")

    # 8) 발화 전 1시간 OI 선행 스텔스 매집 (실제 OI 히스토리 조회)
    oi_chg_1h_pct = fetch_oi_change_before_ignition(coin["symbol"], int(ign_bar["timestamp"]))
    is_oi_stealth = bool(oi_chg_1h_pct >= 3.0)
    if is_oi_stealth:
        alphas.append(f"OI 선행 스텔스매집(발화전 1h OI +{oi_chg_1h_pct:.1f}% 급증)")

    # 9) RSI 과매도 스러스트 & MACD 양전환 골든크로스
    min_rsi = to_float(df.loc[max(0, ign_idx - 8) : max(0, ign_idx - 1)]["rsi14"].min(), default=50.0)
    max_rsi = to_float(df.loc[ign_idx : min(len(df) - 1, ign_idx + 3)]["rsi14"].max(), default=50.0)
    is_rsi_thrust = bool(min_rsi <= 38.0 and (max_rsi - min_rsi) >= 20.0)
    if is_rsi_thrust:
        alphas.append(f"RSI 과매도 스러스트(발화전 RSI {min_rsi:.0f}→{max_rsi:.0f} 급반등)")

    is_macd_cross = bool(to_float(ign_bar["macd_hist"]) > 0 and to_float(pre_bar["macd_hist"]) <= 0)
    if is_macd_cross:
        alphas.append("MACD 히스토그램 양전환 골든크로스 발화")

    # 10) 돌파 후 S/R Flip 리테스트 & 숏스퀴즈 수급
    if is_sr_flip:
        alphas.append(f"돌파후 박스상단 S/R Flip 리테스트(거래량 {dryup_ratio*100:.0f}%로 건조)")

    is_short_sqz = bool(coin["funding_rate"] <= 0.01)
    if is_short_sqz:
        alphas.append(f"숏스퀴즈 수급(펀딩 {coin['funding_rate']:+.4f}%)")

    # 종목별 유동적 손절폭(SL: 스윙저점+15분ATR 버퍼와 박스 변동폭 비례, -2% ~ -5% 유동)
    atr_ign = to_float(ign_bar["atr14"], default=ign_price * 0.012)
    struct_low = min(pullback_low, to_float(pre6["low"].min(), default=pullback_low))
    struct_sl_pct = ((ign_price - (struct_low - 0.85 * atr_ign * 1.732)) / ign_price) * 100.0 if ign_price > 0 else 3.0
    box_based_sl_pct = max(2.2, box_range_pct * 0.65)
    raw_sl_pct = max(struct_sl_pct, box_based_sl_pct)
    dynamic_sl_pct = max(2, min(5, int(round(raw_sl_pct))))

    return {
        "coin": coin,
        "ign_time": ign_time_str,
        "ign_price": ign_price,
        "pump_pct": pump_from_ign_pct,
        "ma_str": ma_str,
        "box_range_pct": box_range_pct,
        "obv_str": obv_str,
        "vol_mult": vol_mult_ign,
        "post_pattern_str": post_pattern_str,
        "alphas": alphas if alphas else ["단기 수급 유입 돌파"],
        "ma_aligned": ma_aligned,
        "ma_full_aligned": ma_full_aligned,
        "obv_up": obv_up,
        "is_spring": is_spring,
        "has_cvd_div": has_cvd_div,
        "is_vol_pocket": is_vol_pocket,
        "is_ribbon": is_ribbon,
        "is_bb_vwap": is_bb_vwap,
        "is_ttm": is_ttm,
        "is_htf_support": is_htf_support,
        "is_oi_stealth": is_oi_stealth,
        "oi_chg_1h_pct": oi_chg_1h_pct,
        "is_rsi_thrust": is_rsi_thrust,
        "is_macd_cross": is_macd_cross,
        "is_short_sqz": is_short_sqz,
        "is_sr_flip": is_sr_flip,
        "is_ma20_sup": is_ma20_sup,
        "dryup_ratio": dryup_ratio,
        "dynamic_sl_pct": dynamic_sl_pct,
    }


def main():
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    top5 = get_top5_pure_crypto_gainers()
    if not top5:
        print("[알림] Top 5 순수 코인 데이터를 가져오지 못했습니다.")
        return

    analyzed = []
    for c in top5:
        res = reverse_engineer_coin(c)
        if res:
            analyzed.append(res)

    if not analyzed:
        return

    emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 신규 전략 발굴]",
        f"🕒 기준 시각: {now_str} (순수 코인 상위 {len(analyzed)}개)\n"
    ]

    for idx, item in enumerate(analyzed):
        c = item["coin"]
        short_sym = c["symbol"].replace("USDT", "")
        lines.append(
            f"{emojis[idx]} ${short_sym} ({c['symbol']} | +{c['change_24h']:.2f}% | 현재가 {fmt_price(c['last_price'])} USDT)\n"
            f"• 거래대금: {c['turnover_24h']/1e6:.1f}M | 미결제약정(OI): {c['oi_value']/1e6:.1f}M | 펀딩비: {c['funding_rate']:+.4f}%\n"
            f"• [초동 발화점]: {item['ign_time']} (발화가 {fmt_price(item['ign_price'])} → 고점까지 +{item['pump_pct']:.1f}% 폭등)\n"
            f"• [발화 직전 컨펌된 초동 전략]:\n"
            f"  - 5분봉 이평선: {item['ma_str']}\n"
            f"  - 사전 매집: 직전 2시간 변동폭 {item['box_range_pct']:.1f}% 응축 | OBV {item['obv_str']}\n"
            f"  - 초동 수급: 평소 대비 거래량 [{item['vol_mult']:.1f}배 폭발]\n"
            f"  - 돌파 후 패턴: {item['post_pattern_str']}\n"
            f"• [🔍 포착된 신규 알파 전략]:\n"
            f"  - {' / '.join(item['alphas'])}\n"
        )

    # 당일 Top 5 적중 통계 집계
    n = len(analyzed)
    cnt_ma = sum(1 for x in analyzed if x["ma_aligned"])
    cnt_ma_full = sum(1 for x in analyzed if x["ma_full_aligned"])
    cnt_obv = sum(1 for x in analyzed if x["obv_up"])
    cnt_spring = sum(1 for x in analyzed if x["is_spring"])
    cnt_cvd = sum(1 for x in analyzed if x["has_cvd_div"])
    cnt_pocket = sum(1 for x in analyzed if x["is_vol_pocket"])
    cnt_ribbon = sum(1 for x in analyzed if x["is_ribbon"])
    cnt_bb_vwap = sum(1 for x in analyzed if x["is_bb_vwap"])
    cnt_ttm = sum(1 for x in analyzed if x["is_ttm"])
    cnt_htf = sum(1 for x in analyzed if x["is_htf_support"])
    cnt_oi = sum(1 for x in analyzed if x["is_oi_stealth"])
    avg_oi_chg = float(np.mean([x["oi_chg_1h_pct"] for x in analyzed]))
    cnt_rsi = sum(1 for x in analyzed if x["is_rsi_thrust"])
    cnt_macd = sum(1 for x in analyzed if x["is_macd_cross"])
    cnt_sqz = sum(1 for x in analyzed if x["is_short_sqz"])
    cnt_sr = sum(1 for x in analyzed if x["is_sr_flip"])
    cnt_ma20 = sum(1 for x in analyzed if x["is_ma20_sup"])

    # 당일 데이터 기반 파라미터 자동 산출
    avg_vol_mult = float(np.mean([x["vol_mult"] for x in analyzed]))
    min_vol_mult = float(np.min([x["vol_mult"] for x in analyzed]))
    rec_vol_min = round(max(2.2, min(3.2, min_vol_mult * 0.95)), 1)

    med_box = float(np.median([x["box_range_pct"] for x in analyzed]))
    rec_box = round(max(4.5, min(14.0, float(np.percentile([x["box_range_pct"] for x in analyzed], 80)))), 1)
    rec_dryup = round(max(0.15, min(0.35, float(np.median([x["dryup_ratio"] for x in analyzed]) + 0.08))), 2)

    # [5번 항목] 유동적 손절폭(SL: -2% ~ -5%) 및 저항·피보나치 비례 3단계 익절(TP1~TP3) 산출
    sl_list = [x["dynamic_sl_pct"] for x in analyzed]
    min_sl, max_sl = int(min(sl_list)), int(max(sl_list))
    rec_sl_int = int(round(float(np.median(sl_list))))

    med_pump = float(np.median([x["pump_pct"] for x in analyzed]))
    max_pump = float(np.max([x["pump_pct"] for x in analyzed]))

    tp1_pct = int(max(rec_sl_int * 2, round(med_pump * 0.65), 6))         # 1차: 최소 1:2 R:R & 단기 매물대(Fib 1.272)
    tp2_pct = int(max(tp1_pct + 5, round(med_pump * 1.25), 12))           # 2차: 1H 스윙고점 저항(Fib 1.618)
    tp3_pct = int(max(tp2_pct + 7, round(max_pump * 1.05), 20))           # 3차: 4H 메이저 고점 저항(Fib 2.618)

    lines.append(
        "━━━━━━━━━━━━━━━━━━\n"
        "💡 [내 전략(scanner.py) 실시간 업데이트 피드백 (제미나이 복붙용)]\n\n"
        "[1. 기존 vs 신규 글로벌 알파 전략 적중률 비교]\n"
        f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {cnt_ma}종목 (완전정배열 {cnt_ma_full}종목)\n"
        f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {cnt_obv}종목\n"
        f"• [신규] 비정배열 유동성 스윕(Spring) 반전: {n}종목 중 {cnt_spring}종목\n"
        f"• [신규] CVD 강세/흡수 다이버전스: {n}종목 중 {cnt_cvd}종목\n"
        f"• [신규] 상위프레임(1H 중심선) 구조적 지지 동조: {n}종목 중 {cnt_htf}종목\n"
        f"• [신규] 발화 직전 거래량 극건조 포켓(<=35%): {n}종목 중 {cnt_pocket}종목\n"
        f"• [신규] 이평선 리본 초밀집 확산(<=1.8%): {n}종목 중 {cnt_ribbon}종목\n"
        f"• [신규] BB스퀴즈 + VWAP 장대양봉 돌파: {n}종목 중 {cnt_bb_vwap}종목 (TTM스퀴즈 {cnt_ttm}종목)\n"
        f"• [신규] OI 선행매집(발화전 1h OI 급증): {n}종목 중 {cnt_oi}종목 (평균 {avg_oi_chg:+.1f}%)\n"
        f"• [신규] RSI 과매도 스러스트: {cnt_rsi}종목 | MACD 양전환 골든크로스: {cnt_macd}종목\n"
        f"• [신규] 숏스퀴즈 수급(펀딩<=0.01%): {n}종목 중 {cnt_sqz}종목\n"
        f"• [신규] 돌파 후 타점: S/R Flip(전고점 지지) {cnt_sr}종목 | MA20 지지 {cnt_ma20}종목\n\n"
        "[2. scanner.py 파라미터 튜닝 권장값]\n"
        f"• VOL_MULTIPLIER_MIN = {rec_vol_min} (초동 평균 {avg_vol_mult:.1f}x / 최소 {min_vol_mult:.1f}x)\n"
        f"• PRE_BOX_RANGE_MAX_PCT = {rec_box}% (직전 2시간 변동폭 중앙값 {med_box:.1f}%)\n"
        f"• PULLBACK_DRYUP_MAX_RATIO = {rec_dryup} (발화봉 대비 눌림 거래량 {int(rec_dryup*100)}% 이하)\n"
        f"• PRE_VOL_POCKET_MAX_RATIO = 0.35 (발화 직전 5캔들 최저거래량 / vol_ma20)\n"
        f"• EMA_RIBBON_MAX_SPREAD_PCT = 1.8% (5·20·50 이평선 밀집 수렴 기준)\n"
        f"• OI_PRE_SURGE_MIN_PCT = 3.0% (발화 전 1시간 OI 증가율)\n\n"
        "[3. scanner.py 추가할 신규 파이썬 조건식]\n"
        "• cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.4) and (close > ma20)\n"
        "• cond_cvd_div = (price_low_sweep) and (cvd_higher_low or cvd_stealth_slope >= 0.025)\n"
        f"• cond_vol_pocket = (pre5_min_vol / vol_ma20 <= 0.35) and (vol_mult >= {rec_vol_min})\n"
        "• cond_ema_ribbon = (pre5_min_ribbon_spread <= 1.8) and (close > ma50)\n"
        "• cond_bb_vwap = (bbw_percentile_4h <= 35) and (close > vwap48) and (body_ratio >= 0.60)\n"
        f"• cond_sr_flip_retest = (abs(pullback_low - prev_box_high) / prev_box_high <= 0.012) and (pullback_vol_ratio <= {rec_dryup})\n\n"
        "[4. 100점 만점 스코어링 배점 조정안]\n"
        "• 추세/반전(정배열 5>20>50 또는 Spring 유동성 스윕 반전 + 1H/4H 추세): 25점\n"
        "• 매집/건조(박스권 응축 + OBV/CVD 다이버전스 또는 발화전 거래량 포켓): 20점\n"
        "• 선물수급(OI 선행 급증 +3%↑ 또는 중립/음수 펀딩 숏스퀴즈): 20점\n"
        "• 변동성돌파(BB/TTM 스퀴즈 + 이평선 밀집 + VWAP/거래량 폭발): 15점\n"
        "• 타점컨펌(S/R Flip 전고점 또는 MA20 리테스트 + 거래량 건조): 20점 (총 100점, 85점↑ 알림)\n\n"
        "[5. 타점 / 유동적 손절(SL) / 3단계 익절(TP) 기준]\n"
        f"• 권장 유동적 손절폭(SL): -{rec_sl_int}% (종목별 스윙저점·ATR 기준 -{min_sl}%~-{max_sl}% 유동 적용)\n"
        f"• 권장 3단계 익절(TP): 1차 +{tp1_pct}% (1:2 R:R·단기저항/Fib 1.272) / 2차 +{tp2_pct}% (1H고점/Fib 1.618) / 3차 +{tp3_pct}% (4H메이저저항/Fib 2.618)"
    )

    report_text = "\n".join(lines)
    print(report_text)
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": report_text},
                timeout=12
            )
        except Exception as e:
            print(f"[Telegram 전송 오류] {e}")


if __name__ == "__main__":
    main()
