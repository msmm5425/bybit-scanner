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
STATE_FILE = "bot_state.json"

# =====================================================================
# [실시간 튜닝 파라미터]
# =====================================================================
SCORE_THRESHOLD = 85
VOL_MULTIPLIER_MIN = 3.5
PRE_BOX_RANGE_MAX_PCT = 8.7
PULLBACK_DRYUP_MAX_RATIO = 0.15
PRE_VOL_POCKET_MAX_RATIO = 0.35
EMA_RIBBON_MAX_SPREAD_PCT = 1.8
OI_PRE_SURGE_MIN_PCT = 3.0
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
        ("raw", f"https://api.bytick.com{path}{query_str}"),
        ("raw", target_bybit),
    ]

    for mode, url in routes:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
            if resp.status_code == 200:
                result = parse_bybit_payload(resp, mode)
                if result is not None:
                    time.sleep(0.8)
                    return result
        except Exception:
            pass
        time.sleep(0.2)

    return {}


def get_scanner_candidates():
    """프록시 소모 없이 미국 IP에서 즉시 열리는 CoinGecko Bybit Derivatives API로 후보군 추출"""
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
            if candidates:
                candidates.sort(
                    key=lambda x: x["change_24h"] * np.log10(max(10.0, x["turnover_24h"])),
                    reverse=True
                )
                return candidates[:15]
    except Exception:
        pass

    return []


def fetch_raw_5m_rows_multi_source(symbol, limit=150):
    """무중단 5분봉 수집 엔진 (Bitget / Gate.io / OKX / Bybit Proxy / CryptoCompare)"""
    base_coin = symbol[:-4].upper()
    clean_base = base_coin
    for prefix in ["1000000", "10000", "1000", "100"]:
        if clean_base.startswith(prefix) and len(clean_base) > len(prefix):
            clean_base = clean_base[len(prefix):]
            break

    # 1순위: Bitget USDT-FUTURES (초고속 직통)
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bg_url = "https://api.bitget.com/api/v2/mix/market/candles"
            resp = requests.get(bg_url, params={
                "symbol": sym_try,
                "productType": "USDT-FUTURES",
                "granularity": "5m",
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=8)
            if resp.status_code == 200:
                bg_data = resp.json().get("data", [])
                if bg_data and len(bg_data) >= 35:
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
            }, headers=BROWSER_HEADERS, timeout=8)
            if resp.status_code == 200:
                gt_data = resp.json()
                if isinstance(gt_data, list) and len(gt_data) >= 35:
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

    # 3순위: Bybit V5 API (프록시)
    res = fetch_bybit("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": "5",
        "limit": limit
    })
    rows = res.get("list", [])
    if rows and len(rows) >= 35:
        return rows

    # 4순위: Gate.io Spot
    for base_try in [base_coin, clean_base]:
        try:
            gs_url = "https://api.gateio.ws/api/v4/spot/candlesticks"
            resp = requests.get(gs_url, params={
                "currency_pair": f"{base_try}_USDT",
                "interval": "5m",
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=8)
            if resp.status_code == 200:
                gs_data = resp.json()
                if isinstance(gs_data, list) and len(gs_data) >= 35:
                    return [
                        [int(float(x[0])) * 1000, x[5], x[3], x[4], x[2], x[6], x[1]]
                        for x in gs_data
                    ]
        except Exception:
            pass

    # 5순위: CryptoCompare 5분봉
    for base_try in [clean_base, base_coin]:
        try:
            cc_url = "https://min-api.cryptocompare.com/data/v2/histominute"
            resp = requests.get(cc_url, params={
                "fsym": base_try,
                "tsym": "USDT",
                "limit": str(limit),
                "aggregate": "5"
            }, headers=BROWSER_HEADERS, timeout=8)
            if resp.status_code == 200:
                cc_rows = resp.json().get("Data", {}).get("Data", [])
                if cc_rows and len(cc_rows) >= 35:
                    converted = []
                    for item in cc_rows:
                        if to_float(item.get("close")) > 0:
                            ts_ms = int(item.get("time", 0)) * 1000
                            v = to_float(item.get("volumeto")) or to_float(item.get("volumefrom"), 1.0)
                            converted.append([
                                ts_ms, item.get("open"), item.get("high"),
                                item.get("low"), item.get("close"), v, v
                            ])
                    if len(converted) >= 35:
                        return converted
        except Exception:
            pass

    return []


def get_klines_5m(symbol, limit=150):
    rows = fetch_raw_5m_rows_multi_source(symbol, limit=limit)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    df["ma5"] = df["close"].rolling(5, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=20).mean()
    df["ma100"] = df["close"].rolling(100, min_periods=30).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=40).mean()

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

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap48"] = (
        (tp * df["volume"]).rolling(48, min_periods=10).sum()
        / df["volume"].rolling(48, min_periods=10).sum().replace(0, np.nan)
    )

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / candle_range).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(lower=0.0, upper=10.0)

    return df


def analyze_symbol(coin):
    symbol = coin["symbol"]
    df = get_klines_5m(symbol, limit=150)
    if len(df) < 40:
        return None

    recent_window = df.iloc[-16:-1].copy()
    breakout_cands = recent_window[
        (recent_window["close"] > recent_window["open"]) &
        (recent_window["vol_mult"] >= VOL_MULTIPLIER_MIN * 0.85)
    ]

    if not breakout_cands.empty:
        ign_idx = breakout_cands["vol_mult"].idxmax()
    else:
        valid_v = recent_window["vol_mult"].dropna()
        if valid_v.empty:
            return None
        ign_idx = valid_v.idxmax()

    ign_row = df.loc[ign_idx]
    curr_row = df.iloc[-1]
    pre_df = df.loc[max(0, ign_idx - 24):ign_idx - 1]
    pre_5_df = df.loc[max(0, ign_idx - 5):ign_idx - 1]
    post_df = df.loc[ign_idx + 1:]

    if pre_df.empty:
        return None

    current_price = to_float(curr_row["close"], default=coin["last_price"])
    prev_box_high = to_float(pre_df["high"].max(), default=current_price)
    prev_box_low = to_float(pre_df["low"].min(), default=current_price)
    box_range_2h_pct = ((prev_box_high - prev_box_low) / prev_box_low) * 100.0 if prev_box_low > 0 else 99.0

    # 1. 추세/반전 조건 (최대 20점)
    ma_short_aligned = bool(curr_row["ma5"] > curr_row["ma20"] > curr_row["ma50"])
    ma_full_aligned = bool(ma_short_aligned and (curr_row["ma50"] > curr_row["ma100"] > curr_row["ma200"]))

    recent_6_df = df.loc[max(0, ign_idx - 6):ign_idx]
    lower_wick_ratio_max6 = to_float(recent_6_df["lower_wick_ratio"].max(), default=0.0)
    cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.4) and (curr_row["close"] > curr_row["ma20"])

    score = 0
    matched_tags = []

    if ma_full_aligned:
        score += 20
        matched_tags.append("5분봉 완전정배열")
    elif ma_short_aligned:
        score += 18
        matched_tags.append("5분봉 단기정배열(5>20>50)")
    elif cond_spring:
        score += 20
        matched_tags.append("신규: 유동성 스윕(Spring 아랫꼬리 반전)")

    # 2. 사전 매집 & 발화 직전 거래량 극건조 포켓 (최대 20점)
    obv_accum = False
    if len(pre_df) >= 5:
        try:
            obv_slope = np.polyfit(np.arange(len(pre_df)), pre_df["obv"].values, 1)[0]
            obv_accum = bool(obv_slope > 0)
        except Exception:
            obv_accum = False

    vol_mult = to_float(ign_row["vol_mult"], default=1.0)
    pre5_min_vol = to_float(pre_5_df["volume"].min()) if not pre_5_df.empty else 0.0
    vol_ma20_ign = to_float(ign_row["vol_ma20"], default=1.0)
    pre_vol_ratio = pre5_min_vol / vol_ma20_ign if vol_ma20_ign > 0 else 1.0

    cond_vol_pocket = (pre_vol_ratio <= PRE_VOL_POCKET_MAX_RATIO) and (vol_mult >= VOL_MULTIPLIER_MIN)
    pre5_min_ribbon = to_float(pre_5_df["ribbon_spread_pct"].min(), default=99.0) if not pre_5_df.empty else 99.0
    cond_ema_ribbon = (pre5_min_ribbon <= EMA_RIBBON_MAX_SPREAD_PCT) and (curr_row["close"] > curr_row["ma50"])

    if cond_vol_pocket:
        score += 20
        matched_tags.append(f"신규: 발화전 거래량 극건조 포켓({pre_vol_ratio*100:.0f}% 수축)")
    elif box_range_2h_pct <= PRE_BOX_RANGE_MAX_PCT and obv_accum:
        score += 18
        matched_tags.append(f"박스권 응축({box_range_2h_pct:.1f}%)+OBV 매집")
    elif cond_ema_ribbon:
        score += 16
        matched_tags.append(f"신규: 이평선 리본 밀집({pre5_min_ribbon:.1f}%) 확산")
    elif obv_accum:
        score += 12
        matched_tags.append("OBV 우상향 매집")

    # 3. 선물 수급 및 숏스퀴즈 조건 (최대 15점)
    if coin["funding_rate"] <= FUNDING_NEUTRAL_MAX:
        score += 15
        matched_tags.append(f"신규: 중립/음수 펀딩 숏스퀴즈 우위({coin['funding_rate']:+.4f}%)")
    elif coin["oi_value"] >= 5_000_000:
        score += 10

    # 4. BB 스퀴즈 + VWAP 장악 + 초동 거래량 폭발 (최대 15점)
    pre_bbw = to_float(pre_df["bbw"].iloc[-1], default=10.0)
    bbw_win = df.loc[max(0, ign_idx - 48):ign_idx, "bbw"].dropna()
    bbw_percentile_4h = float((bbw_win < pre_bbw).mean() * 100.0) if not bbw_win.empty else 50.0
    body_ratio = to_float(ign_row["body_ratio"])
    vwap48 = to_float(ign_row["vwap48"], default=current_price)

    cond_bb_vwap = (bbw_percentile_4h <= 35.0) and (ign_row["close"] > vwap48) and (body_ratio >= 0.60)

    if cond_bb_vwap and vol_mult >= VOL_MULTIPLIER_MIN:
        score += 15
        matched_tags.append(f"신규: BB스퀴즈+VWAP 돌파(거래량 {vol_mult:.1f}x)")
    elif vol_mult >= VOL_MULTIPLIER_MIN:
        score += 13
        matched_tags.append(f"초동 거래량 폭발({vol_mult:.1f}x)")
    elif vol_mult >= 2.5:
        score += 8

    # 5. 돌파 후 눌림목 타점 컨펌 (최대 30점)
    pullback_df = post_df if not post_df.empty else df.iloc[-3:]
    pullback_low = to_float(pullback_df["low"].min(), default=current_price)
    pullback_min_vol = to_float(pullback_df["volume"].min(), default=0.0)
    ign_vol = max(to_float(ign_row["volume"]), 1e-8)
    pullback_vol_ratio = pullback_min_vol / ign_vol

    ma20_curr = to_float(curr_row["ma20"], default=current_price)
    sr_flip_dist = abs(pullback_low - prev_box_high) / prev_box_high if prev_box_high > 0 else 99.0
    ma20_dist = abs(current_price - ma20_curr) / ma20_curr if ma20_curr > 0 else 99.0

    cond_sr_flip_retest = (sr_flip_dist <= 0.012) and (pullback_vol_ratio <= PULLBACK_DRYUP_MAX_RATIO)
    cond_ma20_retest = (ma20_dist <= 0.015 and current_price >= ma20_curr * 0.99) and (pullback_vol_ratio <= 0.30)

    if cond_sr_flip_retest:
        score += 30
        matched_tags.append(f"신규: 전고점 S/R Flip 칼지지(거래량 {pullback_vol_ratio*100:.0f}% 건조)")
    elif cond_ma20_retest:
        score += 28
        matched_tags.append(f"MA20 지지 리테스트(거래량 {pullback_vol_ratio*100:.0f}% 건조)")
    elif pullback_vol_ratio <= 0.25 and current_price > ma20_curr:
        score += 22
        matched_tags.append(f"돌파 후 거래량 건조({pullback_vol_ratio*100:.0f}%) 눌림목")
    elif current_price > ma20_curr:
        score += 12

    if score < SCORE_THRESHOLD:
        return None

    entry_high = current_price
    entry_low = max(prev_box_high * 0.995, ma20_curr) if prev_box_high < current_price else ma20_curr
    if entry_low > entry_high:
        entry_low = current_price * 0.992
    avg_entry = (entry_high + entry_low) / 2.0

    atr5 = to_float(curr_row["atr14"], default=avg_entry * 0.015)
    atr15_approx = atr5 * 1.732

    raw_sl_pct = (1.5 * atr15_approx / avg_entry) * 100.0 if avg_entry > 0 else 3.0
    sl_pct = max(2, min(6, int(round(raw_sl_pct))))
    stop_price = avg_entry * (1.0 - sl_pct / 100.0)

    min_tp1_pct = max(sl_pct * 2, 6)
    tp2_pct = max(int(round(min_tp1_pct * 2.2)), 15)
    tp3_pct = max(int(round(min_tp1_pct * 3.5)), 23)
    tp4_pct = max(int(round(min_tp1_pct * 5.0)), 32)

    tps = [
        avg_entry * (1.0 + min_tp1_pct / 100.0),
        avg_entry * (1.0 + tp2_pct / 100.0),
        avg_entry * (1.0 + tp3_pct / 100.0),
        avg_entry * (1.0 + tp4_pct / 100.0),
    ]

    in_pullback_zone = bool(current_price <= entry_high * 1.005 and current_price >= stop_price * 1.005)

    return {
        "symbol": symbol,
        "score": int(min(100, score)),
        "change_24h": coin["change_24h"],
        "last_price": current_price,
        "turnover_m": coin["turnover_24h"] / 1_000_000.0,
        "oi_m": coin["oi_value"] / 1_000_000.0,
        "funding_rate": coin["funding_rate"],
        "entry_low": entry_low,
        "entry_high": entry_high,
        "avg_entry": avg_entry,
        "stop_price": stop_price,
        "sl_pct": sl_pct,
        "tp_pcts": [min_tp1_pct, tp2_pct, tp3_pct, tp4_pct],
        "tps": tps,
        "matched_tags": matched_tags,
        "in_pullback_zone": in_pullback_zone
    }


def send_telegram(text):
    print(text)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        time.sleep(0.4)
    except Exception as e:
        print(f"[Telegram 전송 오류] {e}")


def main():
    state = load_state()
    tracked = state.get("tracked", {})
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    for sym in list(tracked.keys()):
        if now_ts - tracked[sym].get("created_at", 0) > 86400:
            del tracked[sym]

    candidates = get_scanner_candidates()
    if not candidates:
        print("[알림] 현재 조건을 만족하는 후보군 데이터를 가져오지 못했습니다.")
        save_state(state)
        return

    alerts = []

    for coin in candidates:
        symbol = coin["symbol"]
        curr_p = coin["last_price"]

        if symbol in tracked:
            t_info = tracked[symbol]
            if not t_info.get("sl_alerted") and curr_p <= t_info["stop_price"]:
                t_info["sl_alerted"] = True
                alerts.append(
                    f"⚠️ [손절가 이탈 알림] ${symbol.replace('USDT', '')} ({symbol})\n"
                    f"• 현재가: {fmt_price(curr_p)} USDT (손절 기준가 {fmt_price(t_info['stop_price'])} 이탈)\n"
                    f"• 원칙에 따라 리스크 관리를 권장합니다."
                )
                continue

            if not t_info.get("pullback_alerted") and t_info["entry_low"] <= curr_p <= t_info["entry_high"]:
                t_info["pullback_alerted"] = True
                alerts.append(
                    f"🎯 [눌림목 타점 도달] ${symbol.replace('USDT', '')} ({symbol})\n"
                    f"• 현재가: {fmt_price(curr_p)} USDT (권장 분할매수 구간 진입)\n"
                    f"• 손절가: {fmt_price(t_info['stop_price'])} (-{t_info['sl_pct']}%)\n"
                    f"• 1차 목표가(1:2 R:R): {fmt_price(t_info['tps'][0])} (+{t_info['tp_pcts'][0]}%)"
                )

        try:
            res = analyze_symbol(coin)
            if not res:
                continue

            if symbol not in tracked:
                tracked[symbol] = {
                    "created_at": now_ts,
                    "entry_low": res["entry_low"],
                    "entry_high": res["entry_high"],
                    "stop_price": res["stop_price"],
                    "sl_pct": res["sl_pct"],
                    "tps": res["tps"],
                    "tp_pcts": res["tp_pcts"],
                    "pullback_alerted": res["in_pullback_zone"],
                    "sl_alerted": False
                }

                short_sym = symbol.replace("USDT", "")
                alerts.append(
                    f"🚀 [신규 알파 급등 후보 포착 | {res['score']}점] ${short_sym} ({symbol})\n"
                    f"🕒 포착 시각: {now_str}\n"
                    f"• 현재가: {fmt_price(res['last_price'])} USDT (24h +{res['change_24h']:.2f}%)\n"
                    f"• 거래대금: {res['turnover_m']:.1f}M | OI: {res['oi_m']:.1f}M | 펀딩비: {res['funding_rate']:+.4f}%\n"
                    f"• [포착된 전략 시그널]:\n"
                    f"  - {' / '.join(res['matched_tags'])}\n"
                    f"• [매매 전략 가이드]:\n"
                    f"  - 분할 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n"
                    f"  - 손절가(SL): {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%)\n"
                    f"  - 1차 익절(TP1): {fmt_price(res['tps'][0])} (+{res['tp_pcts'][0]}% | 1:2 R:R 보장)\n"
                    f"  - 2차 익절(TP2): {fmt_price(res['tps'][1])} (+{res['tp_pcts'][1]}%)\n"
                    f"  - 3차 익절(TP3): {fmt_price(res['tps'][2])} (+{res['tp_pcts'][2]}%)\n"
                    f"  - 4차 익절(TP4): {fmt_price(res['tps'][3])} (+{res['tp_pcts'][3]}%)"
                )
        except Exception as e:
            print(f"[스캔 예외] {symbol}: {e}")

    state["tracked"] = tracked
    save_state(state)

    if alerts:
        full_msg = "\n\n━━━━━━━━━━━━━━━━━━\n\n".join(alerts)
        send_telegram(full_msg)
    else:
        print(f"[{now_str}] 85점 이상 신규 포착 종목 없음.")


if __name__ == "__main__":
    main()
