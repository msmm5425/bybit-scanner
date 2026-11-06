import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BYBIT_ENDPOINTS = [
    "https://api.bybit.com",
    "https://api.bytick.com"
]

KST = timezone(timedelta(hours=9))
# 24시간(288캔들) 탐색 구간 전체에서 MA200이 NaN이 되지 않도록 500캔들(워밍업 200 + 탐색 300) 수집
KLINE_LIMIT = 500
REQUEST_DELAY_SEC = 0.12

TRADFI_SYMBOL_TYPES = {
    "stock", "etf", "forex", "commodity", "tradfi", "equity", "index"
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


def safe_idxmax(series, default=None):
    if series is None or series.empty:
        return default
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return default
    return valid.idxmax()


def safe_idxmin(series, default=None):
    if series is None or series.empty:
        return default
    valid = series.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return default
    return valid.idxmin()


def request_bybit(path, params=None):
    headers = {"User-Agent": "Mozilla/5.0"}

    for base_url in BYBIT_ENDPOINTS:
        url = f"{base_url}{path}"
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)

            if resp.status_code != 200:
                print(f"[Bybit] HTTP {resp.status_code}: {url}")
                continue

            data = resp.json()

            if data.get("retCode") != 0:
                print(
                    f"[Bybit] API 오류: retCode={data.get('retCode')}, "
                    f"retMsg={data.get('retMsg')}, url={url}"
                )
                continue

            time.sleep(REQUEST_DELAY_SEC)
            return data.get("result", {})

        except Exception as e:
            print(f"[Bybit] 요청 예외: {url} / {e}")
            continue

    return {}


def get_valid_crypto_symbols():
    """
    바이비트 /v5/market/instruments-info 메타데이터 검증을 통해
    TradFi Perps(주식/ETF/원자재/외환)를 배제한 순수 USDT 무기한 선물 목록 추출
    """
    valid_symbols = set()
    cursor = ""

    for _ in range(10):
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        res = request_bybit("/v5/market/instruments-info", params)
        items = res.get("list", [])

        if not items:
            break

        for item in items:
            symbol = str(item.get("symbol", ""))
            contract_type = str(item.get("contractType", ""))
            status = str(item.get("status", ""))
            quote_coin = str(item.get("quoteCoin", ""))
            base_coin = str(item.get("baseCoin", "")).upper()
            symbol_type = str(item.get("symbolType", "")).lower()
            sub_type = str(item.get("subType", "")).lower()

            if contract_type != "LinearPerpetual":
                continue
            if status != "Trading":
                continue
            if quote_coin != "USDT" or not symbol.endswith("USDT"):
                continue
            if base_coin in TRADFI_BLACKLIST:
                continue
            if symbol_type in TRADFI_SYMBOL_TYPES or sub_type in TRADFI_SYMBOL_TYPES:
                continue
            if any(t in symbol_type for t in TRADFI_SYMBOL_TYPES):
                continue

            valid_symbols.add(symbol)

        cursor = res.get("nextPageCursor", "")
        if not cursor:
            break

    return valid_symbols


def get_sorted_gainer_candidates():
    """바이비트 Futures -> Gainers 탭과 동일하게 24h 상승률 순 정렬"""
    valid_symbols = get_valid_crypto_symbols()

    if not valid_symbols:
        print("[경고] instruments-info 심볼 목록이 비어 있어 블랙리스트 및 캔들 필터로 폴백합니다.")

    res = request_bybit("/v5/market/tickers", {"category": "linear"})
    tickers = res.get("list", [])
    candidates = []

    for t in tickers:
        symbol = str(t.get("symbol", ""))

        if not symbol.endswith("USDT") or "-" in symbol:
            continue

        base_coin = symbol[:-4].upper()
        if base_coin in TRADFI_BLACKLIST:
            continue

        if valid_symbols and symbol not in valid_symbols:
            continue

        last_price = to_float(t.get("lastPrice"))
        turnover_24h = to_float(t.get("turnover24h"))

        if last_price <= 0 or turnover_24h <= 0:
            continue

        candidates.append({
            "symbol": symbol,
            "change_24h": to_float(t.get("price24hPcnt")) * 100.0,
            "turnover_24h": turnover_24h,
            "oi_value": to_float(t.get("openInterestValue")),
            "funding_rate": to_float(t.get("fundingRate")) * 100.0,
            "last_price": last_price
        })

    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates


def get_klines_5m(symbol, limit=KLINE_LIMIT):
    """5분봉 500캔들(약 41.6시간) 수집 후 이평선·OBV·BBW·VWAP·캔들 미세구조 계산"""
    res = request_bybit("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": "5",
        "limit": limit
    })

    rows = res.get("list", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=[
        "timestamp", "open", "high", "low", "close", "volume"
    ]).copy()

    if df.empty:
        return pd.DataFrame()

    # Pandas 2.0+ merge_asof 호환을 위해 int64로 명시적 고정
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    for ma in [5, 20, 50, 100, 200]:
        df[f"ma{ma}"] = df["close"].rolling(ma).mean()

    df["vol_ma20"] = df["volume"].shift(1).rolling(20).mean()
    df["vol_mult"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bbw"] = (
        (df["bb_upper"] - df["bb_lower"]) /
        df["ma20"].replace(0, np.nan)
    ) * 100.0

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol_sum = (tp * df["volume"]).rolling(48).sum()
    vol_sum = df["volume"].rolling(48).sum().replace(0, np.nan)
    df["vwap48"] = tp_vol_sum / vol_sum

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (
        (df["close"] - df["open"]).abs() / candle_range
    ).fillna(0.0)

    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(lower=0.0, upper=10.0)

    return df


def is_24_7_crypto_kline(df, coin):
    """
    최근 24시간(288캔들) 전체의 24/7 연속성을 검사하여
    미국/중국 증시 휴장 시간이 존재하는 TradFi 주식/ETF 선물을 실행 시각과 무관하게 100% 차단
    """
    if len(df) < 80:
        return False

    recent_24h = df.tail(288)

    # 1) 최근 24시간 내 캔들 간 타임스탬프 단절(>15분 = 900,000ms) 존재 시 증시 휴장 종목으로 판정
    max_time_gap = recent_24h["timestamp"].diff().max()
    if pd.notna(max_time_gap) and max_time_gap > 900_000:
        return False

    # 2) 최근 24시간 내 연속 거래량 0 스트릭이 12캔들(1시간) 이상 지속되면 휴장 종목으로 판정
    is_zero_vol = (recent_24h["volume"] <= 0).astype(int)
    if is_zero_vol.sum() >= 24:
        return False

    # 연속 0 거래량 최대 길이 계산
    zero_streaks = is_zero_vol.groupby((is_zero_vol != is_zero_vol.shift()).cumsum()).sum()
    if not zero_streaks.empty and int(zero_streaks.max()) >= 12:
        return False

    # 3) 펀딩비가 0.0000%이면서 변동폭 0(high == low) 더미 캔들이 과다한 미러링 종목 차단
    flat_candles = int((recent_24h["high"] == recent_24h["low"]).sum())
    if abs(coin["funding_rate"]) < 1e-7 and flat_candles >= 20:
        return False

    return True


def get_oi_history_5m(symbol):
    """
    바이비트 /v5/market/open-interest의 1회 한도(200개=16.6시간)를 보완하기 위해
    최대 2페이지(350개=약 29시간)를 수집하여 24시간 전체 발화 구간을 커버
    """
    all_rows = []
    end_time = None

    for _ in range(2):
        params = {
            "category": "linear",
            "symbol": symbol,
            "intervalTime": "5min",
            "limit": 200
        }
        if end_time is not None:
            params["endTime"] = end_time

        res = request_bybit("/v5/market/open-interest", params)
        rows = res.get("list", [])
        if not rows:
            break

        all_rows.extend(rows)

        # 가장 오래된 타임스탬프 이전 구간 조회
        try:
            oldest_ts = min(int(r["timestamp"]) for r in rows if "timestamp" in r)
            end_time = oldest_ts - 1
        except (ValueError, TypeError):
            break

        if len(rows) < 200:
            break

    if not all_rows:
        return pd.DataFrame()

    oi_df = pd.DataFrame(all_rows)
    if "timestamp" not in oi_df.columns or "openInterest" not in oi_df.columns:
        return pd.DataFrame()

    oi_df["timestamp"] = pd.to_numeric(oi_df["timestamp"], errors="coerce")
    oi_df["openInterest"] = pd.to_numeric(oi_df["openInterest"], errors="coerce")
    oi_df = oi_df.dropna(subset=["timestamp", "openInterest"]).copy()

    if oi_df.empty:
        return pd.DataFrame()

    oi_df["timestamp"] = oi_df["timestamp"].astype(np.int64)
    oi_df = oi_df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return oi_df


def analyze_coin(coin):
    symbol = coin["symbol"]
    df = get_klines_5m(symbol)

    if df.empty or not is_24_7_crypto_kline(df, coin):
        return None

    oi_df = get_oi_history_5m(symbol)

    if not oi_df.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            oi_df[["timestamp", "openInterest"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=300_000
        )
    else:
        df["openInterest"] = np.nan

    # 최근 24시간(288캔들) 범위 내에서 초동 발화 캔들 탐색 (앞부분은 MA200 워밍업 구간)
    start_search_idx = max(30, len(df) - 288)
    end_search_idx = len(df) - 3

    if end_search_idx <= start_search_idx:
        return None

    search_df = df.iloc[start_search_idx:end_search_idx].copy()
    if search_df.empty:
        return None

    candidates = search_df[
        (search_df["close"] > search_df["open"]) &
        (search_df["vol_mult"] >= 3.0) &
        ((search_df["close"] - search_df["open"]) /
         search_df["open"].replace(0, np.nan) >= 0.015)
    ]

    ign_idx = None
    if not candidates.empty:
        for idx in candidates.index:
            future_max = to_float(df.loc[idx:, "high"].max())
            ign_close = to_float(df.loc[idx, "close"])
            if ign_close <= 0:
                continue
            if (future_max - ign_close) / ign_close >= 0.08:
                ign_idx = idx
                break

        if ign_idx is None:
            ign_idx = safe_idxmax(candidates["vol_mult"], default=None)

    if ign_idx is None:
        ign_idx = safe_idxmax(search_df["vol_mult"], default=None)

    if ign_idx is None:
        return None

    ign_row = df.loc[ign_idx]
    pre_df = df.loc[max(0, ign_idx - 24):ign_idx - 1]
    post_df = df.loc[ign_idx + 1:]

    ign_time_kst = datetime.fromtimestamp(
        ign_row["timestamp"] / 1000.0, tz=KST
    ).strftime("%m/%d %H:%M KST")

    ign_price = to_float(ign_row["close"])
    post_high = to_float(df.loc[ign_idx:, "high"].max())
    pump_pct = ((post_high - ign_price) / ign_price) * 100.0 if ign_price > 0 else 0.0
    vol_mult = to_float(ign_row["vol_mult"], default=1.0)

    short_align = bool(
        pd.notna(ign_row["ma5"])
        and pd.notna(ign_row["ma20"])
        and pd.notna(ign_row["ma50"])
        and ign_row["ma5"] > ign_row["ma20"] > ign_row["ma50"]
    )

    full_align = bool(
        short_align
        and pd.notna(ign_row["ma100"])
        and pd.notna(ign_row["ma200"])
        and ign_row["ma50"] > ign_row["ma100"] > ign_row["ma200"]
    )

    box_high = to_float(pre_df["high"].max()) if not pre_df.empty else 0.0
    box_low = to_float(pre_df["low"].min()) if not pre_df.empty else 0.0
    box_range_pct = ((box_high - box_low) / box_low) * 100.0 if box_low > 0 else 0.0

    obv_accum = False
    if len(pre_df) >= 5:
        try:
            obv_vals = pre_df["obv"].values
            if np.isfinite(obv_vals).all() and np.std(obv_vals) > 0:
                obv_slope = np.polyfit(np.arange(len(pre_df)), obv_vals, 1)[0]
                obv_accum = bool(obv_slope > 0)
        except Exception:
            obv_accum = False

    new_strategies = []

    # [신규 전략 1] OI 선행 매집 다이버전스
    oi_pre_change = 0.0
    oi_ignition_change = 0.0

    if "openInterest" in df.columns and df["openInterest"].notna().any() and ign_idx >= 12:
        oi_1h_ago = to_float(df.loc[ign_idx - 12, "openInterest"])
        oi_1candle_ago = to_float(df.loc[ign_idx - 1, "openInterest"])
        oi_at_ign = to_float(df.loc[ign_idx, "openInterest"])

        if oi_1h_ago > 0 and oi_1candle_ago > 0:
            oi_pre_change = ((oi_1candle_ago - oi_1h_ago) / oi_1h_ago) * 100.0
        if oi_1candle_ago > 0 and oi_at_ign > 0:
            oi_ignition_change = ((oi_at_ign - oi_1candle_ago) / oi_1candle_ago) * 100.0

    stealth_oi = (oi_pre_change >= 3.0) and (box_range_pct <= 8.0)
    if stealth_oi:
        new_strategies.append(f"OI 선행매집(발화전 1h OI +{oi_pre_change:.1f}%)")

    # [신규 전략 2] BB 스퀴즈 + VWAP 장악 돌파
    pre_bbw = (
        to_float(pre_df["bbw"].iloc[-1], default=-1.0)
        if not pre_df.empty and pd.notna(pre_df["bbw"].iloc[-1])
        else -1.0
    )

    bb_squeeze_break = False
    bbw_percentile = 100.0

    if pre_bbw > 0:
        bbw_window = df.loc[max(0, ign_idx - 48):ign_idx, "bbw"].dropna()
        if not bbw_window.empty:
            bbw_percentile = float((bbw_window < pre_bbw).mean() * 100.0)
            vwap_break = bool(
                pd.notna(ign_row["vwap48"])
                and ign_row["close"] > ign_row["vwap48"]
                and ign_row["body_ratio"] >= 0.65
            )
            bb_squeeze_break = (bbw_percentile <= 35.0) and vwap_break
            if bb_squeeze_break:
                new_strategies.append(
                    f"BB스퀴즈(하위 {bbw_percentile:.0f}%)+VWAP 장악(몸통 {ign_row['body_ratio']*100:.0f}%)"
                )

    # [신규 전략 3] 비정배열 유동성 스윕 (Wyckoff Spring)
    recent_6_df = df.loc[max(0, ign_idx - 6):ign_idx]
    had_wick_sweep = bool((recent_6_df["lower_wick_ratio"] >= 1.5).any()) if not recent_6_df.empty else False
    spring_reversal = bool(
        (not short_align)
        and had_wick_sweep
        and pd.notna(ign_row["ma20"])
        and ign_row["close"] > ign_row["ma20"]
    )
    if spring_reversal:
        new_strategies.append("유동성 스윕(Spring 아랫꼬리 스탑헌팅 후 MA20 돌파)")

    # [신규 전략 4] 발화 직전 거래량 극건조 포켓 (Pre-Ignition Volume Dry-up Pocket)
    pre_5_df = df.loc[max(0, ign_idx - 5):ign_idx - 1]
    pre_vol_pocket = False
    min_pre_vol_ratio = 1.0
    if not pre_5_df.empty and pd.notna(ign_row["vol_ma20"]) and ign_row["vol_ma20"] > 0:
        min_pre_vol_ratio = to_float(pre_5_df["volume"].min() / ign_row["vol_ma20"], default=1.0)
        if min_pre_vol_ratio <= 0.35:
            pre_vol_pocket = True
            new_strategies.append(f"발화 직전 거래량 극건조 포켓(평소 대비 {min_pre_vol_ratio*100:.0f}%로 수축 후 폭발)")

    # [신규 전략 5 & 돌파 후 눌림목 정밀 분석] 1차 파동 고점 형성 후 진성 눌림목 탐색
    pullback_type = "눌림 없이 직행(Runaway)"
    dryup_ratio = 1.0
    retest_sl_pct = 3
    sr_flip_hit = False
    ma20_hit = False

    if len(post_df) >= 4:
        # 1단계: 초동 발화 직후 첫 1~6캔들 내 1차 단기 고점(Wave 1 High) 탐색
        wave1_window = df.loc[ign_idx:min(len(df) - 1, ign_idx + 5)]
        wave1_high_idx = safe_idxmax(wave1_window["high"], default=ign_idx)
        wave1_high_val = to_float(df.loc[wave1_high_idx, "high"], default=ign_price)

        # 2단계: 1차 고점 이후 구간에서 눌림목 저점 탐색
        pb_window = df.loc[wave1_high_idx + 1:min(len(df) - 1, wave1_high_idx + 15)]
        min_pb_idx = safe_idxmin(pb_window["low"], default=None)

        if min_pb_idx is not None:
            min_pb_row = pb_window.loc[min_pb_idx]
            pb_low = to_float(min_pb_row["low"])

            ign_volume = max(to_float(ign_row["volume"]), 1e-8)
            dryup_ratio = to_float(pb_window["volume"].min() / ign_volume, default=1.0)

            sr_diff_pct = abs(pb_low - box_high) / box_high * 100.0 if box_high > 0 else 99.0
            ma20_val = to_float(min_pb_row["ma20"])
            ma50_val = to_float(min_pb_row["ma50"])

            ma20_diff_pct = abs(pb_low - ma20_val) / ma20_val * 100.0 if ma20_val > 0 else 99.0
            ma50_diff_pct = abs(pb_low - ma50_val) / ma50_val * 100.0 if ma50_val > 0 else 99.0

            if sr_diff_pct <= 1.2 and pb_low >= box_high * 0.988:
                sr_flip_hit = True
                pullback_type = f"전고점 S/R Flip 칼지지(오차 {sr_diff_pct:.2f}%)"
                new_strategies.append(
                    f"돌파후 박스상단 S/R Flip 리테스트(거래량 {dryup_ratio*100:.0f}%로 건조)"
                )
            elif ma20_diff_pct <= 1.2:
                ma20_hit = True
                pullback_type = "5분봉 MA 20선 지지 리테스트"
            elif ma50_diff_pct <= 1.5:
                pullback_type = "5분봉 MA 50선 눌림 후 반등"
            else:
                pullback_pct = (
                    (wave1_high_val - pb_low) / wave1_high_val * 100.0
                    if wave1_high_val > 0 else 0.0
                )
                pullback_type = f"1차 고점 대비 -{pullback_pct:.1f}% 자율 눌림 후 재상승"

            pb_close = to_float(min_pb_row["close"])
            raw_sl = ((pb_close - (pb_low * 0.992)) / pb_close * 100.0) if pb_close > 0 else 3.0
            retest_sl_pct = max(2, min(6, int(round(raw_sl))))

    # [신규 전략 6] 숏스퀴즈 발화
    short_squeeze = bool(
        (coin["funding_rate"] <= 0.01)
        and (oi_ignition_change >= 1.5 or oi_pre_change >= 3.0)
    )
    if short_squeeze:
        new_strategies.append(
            f"숏스퀴즈 발화(펀딩 {coin['funding_rate']:+.4f}% + 발화OI +{oi_ignition_change:.1f}%)"
        )

    if not new_strategies:
        new_strategies.append("단독 대량거래량 모멘텀 돌파(평균 대비 고배수 수급 집중)")

    return {
        "symbol": symbol,
        "change_24h": coin["change_24h"],
        "last_price": coin["last_price"],
        "turnover_m": coin["turnover_24h"] / 1_000_000.0,
        "oi_m": coin["oi_value"] / 1_000_000.0,
        "funding_rate": coin["funding_rate"],
        "ign_time_kst": ign_time_kst,
        "ign_price": ign_price,
        "pump_pct": pump_pct,
        "short_align": short_align,
        "full_align": full_align,
        "box_range_pct": box_range_pct,
        "obv_accum": obv_accum,
        "vol_mult": vol_mult,
        "pullback_type": pullback_type,
        "dryup_ratio": dryup_ratio,
        "oi_pre_change": oi_pre_change,
        "oi_ignition_change": oi_ignition_change,
        "stealth_oi": stealth_oi,
        "bb_squeeze_break": bb_squeeze_break,
        "bbw_val": pre_bbw if pre_bbw > 0 else 0.0,
        "body_ratio": to_float(ign_row["body_ratio"]),
        "spring_reversal": spring_reversal,
        "pre_vol_pocket": pre_vol_pocket,
        "sr_flip_hit": sr_flip_hit,
        "ma20_hit": ma20_hit,
        "short_squeeze": short_squeeze,
        "retest_sl_pct": retest_sl_pct,
        "new_strategies": new_strategies
    }


def build_report(results):
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 신규 전략 발굴]",
        f"🕒 기준 시각: {now_str} (순수 암호화폐 선물 전용 / TradFi 제외)\n"
    ]

    num_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]

    for idx, r in enumerate(results):
        emoji = num_emojis[idx] if idx < len(num_emojis) else f"{idx + 1}."
        ticker_short = r["symbol"].replace("USDT", "")

        if r["full_align"]:
            align_str = "✅ 완전 정배열(5>20>50>100>200)"
        elif r["short_align"]:
            align_str = "✅ 단기 정배열(5>20>50)"
        else:
            align_str = "⚠️ 비정배열(역배열/혼조 돌파)"

        obv_str = "✅ 우상향 선행매집" if r["obv_accum"] else "➖ 중립/횡보"

        lines.append(
            f"{emoji} ${ticker_short} ({r['symbol']} | "
            f"+{r['change_24h']:.2f}% | "
            f"현재가 {r['last_price']:g} USDT)\n"
            f"• 거래대금: {r['turnover_m']:.2f}M | "
            f"미결제약정(OI): {r['oi_m']:.2f}M | "
            f"펀딩비: {r['funding_rate']:+.4f}%\n"
            f"• [초동 발화점]: {r['ign_time_kst']} "
            f"(발화가 {r['ign_price']:g} → "
            f"고점까지 +{r['pump_pct']:.1f}% 폭등)\n"
            f"• [기존 스캐너 전략 체크]:\n"
            f"  - 5분봉 이평선: [{align_str}]\n"
            f"  - 사전 매집: 직전 2시간 변동폭 "
            f"{r['box_range_pct']:.1f}% 응축 | OBV [{obv_str}]\n"
            f"  - 초동 수급: 평소 대비 거래량 "
            f"[{r['vol_mult']:.1f}배 폭발]\n"
            f"  - 돌파 후 패턴: {r['pullback_type']} "
            f"(눌림 최저거래량 {r['dryup_ratio']*100:.0f}% 수준)\n"
            f"• [🔍 포착된 신규 알파 전략]:\n"
            f"  - {' / '.join(r['new_strategies'])}\n"
        )

    n = len(results)

    cnt_short_align = sum(1 for r in results if r["short_align"])
    cnt_full_align = sum(1 for r in results if r["full_align"])
    cnt_obv = sum(1 for r in results if r["obv_accum"])
    cnt_stealth_oi = sum(1 for r in results if r["stealth_oi"])
    cnt_bb_vwap = sum(1 for r in results if r["bb_squeeze_break"])
    cnt_spring = sum(1 for r in results if r["spring_reversal"])
    cnt_vol_pocket = sum(1 for r in results if r["pre_vol_pocket"])
    cnt_sr_flip = sum(1 for r in results if r["sr_flip_hit"])
    cnt_ma20 = sum(1 for r in results if r["ma20_hit"])
    cnt_squeeze = sum(1 for r in results if r["short_squeeze"])

    avg_vol_mult = float(np.mean([r["vol_mult"] for r in results]))
    min_vol_mult = float(np.min([r["vol_mult"] for r in results]))
    rec_vol_mult = round(
        max(3.5, min(min_vol_mult * 0.85, avg_vol_mult * 0.6)), 1
    )

    med_box_range = float(np.median([r["box_range_pct"] for r in results]))
    rec_box_max = round(max(4.5, med_box_range * 1.15), 1)

    med_dryup = float(np.median([r["dryup_ratio"] for r in results]))
    rec_dryup_max = round(min(0.45, max(0.15, med_dryup * 1.2)), 2)

    avg_oi_pre = float(np.mean([r["oi_pre_change"] for r in results]))
    rec_sl_pct = int(round(np.median([r["retest_sl_pct"] for r in results])))
    avg_pump = float(np.mean([r["pump_pct"] for r in results]))

    tp1_pct = max(rec_sl_pct * 2, 6)
    tp2_pct = int(round(max(tp1_pct * 1.6, avg_pump * 0.45)))
    tp3_pct = int(round(max(tp2_pct * 1.4, avg_pump * 0.70)))
    tp4_pct = int(round(max(tp3_pct * 1.3, avg_pump * 0.95)))

    w_ma = 20 if cnt_short_align >= 3 else 10
    w_obv = 20 if (cnt_obv >= 3 or cnt_vol_pocket >= 2) else 15
    w_oi = 25 if (cnt_stealth_oi >= 2 or cnt_squeeze >= 2) else 15
    w_bb_vwap = 20 if cnt_bb_vwap >= 2 else 15
    w_retest = max(0, 100 - (w_ma + w_obv + w_oi + w_bb_vwap))

    lines.append(
        "━━━━━━━━━━━━━━━━━━\n"
        "🛠️ [scanner.py 업데이트용 엔지니어링 명세서 (제미나이 복붙용)]\n\n"
        "[1. 기존 vs 신규 전략 적중률 비교]\n"
        f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {cnt_short_align}종목 "
        f"(완전정배열 {cnt_full_align}종목)\n"
        f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {cnt_obv}종목\n"
        f"• [신규] OI 선행매집(발화전 1h OI 급증): {n}종목 중 {cnt_stealth_oi}종목 "
        f"(평균 {avg_oi_pre:+.1f}%)\n"
        f"• [신규] BB스퀴즈 + VWAP 장대양봉 돌파: {n}종목 중 {cnt_bb_vwap}종목\n"
        f"• [신규] 발화 직전 거래량 극건조 포켓(<=35%): {n}종목 중 {cnt_vol_pocket}종목\n"
        f"• [신규] 비정배열 유동성 스윕(Spring) 반전: {n}종목 중 {cnt_spring}종목\n"
        f"• [신규] 숏스퀴즈(펀딩<=0.01% & OI급증): {n}종목 중 {cnt_squeeze}종목\n"
        f"• [신규] 돌파 후 타점: S/R Flip(전고점 지지) {cnt_sr_flip}종목 | "
        f"MA20 지지 {cnt_ma20}종목\n\n"
        "[2. scanner.py 파라미터 수치 튜닝안]\n"
        f"• VOL_MULTIPLIER_MIN = {rec_vol_mult} "
        f"# 발화 평균 {avg_vol_mult:.1f}x / 최소 {min_vol_mult:.1f}x\n"
        f"• PRE_BOX_RANGE_MAX_PCT = {rec_box_max} "
        f"# 직전 2시간 변동폭 중앙값 {med_box_range:.1f}%\n"
        f"• PULLBACK_DRYUP_MAX_RATIO = {rec_dryup_max} "
        f"# 돌파봉 대비 눌림 거래량 {int(rec_dryup_max*100)}% 이하\n"
        "• PRE_VOL_POCKET_MAX_RATIO = 0.35 # 발화 직전 5캔들 최저 거래량 / vol_ma20\n"
        "• OI_PRE_SURGE_MIN_PCT = 3.0 # 발화 전 12캔들(1시간) OI 증가율(%)\n"
        "• FUNDING_NEUTRAL_MAX = 0.0100 # 숏스퀴즈 가산점 펀딩비 상한(%)\n\n"
        "[3. scanner.py 추가할 신규 조건식 (Python Logic)]\n"
        "• cond_stealth_oi = (oi_change_1h_pct >= 3.0) and "
        f"(box_range_2h_pct <= {rec_box_max})\n"
        "• cond_bb_vwap = (bbw_percentile_4h <= 35) and "
        "(close > vwap48) and (body_ratio >= 0.65)\n"
        "• cond_vol_pocket = (pre5_min_vol / vol_ma20 <= 0.35) and "
        f"(vol_mult >= {rec_vol_mult})\n"
        "• cond_spring = (not ma_short_aligned) and "
        "(lower_wick_ratio_max6 >= 1.5) and (close > ma20)\n"
        "• cond_sr_flip_retest = (abs(pullback_low - prev_box_high) / prev_box_high <= 0.012) and "
        f"(pullback_vol_ratio <= {rec_dryup_max})\n\n"
        "[4. 100점 만점 스코어링 배점 재조정안]\n"
        f"• 이평선 정배열(5>20>50) 또는 Spring 반전: {w_ma}점\n"
        f"• 박스권 응축 + OBV 우상향 + 발화전 거래량 포켓: {w_obv}점\n"
        f"• 선물 OI 선행 급증(+3%↑) & 중립/음수 펀딩비: {w_oi}점\n"
        f"• BB 스퀴즈 돌파 + VWAP 안착 + 거래량({rec_vol_mult}x↑): {w_bb_vwap}점\n"
        f"• S/R Flip(전고점) 또는 MA20 리테스트 + 거래량 건조: {w_retest}점 (총 100점, 85점↑ 알림)\n\n"
        "[5. 타점 / 손절(SL) / 4단계 익절(TP) 표준 파라미터]\n"
        "• 진입 타점: 돌파 후 직전 박스 상단(S/R Flip ±1.2%) 또는 5분봉 MA20 리테스트 양봉 마감\n"
        f"• 권장 손절폭(SL): -{rec_sl_pct}% (소수점 제외 정수, 리테스트 스윙 저점 이탈 기준)\n"
        f"• 권장 4단계 익절(TP): "
        f"1차 +{tp1_pct}% (R:R 1:2 보장) / 2차 +{tp2_pct}% / "
        f"3차 +{tp3_pct}% / 4차 +{tp4_pct}%"
    )

    return "\n".join(lines)


def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3900
    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"

    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            resp = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
                timeout=10
            )
            if resp.status_code != 200:
                print(f"[Telegram] 전송 실패 HTTP {resp.status_code}: {resp.text}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[Telegram] 전송 예외: {e}")


def main():
    candidates = get_sorted_gainer_candidates()

    if not candidates:
        send_telegram_message(
            "⚠️ 바이비트 상승률 상위 종목 데이터를 가져오지 못했습니다."
        )
        return

    results = []

    for coin in candidates[:30]:
        try:
            res = analyze_coin(coin)
            if res:
                results.append(res)
        except Exception as e:
            print(f"[분석 예외] {coin.get('symbol')}: {e}")

        if len(results) >= 5:
            break

    if not results:
        send_telegram_message("⚠️ 5분봉 상세 분석 조건을 만족하는 순수 암호화폐 데이터가 부족합니다.")
        return

    report_text = build_report(results)
    send_telegram_message(report_text)


if __name__ == "__main__":
    main()
