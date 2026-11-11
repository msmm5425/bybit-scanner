import numpy as np
import pandas as pd

from config import (
    SCORE_THRESHOLD,
    PULLBACK_DRYUP_STRICT,
    PULLBACK_DRYUP_MAX,
    SR_FLIP_DRYUP_RATIO_MAX,
    SR_FLIP_TOLERANCE_PCT,
    PRE_BOX_RANGE_MAX_PCT,
    PRE_BOX_LOOSE_MAX_PCT,
    PRE_BOX_TIGHT_PCT,
    EMA_RIBBON_MAX_SPREAD_PCT,
    PRE_VOL_POCKET_MAX_RATIO,
    EARLY_VOL_MULT_MIN,
    EARLY_VOL_MULT_MAX,
    EARLY_BREAKOUT_MAX_PCT,
    MAX_DRAWDOWN_FROM_12H_HIGH_PCT,
    MAX_PRIOR_PUMP_PCT,
    MAX_RETRACEMENT_RATIO,
    CUTOFF_RSI6_15M_MAX,
    CUTOFF_KDJ_J_15M_MAX,
    CUTOFF_BB_MID_EXT_MAX_PCT,
    CUTOFF_RSI6_1H_MAX,
    to_float,
)
from api import fetch_raw_klines_multi_source, clean_and_scale_df
from notifier import fmt_price


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
    high_low = (df["high"] - df["low"]).replace(0, np.nan)
    clv = (((df["close"] - df["low"]) - (df["high"] - df["close"])) / high_low).fillna(0.0)

    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    wick_share = (lower_wick / high_low).fillna(0.0)
    absorption_boost = np.where(df["close"] >= df["open"], wick_share * 0.45, wick_share * 0.20)

    effective_delta_ratio = np.clip(clv + absorption_boost, -1.0, 1.0)
    vol_delta = df["volume"] * effective_delta_ratio
    return vol_delta.cumsum()


def detect_cvd_divergence(df, lookback=24):
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


def get_multi_tf_data(symbol, ref_price=0.0):
    rows_5m = fetch_raw_klines_multi_source(symbol, interval="5m", limit=200)
    df = clean_and_scale_df(rows_5m, ref_price=ref_price, min_len=60, allow_fine_scale=False)
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
    df_1h = clean_and_scale_df(rows_1h, ref_price=curr_p, min_len=24, allow_fine_scale=True)
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


def calc_precise_entry_zone(df_5m, df_15m, current_price, track_type, box_high, swing_high_8):
    curr = df_5m.iloc[-1]
    prev = df_5m.iloc[-2]

    atr5 = to_float(curr.get("atr14"), default=current_price * 0.006)
    atr_pct = (atr5 / current_price) * 100.0 if current_price > 0 else 0.6
    atr_pct = max(0.30, min(1.20, atr_pct))

    ma5_5m = to_float(curr.get("ma5"), default=current_price)
    ma20_5m = to_float(curr.get("ma20"), default=current_price)
    vwap12_5m = to_float(curr.get("vwap12"), default=current_price)
    ma20_15m = to_float(df_15m["ma20"].iloc[-1], default=current_price)

    curr_open = to_float(curr.get("open"), default=current_price)
    candle_mid = (curr_open + current_price) / 2.0
    prev_close = to_float(prev.get("close"), default=current_price)

    curr_gain_pct = ((current_price - curr_open) / curr_open) * 100.0 if curr_open > 0 else 0.0
    if curr_gain_pct >= 1.2:
        entry_high = current_price
    else:
        upper_buf_pct = min(0.15, max(0.06, atr_pct * 0.18))
        entry_high = current_price * (1.0 + upper_buf_pct / 100.0)

    min_reach_price = current_price * (1.0 - max(0.45, min(1.10, atr_pct * 1.15)) / 100.0)
    max_reach_price = current_price * (1.0 - max(0.22, min(0.40, atr_pct * 0.40)) / 100.0)

    tech_candidates = [ma5_5m, ma20_5m, vwap12_5m, ma20_15m, candle_mid, prev_close]
    if track_type == "TRACK_B":
        tech_candidates.extend([box_high, swing_high_8])

    valid_supports = [
        p for p in tech_candidates
        if min_reach_price <= p <= max_reach_price
    ]

    atr_pullback_price = current_price - (0.65 * atr5)
    atr_pullback_price = max(min_reach_price, min(max_reach_price, atr_pullback_price))

    if valid_supports:
        struct_anchor = float(np.mean(valid_supports))
        entry_low = (struct_anchor * 0.65) + (atr_pullback_price * 0.35)
    else:
        if min_reach_price <= candle_mid <= max_reach_price:
            entry_low = (candle_mid * 0.6) + (atr_pullback_price * 0.4)
        else:
            entry_low = atr_pullback_price

    if entry_low >= entry_high * 0.9975:
        entry_low = entry_high * (1.0 - max(0.32, atr_pct * 0.55) / 100.0)

    return entry_low, entry_high


def calc_strategic_altcoin_sl(df_5m, df_15m, df_1h, avg_entry, box_low, box_range_pct, track_type, is_spring, symbol):
    """48h 사후검증 피드백 반영: 표준 유동 손절폭 -3% ~ -5% (극단적 스윕 시 최대 -6%)"""
    if not df_1h.empty and len(df_1h) >= 6:
        recent_1h = df_1h.tail(12)
        h_range_pct_series = ((recent_1h["high"] - recent_1h["low"]) / recent_1h["low"].replace(0, np.nan)) * 100.0
        avg_1h_range_pct = to_float(h_range_pct_series.mean(), default=2.0)
        max_1h_range_pct = to_float(h_range_pct_series.max(), default=3.2)
    else:
        avg_1h_range_pct = 2.0
        max_1h_range_pct = 3.2

    swing_low_2h = min(box_low, to_float(df_5m.iloc[-24:]["low"].min(), default=avg_entry * 0.965))
    ma50_15m = to_float(df_15m["ma50"].iloc[-1], default=swing_low_2h)
    structural_support = min(swing_low_2h, ma50_15m) if ma50_15m < avg_entry else swing_low_2h

    dist_to_box_low_pct = max(0.8, ((avg_entry - swing_low_2h) / avg_entry) * 100.0) if avg_entry > 0 else 2.2

    wick_max_12 = to_float(df_5m.iloc[-12:]["lower_wick_ratio"].max(), default=1.0)
    sweep_buffer_pct = max(1.0, avg_1h_range_pct * 0.80)
    if is_spring or wick_max_12 >= 1.8:
        sweep_buffer_pct += 0.65

    sl_by_structure = dist_to_box_low_pct + sweep_buffer_pct
    sl_by_1h_volatility = (avg_1h_range_pct * 1.35) + (max_1h_range_pct * 0.25)
    sl_by_box_range = max(2.8, box_range_pct * 0.50)

    raw_sl = max(sl_by_structure, sl_by_1h_volatility, sl_by_box_range)

    if symbol in ("BTCUSDT", "ETHUSDT"):
        sl_pct = max(2, min(4, int(round(raw_sl * 0.75))))
        sl_reason = "메이저 구조적 지지+ATR버퍼"
    else:
        min_alt_sl = 4 if (track_type == "TRACK_A" or is_spring) else 3
        max_alt_sl = 6 if (is_spring and box_range_pct >= 8.5) else 5
        sl_pct = max(min_alt_sl, min(max_alt_sl, int(round(raw_sl))))

        if sl_pct == 6:
            sl_reason = "알트 딥스윕(털기) 최대방어선"
        elif sl_pct == 5:
            sl_reason = "48h검증 표준 스윕방어선(-5%)"
        elif sl_pct == 4:
            sl_reason = "15m/1H 매집지지선+ATR버퍼"
        else:
            sl_reason = "S/R지지선+단기변동성버퍼"

    stop_price = avg_entry * (1.0 - sl_pct / 100.0)
    return sl_pct, stop_price, structural_support, sl_reason


def calc_structural_fib_3tps(df_5m, df_15m, df_1h, df_4h, avg_entry, sl_pct):
    """48h 사후검증 피드백 반영: Fib 2.53(1차 +10%권) / Fib 3.87(2차 +15%권) / 3차(+22%권)"""
    candidates = []

    recent_low = to_float(df_5m.iloc[-36:]["low"].min(), default=avg_entry * 0.95)
    recent_high = to_float(df_5m.iloc[-36:]["high"].max(), default=avg_entry * 1.03)
    wave_height = max(recent_high - recent_low, avg_entry * 0.028)

    for fib_ratio, fib_name in [
        (1.618, "Fib 1.618"),
        (2.130, "Fib 2.13(1차표준)"),
        (2.530, "Fib 2.53(15m매물대)"),
        (3.060, "Fib 3.06(2차표준)"),
        (3.860, "Fib 3.86(3차확장)"),
        (4.790, "Fib 4.79(대시세)"),
    ]:
        fib_p = recent_low + wave_height * fib_ratio
        if fib_p > avg_entry * 1.03:
            candidates.append((fib_p * 0.998, fib_name))

    if not df_1h.empty:
        htf_low = to_float(df_1h["low"].min(), default=recent_low)
        htf_high = to_float(df_1h["high"].max(), default=recent_high)
        htf_range = max(htf_high - htf_low, avg_entry * 0.08)

        for r_ratio, r_name in [
            (0.618, "1H Fib 0.618"),
            (0.786, "1H Fib 0.786"),
            (1.000, "1H 전고점"),
            (1.120, "1H Fib 1.12(48h고점)"),
            (1.272, "1H Fib 1.272"),
            (1.618, "1H Fib 1.618"),
            (2.130, "1H Fib 2.13"),
        ]:
            htf_p = htf_low + htf_range * r_ratio
            if htf_p > avg_entry * 1.03:
                candidates.append((htf_p * 0.998, r_name))

        for pv in extract_pivot_highs(df_1h, left=2, right=2):
            if pv > avg_entry * 1.03:
                candidates.append((pv * 0.998, "1H 핵심고점"))

        ma99_1h = to_float(df_1h["ma99"].iloc[-1], default=0.0)
        bb_up_1h = to_float(df_1h["bb_upper"].iloc[-1], default=0.0)
        if ma99_1h > avg_entry * 1.035:
            candidates.append((ma99_1h * 0.997, "1H MA99저항"))
        if bb_up_1h > avg_entry * 1.035:
            candidates.append((bb_up_1h * 0.998, "1H BB상단"))

    for pv15 in extract_pivot_highs(df_15m, left=2, right=2):
        if pv15 > avg_entry * 1.03:
            candidates.append((pv15 * 0.998, "15m 스윙매물대"))

    if not df_4h.empty:
        h_4h_max = to_float(df_4h["high"].max(), default=0.0)
        bb_up_4h = to_float(df_4h["bb_upper"].iloc[-1], default=0.0)
        if h_4h_max > avg_entry * 1.05:
            candidates.append((h_4h_max * 0.997, "4H 메이저고점"))
        if bb_up_4h > avg_entry * 1.05:
            candidates.append((bb_up_4h * 0.998, "4H BB상단"))

    valid_levels = []
    for price_val, label in sorted(candidates, key=lambda x: x[0]):
        pct = ((price_val - avg_entry) / avg_entry) * 100.0
        if pct < 4.5 or pct > 70.0:
            continue
        if valid_levels and abs(pct - valid_levels[-1][1]) < 1.5:
            if "고점" in label or "매물대" in label or "표준" in label:
                valid_levels[-1] = (price_val, round(pct, 1), label)
            continue
        valid_levels.append((price_val, round(pct, 1), label))

    atr5 = to_float(df_5m["atr14"].iloc[-1], default=avg_entry * 0.01)
    vol_unit_pct = max(2.2, min(6.5, (atr5 * 1.85 / avg_entry) * 100.0))

    min_tp1_pct = max(round(sl_pct * 1.7, 1), round(vol_unit_pct * 2.4, 1), 7.5)
    tp1_choice = None
    for p, pct, lbl in valid_levels:
        if min_tp1_pct <= pct <= min_tp1_pct + 6.0:
            tp1_choice = (p, pct, lbl)
            break
    if not tp1_choice:
        fallback_tp1_pct = round(max(10.0, min_tp1_pct), 1)
        tp1_choice = (avg_entry * (1.0 + fallback_tp1_pct / 100.0), fallback_tp1_pct, "Fib 2.13(15m매물대)")

    min_tp2_pct = max(tp1_choice[1] + max(4.5, vol_unit_pct * 1.7), round(sl_pct * 2.9, 1), 13.5)
    tp2_choice = None
    for p, pct, lbl in valid_levels:
        if min_tp2_pct <= pct <= min_tp2_pct + 9.0:
            tp2_choice = (p, pct, lbl)
            break
    if not tp2_choice:
        fallback_tp2_pct = round(max(15.0, min_tp2_pct), 1)
        tp2_choice = (avg_entry * (1.0 + fallback_tp2_pct / 100.0), fallback_tp2_pct, "Fib 3.06(1H핵심고점)")

    min_tp3_pct = max(tp2_choice[1] + max(6.5, vol_unit_pct * 2.3), round(sl_pct * 4.2, 1), 21.0)
    tp3_choice = None
    for p, pct, lbl in valid_levels:
        if min_tp3_pct <= pct <= min_tp3_pct + 22.0:
            tp3_choice = (p, pct, lbl)
            break
    if not tp3_choice:
        fallback_tp3_pct = round(max(22.0, min_tp3_pct), 1)
        tp3_choice = (avg_entry * (1.0 + fallback_tp3_pct / 100.0), fallback_tp3_pct, "Fib 3.86(48h고점)")

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

    cvd_div_5m, cvd_tag_5m = detect_cvd_divergence(df, lookback=24)
    cvd_div_15m, cvd_tag_15m = detect_cvd_divergence(df_15m, lookback=14)
    has_cvd_div = bool(cvd_div_5m or cvd_div_15m)
    cvd_label = cvd_tag_5m if cvd_div_5m else cvd_tag_15m

    recent_low_6 = to_float(df.iloc[-6:]["low"].min(), default=current_price)

    ma5_5m = to_float(curr["ma5"], default=current_price)
    ma20_5m = to_float(curr["ma20"], default=current_price)
    ma50_5m = to_float(curr["ma50"], default=current_price)
    ma112_5m = to_float(curr["ma112"], default=current_price)
    ma_short_aligned = bool(ma5_5m > ma20_5m > ma50_5m)
    ma_full_aligned = bool(ma_short_aligned and ma50_5m > ma112_5m)

    is_15m_ma20_falling = bool(bb_mid_15m < prev_bb_mid_15m * 0.999)

    closed_3_min_vol = to_float(df.iloc[-4:-1]["volume"].min(), default=0.0)
    ref_vol_ma20 = to_float(prev["vol_ma20"], default=1.0)
    recent_3_vol_ratio = (closed_3_min_vol / ref_vol_ma20) if ref_vol_ma20 > 0 else 1.0

    max_wick_6 = to_float(df.iloc[-6:-1]["lower_wick_ratio"].max(), default=0.0)
    curr_wick = to_float(curr.get("lower_wick_ratio"), default=0.0)
    max_wick_all6 = max(max_wick_6, curr_wick)

    pre8_min_rsi6 = to_float(df.iloc[-9:-1]["rsi6"].min(), default=50.0)
    curr_rsi6 = to_float(curr["rsi6"], default=50.0)

    cond_rsi_thrust = bool(pre8_min_rsi6 <= 36.0 and curr_rsi6 >= 54.0 and (curr_rsi6 - pre8_min_rsi6) >= 20.0)
    macd_curr = to_float(curr.get("macd_hist"), default=0.0)
    macd_prev = to_float(prev.get("macd_hist"), default=0.0)
    cond_macd_golden = bool(macd_curr > 0 and (macd_prev <= 0 or to_float(df["macd_hist"].iloc[-3], default=0.0) <= 0))

    cond_spring = bool(
        (not ma_short_aligned)
        and max_wick_all6 >= 1.4
        and current_price > ma20_5m
    )

    recent_squeeze = bool(df.iloc[-7:-1]["ttm_squeeze_on"].any())
    cond_ttm_squeeze = bool(recent_squeeze and current_price > ma20_5m and curr["close"] > curr["open"])

    min_bbw_rank_5 = to_float(df.iloc[-6:-1]["bbw_pct_rank"].min(), default=50.0)
    vwap12_val = to_float(curr.get("vwap12"), default=current_price)
    cond_bb_vwap_breakout = bool(
        min_bbw_rank_5 <= 30.0
        and current_price > vwap12_val
        and curr["close"] > curr["open"]
        and to_float(curr["body_ratio"]) >= 0.50
    )

    pre5_min_ribbon = to_float(df.iloc[-6:-1]["ribbon_spread_pct"].min(), default=99.0)
    cond_ema_ribbon = bool(pre5_min_ribbon <= EMA_RIBBON_MAX_SPREAD_PCT and current_price > ma50_5m)
    cond_htf_center_sup = bool(current_price >= ma112_5m * 0.996 and to_float(df.iloc[-4:-1]["low"].min()) <= ma112_5m * 1.015)

    funding_rate = to_float(coin["funding_rate"], default=0.01)
    cond_strong_short_squeeze = bool(funding_rate <= -0.015)
    cond_clean_funding = bool(funding_rate <= 0.010)

    # [신규 알파 4 - NIGHTUSDT 패턴] 미결제약정(OI) 탄탄한 유입 및 포지션 유지 강도
    turnover_m = to_float(coin["turnover_24h"]) / 1_000_000.0
    oi_m = to_float(coin["oi_value"]) / 1_000_000.0
    oi_turnover_ratio = (oi_m / turnover_m) if turnover_m > 0 else 0.0
    cond_strong_oi = bool(oi_m >= 2.0 and oi_turnover_ratio >= 0.25)

    # =====================================================================
    # Track A 채점 (1차 선취매 / 눌림 타점)
    # =====================================================================
    score_a = 0
    tags_a = []
    if kdj_j_15m <= 68.0 and not is_15m_ma20_falling:
        if ma_full_aligned and current_price >= ma20_5m * 0.998:
            score_a += 25
            tags_a.append("완전 정배열(5>20>50>112) 지지")
        elif ma_short_aligned and current_price >= ma20_5m * 0.998:
            score_a += 23
            tags_a.append("단기 정배열(5>20>50) 안착")
        elif cond_spring and (obv_accum or has_cvd_div or cond_rsi_thrust or box_range_pct <= PRE_BOX_RANGE_MAX_PCT):
            score_a += 25
            tags_a.append(f"유동성 스윕(Spring)+MA20 돌파(꼬리 {max_wick_all6:.1f}x)")

        if box_range_pct <= PRE_BOX_TIGHT_PCT and (obv_accum or has_cvd_div):
            score_a += 25
            tags_a.append(f"초응축({box_range_pct:.1f}%)+{'CVD 다이버전스' if has_cvd_div else 'OBV 선행매집'}")
        elif box_range_pct <= 4.5 and (cond_ema_ribbon or recent_3_vol_ratio <= PRE_VOL_POCKET_MAX_RATIO):
            score_a += 22
            tags_a.append(f"극초응축 박스({box_range_pct:.1f}%)+수평에너지 압축")
        elif box_range_pct <= PRE_BOX_RANGE_MAX_PCT and (obv_accum and has_cvd_div):
            score_a += 25
            tags_a.append(f"박스응축({box_range_pct:.1f}%)+OBV·CVD 동시매집")
        elif box_range_pct <= PRE_BOX_LOOSE_MAX_PCT and (obv_accum or has_cvd_div):
            score_a += 20
            tags_a.append(f"박스응축({box_range_pct:.1f}%)+{'CVD 매집' if has_cvd_div else 'OBV 매집'}")

        if 0.001 <= recent_3_vol_ratio <= PULLBACK_DRYUP_STRICT:
            score_a += 20
            tags_a.append(f"거래량 극건조 포켓({recent_3_vol_ratio*100:.0f}%)")
        elif 0.001 <= recent_3_vol_ratio <= PRE_VOL_POCKET_MAX_RATIO:
            score_a += 16
            tags_a.append(f"거래량 건조 포켓({recent_3_vol_ratio*100:.0f}%)")

        if cond_htf_center_sup or htf_bullish:
            score_a += 8
            tags_a.append("상위프레임(1H 중심선) 구조적 지지")

        if cond_ema_ribbon:
            score_a += 10
            tags_a.append(f"이평리본 초밀집({pre5_min_ribbon:.1f}%)")
        elif cond_ttm_squeeze or cond_bb_vwap_breakout:
            score_a += 10
            tags_a.append("TTM/BB 스퀴즈 압축")

        if cond_rsi_thrust and cond_macd_golden:
            score_a += 8
            tags_a.append("RSI 스러스트+MACD 양전환")

        if cond_strong_oi:
            score_a += 8
            tags_a.append(f"OI 실수급 유입(OI {oi_m:.1f}M·비중 {oi_turnover_ratio*100:.0f}%)")

        if cond_strong_short_squeeze:
            score_a += 15
            tags_a.append(f"강력 숏스퀴즈 수급({funding_rate:+.4f}%)")
        elif cond_clean_funding:
            score_a += 10
            tags_a.append(f"청정 펀딩 수급({funding_rate:+.4f}%)")

    # =====================================================================
    # Track B 채점 (2차 돌파 / S/R Flip·눌림목 스탑헌팅 리테스트 타점)
    # =====================================================================
    score_b = 0
    tags_b = []
    swing_high_8 = to_float(df.iloc[-12:-2]["high"].max(), default=current_price)
    breakout_diff_pct = ((current_price - swing_high_8) / swing_high_8) * 100.0 if swing_high_8 > 0 else 99.0
    reclaimed_bb_mid = bool(current_price > bb_mid_15m and current_price > ma20_5m and not is_15m_ma20_falling)
    is_early_breakout = bool(-0.2 <= breakout_diff_pct <= EARLY_BREAKOUT_MAX_PCT and curr["close"] > curr["open"])

    recent_5_bars = df.iloc[-7:-1]
    peak_breakout_vol = to_float(recent_5_bars["volume"].max(), default=0.0)
    peak_breakout_mult = to_float(recent_5_bars["vol_mult"].max(), default=1.0)

    eval_pullback_vol = to_float(curr["volume"]) if to_float(curr["volume"]) >= ref_vol_ma20 * 0.02 else to_float(prev["volume"])
    pullback_vol_vs_peak = (eval_pullback_vol / peak_breakout_vol) if peak_breakout_vol > 0 else 1.0

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
        and (ma20_err_pct <= 0.70 or ma50_err_pct <= 0.70)
        and pullback_vol_vs_peak <= PULLBACK_DRYUP_STRICT
    )

    # [신규 알파 5 - NIGHTUSDT 전문가 타점] 돌파 후 눌림목에서 롱 청산 스윕(아랫꼬리) 후 말아올림
    cond_pullback_sweep = bool(
        (peak_breakout_mult >= 2.5 or cond_strong_oi)
        and 1.0 <= drawdown_from_high_pct <= 5.5
        and max_wick_all6 >= 1.35
        and current_price >= ma50_5m * 0.998
        and htf_bullish
        and (has_cvd_div or obv_accum or cond_strong_oi)
    )

    max_vol_mult_2 = max(curr_vol_mult, prev_vol_mult)

    if cond_sr_flip_retest:
        score_b += 25
        tags_b.append(f"전고점 S/R Flip 칼지지(오차 {sr_flip_err_pct:.2f}%/눌림거래량 {pullback_vol_vs_peak*100:.0f}%)")
    elif cond_ma_pullback_retest:
        sup_name = "MA20" if ma20_err_pct <= 0.70 else "MA50"
        score_b += 25
        tags_b.append(f"돌파후 5m {sup_name} 칼지지(눌림거래량 {pullback_vol_vs_peak*100:.0f}%)")
    elif cond_pullback_sweep:
        score_b += 24
        tags_b.append(f"눌림목 롱청산 스윕(꼬리 {max_wick_all6:.1f}x) 후 지지 반등")
    elif reclaimed_bb_mid and is_early_breakout and EARLY_VOL_MULT_MIN <= max_vol_mult_2 <= EARLY_VOL_MULT_MAX and drawdown_from_high_pct <= 4.5:
        score_b += 22
        if cond_spring:
            tags_b.append(f"유동성 스윕(Spring)+스윙고점({fmt_price(swing_high_8)}) 돌파")
        else:
            tags_b.append(f"스윙고점({fmt_price(swing_high_8)}) 돌파 초입")

    pre5_min_vol_ratio = (
        to_float(df.iloc[-8:-2]["volume"].min() / ref_vol_ma20, default=1.0)
        if ref_vol_ma20 > 0 else 1.0
    )
    if (cond_sr_flip_retest or cond_ma_pullback_retest or cond_pullback_sweep) and peak_breakout_mult >= EARLY_VOL_MULT_MIN:
        score_b += 15
        tags_b.append(f"초동수급 폭발({peak_breakout_mult:.1f}배)")
    elif EARLY_VOL_MULT_MIN <= max_vol_mult_2 <= EARLY_VOL_MULT_MAX and max(to_float(curr["body_ratio"]), to_float(prev["body_ratio"])) >= 0.50:
        score_b += 15
        tags_b.append(f"초동 수급 유입({max_vol_mult_2:.1f}배)")

    if 0.001 <= pre5_min_vol_ratio <= PRE_VOL_POCKET_MAX_RATIO:
        score_b += 15
        tags_b.append(f"발화전 거래량 극건조 포켓({pre5_min_vol_ratio*100:.0f}%)")

    if has_cvd_div and box_range_pct <= PRE_BOX_LOOSE_MAX_PCT:
        score_b += 20
        tags_b.append(f"박스응축({box_range_pct:.1f}%)+{cvd_label}")
    elif obv_accum and box_range_pct <= PRE_BOX_LOOSE_MAX_PCT:
        score_b += 18
        tags_b.append(f"박스응축({box_range_pct:.1f}%)+OBV 선행매집")
    elif box_range_pct <= PRE_BOX_TIGHT_PCT and (cond_ema_ribbon or cond_spring):
        score_b += 18
        tags_b.append(f"초응축 박스({box_range_pct:.1f}%)+스프링 응축돌파")
    elif has_cvd_div:
        score_b += 15
        tags_b.append(cvd_label)

    if (cond_ttm_squeeze or cond_bb_vwap_breakout) and cond_ema_ribbon:
        score_b += 15
        tags_b.append(f"스퀴즈+이평리본 초밀집 확산({pre5_min_ribbon:.1f}%)")
    elif cond_ema_ribbon:
        score_b += 10
        tags_b.append(f"이평리본 초밀집 확산({pre5_min_ribbon:.1f}%)")
    elif cond_ttm_squeeze:
        score_b += 10
        tags_b.append("TTM 켈트너 스퀴즈 폭발")
    elif cond_bb_vwap_breakout:
        score_b += 10
        tags_b.append("BB스퀴즈+VWAP 장대양봉 돌파")

    if cond_rsi_thrust and cond_macd_golden:
        score_b += 8
        tags_b.append("RSI 과매도 스러스트+MACD 골든크로스")

    if cond_strong_oi:
        score_b += 8
        tags_b.append(f"OI 실수급 유입(OI {oi_m:.1f}M·비중 {oi_turnover_ratio*100:.0f}%)")

    if cond_strong_short_squeeze:
        score_b += 15
        tags_b.append(f"강력 숏스퀴즈 수급({funding_rate:+.4f}%)")
    elif cond_clean_funding:
        score_b += 10
        tags_b.append(f"청정 펀딩 수급({funding_rate:+.4f}%)")

    if (cond_htf_center_sup or htf_bullish) and rsi6_15m <= 68.0 and kdj_j_15m <= 80.0:
        score_b += 8
        tags_b.append("상위프레임(1H 중심선) 지지동조")

    if score_a < SCORE_THRESHOLD and score_b < SCORE_THRESHOLD:
        return None

    resistance_line = max(box_high, swing_high_8, current_price * 1.015)

    if score_a >= score_b:
        track_type = "TRACK_A"
        track_title = "🟢 [1차 선취매/눌림 타점 포착]"
        final_score = int(min(100, score_a))
        matched_tags = tags_a
    else:
        track_type = "TRACK_B"
        track_title = "🚀 [2차 돌파/리테스트 타점 포착]"
        final_score = int(min(100, score_b))
        matched_tags = tags_b

    entry_low, entry_high = calc_precise_entry_zone(
        df, df_15m, current_price, track_type, box_high, swing_high_8
    )
    avg_entry = (entry_low + entry_high) / 2.0

    sl_pct, stop_price, structural_support, sl_reason = calc_strategic_altcoin_sl(
        df, df_15m, df_1h, avg_entry, box_low, box_range_pct, track_type, cond_spring, symbol
    )
    support_line = min(structural_support, recent_low_6, entry_low * 0.992)

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
        "turnover_m": turnover_m,
        "oi_m": oi_m,
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
        "sl_reason": sl_reason,
        "tp_pcts": tp_pcts,
        "tps": tps,
        "tp_labels": tp_labels,
        "matched_tags": matched_tags
    }
