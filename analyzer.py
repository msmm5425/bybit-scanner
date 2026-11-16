import os
import json
import time
import numpy as np
import pandas as pd

import config as cfg
import api as api_mod

_BTC_REGIME_CACHE = {"ts": 0, "is_dumping": False, "drop_pct": 0.0}
_DYNAMIC_TP_CACHE = {"ts": 0, "data": {}}


def get_dynamic_tp_params():
    now = time.time()
    if now - _DYNAMIC_TP_CACHE["ts"] < 60 and _DYNAMIC_TP_CACHE["data"]:
        return _DYNAMIC_TP_CACHE["data"]

    params = {
        "tp1_min_pct": cfg.to_float(getattr(cfg, "TP1_MIN_PCT", 8.5), 8.5),
        "tp2_min_pct": cfg.to_float(getattr(cfg, "TP2_MIN_PCT", 16.0), 16.0),
        "tp3_min_pct": cfg.to_float(getattr(cfg, "TP3_MIN_PCT", 24.0), 24.0),
        "tp3_max_cap_pct": cfg.to_float(getattr(cfg, "TP3_MAX_CAP_PCT", 50.0), 50.0),
        "tp1_fib_mult": cfg.to_float(getattr(cfg, "TP1_FIB_MULT", 1.85), 1.85),
        "tp2_fib_mult": cfg.to_float(getattr(cfg, "TP2_FIB_MULT", 2.95), 2.95),
        "tp3_fib_mult": cfg.to_float(getattr(cfg, "TP3_FIB_MULT", 4.10), 4.10),
        "tp1_ma20_ext_pct": 6.4,
        "tp2_ma20_ext_pct": 9.8,
        "tp3_ma20_ext_pct": 13.3,
    }

    hist_file = getattr(cfg, "HISTORY_FILE", "analyzer_history.json")
    if os.path.exists(hist_file):
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and isinstance(raw.get("dynamic_tp_tuning"), dict):
                dt = raw["dynamic_tp_tuning"]
                params["tp1_min_pct"] = float(np.clip(cfg.to_float(dt.get("tp1_min_pct"), params["tp1_min_pct"]), 7.0, 14.0))
                params["tp2_min_pct"] = float(np.clip(cfg.to_float(dt.get("tp2_min_pct"), params["tp2_min_pct"]), 13.0, 24.0))
                params["tp3_min_pct"] = float(np.clip(cfg.to_float(dt.get("tp3_min_pct"), params["tp3_min_pct"]), 19.0, 48.0))
                params["tp1_fib_mult"] = float(np.clip(cfg.to_float(dt.get("tp1_fib_mult"), params["tp1_fib_mult"]), 1.50, 2.35))
                params["tp2_fib_mult"] = float(np.clip(cfg.to_float(dt.get("tp2_fib_mult"), params["tp2_fib_mult"]), 2.30, 3.50))
                params["tp3_fib_mult"] = float(np.clip(cfg.to_float(dt.get("tp3_fib_mult"), params["tp3_fib_mult"]), 3.10, 4.80))
                params["tp1_ma20_ext_pct"] = float(np.clip(cfg.to_float(dt.get("tp1_ma20_ext_pct"), 6.4), 3.0, 9.0))
                params["tp2_ma20_ext_pct"] = float(np.clip(cfg.to_float(dt.get("tp2_ma20_ext_pct"), 9.8), 6.0, 15.0))
                params["tp3_ma20_ext_pct"] = float(np.clip(cfg.to_float(dt.get("tp3_ma20_ext_pct"), 13.3), 9.5, 24.0))
        except Exception:
            pass

    _DYNAMIC_TP_CACHE["ts"] = now
    _DYNAMIC_TP_CACHE["data"] = params
    return params


class SafeAlertDict(dict):
    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        alias_map = {
            "last_price": "price",
            "curr_price": "price",
            "current_price": "price",
            "close": "price",
            "base": "base_coin",
            "ticker": "symbol",
            "title": "track_title",
            "track_label": "track_title",
            "total_score": "score",
            "chg_24h": "change_24h",
            "price24hPcnt": "change_24h",
            "turnover_24h_m": "turnover_m",
            "vol_24h_m": "turnover_m",
            "open_interest_m": "oi_m",
            "funding_rate_pct": "funding_pct",
            "htf_trend": "htf_status",
            "rsi_15m": "rsi6_15m",
            "kdj_j": "kdj_j_15m",
            "sl_price": "stop_price",
            "stop_pct": "sl_pct",
            "tp_prices": "tps",
        }
        target = alias_map.get(key)
        if target and target in self.keys():
            return super().__getitem__(target)
        return 0.0 if key.endswith(("_price", "_pct", "_m", "_mult")) else ""


def fetch_klines(symbol, interval="5", limit=150):
    """오직 api.py의 Bybit V5 공식 캔들만 조회 (타 거래소 우회 완전 제거)"""
    if api_mod is not None and hasattr(api_mod, "fetch_klines"):
        try:
            df = api_mod.fetch_klines(symbol, str(interval), limit=int(limit))
            if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 20:
                return df
        except Exception:
            pass
    return pd.DataFrame()


def calc_rsi(series, period=6):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calc_kdj_j(df, period=9):
    low_min = df["low"].rolling(window=period, min_periods=3).min()
    high_max = df["high"].rolling(window=period, min_periods=3).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = ((df["close"] - low_min) / denom * 100.0).fillna(50.0)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return j


def calc_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=3).mean().bfill()


def calc_obv_and_cvd(df):
    close_diff = df["close"].diff().fillna(0.0)
    direction = np.where(close_diff > 0, 1.0, np.where(close_diff < 0, -1.0, 0.0))
    obv = (df["volume"] * direction).cumsum()

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    buy_ratio = ((df["close"] - df["low"]) / candle_range).fillna(0.5).clip(0.0, 1.0)
    net_delta = df["volume"] * (2.0 * buy_ratio - 1.0)
    cvd = net_delta.cumsum()
    return obv, cvd, net_delta


def get_btc_market_regime():
    now = time.time()
    if now - _BTC_REGIME_CACHE["ts"] < 60:
        return _BTC_REGIME_CACHE["is_dumping"], _BTC_REGIME_CACHE["drop_pct"]

    df_btc = fetch_klines("BTCUSDT", "15", limit=40)
    is_dumping = False
    drop_pct = 0.0
    if not df_btc.empty and len(df_btc) >= 16:
        c_now = cfg.to_float(df_btc["close"].iloc[-1])
        h_recent3 = cfg.to_float(df_btc["high"].iloc[-3:].max(), c_now)
        h_recent12 = cfg.to_float(df_btc["high"].iloc[-12:].max(), c_now)
        ma20_btc = cfg.to_float(df_btc["close"].rolling(20, min_periods=5).mean().iloc[-1], c_now)
        rsi_btc = cfg.to_float(calc_rsi(df_btc["close"], 6).iloc[-1], 50.0)

        drop_15m_pct = (c_now - h_recent3) / max(h_recent3, 1e-8) * 100.0
        drop_3h_pct = (c_now - h_recent12) / max(h_recent12, 1e-8) * 100.0
        drop_pct = min(drop_15m_pct, drop_3h_pct)

        thr_15m = getattr(cfg, "BTC_DUMP_DROP_15M_PCT", -0.70)
        thr_1h = getattr(cfg, "BTC_DUMP_DROP_1H_PCT", -1.20)

        if (
            drop_15m_pct <= thr_15m
            or (drop_3h_pct <= thr_1h and c_now < ma20_btc)
            or (c_now < ma20_btc * 0.992 and rsi_btc < 34.0)
        ):
            is_dumping = True

    _BTC_REGIME_CACHE["ts"] = now
    _BTC_REGIME_CACHE["is_dumping"] = is_dumping
    _BTC_REGIME_CACHE["drop_pct"] = drop_pct
    return is_dumping, drop_pct


def analyze_symbol(coin):
    symbol = str(coin.get("symbol", "")).upper()
    base_coin = symbol[:-4] if symbol.endswith("USDT") else symbol
    to_float = cfg.to_float

    if cfg.is_tradfi_ticker(base_coin):
        return None
    if hasattr(cfg, "is_heavy_major_coin") and cfg.is_heavy_major_coin(base_coin):
        return None

    # 바이비트 실제 상장 여부 재검증
    if api_mod is not None and hasattr(api_mod, "get_valid_bybit_linear_symbols"):
        valid_syms, _ = api_mod.get_valid_bybit_linear_symbols()
        if valid_syms and symbol not in valid_syms:
            return None

    df_5m = fetch_klines(symbol, "5", limit=150)
    if df_5m.empty or len(df_5m) < 60:
        return None

    df_15m = fetch_klines(symbol, "15", limit=120)
    df_1h = fetch_klines(symbol, "60", limit=75)
    if df_15m.empty or len(df_15m) < 30 or df_1h.empty or len(df_1h) < 25:
        return None

    close = to_float(df_5m["close"].iloc[-1])
    open_p = to_float(df_5m["open"].iloc[-1], close)
    high_p = to_float(df_5m["high"].iloc[-1], close)
    low_p = to_float(df_5m["low"].iloc[-1], close)
    if close <= 0:
        return None

    turnover_m = to_float(coin.get("turnover_m"))
    if turnover_m <= 0:
        turnover_m = to_float(coin.get("turnover24h")) / 1e6

    oi_m = to_float(coin.get("oi_m"))
    if oi_m <= 0:
        oi_m = to_float(coin.get("openInterestValue")) / 1e6

    if "funding_pct" in coin:
        raw_f = to_float(coin.get("funding_pct"))
        funding_pct = cfg.normalize_funding_pct(raw_f) if hasattr(cfg, "normalize_funding_pct") else raw_f
    else:
        funding_pct = to_float(coin.get("fundingRate")) * 100.0

    if "change_24h" in coin:
        change_24h = to_float(coin.get("change_24h"))
    else:
        change_24h = to_float(coin.get("price24hPcnt")) * 100.0

    oi_turnover_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0

    ma20_1h = to_float(df_1h["close"].rolling(20, min_periods=10).mean().iloc[-1], close)
    ema20_1h = to_float(df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-1], close)
    htf_support_ok = (close >= ma20_1h * 0.995) or (close >= ema20_1h * 0.995)
    if not htf_support_ok:
        return None

    recent_12h = df_5m.iloc[-144:]
    high_pos_12h = int(df_5m.index.get_loc(recent_12h["high"].idxmax()))
    high_12h = to_float(recent_12h["high"].max(), close)
    low_24h_15m = to_float(df_15m["low"].iloc[-96:].min(), close)
    pre_high_slice = df_5m.iloc[max(0, len(df_5m) - 144): max(1, high_pos_12h + 1)]
    low_before_high = min(to_float(pre_high_slice["low"].min(), close), low_24h_15m)

    drawdown_12h_pct = (high_12h - close) / max(high_12h, 1e-8) * 100.0
    prior_pump_pct = (high_12h - low_before_high) / max(low_before_high, 1e-8) * 100.0
    curr_gain_from_low_pct = (close - low_before_high) / max(low_before_high, 1e-8) * 100.0
    fib_retracement_ratio = (high_12h - close) / max(high_12h - low_before_high, 1e-8)

    leader_min_turn = getattr(cfg, "LEADER_MIN_TURNOVER_M", 25.0)
    leader_min_pump = getattr(cfg, "LEADER_MIN_PRIOR_PUMP_PCT", 35.0)
    fib_ret_min = getattr(cfg, "LEADER_FIB_RETRACE_MIN", 0.38)
    fib_ret_max = getattr(cfg, "LEADER_FIB_RETRACE_MAX", 0.62)
    leader_max_fund = getattr(cfg, "LEADER_MAX_FUNDING_PCT", 0.0150)

    is_leader_fib_candidate = (
        turnover_m >= leader_min_turn
        and prior_pump_pct >= leader_min_pump
        and change_24h >= 18.0
        and (fib_ret_min <= fib_retracement_ratio <= fib_ret_max)
        and (drawdown_12h_pct >= 12.0)
        and htf_support_ok
        and funding_pct <= leader_max_fund
    )

    max_early_pump = getattr(cfg, "MAX_PRIOR_PUMP_PCT", 22.0)
    max_early_chg24 = getattr(cfg, "MAX_CHANGE_24H_FOR_EARLY_TRACK", 26.0)

    if not is_leader_fib_candidate:
        if prior_pump_pct >= max_early_pump or curr_gain_from_low_pct >= 20.0 or change_24h >= max_early_chg24:
            return None
        if drawdown_12h_pct >= cfg.MAX_DRAWDOWN_FROM_12H_HIGH_PCT:
            return None

    rsi6_15m_series = calc_rsi(df_15m["close"], 6)
    rsi6_15m = to_float(rsi6_15m_series.iloc[-1], 50.0)
    kdj_j_15m = to_float(calc_kdj_j(df_15m, 9).iloc[-1], 50.0)
    rsi6_1h = to_float(calc_rsi(df_1h["close"], 6).iloc[-1], 50.0)

    if rsi6_15m >= cfg.CUTOFF_RSI6_15M_MAX or kdj_j_15m >= cfg.CUTOFF_KDJ_J_15M_MAX:
        return None
    if rsi6_1h >= cfg.CUTOFF_RSI6_1H_MAX:
        return None

    recent_6h_max_rsi15 = to_float(rsi6_15m_series.iloc[-24:-1].max(), 50.0)
    if not is_leader_fib_candidate and recent_6h_max_rsi15 >= 82.0:
        return None

    for p in (5, 20, 50, 112):
        df_5m[f"ma{p}"] = df_5m["close"].rolling(p, min_periods=min(p, 20)).mean()
        df_5m[f"ema{p}"] = df_5m["close"].ewm(span=p, adjust=False).mean()

    ma5 = to_float(df_5m["ma5"].iloc[-1], close)
    ma20 = to_float(df_5m["ma20"].iloc[-1], close)
    ma50 = to_float(df_5m["ma50"].iloc[-1], close)
    ma112 = to_float(df_5m["ma112"].iloc[-1], close)

    ma50_15m = to_float(df_15m["close"].rolling(50, min_periods=15).mean().iloc[-1], close)
    ma112_15m = to_float(df_15m["close"].rolling(112, min_periods=25).mean().iloc[-1], close)

    if close < ma112 and ((ma112 - close) / close * 100.0) <= 4.0:
        return None
    if close < ma112_15m and ((ma112_15m - close) / close * 100.0) <= 4.5:
        return None
    if not is_leader_fib_candidate and close < ma50_15m and ((ma50_15m - close) / close * 100.0) <= 3.0:
        return None

    ma112_15m_ext_pct = (close - ma112_15m) / max(ma112_15m, 1e-8) * 100.0
    max_ma112_ext = getattr(cfg, "MAX_MA112_15M_EXT_PCT", 13.5)
    if not is_leader_fib_candidate and ma112_15m_ext_pct > max_ma112_ext:
        return None

    ribbon_max = df_5m[["ema5", "ema20", "ema50"]].max(axis=1)
    ribbon_min = df_5m[["ema5", "ema20", "ema50"]].min(axis=1)
    df_5m["ribbon_spread_pct"] = (ribbon_max - ribbon_min) / df_5m["close"].replace(0, np.nan) * 100.0
    pre5_min_ribbon_spread = to_float(df_5m["ribbon_spread_pct"].iloc[-8:-1].min(), 9.9)

    pre_box_df = df_5m.iloc[-28:-3]
    prev_box_high = to_float(pre_box_df["high"].max(), close)
    prev_box_low = to_float(pre_box_df["low"].min(), close)
    pre_box_range_pct = (prev_box_high - prev_box_low) / max(prev_box_low, 1e-8) * 100.0

    bb_mid_15m = to_float(df_15m["close"].rolling(20, min_periods=10).mean().iloc[-1], close)
    bb_mid_ext_pct = (close - bb_mid_15m) / max(bb_mid_15m, 1e-8) * 100.0
    allowed_bb_ext = cfg.CUTOFF_BB_MID_EXT_MAX_PCT if pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT else 3.8
    if bb_mid_ext_pct > allowed_bb_ext:
        return None

    btc_dumping, _ = get_btc_market_regime()
    if btc_dumping:
        if (not is_leader_fib_candidate and close < bb_mid_15m) or change_24h < 2.5:
            return None

    if close >= ma20_1h and rsi6_1h >= 55.0:
        htf_status = f"1H 상승우위(RSI6 {int(round(rsi6_1h))})"
    else:
        htf_status = f"1H 중심선지지(RSI6 {int(round(rsi6_1h))})"

    vol_ma20_series = df_5m["volume"].rolling(20, min_periods=5).mean().shift(1)
    recent_window = df_5m.iloc[-6:]
    vol_ratios = recent_window["volume"] / vol_ma20_series.iloc[-6:].replace(0, np.nan)
    ignition_idx = vol_ratios.idxmax()
    vol_mult = to_float(vol_ratios.loc[ignition_idx], 1.0)
    ignition_vol = to_float(df_5m.loc[ignition_idx, "volume"], 1.0)
    ignition_open = to_float(df_5m.loc[ignition_idx, "open"], close)
    ignition_high = to_float(df_5m.loc[ignition_idx, "high"], close)
    ignition_low = to_float(df_5m.loc[ignition_idx, "low"], close)

    vol_ma20_curr = to_float(vol_ma20_series.iloc[-1], 1.0)
    curr_vol = to_float(df_5m["volume"].iloc[-1], ignition_vol)
    curr_vol_mult = curr_vol / max(vol_ma20_curr, 1e-8)

    pre5_min_vol = to_float(df_5m["volume"].iloc[-7:-1].min(), vol_ma20_curr)
    pre_vol_pocket_ratio = pre5_min_vol / max(vol_ma20_curr, 1e-8)

    pullback_vol_ratio = curr_vol / max(ignition_vol, 1e-8)
    effective_dryup_ratio = min(pullback_vol_ratio, curr_vol_mult * 0.65)
    pullback_low = to_float(df_5m["low"].iloc[-2:].min(), close)

    obv, cvd, net_delta = calc_obv_and_cvd(df_5m)
    obv_slope_up = to_float(obv.iloc[-1]) > to_float(obv.iloc[-15:-5].mean())
    cvd_slope_up = to_float(cvd.iloc[-1]) > to_float(cvd.iloc[-15:-5].mean())
    price_low_sweep = to_float(df_5m["low"].iloc[-8:].min()) <= prev_box_low * 1.005
    cond_cvd_div = (price_low_sweep and cvd_slope_up) or (cvd_slope_up and to_float(net_delta.iloc[-3:].sum()) > 0)

    ma_full_aligned = (ma5 > ma20 > ma50 > ma112) and (close >= ma20)
    ma_short_aligned = (ma5 > ma20 > ma50) and (close >= ma20)

    min_body_floor = df_5m["close"] * 0.0015
    body_len = np.maximum((df_5m["close"] - df_5m["open"]).abs(), min_body_floor)
    lower_wick = (df_5m[["open", "close"]].min(axis=1) - df_5m["low"]).clip(lower=0.0)
    lower_wick_ratio_max6 = min(15.0, to_float((lower_wick / body_len).iloc[-6:].max(), 0.0))
    cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.4) and (close > ma20)

    bb_std = df_5m["close"].rolling(20, min_periods=10).std()
    bb_upper = df_5m["ma20"] + 2.0 * bb_std
    bb_lower = df_5m["ma20"] - 2.0 * bb_std
    atr_5m = calc_atr(df_5m, 14)
    kc_upper = df_5m["ma20"] + 1.5 * atr_5m
    kc_lower = df_5m["ma20"] - 1.5 * atr_5m
    ttm_squeeze = bool(((bb_upper < kc_upper) & (bb_lower > kc_lower)).iloc[-8:-1].any())

    ma50_ma112_gap_pct = abs(ma50 - ma112) / max(ma112, 1e-8) * 100.0
    cond_ema_ribbon = (pre5_min_ribbon_spread <= cfg.EMA_RIBBON_MAX_SPREAD_PCT) and (close > ma50) and (ma50_ma112_gap_pct <= 8.0)
    cond_vol_pocket = (pre_vol_pocket_ratio <= cfg.PRE_VOL_POCKET_MAX_RATIO) and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)

    breakout_diff_pct = (close - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    sr_error_pct = abs(pullback_low - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    is_current_ignition = (ignition_idx == df_5m.index[-1])

    curr_range = max(high_p - low_p, 1e-8)
    upper_wick_ratio = (high_p - max(open_p, close)) / curr_range
    curr_candle_gain_pct = (close - open_p) / max(open_p, 1e-8) * 100.0

    cond_sr_flip_retest = (
        (not is_leader_fib_candidate)
        and (not is_current_ignition)
        and (close >= prev_box_high * 0.996)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (sr_error_pct <= cfg.SR_FLIP_TOLERANCE_PCT)
        and (pullback_vol_ratio <= cfg.SR_FLIP_DRYUP_RATIO_MAX)
        and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)
    )

    impulse_range = max(ignition_high - prev_box_high, close * 0.004)
    flag_support_line = ignition_high - (impulse_range * cfg.BULL_FLAG_MAX_RETRACE)
    cond_bull_flag = (
        (not is_leader_fib_candidate)
        and (not is_current_ignition)
        and (close > prev_box_high)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (pullback_low >= flag_support_line)
        and (close >= ma5 * 0.998)
        and (pullback_vol_ratio <= cfg.PULLBACK_DRYUP_MAX)
        and (cfg.EARLY_VOL_MULT_MIN <= vol_mult <= cfg.EARLY_VOL_MULT_MAX)
    )

    fvg_mid_line = (ignition_open + ignition_high) * 0.5
    cond_fvg_mid_hold = (
        (not is_leader_fib_candidate)
        and (not is_current_ignition)
        and (close >= prev_box_high * 0.995)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (pullback_low >= fvg_mid_line * 0.998)
        and (pullback_vol_ratio <= 0.35)
        and (cfg.EARLY_VOL_MULT_MIN <= vol_mult <= cfg.EARLY_VOL_MULT_MAX)
    )

    cond_pocket_pivot = (
        (not is_leader_fib_candidate)
        and is_current_ignition
        and (close > open_p)
        and (0.4 <= curr_candle_gain_pct <= 3.8)
        and (upper_wick_ratio <= 0.36)
        and (-0.5 <= breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (cfg.EARLY_VOL_MULT_MIN <= curr_vol_mult <= cfg.EARLY_VOL_MULT_MAX)
        and (pre_vol_pocket_ratio <= 0.40 or pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT or cond_ema_ribbon)
        and htf_support_ok
    )

    cond_track_a_pullback = (
        (not is_leader_fib_candidate)
        and htf_support_ok
        and (close >= ma112)
        and (close <= prev_box_high * 1.008)
        and (close >= ma20 * 0.996)
        and (effective_dryup_ratio <= cfg.PULLBACK_DRYUP_MAX)
        and (vol_mult >= 2.2 or cond_vol_pocket)
        and (ma_short_aligned or cond_spring or cond_ema_ribbon)
    )

    recent_pullback_low = to_float(df_5m["low"].iloc[-12:].min(), close)
    rebound_from_pb_low_pct = (close - recent_pullback_low) / max(recent_pullback_low, 1e-8) * 100.0
    cond_track_c_leader_fib = (
        is_leader_fib_candidate
        and (close >= ma20 * 0.998)
        and (close > open_p or ma5 >= ma20 * 0.998)
        and (0.8 <= rebound_from_pb_low_pct <= 7.5)
        and (obv_slope_up or cvd_slope_up or vol_mult >= 1.8 or lower_wick_ratio_max6 >= 1.3)
    )

    is_track_b = cond_sr_flip_retest or cond_bull_flag or cond_fvg_mid_hold or cond_pocket_pivot
    if not is_track_b and not cond_track_a_pullback and not cond_track_c_leader_fib:
        return None

    if cond_track_c_leader_fib:
        track_type = "TRACK_C"
        track_title = "🔥 [대장주 피보나치 0.5 눌림목 2파동 반등 포착]"
    elif is_track_b:
        track_type = "TRACK_B"
        track_title = "🚀 [2차 돌파/리테스트 타점 포착]"
    else:
        track_type = "TRACK_A"
        track_title = "🟢 [1차 선취매/눌림 타점 포착]"

    score = 0
    reasons = []

    if track_type == "TRACK_C":
        fib_pct_val = int(round(fib_retracement_ratio * 100))
        score += 22
        reasons.append(
            f"대장주(거래대금 {turnover_m:.0f}M·직전 +{prior_pump_pct:.0f}% 폭발) 피보나치 {fib_pct_val}% 눌림목 지지"
        )
        score += 14
        reasons.append("1H 중심선(MA20) 지지 + 5m MA20 재탈환 2파동 시동")
        if obv_slope_up or cvd_slope_up:
            score += 12
            reasons.append("눌림목 구간 OBV/CVD 세력 매수세 유지(이탈 없음)")
        else:
            score += 8
            reasons.append(f"눌림 바닥 아랫꼬리({lower_wick_ratio_max6:.1f}배) 저가 매수 유입")

        if pre_vol_pocket_ratio <= 0.45 or effective_dryup_ratio <= 0.35:
            score += 8
            reasons.append("조정 말기 투매 거래량 건조 확인 후 반등 양봉")
        if vol_mult >= 2.0 or curr_vol_mult >= 1.5:
            score += 6
            reasons.append(f"반등 수급 재유입({max(vol_mult, curr_vol_mult):.1f}배)")
    else:
        if track_type == "TRACK_B":
            if cond_sr_flip_retest:
                score += 20
                reasons.append(f"전고점 S/R Flip 칼지지(오차 {sr_error_pct:.2f}%/눌림거래량 {int(round(pullback_vol_ratio * 100))}%)")
            elif cond_bull_flag:
                score += 20
                reasons.append(f"공중눌림 Bull Flag 칼지지(상단38.2%·MA5 사수/눌림거래량 {int(round(pullback_vol_ratio * 100))}%)")
            elif cond_fvg_mid_hold:
                score += 20
                reasons.append(f"장대양봉 50%(FVG) 중심선 칼지지(눌림거래량 {int(round(pullback_vol_ratio * 100))}%)")
            elif cond_pocket_pivot:
                score += 20
                reasons.append(f"포켓피벗 초동양봉 직격(직전거래량 {int(round(pre_vol_pocket_ratio * 100))}% 건조 후 {curr_vol_mult:.1f}배 발화)")
        else:
            if effective_dryup_ratio <= cfg.PULLBACK_DRYUP_STRICT:
                score += 20
                reasons.append(f"거래량 극건조 눌림 포켓({int(round(effective_dryup_ratio * 100))}%)")
            else:
                score += 17
                reasons.append(f"거래량 건조 선취매 포켓({int(round(effective_dryup_ratio * 100))}%)")

        if ma_full_aligned:
            score += 15
            reasons.append("완전 정배열(5>20>50>112) 지지")
        elif ma_short_aligned:
            score += 14
            reasons.append("단기 정배열(5>20>50) 지지")
        elif cond_spring:
            score += 14
            reasons.append(f"유동성 스윕(Spring 아랫꼬리 {lower_wick_ratio_max6:.1f}배) 반전")
        elif cond_ema_ribbon:
            score += 13
            reasons.append(f"바닥 이평리본 수렴 돌파({pre5_min_ribbon_spread:.1f}%)")

        if htf_support_ok:
            score += 10
            reasons.append("상위프레임(1H 중심선) 구조적 지지")

        allowed_box_max = cfg.PRE_BOX_LOOSE_MAX_PCT if cond_spring else cfg.PRE_BOX_RANGE_MAX_PCT
        if pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT:
            score += 12
            if cond_cvd_div:
                score += 8
                reasons.append(f"박스응축({pre_box_range_pct:.1f}%)+CVD 강세 다이버전스(저점 스윕 매수흡수)")
            elif obv_slope_up:
                score += 8
                reasons.append(f"초응축({pre_box_range_pct:.1f}%)+OBV 선행매집")
            else:
                score += 5
                reasons.append(f"초응축 박스권({pre_box_range_pct:.1f}%)")
        elif pre_box_range_pct <= allowed_box_max:
            score += 8
            if cond_cvd_div or obv_slope_up:
                score += 8
                reasons.append(f"박스응축({pre_box_range_pct:.1f}%)+OBV/CVD 선행매집")

        if cond_ema_ribbon:
            score += 9
            reasons.append(f"이평리본 초밀집 확산({pre5_min_ribbon_spread:.1f}%)")
        elif ttm_squeeze:
            score += 8
            reasons.append("TTM 켈트너 스퀴즈 응축 돌파")

        if cfg.EARLY_VOL_MULT_MIN <= vol_mult <= cfg.EARLY_VOL_MULT_MAX:
            score += 6
            reasons.append(f"초동수급 폭발({vol_mult:.1f}배)")
        elif cond_vol_pocket or pre_vol_pocket_ratio <= 0.35:
            score += 6
            reasons.append(f"발화전 거래량 극건조 포켓({int(round(pre_vol_pocket_ratio * 100))}%)")

    if oi_m >= 1.0 and oi_turnover_ratio >= 8.0:
        score += 10
        reasons.append(f"OI 실수급 유입(OI {oi_m:.1f}M·비중 {int(round(oi_turnover_ratio))}%)")
    elif oi_m >= 0.4:
        score += 7
        reasons.append(f"OI 수급 동조(OI {oi_m:.1f}M)")

    if funding_pct <= 0.0:
        score += 10
        reasons.append(f"음수 펀딩 숏스퀴즈 수급({funding_pct:+.4f}%)")
    elif funding_pct <= 0.0120:
        score += 10
        reasons.append(f"청정 펀딩 수급({funding_pct:+.4f}%)")
    elif funding_pct <= 0.0250:
        score += 5

    score = int(min(100, score))
    active_threshold = getattr(cfg, "BTC_DUMP_SCORE_THRESHOLD", 90) if btc_dumping else cfg.SCORE_THRESHOLD
    if score < active_threshold:
        return None

    if track_type == "TRACK_C":
        support_line = recent_pullback_low
        swing_high_line = high_12h
    else:
        support_line = max(prev_box_low, min(ma50, ignition_low))
        swing_high_line = max(prev_box_high, ignition_high)

    entry_low = close * 0.9975
    entry_high = close * 1.0010

    sl_pct_int = int(getattr(cfg, "FIXED_SL_PCT", 6))
    stop_price = close * (1.0 - sl_pct_int / 100.0)
    sl_note = f"급등전 휩소·스탑헌팅 방어 고정손절(-{sl_pct_int}%)"

    tp_params = get_dynamic_tp_params()
    base_tp1_pct = tp_params["tp1_min_pct"]
    base_tp2_pct = tp_params["tp2_min_pct"]
    base_tp3_pct = tp_params["tp3_min_pct"]

    fib1 = tp_params["tp1_fib_mult"]
    fib2 = tp_params["tp2_fib_mult"]
    fib3 = tp_params["tp3_fib_mult"]

    atr_15m_val = to_float(calc_atr(df_15m, 14).iloc[-1], close * 0.018)
    atr_15m_pct = (atr_15m_val / close) * 100.0
    struct_span_pct = (swing_high_line - support_line) / max(close, 1e-8) * 100.0

    vol_boost = float(np.clip((vol_mult - 2.2) * 0.045, -0.08, 0.25))
    squeeze_boost = 0.12 if funding_pct < 0 else (0.06 if oi_turnover_ratio >= 25.0 else 0.0)
    energy_mult = float(np.clip(1.0 + vol_boost + squeeze_boost, 0.85, 1.45))

    coin_unit_wave_pct = float(np.clip((atr_15m_pct * 2.1 + struct_span_pct * 0.45) * energy_mult, 3.6, 8.5))

    if track_type == "TRACK_C":
        pullback_depth_pct = max((high_12h - close) / close * 100.0, 9.0)
        calc_tp1_pct = float(np.clip(pullback_depth_pct * 0.52 * energy_mult, 8.0, 15.0))
        calc_tp2_pct = float(np.clip(max(calc_tp1_pct + 6.0, pullback_depth_pct * 0.98), 15.0, 28.0))
        calc_tp3_pct = float(np.clip(max(calc_tp2_pct + 8.0, pullback_depth_pct * 1.45 * energy_mult), 24.0, 46.0))
        tp_labels = [
            "1차반등(낙폭50%회복)",
            "2차전고점(고점재테스트)",
            "3차신고가(대시세2파동)"
        ]
    else:
        raw_w1_pct = (coin_unit_wave_pct * fib1 * 0.65) + (base_tp1_pct * 0.35)
        raw_w2_pct = (coin_unit_wave_pct * fib2 * 0.65) + (base_tp2_pct * 0.35)
        raw_w3_pct = (coin_unit_wave_pct * fib3 * 0.65) + (base_tp3_pct * 0.35)

        calc_tp1_pct = float(np.clip(raw_w1_pct, 7.0, 14.0))
        calc_tp2_pct = float(np.clip(max(calc_tp1_pct + 5.5, raw_w2_pct), 13.0, 26.0))
        calc_tp3_pct = float(np.clip(max(calc_tp2_pct + 7.0, raw_w3_pct), 20.0, 45.0))

        tp_labels = [
            f"1차첫조정(Fib {fib1:.2f}·ATR반영)",
            f"2차중간고점(Fib {fib2:.2f}·ATR반영)",
            f"3차최종고점(Fib {fib3:.2f}·클라이맥스)"
        ]

    tp1_pct = round(calc_tp1_pct, 1)
    tp2_pct = round(calc_tp2_pct, 1)
    tp3_pct = round(calc_tp3_pct, 1)

    tp1 = close * (1.0 + tp1_pct / 100.0)
    tp2 = close * (1.0 + tp2_pct / 100.0)
    tp3 = close * (1.0 + tp3_pct / 100.0)

    return SafeAlertDict({
        "symbol": symbol,
        "base_coin": base_coin,
        "base": base_coin,
        "track_type": track_type,
        "track_title": track_title,
        "score": score,
        "price": close,
        "last_price": close,
        "curr_price": close,
        "change_24h": change_24h,
        "turnover_m": turnover_m,
        "oi_m": oi_m,
        "oi_ratio": oi_turnover_ratio,
        "funding_pct": funding_pct,
        "htf_status": htf_status,
        "htf_diag": htf_status,
        "rsi6_15m": round(rsi6_15m, 1),
        "rsi6_1h": round(rsi6_1h, 1),
        "kdj_j_15m": round(kdj_j_15m, 1),
        "j_15m": round(kdj_j_15m, 1),
        "reasons": reasons,
        "reason_str": " / ".join(reasons),
        "support_line": support_line,
        "swing_high_line": swing_high_line,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_price": stop_price,
        "sl_price": stop_price,
        "sl_pct": sl_pct_int,
        "sl_reason": sl_note,
        "sl_note": sl_note,
        "tps": [tp1, tp2, tp3],
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "tp_pcts": [tp1_pct, tp2_pct, tp3_pct],
        "tp1_pct": tp1_pct,
        "tp2_pct": tp2_pct,
        "tp3_pct": tp3_pct,
        "tp_labels": tp_labels,
        "tp_notes": tp_labels,
        "vol_mult": round(vol_mult, 1),
        "pre_box_range_pct": round(pre_box_range_pct, 1),
    })
