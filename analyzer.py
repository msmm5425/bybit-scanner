import os
import json
import time
import inspect
import requests
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
        "tp1_min_pct": cfg.to_float(getattr(cfg, "TP1_MIN_PCT", 9.0), 9.0),
        "tp2_min_pct": cfg.to_float(getattr(cfg, "TP2_MIN_PCT", 17.0), 17.0),
        "tp3_min_pct": cfg.to_float(getattr(cfg, "TP3_MIN_PCT", 29.0), 29.0),
        "tp3_max_cap_pct": cfg.to_float(getattr(cfg, "TP3_MAX_CAP_PCT", 52.0), 52.0),
        "tp1_fib_mult": cfg.to_float(getattr(cfg, "TP1_FIB_MULT", 1.85), 1.85),
        "tp2_fib_mult": cfg.to_float(getattr(cfg, "TP2_FIB_MULT", 2.75), 2.75),
        "tp3_fib_mult": cfg.to_float(getattr(cfg, "TP3_FIB_MULT", 4.04), 4.04),
        "tp1_ma20_ext_pct": 4.5,
        "tp2_ma20_ext_pct": 7.7,
        "tp3_ma20_ext_pct": 12.5,
    }

    hist_file = getattr(cfg, "HISTORY_FILE", "analyzer_history.json")
    if os.path.exists(hist_file):
        try:
            with open(hist_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and isinstance(raw.get("dynamic_tp_tuning"), dict):
                dt = raw["dynamic_tp_tuning"]
                params["tp1_min_pct"] = float(np.clip(cfg.to_float(dt.get("tp1_min_pct"), params["tp1_min_pct"]), 8.5, 14.0))
                params["tp2_min_pct"] = float(np.clip(cfg.to_float(dt.get("tp2_min_pct"), params["tp2_min_pct"]), 15.0, 26.0))
                params["tp3_min_pct"] = float(np.clip(cfg.to_float(dt.get("tp3_min_pct"), params["tp3_min_pct"]), 25.0, 50.0))
                params["tp1_fib_mult"] = float(np.clip(cfg.to_float(dt.get("tp1_fib_mult"), params["tp1_fib_mult"]), 1.65, 2.45))
                params["tp2_fib_mult"] = float(np.clip(cfg.to_float(dt.get("tp2_fib_mult"), params["tp2_fib_mult"]), 2.50, 3.60))
                params["tp3_fib_mult"] = float(np.clip(cfg.to_float(dt.get("tp3_fib_mult"), params["tp3_fib_mult"]), 3.50, 4.80))
                params["tp1_ma20_ext_pct"] = float(np.clip(cfg.to_float(dt.get("tp1_ma20_ext_pct"), 4.5), 3.0, 9.0))
                params["tp2_ma20_ext_pct"] = float(np.clip(cfg.to_float(dt.get("tp2_ma20_ext_pct"), 7.7), 6.0, 16.0))
                params["tp3_ma20_ext_pct"] = float(np.clip(cfg.to_float(dt.get("tp3_ma20_ext_pct"), 12.5), 10.0, 26.0))
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


def _normalize_df(raw, expected_interval_min=None):
    df = pd.DataFrame()
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        df = raw.copy()
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("start", "starttime", "open_time", "time", "date", "datetime", "t"):
                col_map[c] = "timestamp"
            elif cl in ("o", "openprice"):
                col_map[c] = "open"
            elif cl in ("h", "highprice"):
                col_map[c] = "high"
            elif cl in ("l", "lowprice"):
                col_map[c] = "low"
            elif cl in ("c", "closeprice"):
                col_map[c] = "close"
            elif cl in ("v", "vol", "basevol"):
                col_map[c] = "volume"
        if col_map:
            df = df.rename(columns=col_map)
    elif isinstance(raw, list) and len(raw) > 0:
        try:
            if isinstance(raw[0], dict):
                return _normalize_df(pd.DataFrame(raw), expected_interval_min)
            cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(raw[0])]
            df = pd.DataFrame(raw, columns=cols)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            if ts.max() < 1e11:
                ts = ts * 1000.0
            df["timestamp"] = ts
            df = df.sort_values("timestamp")
            if expected_interval_min is not None and len(df) >= 5:
                median_diff_sec = df["timestamp"].diff().median() / 1000.0
                target_sec = expected_interval_min * 60.0
                if not (target_sec * 0.6 <= median_diff_sec <= target_sec * 1.4):
                    return pd.DataFrame()

    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_klines(symbol, interval="5", limit=150):
    if api_mod is not None and hasattr(api_mod, "fetch_klines"):
        try:
            df = api_mod.fetch_klines(symbol, str(interval), limit=int(limit))
            if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 20:
                return df
        except Exception:
            pass

    iv_min = int(interval)
    headers = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
    base = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = cfg.strip_multiplier_prefix(base)

    gate_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h"}.get(iv_min, "5m")
    for contract in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": contract, "interval": gate_iv, "limit": int(limit)},
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200:
                df = _normalize_df(r.json(), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 20:
                    return df
        except Exception:
            pass

    bg_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1H", 240: "4H"}.get(iv_min, "5m")
    for bg_sym in (symbol, f"{clean_base}USDT"):
        try:
            r = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": bg_sym, "productType": "USDT-FUTURES", "granularity": bg_iv, "limit": str(limit)},
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200:
                df = _normalize_df(r.json().get("data", []), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 20:
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
    symbol = coin["symbol"]
    base_coin = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    to_float = cfg.to_float

    if cfg.is_tradfi_ticker(base_coin):
        return None
    if hasattr(cfg, "is_heavy_major_coin") and cfg.is_heavy_major_coin(base_coin):
        return None

    df_5m = fetch_klines(symbol, "5", limit=150)
    if df_5m.empty or len(df_5m) < 60:
        return None

    df_15m = fetch_klines(symbol, "15", limit=90)
    df_1h = fetch_klines(symbol, "60", limit=75)
    if df_15m.empty or len(df_15m) < 25 or df_1h.empty or len(df_1h) < 25:
        return None

    close = to_float(df_5m["close"].iloc[-1])
    open_p = to_float(df_5m["open"].iloc[-1], close)
    high_p = to_float(df_5m["high"].iloc[-1], close)
    low_p = to_float(df_5m["low"].iloc[-1], close)
    if close <= 0:
        return None

    # 1. 설거지 및 역V자 급락 사전 차단
    recent_12h = df_5m.iloc[-144:]
    high_12h = to_float(recent_12h["high"].max(), close)
    low_12h = to_float(recent_12h["low"].min(), close)

    drawdown_12h_pct = (high_12h - close) / max(high_12h, 1e-8) * 100.0
    if drawdown_12h_pct >= cfg.MAX_DRAWDOWN_FROM_12H_HIGH_PCT:
        return None

    prior_pump_pct = (high_12h - low_12h) / max(low_12h, 1e-8) * 100.0
    if prior_pump_pct >= cfg.MAX_PRIOR_PUMP_PCT:
        retracement_ratio = (high_12h - close) / max(high_12h - low_12h, 1e-8)
        if retracement_ratio >= cfg.MAX_RETRACEMENT_RATIO:
            return None

    # 2. 멀티프레임 과열 차단
    rsi6_15m = to_float(calc_rsi(df_15m["close"], 6).iloc[-1], 50.0)
    kdj_j_15m = to_float(calc_kdj_j(df_15m, 9).iloc[-1], 50.0)
    rsi6_1h = to_float(calc_rsi(df_1h["close"], 6).iloc[-1], 50.0)

    if rsi6_15m >= cfg.CUTOFF_RSI6_15M_MAX or kdj_j_15m >= cfg.CUTOFF_KDJ_J_15M_MAX:
        return None
    if rsi6_1h >= cfg.CUTOFF_RSI6_1H_MAX:
        return None

    # 3. 5분봉 이평선, 리본 밀집도, 박스권 계산
    for p in (5, 20, 50, 112):
        df_5m[f"ma{p}"] = df_5m["close"].rolling(p, min_periods=min(p, 20)).mean()
        df_5m[f"ema{p}"] = df_5m["close"].ewm(span=p, adjust=False).mean()

    ma5 = to_float(df_5m["ma5"].iloc[-1], close)
    ma20 = to_float(df_5m["ma20"].iloc[-1], close)
    ma50 = to_float(df_5m["ma50"].iloc[-1], close)
    ma112 = to_float(df_5m["ma112"].iloc[-1], close)

    # [오답노트 $PUMP·$ZRO·$Q 패인 차단]: 머리 바로 위(0%~+4.0%)에 5분봉 또는 15분봉 MA112 장기이평 저항이 가로막고 있으면 진입 차단!
    ma50_15m = to_float(df_15m["close"].rolling(50, min_periods=15).mean().iloc[-1], close)
    ma112_15m = to_float(df_15m["close"].rolling(112, min_periods=25).mean().iloc[-1], close)
    if close < ma112 and ((ma112 - close) / close * 100.0) <= 4.0:
        return None
    if close < ma112_15m and ((ma112_15m - close) / close * 100.0) <= 4.5:
        return None
    if close < ma50_15m and ((ma50_15m - close) / close * 100.0) <= 3.0:
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

    if "change_24h" in coin:
        change_24h = to_float(coin.get("change_24h"))
    else:
        change_24h = to_float(coin.get("price24hPcnt")) * 100.0

    # 4. 1H 상위프레임 구조적 지지 (5종목 중 4종목 적중 필수 필터)
    ma20_1h = to_float(df_1h["close"].rolling(20, min_periods=10).mean().iloc[-1], close)
    ema20_1h = to_float(df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-1], close)
    htf_support_ok = (close >= ma20_1h * 0.995) or (close >= ema20_1h * 0.995)

    if not htf_support_ok:
        return None

    btc_dumping, _ = get_btc_market_regime()
    if btc_dumping:
        if close < bb_mid_15m or change_24h < 2.5:
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

    cond_ema_ribbon = (pre5_min_ribbon_spread <= cfg.EMA_RIBBON_MAX_SPREAD_PCT) and (close > ma50)
    cond_vol_pocket = (pre_vol_pocket_ratio <= cfg.PRE_VOL_POCKET_MAX_RATIO) and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)

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

    oi_turnover_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0

    breakout_diff_pct = (close - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    sr_error_pct = abs(pullback_low - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    is_current_ignition = (ignition_idx == df_5m.index[-1])

    curr_range = max(high_p - low_p, 1e-8)
    upper_wick_ratio = (high_p - max(open_p, close)) / curr_range
    curr_candle_gain_pct = (close - open_p) / max(open_p, 1e-8) * 100.0

    # 1) 전고점 S/R Flip 칼지지 ($SHELL 0.25%, $UNITAS 0.43%, $NIGHT 1.22% 적중)
    cond_sr_flip_retest = (
        (not is_current_ignition)
        and (close >= prev_box_high * 0.996)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (sr_error_pct <= cfg.SR_FLIP_TOLERANCE_PCT)
        and (pullback_vol_ratio <= cfg.SR_FLIP_DRYUP_RATIO_MAX)
        and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)
    )

    # 2) 공중 눌림(High-Tight Bull Flag)
    impulse_range = max(ignition_high - prev_box_high, close * 0.004)
    flag_support_line = ignition_high - (impulse_range * cfg.BULL_FLAG_MAX_RETRACE)
    cond_bull_flag = (
        (not is_current_ignition)
        and (close > prev_box_high)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (pullback_low >= flag_support_line)
        and (close >= ma5 * 0.998)
        and (pullback_vol_ratio <= cfg.PULLBACK_DRYUP_MAX)
        and (cfg.EARLY_VOL_MULT_MIN <= vol_mult <= cfg.EARLY_VOL_MULT_MAX)
    )

    # 3) 장대양봉 50% 중심선(FVG/Equilibrium) 칼지지 ($PIEVERSE, $NIGHT 등 5종목 중 3종목 적중)
    fvg_mid_line = (ignition_open + ignition_high) * 0.5
    cond_fvg_mid_hold = (
        (not is_current_ignition)
        and (close >= prev_box_high * 0.995)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (pullback_low >= fvg_mid_line * 0.998)
        and (pullback_vol_ratio <= 0.35)
        and (cfg.EARLY_VOL_MULT_MIN <= vol_mult <= cfg.EARLY_VOL_MULT_MAX)
    )

    # 4) 포켓 피벗(Pocket Pivot) 초동 양봉 직격
    cond_pocket_pivot = (
        is_current_ignition
        and (close > open_p)
        and (0.4 <= curr_candle_gain_pct <= 3.8)
        and (upper_wick_ratio <= 0.36)
        and (-0.5 <= breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (cfg.EARLY_VOL_MULT_MIN <= curr_vol_mult <= cfg.EARLY_VOL_MULT_MAX)
        and (pre_vol_pocket_ratio <= 0.40 or pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT or cond_ema_ribbon)
        and htf_support_ok
    )

    # 5) Track A (박스권 선취매 - 초응축 및 초동 수급 확인 필수화)
    cond_track_a_pullback = (
        htf_support_ok
        and (close >= ma112)
        and (close <= prev_box_high * 1.008)
        and (close >= ma20 * 0.996)
        and (effective_dryup_ratio <= cfg.PULLBACK_DRYUP_MAX)
        and (vol_mult >= 2.2 or cond_vol_pocket)
        and (ma_short_aligned or cond_spring or cond_ema_ribbon)
    )

    is_track_b = cond_sr_flip_retest or cond_bull_flag or cond_fvg_mid_hold or cond_pocket_pivot
    if not is_track_b and not cond_track_a_pullback:
        return None

    track_type = "TRACK_B" if is_track_b else "TRACK_A"

    score = 0
    reasons = []

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

    support_line = max(prev_box_low, min(ma50, ignition_low))
    swing_high_line = max(prev_box_high, ignition_high)

    entry_low = close * 0.9975
    entry_high = close * 1.0010

    # 1) 고정 손절 -6%
    sl_pct_int = int(getattr(cfg, "FIXED_SL_PCT", 6))
    stop_price = close * (1.0 - sl_pct_int / 100.0)
    sl_note = f"급등전 휩소·스탑헌팅 방어 고정손절(-{sl_pct_int}%)"

    # 2) Top 5 + 오답노트 로데이터 학습 기반 1·2·3차 익절(TP1~TP3) 산출
    tp_params = get_dynamic_tp_params()
    tp1_min_ratio = 1.0 + tp_params["tp1_min_pct"] / 100.0
    tp2_min_ratio = 1.0 + tp_params["tp2_min_pct"] / 100.0
    tp3_min_ratio = 1.0 + tp_params["tp3_min_pct"] / 100.0
    tp3_cap_ratio = 1.0 + tp_params["tp3_max_cap_pct"] / 100.0

    fib1 = tp_params["tp1_fib_mult"]
    fib2 = tp_params["tp2_fib_mult"]
    fib3 = tp_params["tp3_fib_mult"]

    ext1_ratio = 1.0 + tp_params["tp1_ma20_ext_pct"] / 100.0
    ext2_ratio = 1.0 + tp_params["tp2_ma20_ext_pct"] / 100.0
    ext3_ratio = 1.0 + tp_params["tp3_ma20_ext_pct"] / 100.0

    wave_height = max(swing_high_line - support_line, close * 0.042)
    fib_tp1 = support_line + wave_height * fib1
    fib_tp2 = support_line + wave_height * fib2
    fib_tp3 = support_line + wave_height * fib3

    ma_env_tp1 = max(bb_mid_15m, ma20_1h) * ext1_ratio
    ma_env_tp2 = max(bb_mid_15m, ma20_1h) * ext2_ratio
    ma_env_tp3 = max(bb_mid_15m, ma20_1h) * ext3_ratio

    blended_w1 = (fib_tp1 * 0.60) + (ma_env_tp1 * 0.40)
    blended_w2 = (fib_tp2 * 0.60) + (ma_env_tp2 * 0.40)
    blended_w3 = (fib_tp3 * 0.70) + (ma_env_tp3 * 0.30)

    tp1 = max(close * tp1_min_ratio, min(close * 1.14, blended_w1))
    tp2 = max(close * tp2_min_ratio, tp1 * 1.06, min(close * 1.28, blended_w2))
    tp3 = max(close * tp3_min_ratio, tp2 * 1.08, min(close * tp3_cap_ratio, blended_w3))

    tp1_pct = round((tp1 - close) / close * 100.0, 1)
    tp2_pct = round((tp2 - close) / close * 100.0, 1)
    tp3_pct = round((tp3 - close) / close * 100.0, 1)

    tp_labels = [
        f"1차첫조정(Fib {fib1:.2f}·MA이격)",
        f"2차중간고점(Fib {fib2:.2f}·MA이격)",
        f"3차최종고점(Fib {fib3:.2f}·대시세)"
    ]

    return SafeAlertDict({
        "symbol": symbol,
        "base_coin": base_coin,
        "base": base_coin,
        "track_type": track_type,
        "track_title": "🚀 [2차 돌파/리테스트 타점 포착]" if track_type == "TRACK_B" else "🟢 [1차 선취매/눌림 타점 포착]",
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
