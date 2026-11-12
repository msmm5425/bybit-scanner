import time
import inspect
import requests
import numpy as np
import pandas as pd

import config as cfg
import api as api_mod

_BTC_REGIME_CACHE = {"ts": 0, "is_dumping": False, "drop_pct": 0.0}


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
    iv_min = int(interval)
    if api_mod is not None:
        excluded = {
            "get_scanner_candidates", "fetch_verified_1m_klines",
            "to_float", "is_tradfi_ticker", "strip_multiplier_prefix", "strip_tokenized_suffix",
        }
        for name, fn in inspect.getmembers(api_mod, inspect.isfunction):
            if name in excluded or name.startswith("_") or fn.__module__ != api_mod.__name__:
                continue
            for args in [(symbol, str(interval), int(limit)), (symbol, int(interval), int(limit))]:
                try:
                    df = _normalize_df(fn(*args), expected_interval_min=iv_min)
                    if not df.empty and len(df) >= 20:
                        return df
                except Exception:
                    continue

    headers = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
    base = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = cfg.strip_multiplier_prefix(base)

    for host in ("https://api.bybit-tr.com", "https://api.bybit.kz", "https://api.bytick.com", "https://api.bybit.com"):
        try:
            r = requests.get(
                f"{host}/v5/market/kline",
                params={"category": "linear", "symbol": symbol, "interval": str(interval), "limit": int(limit)},
                headers=headers,
                timeout=5,
            )
            if r.status_code == 200:
                df = _normalize_df(r.json().get("result", {}).get("list", []), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 20:
                    return df
        except Exception:
            pass

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

    df_btc = fetch_klines("BTCUSDT", "15", limit=35)
    is_dumping = False
    drop_pct = 0.0
    if not df_btc.empty and len(df_btc) >= 10:
        c_now = cfg.to_float(df_btc["close"].iloc[-1])
        h_recent3 = cfg.to_float(df_btc["high"].iloc[-3:].max(), c_now)
        ma20_btc = cfg.to_float(df_btc["close"].rolling(20, min_periods=5).mean().iloc[-1], c_now)
        rsi_btc = cfg.to_float(calc_rsi(df_btc["close"], 6).iloc[-1], 50.0)
        drop_pct = (c_now - h_recent3) / max(h_recent3, 1e-8) * 100.0
        drop_threshold = getattr(cfg, "BTC_DUMP_DROP_15M_PCT", -0.70)
        if drop_pct <= drop_threshold or (c_now < ma20_btc * 0.992 and rsi_btc < 32.0):
            is_dumping = True

    _BTC_REGIME_CACHE["ts"] = now
    _BTC_REGIME_CACHE["is_dumping"] = is_dumping
    _BTC_REGIME_CACHE["drop_pct"] = drop_pct
    return is_dumping, drop_pct


def analyze_symbol(coin):
    symbol = coin["symbol"]
    to_float = cfg.to_float

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

    # 1. 설거지 및 역V자 급락 사전 차단 (최근 12시간)
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

    # 2. 멀티프레임 과열 차단 (15m RSI6, KDJ J, BB 이격, 1H RSI6)
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

    # [오답노트 유형5 처방 반영] 1H 중심선 구조적 지지 필수화 (Top 5 100% 적중 조건)
    ma20_1h = to_float(df_1h["close"].rolling(20, min_periods=10).mean().iloc[-1], close)
    ema20_1h = to_float(df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-1], close)
    htf_support_ok = (close >= ma20_1h * 0.990) or (close >= ema20_1h * 0.990)
    if not htf_support_ok:
        return None

    btc_dumping, _ = get_btc_market_regime()

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

    pre_slice_vols = df_5m["volume"].iloc[-7:-1]
    non_zero_pre_vols = pre_slice_vols[pre_slice_vols > 0]
    pre5_min_vol = to_float(non_zero_pre_vols.min() if not non_zero_pre_vols.empty else pre_slice_vols.min(), vol_ma20_curr)
    pre_vol_pocket_ratio = max(0.05, pre5_min_vol / max(vol_ma20_curr, 1e-8))

    pullback_vol_ratio = max(0.03, curr_vol / max(ignition_vol, 1e-8))
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

    cond_ema_ribbon = (pre5_min_ribbon_spread <= cfg.EMA_RIBBON_MAX_SPREAD_PCT) and (close > ma50 * 0.996)
    cond_vol_pocket = (pre_vol_pocket_ratio <= cfg.PRE_VOL_POCKET_MAX_RATIO) and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)

    turnover_m = to_float(coin.get("turnover_m"))
    if turnover_m <= 0:
        turnover_m = to_float(coin.get("turnover24h")) / 1e6

    oi_m = to_float(coin.get("oi_m"))
    if oi_m <= 0:
        oi_m = to_float(coin.get("openInterestValue")) / 1e6

    # 펀딩비 100배 중복 곱셈 방지 보정
    if "funding_pct" in coin and coin.get("funding_pct") is not None:
        funding_pct = cfg.normalize_funding_pct(coin.get("funding_pct"), already_pct=True)
    else:
        funding_pct = cfg.normalize_funding_pct(coin.get("fundingRate"), already_pct=False)

    if "change_24h" in coin:
        change_24h = to_float(coin.get("change_24h"))
    else:
        change_24h = to_float(coin.get("price24hPcnt")) * 100.0

    oi_turnover_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0

    breakout_diff_pct = (close - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    sr_error_pct = abs(pullback_low - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    is_current_ignition = (ignition_idx == df_5m.index[-1])

    curr_range = max(high_p - low_p, 1e-8)
    upper_wick_ratio = (high_p - max(open_p, close)) / curr_range
    curr_candle_gain_pct = (close - open_p) / max(open_p, 1e-8) * 100.0

    # 1) 전고점 S/R Flip 칼지지
    cond_sr_flip_retest = (
        (not is_current_ignition)
        and (close >= prev_box_high * 0.996)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (sr_error_pct <= cfg.SR_FLIP_TOLERANCE_PCT)
        and (pullback_vol_ratio <= cfg.SR_FLIP_DRYUP_RATIO_MAX)
        and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)
    )

    # 2) CROSSUSDT 18:10 패턴: 공중 눌림(High-Tight Bull Flag)
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

    # 3) 장대양봉 50% 중심선(FVG/Equilibrium) 칼지지
    fvg_mid_line = (ignition_open + ignition_high) * 0.5
    cond_fvg_mid_hold = (
        (not is_current_ignition)
        and (close > prev_box_high)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (pullback_low >= fvg_mid_line * 0.998)
        and (pullback_vol_ratio <= cfg.PULLBACK_DRYUP_MAX)
        and (cfg.EARLY_VOL_MULT_MIN <= vol_mult <= cfg.EARLY_VOL_MULT_MAX)
    )

    # 4) SOONUSDT 19:50 패턴: 포켓 피벗(Pocket Pivot) 초동 양봉 직격 (Top 5 중 3종목 적중)
    cond_pocket_pivot = (
        is_current_ignition
        and (close > open_p)
        and (0.4 <= curr_candle_gain_pct <= 3.5)
        and (upper_wick_ratio <= 0.36)
        and (-0.5 <= breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (cfg.EARLY_VOL_MULT_MIN <= curr_vol_mult <= cfg.EARLY_VOL_MULT_MAX)
        and (pre_vol_pocket_ratio <= 0.40 or pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT or cond_ema_ribbon)
        and (ma_short_aligned or cond_spring or cond_ema_ribbon)
    )

    # 5) Track A (박스권 선취매 & 거래량 건조 눌림)
    cond_track_a_pullback = (
        (close <= prev_box_high * 1.008)
        and (close >= ma20 * 0.995)
        and (effective_dryup_ratio <= cfg.PULLBACK_DRYUP_MAX)
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
        score += 13
        reasons.append("단기 정배열(5>20>50) 지지")
    elif cond_spring:
        score += 14
        reasons.append(f"유동성 스윕(Spring 아랫꼬리 {lower_wick_ratio_max6:.1f}배) 반전")
    elif cond_ema_ribbon:
        score += 13
        reasons.append(f"바닥 이평리본 초밀집({pre5_min_ribbon_spread:.1f}%) 반전")

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

    # [오답노트 유형2($Q 스탑헌팅) 처방 반영] ATR 버퍼 확대(1.40~1.45배) 및 최소 손절폭 -3% 보장
    atr_val = to_float(atr_5m.iloc[-1], close * 0.008)
    support_line = max(prev_box_low, min(ma50, ignition_low))
    swing_high_line = max(prev_box_high, ignition_high)

    entry_low = close * 0.9975
    entry_high = close * 1.0010

    if track_type == "TRACK_B":
        atr_mult = getattr(cfg, "SL_ATR_MULT_TRACK_B", 1.40)
        raw_stop = min(prev_box_high, flag_support_line, ignition_low) - (atr_mult * atr_val)
        sl_note = "S/R·포켓피벗지지선+스탑헌팅방어버퍼"
    else:
        atr_mult = getattr(cfg, "SL_ATR_MULT_TRACK_A", 1.45)
        raw_stop = support_line - (atr_mult * atr_val)
        sl_note = "15m/1H 매집지지선+스탑헌팅방어버퍼"

    raw_sl_pct = (close - raw_stop) / max(close, 1e-8) * 100.0
    sl_min = getattr(cfg, "SL_MIN_PCT", 3)
    sl_max = getattr(cfg, "SL_MAX_PCT", 5)
    sl_pct_int = int(np.clip(round(raw_sl_pct), sl_min, sl_max))
    stop_price = close * (1.0 - sl_pct_int / 100.0)

    wave_height = max(swing_high_line - support_line, close * 0.032)
    tp1 = max(close * 1.080, support_line + wave_height * 2.13)
    tp2 = max(close * 1.145, support_line + wave_height * 3.06)
    tp3 = max(close * 1.210, support_line + wave_height * 3.86)

    tp1_pct = round((tp1 - close) / close * 100.0, 1)
    tp2_pct = round((tp2 - close) / close * 100.0, 1)
    tp3_pct = round((tp3 - close) / close * 100.0, 1)

    tp_labels = [
        "Fib 2.13(1차표준)",
        "1H Fib 3.06(중기저항)",
        "1H Fib 3.86(48h고점)"
    ]

    return {
        "symbol": symbol,
        "base_coin": symbol.replace("USDT", ""),
        "track_type": track_type,
        "track_title": "🚀 [2차 돌파/리테스트 타점 포착]" if track_type == "TRACK_B" else "🟢 [1차 선취매/눌림 타점 포착]",
        "score": score,
        "price": close,
        "change_24h": change_24h,
        "turnover_m": turnover_m,
        "oi_m": oi_m,
        "funding_pct": funding_pct,
        "htf_status": htf_status,
        "htf_diag": htf_status,
        "rsi6_15m": round(rsi6_15m, 1),
        "kdj_j_15m": round(kdj_j_15m, 1),
        "j_15m": round(kdj_j_15m, 1),
        "reasons": reasons,
        "reason_str": " / ".join(reasons),
        "support_line": support_line,
        "swing_high_line": swing_high_line,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "stop_price": stop_price,
        "sl_pct": sl_pct_int,
        "sl_reason": sl_note,
        "sl_note": sl_note,
        "tps": [tp1, tp2, tp3],
        "tp_pcts": [tp1_pct, tp2_pct, tp3_pct],
        "tp_labels": tp_labels,
        "tp_notes": tp_labels,
        "vol_mult": round(vol_mult, 1),
    }
