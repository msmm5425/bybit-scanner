import numpy as np
import pandas as pd

import config as cfg
from api import fetch_klines


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

    # 직전 2시간 박스권 (최근 3캔들 제외한 직전 24캔들)
    pre_box_df = df_5m.iloc[-28:-3]
    prev_box_high = to_float(pre_box_df["high"].max(), close)
    prev_box_low = to_float(pre_box_df["low"].min(), close)
    pre_box_range_pct = (prev_box_high - prev_box_low) / max(prev_box_low, 1e-8) * 100.0

    # 15m BB 중심선 이격 체크 (초응축 5.2% 이하일 때는 4.8%까지 허용, 일반은 3.8%)
    bb_mid_15m = to_float(df_15m["close"].rolling(20, min_periods=10).mean().iloc[-1], close)
    bb_mid_ext_pct = (close - bb_mid_15m) / max(bb_mid_15m, 1e-8) * 100.0
    allowed_bb_ext = cfg.CUTOFF_BB_MID_EXT_MAX_PCT if pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT else 3.8
    if bb_mid_ext_pct > allowed_bb_ext:
        return None

    # 1H 상위프레임 구조적 지지 확인
    ma20_1h = to_float(df_1h["close"].rolling(20, min_periods=10).mean().iloc[-1], close)
    ema20_1h = to_float(df_1h["close"].ewm(span=20, adjust=False).mean().iloc[-1], close)
    htf_support_ok = (close >= ma20_1h * 0.992) or (close >= ema20_1h * 0.992)

    if close >= ma20_1h and rsi6_1h >= 55.0:
        htf_status = f"1H 상승우위(RSI6 {int(round(rsi6_1h))})"
    elif htf_support_ok:
        htf_status = f"1H 바닥반전시도(RSI6 {int(round(rsi6_1h))})"
    else:
        htf_status = f"1H 단기반등시도(RSI6 {int(round(rsi6_1h))})"

    # 거래량 및 발화 캔들 탐색 (최근 6캔들 내 최대 양봉 발화 탐색)
    vol_ma20_series = df_5m["volume"].rolling(20, min_periods=5).mean().shift(1)
    recent_window = df_5m.iloc[-6:]
    vol_ratios = recent_window["volume"] / vol_ma20_series.iloc[-6:].replace(0, np.nan)
    ignition_idx = vol_ratios.idxmax()
    vol_mult = to_float(vol_ratios.loc[ignition_idx], 1.0)
    ignition_vol = to_float(df_5m.loc[ignition_idx, "volume"], 1.0)
    ignition_high = to_float(df_5m.loc[ignition_idx, "high"], close)
    ignition_low = to_float(df_5m.loc[ignition_idx, "low"], close)

    vol_ma20_curr = to_float(vol_ma20_series.iloc[-1], 1.0)
    curr_vol = to_float(df_5m["volume"].iloc[-1], ignition_vol)
    curr_vol_mult = curr_vol / max(vol_ma20_curr, 1e-8)

    # 직전 5캔들 최저 거래량 포켓 비율 (현재봉이 발화봉일 때를 대비해 직전 구간[-7:-1] 탐색)
    pre5_min_vol = to_float(df_5m["volume"].iloc[-7:-1].min(), vol_ma20_curr)
    pre_vol_pocket_ratio = pre5_min_vol / max(vol_ma20_curr, 1e-8)

    # 눌림 구간 거래량 비율 (발화봉 대비 비율 & 평소 거래량 대비 비율 중 유리한 값 반영)
    pullback_vol_ratio = curr_vol / max(ignition_vol, 1e-8)
    effective_dryup_ratio = min(pullback_vol_ratio, curr_vol_mult * 0.65)
    pullback_low = to_float(df_5m["low"].iloc[-2:].min(), close)

    # OBV & CVD 다이버전스 / 매집 확인
    obv, cvd, net_delta = calc_obv_and_cvd(df_5m)
    obv_slope_up = to_float(obv.iloc[-1]) > to_float(obv.iloc[-15:-5].mean())
    cvd_slope_up = to_float(cvd.iloc[-1]) > to_float(cvd.iloc[-15:-5].mean())
    price_low_sweep = to_float(df_5m["low"].iloc[-8:].min()) <= prev_box_low * 1.005
    cond_cvd_div = (price_low_sweep and cvd_slope_up) or (cvd_slope_up and to_float(net_delta.iloc[-3:].sum()) > 0)

    # 유동성 스윕(Spring) & 이평선 정배열 확인
    ma_full_aligned = (ma5 > ma20 > ma50 > ma112) and (close >= ma20)
    ma_short_aligned = (ma5 > ma20 > ma50) and (close >= ma20)

    body_len = (df_5m["close"] - df_5m["open"]).abs().replace(0, 1e-8)
    lower_wick = df_5m[["open", "close"]].min(axis=1) - df_5m["low"]
    lower_wick_ratio_max6 = to_float((lower_wick / body_len).iloc[-6:].max(), 0.0)
    cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.4) and (close > ma20)

    # BB 스퀴즈 & TTM 켈트너 스퀴즈 확인
    bb_std = df_5m["close"].rolling(20, min_periods=10).std()
    bb_upper = df_5m["ma20"] + 2.0 * bb_std
    bb_lower = df_5m["ma20"] - 2.0 * bb_std
    atr_5m = calc_atr(df_5m, 14)
    kc_upper = df_5m["ma20"] + 1.5 * atr_5m
    kc_lower = df_5m["ma20"] - 1.5 * atr_5m
    ttm_squeeze = bool(((bb_upper < kc_upper) & (bb_lower > kc_lower)).iloc[-8:-1].any())

    cond_ema_ribbon = (pre5_min_ribbon_spread <= cfg.EMA_RIBBON_MAX_SPREAD_PCT) and (close > ma50)
    cond_vol_pocket = (pre_vol_pocket_ratio <= 0.40) and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)

    # 선물 수급 데이터 추출 (거래대금, OI, 펀딩비)
    turnover_m = to_float(coin.get("turnover_m"))
    if turnover_m <= 0:
        turnover_m = to_float(coin.get("turnover24h")) / 1e6

    oi_m = to_float(coin.get("oi_m"))
    if oi_m <= 0:
        oi_m = to_float(coin.get("openInterestValue")) / 1e6

    if "funding_pct" in coin:
        funding_pct = to_float(coin.get("funding_pct"))
    else:
        funding_pct = to_float(coin.get("fundingRate")) * 100.0

    if "change_24h" in coin:
        change_24h = to_float(coin.get("change_24h"))
    else:
        change_24h = to_float(coin.get("price24hPcnt")) * 100.0

    oi_turnover_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0

    # =====================================================================
    # [Track A (선취매/눌림) vs Track B (돌파/Bull Flag/Pocket Pivot) 판별]
    # =====================================================================
    breakout_diff_pct = (close - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    sr_error_pct = abs(pullback_low - prev_box_high) / max(prev_box_high, 1e-8) * 100.0
    is_current_ignition = (ignition_idx == df_5m.index[-1])

    # 현재 캔들의 양봉 몸통 및 윗꼬리 비율 (포켓 피벗 첫 양봉 검증용)
    curr_range = max(high_p - low_p, 1e-8)
    upper_wick_ratio = (high_p - max(open_p, close)) / curr_range
    curr_candle_gain_pct = (close - open_p) / max(open_p, 1e-8) * 100.0

    # 1) 전고점 S/R Flip 칼지지 조건 (발화 이후 눌림 캔들)
    cond_sr_flip_retest = (
        (not is_current_ignition)
        and (close >= prev_box_high * 0.996)
        and (breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (sr_error_pct <= cfg.SR_FLIP_TOLERANCE_PCT)
        and (pullback_vol_ratio <= cfg.SR_FLIP_DRYUP_RATIO_MAX)
        and (vol_mult >= cfg.EARLY_VOL_MULT_MIN)
    )

    # 2) CROSSUSDT 18:10 패턴: 돌파 후 공중 눌림(High-Tight Bull Flag)
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

    # 3) [신규] SOONUSDT 19:50 패턴: 직전 거래량 건조/응축 후 첫 초동 양봉(Pocket Pivot) 즉시 포착
    cond_pocket_pivot = (
        is_current_ignition
        and (close > open_p)
        and (0.4 <= curr_candle_gain_pct <= 3.5)
        and (upper_wick_ratio <= 0.42)
        and (-0.5 <= breakout_diff_pct <= cfg.EARLY_BREAKOUT_MAX_PCT)
        and (cfg.EARLY_VOL_MULT_MIN <= curr_vol_mult <= cfg.EARLY_VOL_MULT_MAX)
        and (pre_vol_pocket_ratio <= 0.40 or pre_box_range_pct <= cfg.PRE_BOX_TIGHT_PCT or cond_ema_ribbon)
        and (ma_short_aligned or cond_spring or htf_support_ok)
    )

    # 4) Track A (박스권 상단 턱밑 선취매 & 거래량 건조 눌림)
    cond_track_a_pullback = (
        (close <= prev_box_high * 1.008)
        and (close >= ma20 * 0.995)
        and (effective_dryup_ratio <= cfg.PULLBACK_DRYUP_MAX)
        and (ma_short_aligned or cond_spring)
    )

    is_track_b = cond_sr_flip_retest or cond_bull_flag or cond_pocket_pivot
    if not is_track_b and not cond_track_a_pullback:
        return None

    track_type = "TRACK_B" if is_track_b else "TRACK_A"

    # =====================================================================
    # [100점 만점 정밀 채점 시스템]
    # =====================================================================
    score = 0
    reasons = []

    # [카테고리 1] 타점 컨펌 (20점)
    if track_type == "TRACK_B":
        if cond_sr_flip_retest:
            score += 20
            reasons.append(f"전고점 S/R Flip 칼지지(오차 {sr_error_pct:.2f}%/눌림거래량 {int(round(pullback_vol_ratio * 100))}%)")
        elif cond_bull_flag:
            score += 20
            reasons.append(f"공중눌림 Bull Flag 칼지지(상단38.2%·MA5 사수/눌림거래량 {int(round(pullback_vol_ratio * 100))}%)")
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

    # [카테고리 2] 추세/반전 및 상위프레임 동조 (25점)
    if ma_full_aligned:
        score += 15
        reasons.append("완전 정배열(5>20>50>112) 지지")
    elif ma_short_aligned:
        score += 13
        reasons.append("단기 정배열(5>20>50) 지지")
    elif cond_spring:
        score += 14
        reasons.append(f"유동성 스윕(Spring 아랫꼬리 {lower_wick_ratio_max6:.1f}배) 반전")

    if htf_support_ok:
        score += 10
        reasons.append("상위프레임(1H 중심선) 구조적 지지")

    # [카테고리 3] 매집/박스권 응축 및 CVD/OBV (20점)
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

    # [카테고리 4] 변동성 돌파 & 이평선 리본 초밀집 (15점)
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

    # [카테고리 5] 선물 실수급 OI & 펀딩비 숏스퀴즈 (20점)
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
    if score < cfg.SCORE_THRESHOLD:
        return None

    # =====================================================================
    # [핵심 기준선, 유동적 손절가(SL -2%~-5%), 3단계 피보나치 익절가(TP) 산출]
    # =====================================================================
    atr_val = to_float(atr_5m.iloc[-1], close * 0.008)
    support_line = max(prev_box_low, min(ma50, ignition_low))
    swing_high_line = max(prev_box_high, ignition_high)

    entry_low = close * 0.9975
    entry_high = close * 1.0010

    if track_type == "TRACK_B":
        raw_stop = min(prev_box_high, flag_support_line, ignition_low) - (1.0 * atr_val)
        sl_note = "S/R·포켓피벗지지선+단기변동성버퍼"
    else:
        raw_stop = support_line - (1.2 * atr_val)
        sl_note = "15m/1H 매집지지선+ATR버퍼"

    raw_sl_pct = (close - raw_stop) / max(close, 1e-8) * 100.0
    sl_pct_int = int(np.clip(round(raw_sl_pct), 2, 5))
    stop_price = close * (1.0 - sl_pct_int / 100.0)

    # 파동 진폭 기반 3단계 피보나치 목표가 (최소 +8% / +15% / +22% 보장)
    wave_height = max(swing_high_line - support_line, close * 0.032)
    tp1 = max(close * 1.078, support_line + wave_height * 2.13)
    tp2 = max(close * 1.150, support_line + wave_height * 3.06)
    tp3 = max(close * 1.225, support_line + wave_height * 3.86)

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
