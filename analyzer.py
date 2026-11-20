import os, re, json
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

import config as cfg
from config import to_float, normalize_funding_pct, is_tradfi_ticker
from api import fetch_klines

try:
    from api import is_stock_or_tradfi_symbol
except ImportError:
    _FALLBACK_STOCKS = {
        "SAMSUNG", "HYNIX", "SKHYNIX", "HYUNDAI", "POSCO", "NAVER", "KAKAO",
        "KOSPI", "LITE", "ANTA", "TSLA", "NVDA", "AAPL", "MSFT", "GOOG", "AMZN",
        "META", "NFLX", "MSTR", "COIN", "SPY", "QQQ", "XAU", "XAG", "WTI", "BRENT", "SPX", "NDX",
    }
    def is_stock_or_tradfi_symbol(sym_or_base):
        s = str(sym_or_base or "").upper().strip()
        base = s[:-4] if s.endswith("USDT") else s
        if not base or base in _FALLBACK_STOCKS or is_tradfi_ticker(base):
            return True
        return any(k in base for k in ("SAMSUNG", "HYNIX", "HYUNDAI", "KOSPI", "TSLA", "NVDA", "AAPL"))

try:
    import wave_engine
except Exception:
    wave_engine = None

try:
    import postmortem
except Exception:
    postmortem = None

KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
HISTORY_FILE = getattr(cfg, "ANALYZER_HISTORY_FILE", "analyzer_history.json")
ELITE_SCORE_THRESHOLD = max(90, int(getattr(cfg, "SCORE_THRESHOLD", 90)))
FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
LONGTERM_SL_PCT = 20


def get_dynamic_tp_params():
    default_params = {"tp1_min_pct": 8.0, "tp2_min_pct": 15.0, "tp3_min_pct": 22.0}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                learned = raw.get("dynamic_tp_tuning") or raw.get("learned_tp_params") or {}
                if isinstance(learned, dict):
                    p1 = float(np.clip(to_float(learned.get("tp1_min_pct"), 8.0), 6.5, 11.5))
                    p2 = float(np.clip(to_float(learned.get("tp2_min_pct"), 15.0), 13.0, 20.0))
                    p3 = float(np.clip(to_float(learned.get("tp3_min_pct"), 22.0), 20.0, 34.0))
                    return {"tp1_min_pct": round(p1, 1), "tp2_min_pct": round(p2, 1), "tp3_min_pct": round(p3, 1)}
        except Exception:
            pass
    return default_params


def _get_session_adjustment():
    """
    [글로벌 유동성 세션 가중치 필터 (KST 기준)]
    - 런던 개장(15:30~18:30 KST) & 뉴욕 개장(21:00~00:30 KST): +4점 가산
    - 아시아 점심 저유동성(11:30~14:30 KST): -4점 감점
    """
    now_kst = datetime.now(KST)
    hm = now_kst.hour + (now_kst.minute / 60.0)
    if 15.5 <= hm <= 18.5:
        return 4, "런던 유동성 세션(+4점)"
    if hm >= 21.0 or hm <= 0.5:
        return 4, "뉴욕 유동성 세션(+4점)"
    if 11.5 <= hm <= 14.5:
        return -4, ""
    return 0, ""


def _evaluate_oi_quadrant(price, oi_m, oi_snapshot_prev):
    """
    [OI × 가격 델타 4분면 분석]
    - Q1 (가격 + & OI +): 건전한 신규 롱 매집 → +10~15점 가산
    - Q2 (가격 + & OI 급감): 숏스퀴즈 반등 → TP1·TP2 보수적 하향(×0.88) & 경고
    - Q3 (가격 - & OI 급증): 공격적 숏 세력 진입 → 진입 원천 차단(block=True)
    - Q4 (가격 - & OI 감소): 롱 청산/털기 → 중립 (Track C 바닥 확인 시 +3점)
    """
    res = {"quadrant": "NEUTRAL", "bonus": 0, "block": False, "squeeze_warn": False, "note": ""}
    if not isinstance(oi_snapshot_prev, dict):
        return res
    p_prev = to_float(oi_snapshot_prev.get("price"), 0.0)
    oi_prev = to_float(oi_snapshot_prev.get("oi_m"), 0.0)
    if p_prev <= 0 or oi_prev <= 0:
        return res

    dp_pct = ((price - p_prev) / p_prev) * 100.0
    doi_pct = ((oi_m - oi_prev) / oi_prev) * 100.0

    # Q3: 가격 하락 + OI 급증 → 공격적 숏 진입 (진입 금지)
    if dp_pct <= -0.5 and doi_pct >= 2.5:
        res.update({"quadrant": "Q3", "block": True, "note": f"숏세력 유입 차단(가격 {dp_pct:+.1f}%, OI {doi_pct:+.1f}%)"})
        return res

    # Q1: 가격 상승 + OI 증가 → 건전한 신규 롱 매집 (+10~15점)
    if dp_pct >= 0.4 and doi_pct >= 2.0:
        bonus = 15 if doi_pct >= 5.0 else 10
        res.update({
            "quadrant": "Q1",
            "bonus": bonus,
            "note": f"[건전한 롱 매집] OI {doi_pct:+.1f}% 동반 상승(+{bonus}점)",
        })
        return res

    # Q2: 가격 상승 + OI 급감 → 숏스퀴즈 반등 (보수적 TP 하향)
    if dp_pct >= 1.0 and doi_pct <= -2.5:
        res.update({
            "quadrant": "Q2",
            "squeeze_warn": True,
            "note": f"[숏스퀴즈 주의] OI {doi_pct:+.1f}% 감소 반등(TP1·2 보수적 하향)",
        })
        return res

    # Q4: 가격 하락 + OI 감소 → 롱 청산 털기
    if dp_pct <= -0.5 and doi_pct <= -2.0:
        res.update({"quadrant": "Q4", "note": f"롱 청산 매물 소화(OI {doi_pct:+.1f}%)"})
        return res

    return res


def _calc_rsi(series, period=6):
    delta = series.diff()
    up = delta.clip(lower=0.0).ewm(com=period - 1, adjust=True, min_periods=period).mean()
    down = (-1.0 * delta.clip(upper=0.0)).ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = up / down.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _calc_kdj_j(df, n=9):
    low_n = df["low"].rolling(n, min_periods=3).min()
    high_n = df["high"].rolling(n, min_periods=3).max()
    denom = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["close"] - low_n) / denom * 100.0).fillna(50.0)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return (3.0 * k - 2.0 * d).fillna(50.0)


def _calc_ttm_squeeze_and_cvd(df_15m):
    """
    [Track A 심층 업그레이드]
    1) 존 카터 TTM Squeeze: BB(20, 2.0)가 Keltner Channel(20, 1.5 ATR) 안으로 진입(ON) 후 이탈(Fired)
    2) 캔들 누적 델타 추정치(CLV 기반 CVD Absorption): 가격 횡보 중 매수 델타 흡수 선행 상승 판별
    """
    closes = df_15m["close"].astype(float)
    highs = df_15m["high"].astype(float)
    lows = df_15m["low"].astype(float)
    vols = df_15m["volume"].astype(float).fillna(0.0)

    ma20 = closes.rolling(20, min_periods=5).mean()
    std20 = closes.rolling(20, min_periods=5).std().fillna(0.0)
    bb_upper = ma20 + (2.0 * std20)
    bb_lower = ma20 - (2.0 * std20)

    tr = pd.concat([
        highs - lows,
        (highs - closes.shift(1)).abs(),
        (lows - closes.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr20 = tr.rolling(20, min_periods=5).mean().fillna(0.0)
    kc_upper = ma20 + (1.5 * atr20)
    kc_lower = ma20 - (1.5 * atr20)

    squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    was_squeezed = bool(squeeze_on.iloc[-9:-1].any()) if len(squeeze_on) >= 10 else False
    curr_squeezed = bool(squeeze_on.iloc[-1]) if len(squeeze_on) >= 1 else False
    mom_up = to_float(closes.iloc[-1]) > to_float(ma20.iloc[-1]) and to_float(closes.iloc[-1]) > to_float(closes.iloc[-3])
    ttm_fired = bool(was_squeezed and (not curr_squeezed) and mom_up)

    # CLV 기반 캔들 델타(CVD Proxy) 흡수 검증
    bar_range = (highs - lows).replace(0, np.nan)
    clv = (((closes - lows) - (highs - closes)) / bar_range).clip(-1.0, 1.0).fillna(0.0)
    bar_delta = clv * vols
    cvd = bar_delta.cumsum()
    cvd_ma10 = cvd.rolling(10, min_periods=3).mean()

    recent_delta_sum = to_float(bar_delta.iloc[-3:].sum(), 0.0)
    cvd_breakout = to_float(cvd.iloc[-1]) >= to_float(cvd.iloc[-12:-1].max(), -1e18)
    cvd_absorption = bool((to_float(cvd.iloc[-1]) > to_float(cvd_ma10.iloc[-1])) and (recent_delta_sum > 0) and cvd_breakout)

    return ttm_fired, was_squeezed, cvd_absorption


def _check_5m_mss_trigger(symbol):
    """
    [Track C 5분봉 하위 프레임 시장 구조 전환(MSS / ChOCH) 또는 상승 장악형 확인]
    - 15분봉 심층 눌림 + 거래량 건조를 통과한 소수 정예 후보에만 5분봉을 추가 조회
    """
    df_5m = fetch_klines(symbol, interval="5", limit=36)
    if df_5m is None or df_5m.empty or len(df_5m) < 12:
        return None  # API 조회 불가 시 None 반환 (호출부에서 15m 강세 반전으로 대체 확인)

    o5 = df_5m["open"].astype(float)
    h5 = df_5m["high"].astype(float)
    l5 = df_5m["low"].astype(float)
    c5 = df_5m["close"].astype(float)

    # 직전 저점을 높이는 Higher Low 구조 확인
    recent_low = to_float(l5.iloc[-4:].min())
    prior_low = to_float(l5.iloc[-12:-4].min())
    higher_low = recent_low >= prior_low * 0.996

    # 단기 5분봉 스윙 고점 돌파(MSS / ChOCH)
    prev_swing_h = to_float(h5.iloc[-7:-2].max())
    mss_break = to_float(c5.iloc[-1]) > prev_swing_h and to_float(c5.iloc[-1]) > to_float(o5.iloc[-1])

    # 5분봉 상승 장악형 캔들(Bullish Engulfing)
    bull_engulf = (
        to_float(c5.iloc[-1]) > to_float(o5.iloc[-1])
        and to_float(c5.iloc[-2]) < to_float(o5.iloc[-2])
        and to_float(c5.iloc[-1]) >= to_float(o5.iloc[-2])
        and to_float(o5.iloc[-1]) <= to_float(c5.iloc[-2]) * 1.003
    )

    return bool((higher_low and mss_break) or bull_engulf)


def _calculate_resistance_based_tps(price, df_15m, df_1h, conservative_scale=1.0):
    """단기·스윙(Track A·B·C) 전용 실제 차트 저항선 기반 개별 TP 산출"""
    highs_15 = df_15m["high"].astype(float)
    lows_15 = df_15m["low"].astype(float)
    highs_1h = df_1h["high"].astype(float)
    lows_1h = df_1h["low"].astype(float)

    atr_1h_pct = to_float(((highs_1h - lows_1h) / df_1h["close"].replace(0, np.nan)).tail(24).mean(), 0.025) * 100.0
    vol_scale = float(np.clip(atr_1h_pct / 2.2, 0.82, 1.45))

    swing_h_24 = to_float(highs_15.iloc[-96:].max(), price * 1.08)
    swing_l_24 = to_float(lows_15.iloc[-96:].min(), price * 0.94)
    swing_h_1h = to_float(highs_1h.iloc[-72:].max(), price * 1.15)
    wave_span = max(swing_h_24 - swing_l_24, price * 0.04)

    res_candidates = []
    for i in range(3, len(highs_15) - 2):
        h_v = float(highs_15.iloc[i])
        if h_v > price * 1.055 and h_v >= float(highs_15.iloc[i - 2 : i + 3].max()):
            res_candidates.append(((h_v - price) / price * 100.0, "15m 전고점 매물대"))

    for i in range(2, len(highs_1h) - 1):
        h_v = float(highs_1h.iloc[i])
        if h_v > price * 1.06 and h_v >= float(highs_1h.iloc[i - 2 : i + 2].max()):
            res_candidates.append(((h_v - price) / price * 100.0, "1H 스윙 저항선"))

    res_candidates.extend([
        (((swing_h_24 - price) / price * 100.0), "24H 전고점 저항"),
        ((((max(swing_h_24, price) + wave_span * 0.272) - price) / price * 100.0), "Fib 1.272 저항대"),
        ((((max(swing_h_24, price) + wave_span * 0.618) - price) / price * 100.0), "Fib 1.618 확장저항"),
        ((((max(swing_h_1h, price) + wave_span * 1.272) - price) / price * 100.0), "Fib 2.272 상단채널"),
    ])

    tp1_pool = sorted([(pct, lbl) for pct, lbl in res_candidates if 6.5 <= pct <= 11.8], key=lambda x: x[0])
    p1_pct, lbl1 = tp1_pool[len(tp1_pool) // 2] if tp1_pool else (float(np.clip(7.2 * vol_scale + (wave_span / price * 18.0), 6.8, 11.4)), "단기 매물대 저항")

    tp2_min = max(p1_pct + 5.2, 13.0)
    tp2_pool = sorted([(pct, lbl) for pct, lbl in res_candidates if tp2_min <= pct <= 21.5], key=lambda x: x[0])
    p2_pct, lbl2 = tp2_pool[len(tp2_pool) // 2] if tp2_pool else (float(np.clip(p1_pct * 1.85 + 1.2, tp2_min, 20.8)), "1H 주요 저항구간")

    tp3_min = max(p2_pct + 6.5, 20.5)
    tp3_pool = sorted([(pct, lbl) for pct, lbl in res_candidates if tp3_min <= pct <= 35.0], key=lambda x: x[0])
    p3_pct, lbl3 = tp3_pool[len(tp3_pool) // 2] if tp3_pool else (float(np.clip(p2_pct * 1.48 + 1.5, tp3_min, 33.5)), "Fib 1.618 클라이맥스")

    # OI 2사분면(숏스퀴즈 주의) 시 TP1·TP2 보수적 하향으로 TP1 방어선 조기 잠금 유도
    if conservative_scale < 0.99:
        p1_pct = max(6.5, p1_pct * conservative_scale)
        p2_pct = max(p1_pct + 4.8, p2_pct * conservative_scale)
        lbl1 = f"{lbl1}·보수적"
        lbl2 = f"{lbl2}·보수적"

    p1_pct, p2_pct, p3_pct = round(float(p1_pct), 1), round(float(p2_pct), 1), round(float(p3_pct), 1)
    tps = [round(price * (1.0 + p1_pct / 100.0), 6), round(price * (1.0 + p2_pct / 100.0), 6), round(price * (1.0 + p3_pct / 100.0), 6)]
    return tps, [p1_pct, p2_pct, p3_pct], [lbl1, lbl2, lbl3]


def _calculate_macro_longterm_tps(price, turnover_m, df_4h, df_1d):
    """장기 포지션(Track D) 전용 큰 시간 프레임(4H·1D) 종목별 개별 TP 산출"""
    h4 = df_4h["high"].astype(float)
    l4 = df_4h["low"].astype(float)
    box_h_30d = to_float(h4.max(), price * 1.15)
    box_l_30d = to_float(l4.min(), price * 0.85)
    box_span = max(box_h_30d - box_l_30d, price * 0.12)

    if df_1d is not None and not df_1d.empty and len(df_1d) >= 15:
        h1d = df_1d["high"].astype(float)
        l1d = df_1d["low"].astype(float)
        c1d = df_1d["close"].astype(float)
        macro_h_120d = to_float(h1d.max(), box_h_30d * 1.6)
        atr_1d_pct = to_float(((h1d - l1d) / c1d.replace(0, np.nan)).tail(20).mean(), 0.065) * 100.0
    else:
        macro_h_120d = box_h_30d * 1.75
        atr_1d_pct = 6.5

    if turnover_m < 12.0:
        tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier1 초대형확장", 1.32, 145.0, 250.0
    elif turnover_m < 45.0:
        tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier2 계단식확장", 1.12, 115.0, 190.0
    else:
        tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier3 중형추세확장", 0.95, 90.0, 145.0

    macro_scale = float(np.clip((atr_1d_pct / 5.8) * tier_mult, 0.85, 1.65))

    fib1618_pct = ((box_h_30d + box_span * 0.618) - price) / price * 100.0
    p1_pct = round(float(np.clip(max(20.0, fib1618_pct * 0.85, 22.0 * macro_scale), 20.0, 40.0)), 1)

    fib2618_pct = ((box_h_30d + box_span * 1.618) - price) / price * 100.0
    macro_mid_pct = ((macro_h_120d - price) / price * 100.0) * 0.65
    p2_pct = round(float(np.clip(max(p1_pct + 26.0, fib2618_pct * 0.80, macro_mid_pct, 50.0 * macro_scale), 48.0, 95.0)), 1)

    fib4236_pct = ((box_h_30d + box_span * 3.236) - price) / price * 100.0
    macro_top_pct = (macro_h_120d - price) / price * 100.0
    p3_pct = round(float(np.clip(max(p2_pct + 48.0, fib4236_pct * 0.85, macro_top_pct, 105.0 * macro_scale), tp3_min_cap, tp3_max_cap)), 1)

    tps = [
        round(price * (1.0 + p1_pct / 100.0), 6),
        round(price * (1.0 + p2_pct / 100.0), 6),
        round(price * (1.0 + p3_pct / 100.0), 6),
    ]
    return tps, [p1_pct, p2_pct, p3_pct], ["4H/1D 1차 저항대", "1D 주요 전고점 저항", f"1D 대시세목표({tier_lbl})"]


def _evaluate_macro_track_d(price, change_24h, oi_ratio, funding_pct, df_4h):
    """신규 장기 포지션 Track D 3대 전략 판별 (D1·D2·D3 — 기존 유지)"""
    if df_4h is None or df_4h.empty or len(df_4h) < 45:
        return None

    c4 = df_4h["close"].astype(float)
    h4 = df_4h["high"].astype(float)
    l4 = df_4h["low"].astype(float)
    o4 = df_4h["open"].astype(float)
    v4 = df_4h["volume"].astype(float).fillna(0.0)

    obv4 = (np.sign(c4.diff().fillna(0.0)) * v4).cumsum()
    v4_ma20 = v4.rolling(20, min_periods=5).mean().replace(0, np.nan)
    v4_ratio = (v4 / v4_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ma20_4h = to_float(c4.rolling(20, min_periods=5).mean().iloc[-1], price)
    ma50_4h = to_float(c4.rolling(50, min_periods=10).mean().iloc[-1], price)

    hi_10d = to_float(h4.iloc[-60:-1].max(), price * 1.05)
    obv_hi_10d = to_float(obv4.iloc[-60:-1].max(), 0.0)
    if -3.0 <= change_24h <= 11.0 and oi_ratio >= 28.0 and funding_pct <= 0.008:
        if price < hi_10d * 0.995 and to_float(obv4.iloc[-1]) > obv_hi_10d and price >= ma20_4h * 0.985:
            score_d1 = int(np.clip(88 + min(7, int((oi_ratio - 26) / 3)) + (4 if funding_pct < 0 else 1), 88, 99))
            return {
                "sub_strategy": "D1 · 스텔스 OI·OBV 선행 매집",
                "score": score_d1,
                "reasons": [
                    "가격 횡보 중 4H OBV 10일 최고치 선행 돌파",
                    f"스텔스 미결제약정(OI) 집중도 {oi_ratio:.0f}% 축적",
                    f"펀딩비 {funding_pct:+.4f}% 저과열 바닥 매집 구조",
                ],
            }

    recent_30 = df_4h.iloc[-30:-2]
    vr_30 = v4_ratio.iloc[-30:-2]
    if not vr_30.empty and float(vr_30.max()) >= 3.8:
        pump_pos = int(np.argmax(vr_30.values))
        pump_row = recent_30.iloc[pump_pos]
        pump_vol = to_float(pump_row["volume"], 1.0)
        pump_open = to_float(pump_row["open"], price)
        pump_high = to_float(pump_row["high"], price)
        pump_mid = (pump_open + pump_high) * 0.5
        curr_vol_4h = to_float(v4.iloc[-2], pump_vol)
        is_dry_up = pump_vol > 0 and (curr_vol_4h / pump_vol) <= 0.30
        is_turn_up = c4.iloc[-1] >= o4.iloc[-1] and price >= max(pump_open * 0.99, ma20_4h * 0.985) and price <= pump_high * 1.02
        if is_dry_up and is_turn_up:
            score_d2 = int(np.clip(88 + (5 if price >= pump_mid else 2) + (5 if oi_ratio >= 15.0 else 1), 88, 99))
            return {
                "sub_strategy": "D2 · 매집봉 출현 후 거래량 건조 눌림",
                "score": score_d2,
                "reasons": [
                    f"최근 5일 내 4H 매집봉(거래량 {float(vr_30.max()):.1f}배) 출현",
                    "조정 구간 거래량 70% 이상 건조(세력 이탈 없음)",
                    "매집봉 지지선 & 4H 20MA 위 양봉 반등 턴",
                ],
            }

    if len(df_4h) >= 120:
        hi_30d = to_float(h4.iloc[:-1].max(), price)
        lo_30d = to_float(l4.iloc[:-1].min(), price * 0.8)
        range_30d_pct = (hi_30d - lo_30d) / max(lo_30d, 1e-8) * 100.0
        curr_vr_4h = to_float(v4_ratio.iloc[-1], 1.0)
        if hi_30d * 0.996 <= price <= hi_30d * 1.045 and curr_vr_4h >= 2.4 and range_30d_pct <= 55.0:
            score_d3 = int(np.clip(89 + min(7, int(curr_vr_4h * 1.5)) + (3 if price >= ma50_4h else 0), 89, 99))
            return {
                "sub_strategy": "D3 · 4H 30일 장기 박스권 첫 돌파",
                "score": score_d3,
                "reasons": [
                    "4H 30일 장기 박스권 상단 저항 첫 돌파 초입",
                    f"4H 돌파 거래량 평균 대비 {curr_vr_4h:.1f}배 폭발",
                    f"장기 박스권 응축폭 {range_30d_pct:.1f}% 소화 후 1파동 개시",
                ],
            }

    return None


def analyze_symbol(coin, oi_snapshot_prev=None, market_regime=None):
    symbol = str(coin.get("symbol", "")).upper().strip()
    base_coin = coin.get("base_coin") or symbol.replace("USDT", "")
    if is_stock_or_tradfi_symbol(symbol):
        return None

    # BTC 마스터 스위치가 잠겨 있으면 신규 진입 전면 차단
    if isinstance(market_regime, dict) and market_regime.get("gate_locked"):
        return None

    price = to_float(coin.get("price"))
    change_24h = to_float(coin.get("change_24h"))
    turnover_m = to_float(coin.get("turnover_m"))
    oi_m = to_float(coin.get("oi_m"))
    funding_pct = normalize_funding_pct(to_float(coin.get("funding_pct")))

    if price <= 0 or turnover_m < 2.5:
        return None

    df_15m = fetch_klines(symbol, interval="15", limit=120)
    df_1h = fetch_klines(symbol, interval="60", limit=80)
    if df_15m.empty or len(df_15m) < 35 or df_1h.empty or len(df_1h) < 20:
        return None

    closes = df_15m["close"].astype(float)
    highs = df_15m["high"].astype(float)
    lows = df_15m["low"].astype(float)
    opens = df_15m["open"].astype(float)
    vols = df_15m["volume"].astype(float).fillna(0.0)

    price = to_float(closes.iloc[-1], price)
    if abs(change_24h) < 0.05 and len(df_15m) >= 90:
        p_24h = to_float(opens.iloc[-min(len(df_15m), 96)], price)
        if p_24h > 0:
            change_24h = round((price - p_24h) / p_24h * 100.0, 2)

    # OI × 가격 델타 4분면 검증 (Q3 공격적 숏 유입 시 즉시 진입 차단)
    oi_q = _evaluate_oi_quadrant(price, oi_m, oi_snapshot_prev)
    if oi_q["block"]:
        return None

    sess_adj, sess_note = _get_session_adjustment()

    ma7 = closes.rolling(7, min_periods=3).mean()
    ma20 = closes.rolling(20, min_periods=5).mean()
    std20 = closes.rolling(20, min_periods=5).std().fillna(0.0)
    bb_width_pct = ((std20 * 4.0) / ma20.replace(0, np.nan) * 100.0).fillna(5.0)

    ma20_15m_val = to_float(ma20.iloc[-1], price)
    ma20_disp_pct = ((price - ma20_15m_val) / max(ma20_15m_val, 1e-8)) * 100.0

    rsi6_s = _calc_rsi(closes, 6)
    rsi6_15m = round(to_float(rsi6_s.iloc[-1], 50.0), 1)
    rsi6_prev = to_float(rsi6_s.iloc[-2], 50.0)
    min_rsi6_6 = round(to_float(rsi6_s.iloc[-6:].min(), 50.0), 1)

    kdj_j_s = _calc_kdj_j(df_15m, 9)
    kdj_j_15m = round(to_float(kdj_j_s.iloc[-1], 50.0), 1)
    kdj_j_prev = to_float(kdj_j_s.iloc[-2], 50.0)
    min_kdj_j_6 = round(to_float(kdj_j_s.iloc[-6:].min(), 50.0), 1)

    vol_ma20 = vols.rolling(20, min_periods=5).mean().replace(0, np.nan)
    vol_ratios = (vols / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recent_vr = vol_ratios.iloc[-6:]
    vol_mult = round(float(recent_vr.max()), 2) if (not recent_vr.empty and float(recent_vr.max()) > 0.0) else 1.0
    curr_vol_ratio = round(to_float(vol_ratios.iloc[-1], 1.0), 2)

    c_1h = df_1h["close"].astype(float)
    ma20_1h = to_float(c_1h.rolling(20, min_periods=5).mean().iloc[-1], price)
    ma50_1h = to_float(c_1h.rolling(50, min_periods=10).mean().iloc[-1], price)
    htf_bull = price >= ma20_1h * 0.995
    htf_status = "1H 정배열지지" if (price >= ma20_1h >= ma50_1h * 0.995) else ("1H 중심선지지" if htf_bull else "1H 눌림목지지")
    oi_ratio = round((oi_m / max(turnover_m, 0.1)) * 100.0, 1)

    # [1순위] 신규 장기 포지션(Track D: D1·D2·D3) — 장기 고정손절 -20% 유지
    if oi_m >= 0.4 and turnover_m >= 2.5 and rsi6_15m < 80.0 and ma20_disp_pct < 8.0:
        df_4h = fetch_klines(symbol, interval="240", limit=180)
        macro_d = _evaluate_macro_track_d(price, change_24h, oi_ratio, funding_pct, df_4h)
        if macro_d is not None:
            d_score = int(np.clip(macro_d["score"] + (5 if oi_q["quadrant"] == "Q1" else 0) + max(0, sess_adj // 2), 0, 99))
            if postmortem is not None and hasattr(postmortem, "get_setup_penalty"):
                try:
                    d_score = max(0, d_score - int(postmortem.get_setup_penalty(symbol, "TRACK_D", rsi6_15m, vol_mult)))
                except Exception:
                    pass
            if d_score >= ELITE_SCORE_THRESHOLD:
                df_1d = fetch_klines(symbol, interval="D", limit=120)
                tps, tp_pcts, tp_labels = _calculate_macro_longterm_tps(price, turnover_m, df_4h, df_1d)
                stop_price = round(price * (1.0 - LONGTERM_SL_PCT / 100.0), 6)
                reasons = [f"장기 전략: {macro_d['sub_strategy']}"] + macro_d["reasons"]
                if oi_q["note"]:
                    reasons.append(oi_q["note"])
                return {
                    "symbol": symbol,
                    "base_coin": base_coin,
                    "track_type": "TRACK_D",
                    "is_long_term": True,
                    "position_horizon": "장기 포지션 (1개월 이상 홀딩 예상)",
                    "sub_strategy": macro_d["sub_strategy"],
                    "track_title": "🐋 [장기 포지션 · 세력 대시세 매집 초입]",
                    "score": d_score,
                    "price": price,
                    "change_24h": round(change_24h, 2),
                    "turnover_m": round(turnover_m, 1),
                    "oi_m": round(oi_m, 1),
                    "oi_ratio": oi_ratio,
                    "funding_pct": funding_pct,
                    "htf_status": f"4H·1D 매집구조 ({htf_status})",
                    "rsi6_15m": rsi6_15m,
                    "kdj_j_15m": kdj_j_15m,
                    "vol_mult": vol_mult,
                    "swing_high_line": to_float(df_4h["high"].max(), price * 1.1),
                    "entry_low": round(price * 0.9920, 6),
                    "entry_high": round(price * 1.0050, 6),
                    "stop_price": stop_price,
                    "sl_pct": LONGTERM_SL_PCT,
                    "sl_note": f"장기 고정손절(-{LONGTERM_SL_PCT}%)",
                    "tps": tps,
                    "tp_pcts": tp_pcts,
                    "tp_labels": tp_labels,
                    "reasons": reasons[:4],
                    "reason_str": " / ".join(reasons[:4]),
                }

    # 단기·스윙(Track A·B·C) 과열 추격 차단
    if rsi6_15m >= 78.0 or ma20_disp_pct >= 6.0:
        return None

    obv = (np.sign(closes.diff().fillna(0.0)) * vols).cumsum()
    obv_strong = bool(to_float(obv.iloc[-1]) >= to_float(obv.rolling(10, min_periods=3).mean().iloc[-1]))

    # 24시간(96개봉) 주요 스윙 고점 구조 (Track B 확장)
    span_24h = min(len(highs) - 2, 96)
    high_24h = to_float(highs.iloc[-span_24h - 2 : -2].max(), price * 1.05)

    # Track B 판별: 24시간 구조 돌파 후 종가 안착(Confirmation) 또는 S-R Flip 리테스트 확인
    prev_c, prev_l = to_float(closes.iloc[-2], price), to_float(lows.iloc[-2], price)
    curr_o, curr_c, curr_l = to_float(opens.iloc[-1], price), to_float(closes.iloc[-1], price), to_float(lows.iloc[-1], price)
    close_confirmed_b = (prev_c >= high_24h * 0.998) and (curr_c >= high_24h * 0.996) and (curr_c >= curr_o)
    had_break_recent = to_float(highs.iloc[-5:-1].max(), 0.0) >= high_24h * 1.001
    sr_flip_retest_b = (
        had_break_recent
        and (min(curr_l, prev_l) <= high_24h * 1.008)
        and (min(curr_l, prev_l) >= high_24h * 0.985)
        and (curr_c >= high_24h * 0.998)
        and (curr_c >= curr_o)
    )
    is_breakout_b = (close_confirmed_b or sr_flip_retest_b) and vol_mult >= 2.0 and change_24h >= 3.0

    swing_high = to_float(highs.iloc[-48:-1].max(), price * 1.03)
    swing_low = to_float(lows.iloc[-48:-1].min(), price * 0.95)
    fib_retracement = (swing_high - price) / max(swing_high - swing_low, price * 0.01)
    pullback_from_high_pct = ((swing_high - price) / max(swing_high, 1e-8)) * 100.0

    reasons = []
    supply_score = (10 if oi_ratio >= 18.0 else (7 if oi_ratio >= 8.0 else 4)) + (10 if funding_pct < -0.005 else (7 if funding_pct <= 0.012 else 3))
    if oi_ratio >= 18.0:
        reasons.append(f"OI 집중도 최상({oi_ratio:.0f}%)")
    if funding_pct < -0.005:
        reasons.append(f"음수 펀딩 숏스퀴즈({funding_pct:+.4f}%)")

    # [Track C] 급등 대장주 심층 눌림목 + 거래량 건조(<= 35%) + 5분봉 MSS/ChOCH 필수화
    if change_24h >= 12.0 and not is_breakout_b:
        if not (0.48 <= fib_retracement <= 0.78 and pullback_from_high_pct >= 6.5 and price >= swing_low * 1.015):
            return None

        touched_ma20_flush = to_float(lows.iloc[-6:].min(), price) <= ma20_15m_val * 0.995
        had_flush = (min_rsi6_6 <= 38.0) or (min_kdj_j_6 <= 22.0) or touched_ma20_flush
        if not had_flush:
            return None

        # 정량 거래량 건조(Volume Dry-up <= 35%): 1파 상승 핵심 거래량 대비 2파 조정 거래량 35% 이하 필수
        win_48_h = highs.iloc[-48:-2]
        peak_rel_idx = int(np.argmax(win_48_h.values))
        peak_abs_idx = len(df_15m) - 48 + peak_rel_idx
        wave1_vols = vols.iloc[max(0, peak_abs_idx - 8) : peak_abs_idx + 1]
        avg_v_wave1 = max(to_float(wave1_vols.mean(), 1.0), to_float(vols.iloc[-48:-3].nlargest(5).mean(), 1.0) * 0.75)
        wave2_vols = vols.iloc[peak_abs_idx + 1 : -1] if (peak_abs_idx + 3) < len(vols) else vols.iloc[-4:-1]
        avg_v_wave2 = to_float(wave2_vols.mean(), avg_v_wave1)
        vol_dry_ratio = avg_v_wave2 / max(avg_v_wave1, 1e-8)
        if vol_dry_ratio > 0.35:
            return None

        # 15m 반등 턴 확인
        c_high, p_open, p_close, p_high = to_float(highs.iloc[-1], price), to_float(opens.iloc[-2], price), prev_c, to_float(highs.iloc[-2], price)
        curr_lwick = (min(curr_o, curr_c) - curr_l) / max(c_high - curr_l, 1e-10)
        prev_lwick = (min(p_open, p_close) - prev_l) / max(p_high - prev_l, 1e-10)
        max_lwick = max(curr_lwick, prev_lwick)

        is_bullish_turn = (curr_c > curr_o) or (curr_lwick >= 0.45 and curr_c >= prev_l)
        is_momentum_turn = (kdj_j_15m > kdj_j_prev) and (rsi6_15m > rsi6_prev) and (rsi6_15m <= 60.0)
        is_holding_low = curr_c >= to_float(lows.iloc[-4:].min(), curr_l) * 1.003
        if not (is_bullish_turn and is_momentum_turn and is_holding_low):
            return None

        # 5분봉 하위 프레임 시장 구조 전환(MSS / ChOCH) 또는 상승 장악형 필수 확인
        mss_ok = _check_5m_mss_trigger(symbol)
        if mss_ok is False:
            return None
        if mss_ok is None and not (curr_c > curr_o and curr_c > p_open):
            return None

        track_type = "TRACK_C"
        track_title = "🚀 [Track C · 대장주 심층 눌림목(5m MSS 전환) 반등 타점]"
        tech_score = (
            (34 if 0.52 <= fib_retracement <= 0.72 else 29)
            + (18 if price >= ma20_1h * 0.985 else (14 if price >= ma50_1h * 0.985 else 8))
            + (14 if obv_strong else 11)
            + (14 if (min_rsi6_6 <= 30.0 or max_lwick >= 0.35) else 10)
            + (10 if mss_ok else 6)
            + (3 if oi_q["quadrant"] == "Q4" else 0)
        )
        reasons.insert(0, f"5m MSS/장악형 반등 확인 + 거래량 건조({vol_dry_ratio * 100:.0f}%)")
        reasons.insert(0, f"대장주(+{change_24h:.1f}%) 고점대비 -{pullback_from_high_pct:.1f}% · Fib {fib_retracement:.2f} 심층 눌림")

    # [Track B] 24시간 스윙 구조 돌파 + 종가 안착 / S-R Flip 리테스트
    elif is_breakout_b:
        track_type = "TRACK_B"
        flip_lbl = "S-R Flip 리테스트 지지" if sr_flip_retest_b else "24H 전고점 종가 안착 돌파"
        track_title = f"🔥 [Track B · 24H 구조 돌파 ({'S-R Flip' if sr_flip_retest_b else '종가 안착'})]"
        tech_score = (
            (35 if vol_mult >= 2.8 else 30)
            + (18 if htf_bull else 0)
            + (14 if obv_strong else 0)
            + (13 if (curr_c > ma20_15m_val and to_float(ma7.iloc[-1]) >= ma20_15m_val) else 0)
            + (8 if 48.0 <= rsi6_15m <= 74.0 else 0)
            + (4 if sr_flip_retest_b else 2)
        )
        reasons.insert(0, f"{flip_lbl} + 15m 거래량 {vol_mult:.1f}배 폭발")

    # [Track A] TTM Squeeze Fired + 델타(CVD) 흡수 검증
    else:
        track_type = "TRACK_A"
        ttm_fired, was_squeezed, cvd_absorption = _calc_ttm_squeeze_and_cvd(df_15m)
        min_bw = to_float(bb_width_pct.iloc[-10:-1].min(), 5.0)
        # 페이크 돌파 원천 차단: TTM 스퀴즈 발사 또는 (초응축 + 델타 매집 흡수) 미충족 시 탈락
        if not (ttm_fired or (was_squeezed and cvd_absorption) or (min_bw <= 4.5 and cvd_absorption)):
            return None

        track_title = "💎 [Track A · TTM 스퀴즈 발사 & 델타 흡수 초입]" if ttm_fired else "💎 [Track A · 박스권 초응축 & 델타 매집 초입]"
        tech_score = (
            (35 if ttm_fired else (28 if was_squeezed else 22))
            + (18 if (vol_mult >= 1.9 and curr_c > curr_o) else (12 if curr_vol_ratio >= 1.3 else 0))
            + (15 if htf_bull else 0)
            + (15 if (cvd_absorption and obv_strong) else (10 if obv_strong else 0))
            + (10 if (kdj_j_15m > kdj_j_prev and 35.0 <= rsi6_15m <= 70.0) else 0)
        )
        if ttm_fired:
            reasons.insert(0, f"TTM Squeeze(BB<KC) 상방 발사(응축폭 {min_bw:.1f}%)")
        else:
            reasons.insert(0, f"볼린저/켈트너 초응축({min_bw:.1f}%) 후 전환")
        if cvd_absorption:
            reasons.append("15m 매수 델타(CVD)·OBV 바닥 흡수 확인")

    # OI 1사분면 가산점 & 글로벌 세션 가중치 합산
    total_score = int(np.clip(tech_score + supply_score + oi_q["bonus"] + sess_adj, 0, 99))
    if oi_q["note"]:
        reasons.append(oi_q["note"])
    if sess_note:
        reasons.append(sess_note)

    if wave_engine is not None and hasattr(wave_engine, "evaluate_wave_bonus"):
        try:
            wb, wn = wave_engine.evaluate_wave_bonus(df_15m, df_1h)
            total_score = int(np.clip(total_score + int(wb), 0, 99))
            if wn:
                reasons.append(str(wn))
        except Exception:
            pass

    if postmortem is not None and hasattr(postmortem, "get_setup_penalty"):
        try:
            pen = int(postmortem.get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult))
            total_score = max(0, total_score - pen)
        except Exception:
            pass

    if total_score < ELITE_SCORE_THRESHOLD:
        return None

    # OI 2사분면(숏스퀴즈 주의)이면 TP1·TP2 보수적 하향(×0.88) 적용
    tp_scale = 0.88 if oi_q["squeeze_warn"] else 1.0
    tps, tp_pcts, tp_labels = _calculate_resistance_based_tps(price, df_15m, df_1h, conservative_scale=tp_scale)
    stop_price = round(price * (1.0 - FIXED_SL_PCT / 100.0), 6)

    return {
        "symbol": symbol,
        "base_coin": base_coin,
        "track_type": track_type,
        "is_long_term": False,
        "position_horizon": "단기·스윙 포지션",
        "track_title": track_title,
        "score": total_score,
        "price": price,
        "change_24h": round(change_24h, 2),
        "turnover_m": round(turnover_m, 1),
        "oi_m": round(oi_m, 1),
        "oi_ratio": oi_ratio,
        "funding_pct": funding_pct,
        "htf_status": htf_status,
        "rsi6_15m": rsi6_15m,
        "kdj_j_15m": kdj_j_15m,
        "vol_mult": vol_mult,
        "swing_high_line": swing_high,
        "entry_low": round(price * 0.9970, 6),
        "entry_high": round(price * 1.0015, 6),
        "stop_price": stop_price,
        "sl_pct": FIXED_SL_PCT,
        "sl_note": f"고정손절(-{FIXED_SL_PCT}%)",
        "tps": tps,
        "tp_pcts": tp_pcts,
        "tp_labels": tp_labels,
        "reasons": reasons[:4],
        "reason_str": " / ".join(reasons[:4]),
    }
# === END OF FILE ===
