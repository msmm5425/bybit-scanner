import os, re, json
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
        "KOSPI", "TSLA", "NVDA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "NFLX",
        "MSTR", "COIN", "SPY", "QQQ", "XAU", "XAG", "WTI", "BRENT", "SPX", "NDX",
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

HISTORY_FILE = getattr(cfg, "ANALYZER_HISTORY_FILE", "analyzer_history.json")
SCORE_THRESHOLD = int(getattr(cfg, "SCORE_THRESHOLD", 85))
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


def _calculate_resistance_based_tps(price, df_15m, df_1h):
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
    """신규 장기 포지션 Track D 3대 전략 판별 (D1·D2·D3)"""
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

    # D1: 스텔스 OI · OBV 선행 매집
    hi_10d = to_float(h4.iloc[-60:-1].max(), price * 1.05)
    obv_hi_10d = to_float(obv4.iloc[-60:-1].max(), 0.0)
    if -3.0 <= change_24h <= 11.0 and oi_ratio >= 26.0 and funding_pct <= 0.008:
        if price < hi_10d * 0.995 and to_float(obv4.iloc[-1]) > obv_hi_10d and price >= ma20_4h * 0.985:
            score_d1 = int(np.clip(86 + min(8, int((oi_ratio - 26) / 4)) + (4 if funding_pct < 0 else 2), 86, 98))
            return {
                "sub_strategy": "D1 · 스텔스 OI·OBV 선행 매집",
                "score": score_d1,
                "reasons": [
                    "가격 횡보 중 4H OBV 10일 최고치 선행 돌파",
                    f"스텔스 미결제약정(OI) 집중도 {oi_ratio:.0f}% 축적",
                    f"펀딩비 {funding_pct:+.4f}% 저과열 바닥 매집 구조",
                ],
            }

    # D2: 매집봉(Test Pump) 출현 후 거래량 건조 눌림목 반등
    recent_30 = df_4h.iloc[-30:-2]
    vr_30 = v4_ratio.iloc[-30:-2]
    if not vr_30.empty and float(vr_30.max()) >= 3.5:
        pump_pos = int(np.argmax(vr_30.values))
        pump_row = recent_30.iloc[pump_pos]
        pump_vol = to_float(pump_row["volume"], 1.0)
        pump_open = to_float(pump_row["open"], price)
        pump_high = to_float(pump_row["high"], price)
        pump_mid = (pump_open + pump_high) * 0.5
        curr_vol_4h = to_float(v4.iloc[-2], pump_vol)
        is_dry_up = pump_vol > 0 and (curr_vol_4h / pump_vol) <= 0.32
        is_turn_up = c4.iloc[-1] >= o4.iloc[-1] and price >= max(pump_open * 0.99, ma20_4h * 0.985) and price <= pump_high * 1.02
        if is_dry_up and is_turn_up:
            score_d2 = int(np.clip(87 + (4 if price >= pump_mid else 2) + (4 if oi_ratio >= 15.0 else 1), 87, 98))
            return {
                "sub_strategy": "D2 · 매집봉 출현 후 거래량 건조 눌림",
                "score": score_d2,
                "reasons": [
                    f"최근 5일 내 4H 매집봉(거래량 {float(vr_30.max()):.1f}배) 출현",
                    "조정 구간 거래량 70% 이상 건조(세력 이탈 없음)",
                    "매집봉 지지선 & 4H 20MA 위 양봉 반등 턴",
                ],
            }

    # D3: 4H · 1D 장기 박스권 초응축 30일 신고가 첫 돌파
    if len(df_4h) >= 120:
        hi_30d = to_float(h4.iloc[:-1].max(), price)
        lo_30d = to_float(l4.iloc[:-1].min(), price * 0.8)
        range_30d_pct = (hi_30d - lo_30d) / max(lo_30d, 1e-8) * 100.0
        curr_vr_4h = to_float(v4_ratio.iloc[-1], 1.0)
        if hi_30d * 0.996 <= price <= hi_30d * 1.045 and curr_vr_4h >= 2.2 and range_30d_pct <= 55.0:
            score_d3 = int(np.clip(88 + min(6, int(curr_vr_4h * 1.5)) + (3 if price >= ma50_4h else 0), 88, 99))
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


def analyze_symbol(coin):
    symbol = str(coin.get("symbol", "")).upper().strip()
    base_coin = coin.get("base_coin") or symbol.replace("USDT", "")
    if is_stock_or_tradfi_symbol(symbol):
        return None

    price = to_float(coin.get("price"))
    change_24h = to_float(coin.get("change_24h"))
    turnover_m = to_float(coin.get("turnover_m"))
    oi_m = to_float(coin.get("oi_m"))
    funding_pct = normalize_funding_pct(to_float(coin.get("funding_pct")))

    if price <= 0 or turnover_m < 2.0:
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

    rsi6_15m = round(to_float(_calc_rsi(closes, 6).iloc[-1], 50.0), 1)
    kdj_j_s = _calc_kdj_j(df_15m, 9)
    kdj_j_15m = round(to_float(kdj_j_s.iloc[-1], 50.0), 1)
    kdj_j_prev = to_float(kdj_j_s.iloc[-2], 50.0)

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

    # [1순위] 신규 장기 포지션(Track D: D1·D2·D3) — 장기 고정손절 -20%
    if oi_m >= 0.4 and turnover_m >= 2.5:
        df_4h = fetch_klines(symbol, interval="240", limit=180)
        macro_d = _evaluate_macro_track_d(price, change_24h, oi_ratio, funding_pct, df_4h)
        if macro_d is not None and macro_d["score"] >= SCORE_THRESHOLD:
            df_1d = fetch_klines(symbol, interval="D", limit=120)
            tps, tp_pcts, tp_labels = _calculate_macro_longterm_tps(price, turnover_m, df_4h, df_1d)
            stop_price = round(price * (1.0 - LONGTERM_SL_PCT / 100.0), 6)
            reasons = [f"장기 전략: {macro_d['sub_strategy']}"] + macro_d["reasons"]
            return {
                "symbol": symbol,
                "base_coin": base_coin,
                "track_type": "TRACK_D",
                "is_long_term": True,
                "position_horizon": "장기 포지션 (1개월 이상 홀딩 예상)",
                "sub_strategy": macro_d["sub_strategy"],
                "track_title": "🐋 [장기 포지션 · 세력 대시세 매집 초입]",
                "score": macro_d["score"],
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

    # [2순위] 기존 단기·스윙 전략 (Track A · B · C) — 단기 고정손절 -6%
    ma7 = closes.rolling(7, min_periods=3).mean()
    ma20 = closes.rolling(20, min_periods=5).mean()
    std20 = closes.rolling(20, min_periods=5).std().fillna(0.0)
    bb_width_pct = ((std20 * 4.0) / ma20.replace(0, np.nan) * 100.0).fillna(5.0)

    obv = (np.sign(closes.diff().fillna(0.0)) * vols).cumsum()
    obv_strong = bool(to_float(obv.iloc[-1]) >= to_float(obv.rolling(10, min_periods=3).mean().iloc[-1]))

    swing_high = to_float(highs.iloc[-48:-1].max(), price * 1.03)
    swing_low = to_float(lows.iloc[-48:-1].min(), price * 0.95)
    fib_retracement = (swing_high - price) / max(swing_high - swing_low, price * 0.01)

    reasons = []
    supply_score = (10 if oi_ratio >= 18.0 else (7 if oi_ratio >= 8.0 else 4)) + (10 if funding_pct < -0.005 else (7 if funding_pct <= 0.012 else 3))
    if oi_ratio >= 18.0:
        reasons.append(f"OI 집중도 최상({oi_ratio:.0f}%)")
    if funding_pct < -0.005:
        reasons.append(f"음수 펀딩 숏스퀴즈({funding_pct:+.4f}%)")

    if change_24h >= 12.0 and 0.28 <= fib_retracement <= 0.68 and price >= swing_low * 1.02:
        track_type = "TRACK_C"
        track_title = "🚀 [Track C · 대장주 핵심 눌림목 2파동 타점]"
        tech_score = (36 if 0.36 <= fib_retracement <= 0.62 else 31) + (18 if price >= ma20_1h else (14 if price >= ma50_1h * 0.99 else 0)) + (14 if obv_strong else 9) + (14 if (closes.iloc[-1] >= opens.iloc[-1] and kdj_j_15m >= kdj_j_prev) else 9) + (8 if 35.0 <= rsi6_15m <= 68.0 else 0)
        reasons.insert(0, f"대장주(+{change_24h:.1f}%) Fib {fib_retracement:.2f} 눌림 지지")
    elif price >= to_float(highs.iloc[-24:-2].max(), price) * 0.992 and vol_mult >= 1.8 and change_24h >= 3.0:
        track_type = "TRACK_B"
        track_title = "🔥 [Track B · 수급 폭발 전고점 돌파 타점]"
        tech_score = (34 if vol_mult >= 2.5 else 29) + (18 if htf_bull else 0) + (14 if obv_strong else 0) + (12 if closes.iloc[-1] > to_float(ma20.iloc[-1]) else 0) + (8 if 48.0 <= rsi6_15m <= 78.0 else 0)
        reasons.insert(0, f"15분봉 거래량 {vol_mult:.1f}배 포켓피벗 돌파")
    else:
        track_type = "TRACK_A"
        track_title = "💎 [Track A · 바닥권 에너지 응축 초입 타점]"
        min_bw = to_float(bb_width_pct.iloc[-10:-1].min(), 5.0)
        tech_score = (34 if min_bw <= 4.2 else (28 if min_bw <= 6.5 else 20)) + (18 if (vol_mult >= 1.7 and closes.iloc[-1] > opens.iloc[-1]) else (12 if curr_vol_ratio >= 1.2 else 0)) + (15 if htf_bull else 0) + (12 if obv_strong else 0) + (10 if (kdj_j_15m > kdj_j_prev and 35.0 <= rsi6_15m <= 72.0) else 0)
        reasons.insert(0, f"볼린저/켈트너 응축({min_bw:.1f}%) 후 전환")

    total_score = int(np.clip(tech_score + supply_score, 0, 99))
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
            total_score = max(0, total_score - int(postmortem.get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult)))
        except Exception:
            pass

    if total_score < SCORE_THRESHOLD:
        return None

    tps, tp_pcts, tp_labels = _calculate_resistance_based_tps(price, df_15m, df_1h)
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
