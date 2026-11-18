import os, json
import numpy as np
import pandas as pd

import config as cfg
from config import to_float, normalize_funding_pct
from api import fetch_klines, is_stock_or_tradfi_symbol

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
    """
    [코인의 실제 차트 저항대 기반 개별 TP1·TP2·TP3 산출]
    - 15m/1H 스윙 고점(저항선), 볼린저 상단, 피보나치 확장(1.272 / 1.618 / 2.272) 및 ATR 변동성을 결합하여
      종목마다 고유한 실제 저항 목표가와 수익률(%)을 산출합니다.
    """
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

    # 차트 상단 실제 저항 후보군 수집 (현재가 대비 상승률 %)
    res_candidates = []
    for i in range(3, len(highs_15) - 2):
        h_v = float(highs_15.iloc[i])
        if h_v > price * 1.055 and h_v >= float(highs_15.iloc[i - 2 : i + 3].max()):
            res_candidates.append(((h_v - price) / price * 100.0, "15m 전고점 매물대"))

    for i in range(2, len(highs_1h) - 1):
        h_v = float(highs_1h.iloc[i])
        if h_v > price * 1.06 and h_v >= float(highs_1h.iloc[i - 2 : i + 2].max()):
            res_candidates.append(((h_v - price) / price * 100.0, "1H 스윙 저항선"))

    # 피보나치 확장 & 변동성 저항 후보 추가
    fib_1272_p = max(swing_h_24, price) + wave_span * 0.272
    fib_1618_p = max(swing_h_24, price) + wave_span * 0.618
    fib_2272_p = max(swing_h_1h, price) + wave_span * 1.272

    res_candidates.extend([
        (((swing_h_24 - price) / price * 100.0), "24H 전고점 저항"),
        (((fib_1272_p - price) / price * 100.0), "Fib 1.272 저항대"),
        (((fib_1618_p - price) / price * 100.0), "Fib 1.618 확장저항"),
        (((fib_2272_p - price) / price * 100.0), "Fib 2.272 상단채널"),
    ])

    # 1차 목표(TP1): +6.5% ~ +11.8% 구간 내 실제 저항선 선택 (없으면 ATR 비례 산출)
    tp1_pool = [(pct, lbl) for pct, lbl in res_candidates if 6.5 <= pct <= 11.8]
    if tp1_pool:
        tp1_pool.sort(key=lambda x: x[0])
        p1_pct, lbl1 = tp1_pool[len(tp1_pool) // 2]
    else:
        p1_pct = float(np.clip(7.2 * vol_scale + (wave_span / price * 18.0), 6.8, 11.4))
        lbl1 = "단기 매물대 저항"

    # 2차 목표(TP2): max(p1_pct + 5.2%, 13.0%) ~ +21.5% 구간 내 실제 저항선 선택
    tp2_min = max(p1_pct + 5.2, 13.0)
    tp2_pool = [(pct, lbl) for pct, lbl in res_candidates if tp2_min <= pct <= 21.5]
    if tp2_pool:
        tp2_pool.sort(key=lambda x: x[0])
        p2_pct, lbl2 = tp2_pool[len(tp2_pool) // 2]
    else:
        p2_pct = float(np.clip(p1_pct * 1.85 + 1.2, tp2_min, 20.8))
        lbl2 = "1H 주요 저항구간"

    # 3차 목표(TP3): max(p2_pct + 6.5%, 20.5%) ~ +35.0% 구간 내 실제 저항선 선택
    tp3_min = max(p2_pct + 6.5, 20.5)
    tp3_pool = [(pct, lbl) for pct, lbl in res_candidates if tp3_min <= pct <= 35.0]
    if tp3_pool:
        tp3_pool.sort(key=lambda x: x[0])
        p3_pct, lbl3 = tp3_pool[len(tp3_pool) // 2]
    else:
        p3_pct = float(np.clip(p2_pct * 1.48 + 1.5, tp3_min, 33.5))
        lbl3 = "Fib 1.618 클라이맥스"

    p1_pct = round(float(p1_pct), 1)
    p2_pct = round(float(p2_pct), 1)
    p3_pct = round(float(p3_pct), 1)

    tps = [
        round(price * (1.0 + p1_pct / 100.0), 6),
        round(price * (1.0 + p2_pct / 100.0), 6),
        round(price * (1.0 + p3_pct / 100.0), 6),
    ]
    return tps, [p1_pct, p2_pct, p3_pct], [lbl1, lbl2, lbl3]


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

    ma7 = closes.rolling(7, min_periods=3).mean()
    ma20 = closes.rolling(20, min_periods=5).mean()
    std20 = closes.rolling(20, min_periods=5).std().fillna(0.0)
    bb_width_pct = ((std20 * 4.0) / ma20.replace(0, np.nan) * 100.0).fillna(5.0)

    rsi6_s = _calc_rsi(closes, 6)
    kdj_j_s = _calc_kdj_j(df_15m, 9)
    rsi6_15m = round(to_float(rsi6_s.iloc[-1], 50.0), 1)
    kdj_j_15m = round(to_float(kdj_j_s.iloc[-1], 50.0), 1)
    kdj_j_prev = to_float(kdj_j_s.iloc[-2], 50.0)

    vol_ma20 = vols.rolling(20, min_periods=5).mean().replace(0, np.nan)
    vol_ratios = (vols / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recent_vr = vol_ratios.iloc[-6:]
    vol_mult = round(float(recent_vr.max()), 2) if (not recent_vr.empty and float(recent_vr.max()) > 0.0) else 1.0
    curr_vol_ratio = round(to_float(vol_ratios.iloc[-1], 1.0), 2)

    direction = np.sign(closes.diff().fillna(0.0))
    obv = (direction * vols).cumsum()
    obv_ma10 = obv.rolling(10, min_periods=3).mean()
    obv_strong = bool(to_float(obv.iloc[-1]) >= to_float(obv_ma10.iloc[-1]))

    c_1h = df_1h["close"].astype(float)
    ma20_1h = to_float(c_1h.rolling(20, min_periods=5).mean().iloc[-1], price)
    ma50_1h = to_float(c_1h.rolling(50, min_periods=10).mean().iloc[-1], price)
    htf_bull = price >= ma20_1h * 0.995
    htf_status = "1H 정배열지지" if (price >= ma20_1h >= ma50_1h * 0.995) else ("1H 중심선지지" if htf_bull else "1H 눌림목지지")

    swing_high = to_float(highs.iloc[-48:-1].max(), price * 1.03)
    swing_low = to_float(lows.iloc[-48:-1].min(), price * 0.95)
    swing_span = max(swing_high - swing_low, price * 0.01)
    fib_retracement = (swing_high - price) / swing_span

    oi_ratio = round((oi_m / max(turnover_m, 0.1)) * 100.0, 1)
    reasons = []

    supply_score = 0
    if oi_ratio >= 18.0:
        supply_score += 10
        reasons.append(f"OI 집중도 최상({oi_ratio:.0f}%)")
    elif oi_ratio >= 8.0:
        supply_score += 7
    else:
        supply_score += 4

    if funding_pct < -0.005:
        supply_score += 10
        reasons.append(f"음수 펀딩 숏스퀴즈({funding_pct:+.4f}%)")
    elif funding_pct <= 0.012:
        supply_score += 7
    else:
        supply_score += 3

    track_type = "TRACK_A"
    track_title = "💎 [Track A · 박스권 스퀴즈 초입 타점]"
    tech_score = 0

    if change_24h >= 12.0 and 0.28 <= fib_retracement <= 0.68 and price >= swing_low * 1.02:
        track_type = "TRACK_C"
        track_title = "🚀 [Track C · 대장주 핵심 눌림목 2파동 타점]"
        if 0.36 <= fib_retracement <= 0.62:
            tech_score += 36
            reasons.append(f"대장주(+{change_24h:.1f}%) Fib {fib_retracement:.2f} 골든존 지지")
        else:
            tech_score += 31
            reasons.append(f"대장주(+{change_24h:.1f}%) 눌림목 지지")

        if price >= ma20_1h:
            tech_score += 18
            reasons.append("1H 20MA 추세선 방어")
        elif price >= ma50_1h * 0.99:
            tech_score += 14

        if obv_strong:
            tech_score += 14
            reasons.append("눌림 구간 OBV 매집세 유지")
        else:
            tech_score += 9

        if closes.iloc[-1] >= opens.iloc[-1] and kdj_j_15m >= kdj_j_prev:
            tech_score += 14
            reasons.append("거래량 건조 후 15m J선 반등 턴")
        else:
            tech_score += 9

        if 35.0 <= rsi6_15m <= 68.0:
            tech_score += 8

    elif price >= to_float(highs.iloc[-24:-2].max(), price) * 0.992 and vol_mult >= 1.8 and change_24h >= 3.0:
        track_type = "TRACK_B"
        track_title = "🔥 [Track B · 수급 폭발 전고점 돌파 타점]"
        tech_score += 34 if vol_mult >= 2.5 else 29
        reasons.append(f"15분봉 거래량 {vol_mult:.1f}배 포켓피벗 돌파")
        if htf_bull:
            tech_score += 18
            reasons.append("1H 상단 추세 동조")
        if obv_strong:
            tech_score += 14
            reasons.append("OBV 전고점 선행 돌파")
        if closes.iloc[-1] > to_float(ma20.iloc[-1]) and to_float(ma7.iloc[-1]) >= to_float(ma20.iloc[-1]):
            tech_score += 12
        if 48.0 <= rsi6_15m <= 78.0:
            tech_score += 8

    else:
        track_type = "TRACK_A"
        track_title = "💎 [Track A · 바닥권 에너지 응축 초입 타점]"
        min_bw = to_float(bb_width_pct.iloc[-10:-1].min(), 5.0)
        if min_bw <= 4.2:
            tech_score += 34
            reasons.append(f"볼린저/켈트너 초응축({min_bw:.1f}%) 후 확산")
        elif min_bw <= 6.5:
            tech_score += 28
            reasons.append(f"박스권 에너지 응축({min_bw:.1f}%)")
        else:
            tech_score += 20

        if vol_mult >= 1.7 and closes.iloc[-1] > opens.iloc[-1]:
            tech_score += 18
            reasons.append(f"응축 돌파 거래량 {vol_mult:.1f}배 유입")
        elif curr_vol_ratio >= 1.2:
            tech_score += 12

        if htf_bull:
            tech_score += 15
        if obv_strong:
            tech_score += 12
            reasons.append("OBV 바닥 다이버전스 매집")
        if kdj_j_15m > kdj_j_prev and 35.0 <= rsi6_15m <= 72.0:
            tech_score += 10

    total_score = int(np.clip(tech_score + supply_score, 0, 99))

    if wave_engine is not None and hasattr(wave_engine, "evaluate_wave_bonus"):
        try:
            w_bonus, w_note = wave_engine.evaluate_wave_bonus(df_15m, df_1h)
            total_score = int(np.clip(total_score + int(w_bonus), 0, 99))
            if w_note:
                reasons.append(str(w_note))
        except Exception:
            pass

    if postmortem is not None and hasattr(postmortem, "get_setup_penalty"):
        try:
            pen = int(postmortem.get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult))
            total_score = max(0, total_score - pen)
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
