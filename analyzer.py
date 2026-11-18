import os, json
import numpy as np
import pandas as pd

import config as cfg
from config import to_float, normalize_funding_pct
from api import fetch_klines

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
    """learned_tp_params 와 dynamic_tp_tuning 키를 모두 호환하여 자율학습 익절폭 로드"""
    default_params = {"tp1_min_pct": 8.0, "tp2_min_pct": 15.0, "tp3_min_pct": 22.0}
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                learned = raw.get("dynamic_tp_tuning") or raw.get("learned_tp_params") or {}
                if isinstance(learned, dict):
                    p1 = float(np.clip(to_float(learned.get("tp1_min_pct"), 8.0), 7.0, 12.0))
                    p2 = float(np.clip(to_float(learned.get("tp2_min_pct"), 15.0), 13.0, 22.0))
                    p3 = float(np.clip(to_float(learned.get("tp3_min_pct"), 22.0), 20.0, 35.0))
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
    j = 3.0 * k - 2.0 * d
    return j.fillna(50.0)


def analyze_symbol(coin):
    symbol = str(coin.get("symbol", "")).upper().strip()
    base_coin = coin.get("base_coin") or symbol.replace("USDT", "")
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
    ma50 = closes.rolling(50, min_periods=10).mean()
    std20 = closes.rolling(20, min_periods=5).std().fillna(0.0)
    bb_width_pct = ((std20 * 4.0) / ma20.replace(0, np.nan) * 100.0).fillna(5.0)

    rsi6_s = _calc_rsi(closes, 6)
    kdj_j_s = _calc_kdj_j(df_15m, 9)
    rsi6_15m = round(to_float(rsi6_s.iloc[-1], 50.0), 1)
    kdj_j_15m = round(to_float(kdj_j_s.iloc[-1], 50.0), 1)
    kdj_j_prev = to_float(kdj_j_s.iloc[-2], 50.0)

    # [Pandas idxmax 크래시 완벽 방어]
    vol_ma20 = vols.rolling(20, min_periods=5).mean().replace(0, np.nan)
    vol_ratios = (vols / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recent_vr = vol_ratios.iloc[-6:]
    if not recent_vr.empty and float(recent_vr.max()) > 0.0:
        vol_mult = round(float(recent_vr.max()), 2)
    else:
        vol_mult = 1.0
    curr_vol_ratio = round(to_float(vol_ratios.iloc[-1], 1.0), 2)

    # OBV / 수급 방어 계산
    direction = np.sign(closes.diff().fillna(0.0))
    obv = (direction * vols).cumsum()
    obv_ma10 = obv.rolling(10, min_periods=3).mean()
    obv_strong = bool(to_float(obv.iloc[-1]) >= to_float(obv_ma10.iloc[-1]))

    # 1시간봉 멀티프레임 구조
    c_1h = df_1h["close"].astype(float)
    ma20_1h = to_float(c_1h.rolling(20, min_periods=5).mean().iloc[-1], price)
    ma50_1h = to_float(c_1h.rolling(50, min_periods=10).mean().iloc[-1], price)
    htf_bull = price >= ma20_1h * 0.995
    htf_status = "1H 정배열지지" if (price >= ma20_1h >= ma50_1h * 0.995) else ("1H 중심선지지" if htf_bull else "1H 눌림목지지")

    # 최근 스윙 고점/저점 및 피보나치 되돌림
    swing_high = to_float(highs.iloc[-48:-1].max(), price * 1.03)
    swing_low = to_float(lows.iloc[-48:-1].min(), price * 0.95)
    swing_span = max(swing_high - swing_low, price * 0.01)
    fib_retracement = (swing_high - price) / swing_span

    oi_ratio = round((oi_m / max(turnover_m, 0.1)) * 100.0, 1)
    reasons = []

    # 공통 수급 가산점 (최대 20점: OI 10점 + 펀딩 10점)
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

    # [Critical #1 해결] Track C: 급등 대장주(+12% 이상) 피보나치 0.30~0.68 눌림목 2파동 타점 (최대 100점 도달 가능)
    if change_24h >= 12.0 and 0.28 <= fib_retracement <= 0.68 and price >= swing_low * 1.02:
        track_type = "TRACK_C"
        track_title = "🚀 [Track C · 대장주 핵심 눌림목 2파동 타점]"
        # 1) 피보나치 골든존 기본 배점 (최대 36점)
        if 0.36 <= fib_retracement <= 0.62:
            tech_score += 36
            reasons.append(f"대장주(+{change_24h:.1f}%) Fib {fib_retracement:.2f} 골든존 지지")
        else:
            tech_score += 31
            reasons.append(f"대장주(+{change_24h:.1f}%) 눌림목 지지")

        # 2) 1H 이평 지지 배점 (최대 18점)
        if price >= ma20_1h:
            tech_score += 18
            reasons.append("1H 20MA 추세선 방어")
        elif price >= ma50_1h * 0.99:
            tech_score += 14

        # 3) OBV/CVD 세력 수급 잔존 배점 (최대 14점)
        if obv_strong:
            tech_score += 14
            reasons.append("눌림 구간 OBV 매집세 유지")
        else:
            tech_score += 9

        # 4) 조정 거래량 건조 또는 반등 양봉 유입 (최대 14점)
        is_bull_turn = closes.iloc[-1] >= opens.iloc[-1] and kdj_j_15m >= kdj_j_prev
        if is_bull_turn and (curr_vol_ratio <= 1.2 or vol_mult >= 1.5):
            tech_score += 14
            reasons.append("거래량 건조 후 15m J선 반등 턴")
        else:
            tech_score += 9

        # 5) RSI 과열 해소 가산점 (최대 8점)
        if 35.0 <= rsi6_15m <= 68.0:
            tech_score += 8

    # Track B: 수급 폭발 돌파 / S-R Flip 리테스트
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

    # Track A: 박스권 에너지 응축 / 유동성 스윕 후 추세 전환
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

    # wave_engine 모듈이 존재할 경우 파동 가산점 및 근거 연동
    if wave_engine is not None:
        try:
            if hasattr(wave_engine, "evaluate_wave_bonus"):
                w_bonus, w_note = wave_engine.evaluate_wave_bonus(df_15m, df_1h)
                total_score = int(np.clip(total_score + int(w_bonus), 0, 99))
                if w_note:
                    reasons.append(str(w_note))
        except Exception:
            pass

    # postmortem 모듈이 존재할 경우 오답노트 반복 진입 감점 연동
    if postmortem is not None:
        try:
            if hasattr(postmortem, "get_setup_penalty"):
                pen = int(postmortem.get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult))
                total_score = max(0, total_score - pen)
        except Exception:
            pass

    if total_score < SCORE_THRESHOLD:
        return None

    tp_params = get_dynamic_tp_params()
    p1_pct = tp_params["tp1_min_pct"]
    p2_pct = tp_params["tp2_min_pct"]
    p3_pct = tp_params["tp3_min_pct"]

    tps = [
        round(price * (1.0 + p1_pct / 100.0), 6),
        round(price * (1.0 + p2_pct / 100.0), 6),
        round(price * (1.0 + p3_pct / 100.0), 6),
    ]
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
        "tp_pcts": [p1_pct, p2_pct, p3_pct],
        "tp_labels": ["1차 파동목표", "2차 파동목표", "3차 최종목표"],
        "reasons": reasons[:4],
        "reason_str": " / ".join(reasons[:4]),
    }
# === END OF FILE ===
