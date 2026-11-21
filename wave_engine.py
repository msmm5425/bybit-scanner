import numpy as np
import pandas as pd


def _to_float(val, default=0.0):
    try:
        v = float(val)
        return default if (np.isnan(v) or np.isinf(v)) else v
    except Exception:
        return default


def calc_ppo(series, fast=12, slow=26, signal=9):
    """비율 오실레이터 PPO (12, 26, 9 EMA) 산출"""
    s = series.astype(float)
    ema_fast = s.ewm(span=fast, adjust=False, min_periods=min(fast, 4)).mean()
    ema_slow = s.ewm(span=slow, adjust=False, min_periods=min(slow, 6)).mean()
    ppo = ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)) * 100.0
    ppo = ppo.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig = ppo.ewm(span=signal, adjust=False, min_periods=min(signal, 3)).mean().fillna(0.0)
    hist = (ppo - sig).fillna(0.0)
    return ppo, sig, hist


def evaluate_wave_bonus(df_15m, df_1h):
    """
    [PPO(12, 26, 9) 멀티프레임 파동 가감점 엔진 (-3 ~ +5점)]
    - 선형 MACD 대신 가격 비율 기반 PPO로 골든크로스 · 0선 돌파 · 히스토그램 확장 검증
    """
    if df_15m is None or df_15m.empty or len(df_15m) < 30:
        return 0, ""

    c15 = df_15m["close"].astype(float)
    p15, s15, h15 = calc_ppo(c15, 12, 26, 9)

    p15_now, p15_prev = _to_float(p15.iloc[-1]), _to_float(p15.iloc[-2])
    s15_now, s15_prev = _to_float(s15.iloc[-1]), _to_float(s15.iloc[-2])
    h15_now, h15_prev, h15_prev2 = _to_float(h15.iloc[-1]), _to_float(h15.iloc[-2]), _to_float(h15.iloc[-3])

    bonus = 0
    tags = []

    # 1) 15m PPO 0선 상방 돌파 또는 골든크로스
    zero_cross_15 = (p15_prev <= 0.0) and (p15_now > 0.0) and (p15_now > s15_now)
    golden_cross_15 = (p15_prev <= s15_prev) and (p15_now > s15_now)

    if zero_cross_15:
        bonus += 3
        tags.append("15m PPO 0선돌파")
    elif golden_cross_15:
        bonus += 3
        tags.append("15m PPO 골든크로스")
    elif h15_now > h15_prev > h15_prev2 and h15_now > 0.0:
        bonus += 2
        tags.append("15m PPO 히스토그램 확장")

    # 2) 1H 상위 프레임 PPO 정렬 검증
    if df_1h is not None and not df_1h.empty and len(df_1h) >= 28:
        c1h = df_1h["close"].astype(float)
        p1h, s1h, h1h = calc_ppo(c1h, 12, 26, 9)
        p1h_now, s1h_now = _to_float(p1h.iloc[-1]), _to_float(s1h.iloc[-1])
        h1h_now, h1h_prev = _to_float(h1h.iloc[-1]), _to_float(h1h.iloc[-2])

        if p1h_now > 0.0 and p1h_now > s1h_now:
            bonus += 2
            tags.append("1H PPO 0선강세")
        elif h1h_now > h1h_prev:
            bonus += 1
        elif p1h_now < 0.0 and p1h_now < s1h_now and h1h_now < h1h_prev:
            bonus -= 3

    bonus = int(np.clip(bonus, -3, 5))
    note = f"PPO 파동 정렬({' · '.join(tags)})" if (bonus > 0 and tags) else ""
    return bonus, note
# === END OF FILE ===
