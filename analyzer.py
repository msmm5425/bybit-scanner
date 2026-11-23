import os, re, json, time
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
        "USDC", "USDT", "USDE", "FDUSD", "DAI", "TUSD", "BUSD", "PYUSD", "RLUSD", "USDS",
        "EUR", "EURC", "EURT", "EURI", "GBP", "JPY", "KRW", "AUD", "CAD", "CHF", "CNH", "HKD", "SGD",
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
DNA_PROFILE_FILE = "top5_dna_profile.json"
ELITE_SCORE_THRESHOLD = max(90, int(getattr(cfg, "SCORE_THRESHOLD", 90)))
FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
LONGTERM_SL_PCT = 20

_DNA_FORMULA_CACHE = {"ts": 0, "formula": None}


def _load_active_top5_dna_formula():
    """top5_dna_profile.json에서 최근 14일 학습된 활성 Top 5 발사 직전 DNA 공식 로드 (10분 캐시)"""
    now = time.time()
    if now - _DNA_FORMULA_CACHE["ts"] < 600:
        return _DNA_FORMULA_CACHE["formula"]
    _DNA_FORMULA_CACHE["ts"] = now
    formula = None
    if os.path.exists(DNA_PROFILE_FILE):
        try:
            with open(DNA_PROFILE_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                af = raw.get("active_dna_formula")
                if isinstance(af, dict) and af.get("enabled") and int(to_float(af.get("sample_count"), 0)) >= 5:
                    formula = af
        except Exception:
            pass
    _DNA_FORMULA_CACHE["formula"] = formula
    return formula


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
    res = {"quadrant": "NEUTRAL", "bonus": 0, "block": False, "squeeze_warn": False, "note": ""}
    if not isinstance(oi_snapshot_prev, dict):
        return res
    p_prev = to_float(oi_snapshot_prev.get("price"), 0.0)
    oi_prev = to_float(oi_snapshot_prev.get("oi_m"), 0.0)
    if p_prev <= 0 or oi_prev <= 0:
        return res

    dp_pct = ((price - p_prev) / p_prev) * 100.0
    doi_pct = ((oi_m - oi_prev) / oi_prev) * 100.0

    if dp_pct <= -0.5 and doi_pct >= 2.5:
        res.update({"quadrant": "Q3", "block": True, "note": f"숏세력 유입 차단(가격 {dp_pct:+.1f}%, OI {doi_pct:+.1f}%)"})
        return res

    if dp_pct >= 0.4 and doi_pct >= 2.0:
        bonus = 15 if doi_pct >= 5.0 else 10
        res.update({
            "quadrant": "Q1",
            "bonus": bonus,
            "note": f"[건전한 롱 매집] OI {doi_pct:+.1f}% 동반 상승(+{bonus}점)",
        })
        return res

    if dp_pct >= 1.0 and doi_pct <= -2.5:
        res.update({
            "quadrant": "Q2",
            "squeeze_warn": True,
            "note": f"[숏스퀴즈 주의] OI {doi_pct:+.1f}% 감소 반등(TP1·2 보수적 하향)",
        })
        return res

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


def _calc_ppo(series, fast=12, slow=26, signal=9):
    """
    [트레이딩뷰 공식 Price Oscillator (PPO, 12/26/9 EMA)]
    - 선형 MACD의 가격 압축 착시를 제거한 비율(%) 오실레이터
    - PPO = ((EMA12 - EMA26) / EMA26) * 100
    """
    s = series.astype(float)
    ema_fast = s.ewm(span=fast, adjust=False, min_periods=min(fast, 4)).mean()
    ema_slow = s.ewm(span=slow, adjust=False, min_periods=min(slow, 6)).mean()
    ppo = ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)) * 100.0
    ppo = ppo.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig = ppo.ewm(span=signal, adjust=False, min_periods=min(signal, 3)).mean().fillna(0.0)
    hist = (ppo - sig).fillna(0.0)
    return ppo, sig, hist


def _resample_1d_to_1w(df_1d):
    """주봉 API 일시 누락 시 일봉(1D) 데이터를 월요일 00:00 UTC 기준 주봉(1W)으로 정확히 리샘플링"""
    if df_1d is None or df_1d.empty or len(df_1d) < 14:
        return pd.DataFrame()
    try:
        tmp = df_1d.copy()
        dt_utc = pd.to_datetime(tmp["timestamp"], unit="ms", utc=True)
        tmp["week_start"] = dt_utc.dt.floor("D") - pd.to_timedelta(dt_utc.dt.weekday, unit="D")
        grouped = tmp.groupby("week_start", as_index=False).agg({
            "timestamp": "first",
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        return grouped.sort_values("timestamp", ascending=True).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _fetch_1d_and_1w(symbol, limit_1d=180, limit_1w=60):
    """후보 종목의 일봉(1D) + 주봉(1W) 데이터를 안전하게 조회 (주봉 누락 시 1D 리샘플링 백업)"""
    df_1d = fetch_klines(symbol, interval="D", limit=limit_1d)
    df_1w = fetch_klines(symbol, interval="W", limit=limit_1w)
    if (df_1w is None or df_1w.empty or len(df_1w) < 4) and (df_1d is not None and len(df_1d) >= 21):
        df_1w = _resample_1d_to_1w(df_1d)
    return df_1d, df_1w


def _evaluate_weekly_daily_pro_filter(
    price,
    df_1d,
    df_1w,
    mode="LONG",
    track_type="TRACK_D",
    sub_strategy="",
    funding_pct=0.0,
):
    """
    [전문 트레이더 기준 주봉(1W) · 일봉(1D) 탑다운 + PPO(12,26,9) 검증 엔진]
    """
    out = {"allow": True, "score_adj": 0, "htf_tag": "", "reasons": [], "block_reason": ""}
    is_spring_d6 = "D6" in str(sub_strategy or "").upper()

    # 1) 일봉(1D) 구조 및 1D PPO 진단
    has_1d = df_1d is not None and not df_1d.empty and len(df_1d) >= 14
    dist_1d_ma20 = 0.0
    dd_30d_pct = 0.0
    dd_macro_pct = 0.0
    made_new_30d_low_recently = False
    has_floor_base = False
    obv1d_alive = True
    obv1d_dead = False
    daily_bull = False
    ppo1d_zero_up = False
    ppo1d_turn_up = False
    ppo1d_bear_accel = False

    if has_1d:
        c1d = df_1d["close"].astype(float)
        h1d = df_1d["high"].astype(float)
        l1d = df_1d["low"].astype(float)
        v1d = df_1d["volume"].astype(float).fillna(0.0)

        ma20_1d = to_float(c1d.rolling(20, min_periods=7).mean().iloc[-1], price)
        dist_1d_ma20 = ((price - ma20_1d) / max(ma20_1d, 1e-8)) * 100.0
        daily_bull = price >= ma20_1d * 0.99

        win_30 = min(len(df_1d), 30)
        hi_30d = to_float(h1d.iloc[-win_30:].max(), price)
        lo_30d = to_float(l1d.iloc[-win_30:].min(), price)
        dd_30d_pct = ((hi_30d - price) / max(hi_30d, 1e-8)) * 100.0
        lo_recent_4d = to_float(l1d.iloc[-4:].min(), price)
        made_new_30d_low_recently = lo_recent_4d <= lo_30d * 1.005

        macro_hi_1d = to_float(h1d.max(), price)
        dd_macro_pct = ((macro_hi_1d - price) / max(macro_hi_1d, 1e-8)) * 100.0

        if len(df_1d) >= 14:
            base_hi = to_float(h1d.iloc[-14:-2].max(), price)
            base_lo = to_float(l1d.iloc[-14:-2].min(), price)
            base_range_pct = ((base_hi - base_lo) / max(base_lo, 1e-8)) * 100.0
            has_floor_base = (not made_new_30d_low_recently) or (base_range_pct <= 30.0)
        else:
            has_floor_base = not made_new_30d_low_recently

        obv1d = (np.sign(c1d.diff().fillna(0.0)) * v1d).cumsum()
        obv1d_ma20 = obv1d.rolling(20, min_periods=7).mean()
        obv1d_alive = to_float(obv1d.iloc[-1]) >= to_float(obv1d_ma20.iloc[-1])
        obv1d_rising_7d = to_float(obv1d.iloc[-1]) >= to_float(obv1d.iloc[-min(len(obv1d), 7)])
        obv1d_dead = (not obv1d_alive) and (not obv1d_rising_7d)

        if len(df_1d) >= 30:
            p1d, s1d, h1d_ppo = _calc_ppo(c1d, 12, 26, 9)
            p1d_now, s1d_now = to_float(p1d.iloc[-1]), to_float(s1d.iloc[-1])
            h1d_now, h1d_prev = to_float(h1d_ppo.iloc[-1]), to_float(h1d_ppo.iloc[-2])
            ppo1d_zero_up = (p1d_now > 0.0) and (p1d_now >= s1d_now)
            ppo1d_turn_up = (h1d_now > h1d_prev) and (p1d_now >= to_float(p1d.iloc[-2]))
            ppo1d_bear_accel = (p1d_now < 0.0) and (p1d_now < s1d_now) and (h1d_now < h1d_prev)

    # 2) 주봉(1W) 구조 및 1W PPO 진단
    has_1w = df_1w is not None and not df_1w.empty and len(df_1w) >= 4
    dist_1w_ma10 = 0.0
    dd_3w_pct = 0.0
    making_weekly_ll = False
    weekly_red_body = False
    consec_red_3w = False
    weekly_blowoff_top = False
    weekly_floor_hold = False
    weekly_spring_hammer = False
    weekly_ma10_reclaim = False
    weekly_bull_trend = False
    obv1w_rising = True
    ppo1w_zero_up = False
    ppo1w_turn_up = False
    ppo1w_bear_accel = False

    if has_1w:
        o1w = df_1w["open"].astype(float)
        h1w = df_1w["high"].astype(float)
        l1w = df_1w["low"].astype(float)
        c1w = df_1w["close"].astype(float)
        v1w = df_1w["volume"].astype(float).fillna(0.0)

        w_open, w_close = to_float(o1w.iloc[-1], price), to_float(c1w.iloc[-1], price)
        w_high, w_low = to_float(h1w.iloc[-1], price), to_float(l1w.iloc[-1], price)
        w_range = max(w_high - w_low, 1e-10)
        w_lwick = (min(w_open, w_close) - w_low) / w_range

        ma10_1w = to_float(c1w.rolling(10, min_periods=4).mean().iloc[-1], price)
        dist_1w_ma10 = ((price - ma10_1w) / max(ma10_1w, 1e-8)) * 100.0

        obv1w = (np.sign(c1w.diff().fillna(0.0)) * v1w).cumsum()
        obv1w_rising = to_float(obv1w.iloc[-1]) >= to_float(obv1w.iloc[-min(len(obv1w), 4)])

        hi_3w = to_float(h1w.iloc[-min(len(h1w), 3):].max(), price)
        dd_3w_pct = ((hi_3w - price) / max(hi_3w, 1e-8)) * 100.0
        prev_w_low = to_float(l1w.iloc[-2], price)
        making_weekly_ll = w_low < prev_w_low * 0.995
        weekly_red_body = (w_close < w_open) and (w_lwick < 0.35)
        consec_red_3w = len(df_1w) >= 4 and all(to_float(c1w.iloc[-i]) < to_float(o1w.iloc[-i]) for i in (2, 3, 4))

        weekly_spring_hammer = making_weekly_ll and (w_lwick >= 0.42) and (w_close >= prev_w_low * 1.002) and (w_close >= w_open * 0.995)

        v1w_ma = to_float(v1w.iloc[-min(len(v1w), 12):-1].mean(), 1.0)
        for idx in (-1, -2):
            if abs(idx) <= len(df_1w):
                bo_o, bo_h, bo_l, bo_c, bo_v = (
                    to_float(o1w.iloc[idx]), to_float(h1w.iloc[idx]),
                    to_float(l1w.iloc[idx]), to_float(c1w.iloc[idx]), to_float(v1w.iloc[idx]),
                )
                uwick = (bo_h - max(bo_o, bo_c)) / max(bo_h - bo_l, 1e-10)
                if uwick >= 0.62 and bo_c < bo_o and bo_v >= v1w_ma * 2.2 and bo_h >= ma10_1w * 1.15:
                    weekly_blowoff_top = True
                    break

        if len(df_1w) >= 6:
            recent_2w_l = to_float(l1w.iloc[-2:].min(), w_low)
            prior_6w_l = to_float(l1w.iloc[-min(len(df_1w), 8):-2].min(), w_low)
            weekly_floor_hold = (recent_2w_l >= prior_6w_l * 0.985) and (w_close >= w_open or w_lwick >= 0.45)

        weekly_ma10_reclaim = (price >= ma10_1w * 0.99) and (to_float(c1w.iloc[-2], price) <= ma10_1w * 1.03) and (w_close >= w_open)
        weekly_bull_trend = (price >= ma10_1w * 0.99) and (w_close >= w_open or w_lwick >= 0.40)

        if len(df_1w) >= 14:
            p1w, s1w, h1w_ppo = _calc_ppo(c1w, 12, 26, 9)
            p1w_now, s1w_now = to_float(p1w.iloc[-1]), to_float(s1w.iloc[-1])
            h1w_now, h1w_prev = to_float(h1w_ppo.iloc[-1]), to_float(h1w_ppo.iloc[-2])
            ppo1w_zero_up = (p1w_now > 0.0) and (p1w_now >= s1w_now)
            ppo1w_turn_up = (h1w_now > h1w_prev) and (p1w_now >= to_float(p1w.iloc[-2]))
            ppo1w_bear_accel = (p1w_now < 0.0) and (p1w_now < s1w_now) and (h1w_now < h1w_prev)

    # 3) 장기 포지션 (Track D1~D6) 전용 판정
    if mode == "LONG":
        if has_1d:
            if (not is_spring_d6) and made_new_30d_low_recently and dd_30d_pct >= 45.0:
                out.update({"allow": False, "block_reason": f"1D 신저가 폭락 진행형(-{dd_30d_pct:.0f}%)"})
                return out
            if obv1d_dead:
                out.update({"allow": False, "block_reason": "1D OBV 하락 이탈(물량 떠넘기기 의심)"})
                return out
            ma20_cut = -15.0 if is_spring_d6 else -12.0
            if dist_1d_ma20 <= ma20_cut:
                out.update({"allow": False, "block_reason": f"1D 20MA 극단 역배열({dist_1d_ma20:+.1f}%)"})
                return out

        if has_1w:
            if is_spring_d6:
                if weekly_red_body and (dd_3w_pct >= 45.0 or consec_red_3w):
                    out.update({"allow": False, "block_reason": f"1W 주봉 음봉 붕괴 미해소(-{dd_3w_pct:.0f}%)"})
                    return out
            else:
                if (dd_3w_pct >= 45.0 and making_weekly_ll) or (consec_red_3w and weekly_red_body and making_weekly_ll):
                    out.update({"allow": False, "block_reason": f"1W 주봉 수직 붕괴 진행형(-{dd_3w_pct:.0f}%)"})
                    return out
                if ppo1w_bear_accel and dd_3w_pct >= 35.0 and making_weekly_ll:
                    out.update({"allow": False, "block_reason": "1W 주봉 PPO 하락 가속 붕괴"})
                    return out
            if weekly_blowoff_top:
                out.update({"allow": False, "block_reason": "1W 주봉 고점 피뢰침 설거지 직후"})
                return out
            if (not is_spring_d6) and dist_1w_ma10 <= -35.0 and not has_floor_base:
                out.update({"allow": False, "block_reason": f"1W 10MA 극단 괴리({dist_1w_ma10:+.1f}%)"})
                return out

            if ppo1w_zero_up:
                out["score_adj"] += 4
                out["reasons"].append("1W 주봉 PPO 0선 위 상승체제 안착")
            elif weekly_spring_hammer:
                out["score_adj"] += 5
                out["reasons"].append("1W 주봉 신저가 스윕 후 긴 아래꼬리 스프링 복귀")
            elif weekly_ma10_reclaim:
                out["score_adj"] += 4
                out["reasons"].append("1W 주봉 10MA 추세 전환 초입 확인")
            elif weekly_floor_hold and (obv1w_rising or ppo1w_turn_up):
                out["score_adj"] += 4
                out["reasons"].append("1W 주봉 바닥 방어 + PPO·OBV 상방 전환")
            elif ppo1w_turn_up:
                out["score_adj"] += 2
                out["reasons"].append("1W 주봉 PPO 바닥 히스토그램 상방 턴")
            elif weekly_bull_trend:
                out["score_adj"] += 2
                out["reasons"].append("1W 주봉 상승 추세 안착")

            if is_spring_d6 or weekly_spring_hammer:
                out["htf_tag"] = "1W스프링·1D탈환"
            elif ppo1w_zero_up and (daily_bull or ppo1d_zero_up):
                out["htf_tag"] = "1W·1D PPO 0선강세"
            elif weekly_bull_trend and daily_bull:
                out["htf_tag"] = "1W상승·1D정배열"
            elif weekly_floor_hold or obv1w_rising or ppo1w_turn_up:
                out["htf_tag"] = "1W바닥숙성·PPO반전"
            else:
                out["htf_tag"] = "1W·1D바닥전환"
        else:
            out["htf_tag"] = "1D스프링탈환" if is_spring_d6 else ("1D PPO강세" if ppo1d_zero_up else "4H·1D매집")

        if has_1d and ppo1d_zero_up and not ppo1w_zero_up:
            out["score_adj"] += 1
            if len(out["reasons"]) < 2:
                out["reasons"].append("1D 일봉 PPO 0선 돌파 모멘텀 정렬")
        if has_1d and dd_macro_pct >= 60.0 and funding_pct <= -0.005:
            out["score_adj"] += 2
            out["reasons"].append(f"심층 낙폭(-{dd_macro_pct:.0f}%) + 음수펀딩 숏함정 역발상 연료")
        elif has_1d and obv1d_alive and not out["reasons"]:
            out["reasons"].append("1D 일봉 OBV 매집 생존 확인")

        return out

    # 4) 단기·스윙 포지션 (Track A~H) 전용 판정
    tk = str(track_type or "TRACK_A").upper()
    if has_1w:
        crash_cut = 65.0 if tk == "TRACK_E" else 50.0
        if dd_3w_pct >= crash_cut and making_weekly_ll and weekly_red_body:
            out.update({"allow": False, "block_reason": f"1W 주봉 수직 폭락 차단(-{dd_3w_pct:.0f}%)"})
            return out
        if weekly_blowoff_top:
            out.update({"allow": False, "block_reason": "1W 주봉 고점 피뢰침 설거지 차단"})
            return out

    if has_1d and tk in ("TRACK_A", "TRACK_B", "TRACK_G", "TRACK_H"):
        if made_new_30d_low_recently and dd_30d_pct >= 45.0 and dist_1d_ma20 <= -10.0:
            out.update({"allow": False, "block_reason": f"1D 신저가 폭락 역배열 차단(-{dd_30d_pct:.0f}%)"})
            return out

    if has_1w and has_1d:
        if (weekly_bull_trend or ppo1w_zero_up) and (daily_bull or ppo1d_zero_up):
            out["score_adj"] = 3
            out["htf_tag"] = "1W·1D PPO강세" if (ppo1w_zero_up or ppo1d_zero_up) else "1W상승·1D강세"
            out["reasons"].append("1W 주봉·1D 일봉 PPO 상위 추세 정렬 우위")
        elif (weekly_ma10_reclaim or weekly_floor_hold or weekly_spring_hammer or ppo1w_turn_up) and (dist_1d_ma20 >= -3.0 or ppo1d_turn_up):
            out["score_adj"] = 2
            out["htf_tag"] = "1W바닥지지·1D PPO회복"
        elif dist_1w_ma10 <= -8.0 and dist_1d_ma20 <= -4.0:
            out["score_adj"] = -2 if tk == "TRACK_E" else -4
            out["htf_tag"] = "1W·1D역배열(단기반등)"
            out["reasons"].append("⚠️ 1W·1D 상위 역배열 구간(단기 저항 주의)")
        else:
            out["htf_tag"] = "1W중립·1D강세" if daily_bull else "1W횡보·1D중립"
    elif has_1d:
        out["htf_tag"] = "1D PPO강세" if ppo1d_zero_up else ("1D강세" if daily_bull else "1D조정")

    if has_1d and ppo1d_bear_accel and tk in ("TRACK_A", "TRACK_B", "TRACK_G", "TRACK_H") and dist_1d_ma20 <= -6.0:
        out["score_adj"] -= 2

    return out


def _calc_ttm_squeeze_and_cvd(df_15m):
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

    bar_range = (highs - lows).replace(0, np.nan)
    clv = (((closes - lows) - (highs - closes)) / bar_range).clip(-1.0, 1.0).fillna(0.0)
    bar_delta = clv * vols
    cvd = bar_delta.cumsum()
    cvd_ma10 = cvd.rolling(10, min_periods=3).mean()

    recent_delta_sum = to_float(bar_delta.iloc[-3:].sum(), 0.0)
    curr_bar_delta = to_float(bar_delta.iloc[-1], 0.0)
    cvd_breakout = to_float(cvd.iloc[-1]) >= to_float(cvd.iloc[-12:-1].max(), -1e18)
    cvd_absorption = bool((to_float(cvd.iloc[-1]) > to_float(cvd_ma10.iloc[-1])) and (recent_delta_sum > 0) and cvd_breakout)

    return ttm_fired, was_squeezed, cvd_absorption, recent_delta_sum, curr_bar_delta


def _check_5m_mss_trigger(symbol):
    df_5m = fetch_klines(symbol, interval="5", limit=36)
    if df_5m is None or df_5m.empty or len(df_5m) < 12:
        return None

    o5 = df_5m["open"].astype(float)
    h5 = df_5m["high"].astype(float)
    l5 = df_5m["low"].astype(float)
    c5 = df_5m["close"].astype(float)

    recent_low = to_float(l5.iloc[-4:].min())
    prior_low = to_float(l5.iloc[-12:-4].min())
    higher_low = recent_low >= prior_low * 0.996

    prev_swing_h = to_float(h5.iloc[-7:-2].max())
    mss_break = to_float(c5.iloc[-1]) > prev_swing_h and to_float(c5.iloc[-1]) > to_float(o5.iloc[-1])

    bull_engulf = (
        to_float(c5.iloc[-1]) > to_float(o5.iloc[-1])
        and to_float(c5.iloc[-2]) < to_float(o5.iloc[-2])
        and to_float(c5.iloc[-1]) >= to_float(o5.iloc[-2])
        and to_float(o5.iloc[-1]) <= to_float(c5.iloc[-2]) * 1.003
    )

    return bool((higher_low and mss_break) or bull_engulf)


def _calculate_resistance_based_tps(price, df_15m, df_1h, conservative_scale=1.0):
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

    if conservative_scale < 0.99:
        p1_pct = max(6.5, p1_pct * conservative_scale)
        p2_pct = max(p1_pct + 4.8, p2_pct * conservative_scale)
        lbl1 = f"{lbl1}·보수적"
        lbl2 = f"{lbl2}·보수적"

    p1_pct, p2_pct, p3_pct = round(float(p1_pct), 1), round(float(p2_pct), 1), round(float(p3_pct), 1)
    tps = [round(price * (1.0 + p1_pct / 100.0), 6), round(price * (1.0 + p2_pct / 100.0), 6), round(price * (1.0 + p3_pct / 100.0), 6)]
    return tps, [p1_pct, p2_pct, p3_pct], [lbl1, lbl2, lbl3]


def _calculate_macro_longterm_tps(price, turnover_m, df_4h, df_1d, df_1w=None):
    """장기 포지션(Track D) 전용 4H · 1D · 1W(주봉) 매물대 기반 종목별 개별 TP 산출"""
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

    weekly_res_pct = 0.0
    if df_1w is not None and not df_1w.empty and len(df_1w) >= 6:
        h1w = df_1w["high"].astype(float)
        w_peaks = [
            ((float(h1w.iloc[i]) - price) / price * 100.0)
            for i in range(1, len(h1w) - 1)
            if float(h1w.iloc[i]) > price * 1.45 and float(h1w.iloc[i]) >= float(h1w.iloc[i - 1 : i + 2].max())
        ]
        if w_peaks:
            weekly_res_pct = sorted(w_peaks)[0]

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
    p2_base = max(p1_pct + 26.0, fib2618_pct * 0.80, macro_mid_pct, 50.0 * macro_scale)
    if 48.0 <= weekly_res_pct <= 95.0:
        p2_base = (p2_base * 0.5) + (weekly_res_pct * 0.5)
    p2_pct = round(float(np.clip(p2_base, 48.0, 95.0)), 1)

    fib4236_pct = ((box_h_30d + box_span * 3.236) - price) / price * 100.0
    macro_top_pct = (macro_h_120d - price) / price * 100.0
    p3_base = max(p2_pct + 48.0, fib4236_pct * 0.85, macro_top_pct, 105.0 * macro_scale)
    if tp3_min_cap <= weekly_res_pct <= tp3_max_cap:
        p3_base = max(p3_base * 0.7, weekly_res_pct)
    p3_pct = round(float(np.clip(p3_base, tp3_min_cap, tp3_max_cap)), 1)

    tps = [
        round(price * (1.0 + p1_pct / 100.0), 6),
        round(price * (1.0 + p2_pct / 100.0), 6),
        round(price * (1.0 + p3_pct / 100.0), 6),
    ]
    return tps, [p1_pct, p2_pct, p3_pct], ["4H/1D 1차 저항대", "1D·1W 주요 전고점 저항", f"1W·1D 대시세목표({tier_lbl})"]


def _evaluate_macro_track_d(price, change_24h, turnover_m, oi_m, oi_ratio, funding_pct, df_4h):
    """
    [장기 포지션 Track D 6대 전략 판별 — 전 트랙 4H PPO(12,26,9) 통합 + 기급등/초저변동성 페깅 차단]
    """
    if df_4h is None or df_4h.empty or len(df_4h) < 55:
        return None
    if change_24h >= 22.0:
        return None

    c4 = df_4h["close"].astype(float)
    h4 = df_4h["high"].astype(float)
    l4 = df_4h["low"].astype(float)
    o4 = df_4h["open"].astype(float)
    v4 = df_4h["volume"].astype(float).fillna(0.0)

    # [2차 수학적 안전장치] 최근 4H 전체(최대 30일) 고저 변동폭이 5.0% 미만인 스테이블/페깅 토큰 영구 차단
    hi_all_4h = to_float(h4.max(), price)
    lo_all_4h = to_float(l4.min(), price)
    if lo_all_4h <= 0 or ((hi_all_4h - lo_all_4h) / lo_all_4h * 100.0) < 5.0:
        return None

    obv4 = (np.sign(c4.diff().fillna(0.0)) * v4).cumsum()
    rsi14_4h = _calc_rsi(c4, 14)
    ppo4, ppo4_sig, ppo4_hist = _calc_ppo(c4, 12, 26, 9)
    ppo4_now, ppo4_sig_now = to_float(ppo4.iloc[-1]), to_float(ppo4_sig.iloc[-1])
    ppo4_hist_now, ppo4_hist_prev = to_float(ppo4_hist.iloc[-1]), to_float(ppo4_hist.iloc[-2])
    ppo4_turn_up = ppo4_hist_now > ppo4_hist_prev
    ppo4_bull = (ppo4_now > ppo4_sig_now) and (ppo4_hist_now >= 0.0)
    ppo4_zero_up = (ppo4_now > 0.0) and (ppo4_now > ppo4_sig_now)

    v4_ma20 = v4.rolling(20, min_periods=5).mean().replace(0, np.nan)
    v4_ratio = (v4 / v4_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ma20_4h_s = c4.rolling(20, min_periods=5).mean()
    ma50_4h_s = c4.rolling(50, min_periods=10).mean()
    ma20_4h = to_float(ma20_4h_s.iloc[-1], price)
    ma20_4h_prev3 = to_float(ma20_4h_s.iloc[-4], ma20_4h)
    ma50_4h = to_float(ma50_4h_s.iloc[-1], price)

    candidates_d = []

    # D1: 스텔스 OI · OBV 선행 매집 + 4H PPO 상방 정렬
    hi_10d = to_float(h4.iloc[-60:-1].max(), price * 1.05)
    obv_hi_10d = to_float(obv4.iloc[-60:-1].max(), 0.0)
    if -3.0 <= change_24h <= 11.0 and oi_ratio >= 28.0 and funding_pct <= 0.008:
        if price < hi_10d * 0.995 and to_float(obv4.iloc[-1]) > obv_hi_10d and price >= ma20_4h * 0.985 and (ppo4_bull or ppo4_turn_up):
            score_d1 = int(np.clip(87 + min(7, int((oi_ratio - 26) / 3)) + (4 if funding_pct < 0 else 1) + (2 if ppo4_zero_up else 1), 88, 99))
            candidates_d.append({
                "sub_strategy": "D1 · 스텔스 OI·OBV 선행 매집",
                "score": score_d1,
                "reasons": [
                    f"가격 횡보 중 4H OBV 10일 최고치 선행 돌파 + PPO({ppo4_now:+.2f}%) 상방 전환",
                    f"스텔스 미결제약정(OI) 집중도 {oi_ratio:.0f}% 축적",
                    f"펀딩비 {funding_pct:+.4f}% 저과열 바닥 매집 구조",
                ],
            })

    # D2: 매집봉 출현 후 거래량 건조 눌림 + 4H PPO 반전 확인
    recent_30 = df_4h.iloc[-30:-2]
    vr_30 = v4_ratio.iloc[-30:-2]
    if not vr_30.empty and float(vr_30.max()) >= 3.8:
        pump_pos = int(np.argmax(vr_30.values))
        pump_row = recent_30.iloc[pump_pos]
        pump_vol = to_float(pump_row["volume"], 1.0)
        pump_open = to_float(pump_row["open"], price)
        pump_high = to_float(pump_row["high"], price)
        pump_mid = (pump_open + pump_high) * 0.5
        pump_span_pct = ((pump_high - pump_open) / max(pump_open, 1e-8)) * 100.0
        curr_vol_4h = to_float(v4.iloc[-2], pump_vol)
        is_dry_up = pump_vol > 0 and (curr_vol_4h / pump_vol) <= 0.30
        is_turn_up = c4.iloc[-1] >= o4.iloc[-1] and price >= max(pump_open * 0.99, ma20_4h * 0.985) and price <= pump_high * 1.02
        if pump_span_pct >= 2.0 and is_dry_up and is_turn_up and (ppo4_turn_up or ppo4_zero_up):
            score_d2 = int(np.clip(87 + (5 if price >= pump_mid else 2) + (5 if oi_ratio >= 15.0 else 1) + (2 if ppo4_zero_up else 1), 88, 99))
            candidates_d.append({
                "sub_strategy": "D2 · 매집봉 출현 후 거래량 건조 눌림",
                "score": score_d2,
                "reasons": [
                    f"최근 5일 내 4H 매집봉(거래량 {float(vr_30.max()):.1f}배) 출현",
                    "조정 구간 거래량 70% 이상 건조(세력 이탈 없음)",
                    f"매집봉 지지선 & 4H 20MA 위 양봉 반등 + PPO({ppo4_now:+.2f}%) 상방 턴",
                ],
            })

    # D3: 4H 30일 장기 박스권 첫 돌파 + 4H 20MA 이격(+15% 이내) 초입 검증
    if len(df_4h) >= 120:
        hi_30d = to_float(h4.iloc[:-1].max(), price)
        lo_30d = to_float(l4.iloc[:-1].min(), price * 0.8)
        range_30d_pct = (hi_30d - lo_30d) / max(lo_30d, 1e-8) * 100.0
        curr_vr_4h = to_float(v4_ratio.iloc[-1], 1.0)
        if (
            hi_30d * 0.996 <= price <= hi_30d * 1.045
            and price <= ma20_4h * 1.15
            and curr_vr_4h >= 2.4
            and 5.0 <= range_30d_pct <= 55.0
            and ppo4_bull
        ):
            score_d3 = int(np.clip(88 + min(7, int(curr_vr_4h * 1.5)) + (2 if price >= ma50_4h else 0) + (2 if ppo4_zero_up else 1), 89, 99))
            candidates_d.append({
                "sub_strategy": "D3 · 4H 30일 장기 박스권 첫 돌파",
                "score": score_d3,
                "reasons": [
                    f"4H 30일 장기 박스권 첫 돌파 + PPO({ppo4_now:+.2f}%) {'0선 돌파' if ppo4_zero_up else '강세확장'}",
                    f"4H 돌파 거래량 평균 대비 {curr_vr_4h:.1f}배 폭발",
                    f"장기 박스권 응축폭 {range_30d_pct:.1f}% 소화 후 1파동 개시",
                ],
            })

    # D4: 4H 장기 하락추세 탈출 & 50MA 돌파 + 4H PPO 골든크로스/0선 돌파
    if len(df_4h) >= 85:
        hi_30d_d4 = to_float(h4.iloc[:-1].max(), price)
        lo_recent_30 = to_float(l4.iloc[-30:].min(), price)
        dd_30d_pct = ((hi_30d_d4 - lo_recent_30) / max(hi_30d_d4, 1e-8)) * 100.0
        higher_low_4h = to_float(l4.iloc[-12:].min(), price) >= to_float(l4.iloc[-36:-12].min(), price) * 0.995
        fresh_ma50_break = (
            ma50_4h > 0
            and price > ma50_4h
            and price <= ma50_4h * 1.08
            and to_float(c4.iloc[-5:-1].min(), price) <= ma50_4h * 1.015
            and c4.iloc[-1] > o4.iloc[-1]
        )
        vol_break_4h = float(v4_ratio.iloc[-2:].max())
        ma20_turning = ma20_4h > ma20_4h_prev3 and ma20_4h >= ma50_4h * 0.94
        obv_14d_hi = to_float(obv4.iloc[-84:-1].max(), -1e18)
        obv_14d_break = to_float(obv4.iloc[-1]) >= obv_14d_hi

        if dd_30d_pct >= 25.0 and higher_low_4h and fresh_ma50_break and vol_break_4h >= 2.2 and ma20_turning and obv_14d_break and ppo4_bull:
            score_d4 = int(np.clip(88 + min(6, int(vol_break_4h * 1.5)) + (3 if ma20_4h >= ma50_4h else 1) + (2 if ppo4_zero_up else 1), 89, 99))
            candidates_d.append({
                "sub_strategy": "D4 · 4H 장기 하락추세 탈출 & 50MA 돌파",
                "score": score_d4,
                "reasons": [
                    f"30일 고점 대비 -{dd_30d_pct:.0f}% 심층 조정 후 4H 이중바닥 완성",
                    f"4H 50MA 거래량 {vol_break_4h:.1f}배 돌파 + PPO({ppo4_now:+.2f}%) {'0선 안착' if ppo4_zero_up else '골든크로스'}",
                    "4H OBV 14일 최고치 선행 돌파(장기 추세 1파동 초입)",
                ],
            })

    # D5: 4H 매크로 PPO·RSI14·OBV 3중 강세 다이버전스 바닥 매집
    if len(df_4h) >= 65:
        lo_14d = to_float(l4.iloc[-min(len(l4), 84):].min(), price)
        in_bottom_zone = price <= lo_14d * 1.12
        prev_win = l4.iloc[-42:-12]
        rec_win = l4.iloc[-12:]
        p_idx = int(np.argmin(prev_win.values))
        r_idx = int(np.argmin(rec_win.values))
        p_abs = len(df_4h) - 42 + p_idx
        r_abs = len(df_4h) - 12 + r_idx

        l_prev_v = to_float(l4.iloc[p_abs])
        l_rec_v = to_float(l4.iloc[r_abs])
        rsi_prev_v = to_float(rsi14_4h.iloc[p_abs], 50.0)
        rsi_rec_v = to_float(rsi14_4h.iloc[r_abs], 50.0)
        ppo_prev_v = to_float(ppo4.iloc[p_abs], 0.0)
        ppo_rec_v = to_float(ppo4.iloc[r_abs], 0.0)
        obv_prev_v = to_float(obv4.iloc[p_abs], 0.0)
        obv_curr_v = to_float(obv4.iloc[-1], 0.0)

        ppo_bull_div_4h = (ppo_rec_v >= ppo_prev_v + 0.35) and ppo4_turn_up
        bull_div_4h = (l_rec_v <= l_prev_v * 1.03) and (rsi_rec_v >= rsi_prev_v + 4.0) and ppo_bull_div_4h and (obv_curr_v > obv_prev_v)
        body_4h_pct = ((to_float(c4.iloc[-1]) - to_float(o4.iloc[-1])) / max(to_float(o4.iloc[-1]), 1e-8)) * 100.0
        strong_bull_4h = body_4h_pct >= 1.5 and to_float(c4.iloc[-1]) >= to_float(h4.iloc[-1]) * 0.985

        if in_bottom_zone and bull_div_4h and strong_bull_4h and oi_ratio >= 20.0 and funding_pct <= 0.005:
            score_d5 = int(np.clip(89 + min(5, int((rsi_rec_v - rsi_prev_v) / 2.5)) + (4 if funding_pct < 0 else 2), 89, 99))
            candidates_d.append({
                "sub_strategy": "D5 · 4H 매크로 PPO·RSI 강세 다이버전스 바닥 매집",
                "score": score_d5,
                "reasons": [
                    f"14일 바닥권 4H PPO({ppo_prev_v:+.1f}%→{ppo_rec_v:+.1f}%)·RSI14·OBV 3중 강세 다이버전스",
                    f"고OI 집중도({oi_ratio:.0f}%) + 저과열 펀딩({funding_pct:+.4f}%) 세력 바닥 흡수",
                    f"4H 장대양봉(+{body_4h_pct:.1f}%) 출현으로 장기 턴어라운드 개시",
                ],
            })

    # D6: 와이코프 스프링(신저가 개미털기 후 다개월 바닥선 거래량 동반 재탈환 + 4H PPO 상방 턴)
    if len(df_4h) >= 100 and turnover_m >= 3.0 and oi_m >= 0.8 and oi_ratio >= 15.0 and funding_pct <= 0.006:
        prior_floor = to_float(l4.iloc[:-48].min(), price)
        prior_high_30 = to_float(h4.iloc[:-12].max(), price)
        spring_low = to_float(l4.iloc[-48:].min(), price)
        sweep_depth_pct = ((prior_floor - spring_low) / max(prior_floor, 1e-8)) * 100.0

        had_spring_sweep = (2.5 <= sweep_depth_pct <= 35.0) and (prior_high_30 >= prior_floor * 1.15)
        reclaimed_floor = (prior_floor * 1.003 <= price <= prior_floor * 1.18) and (price >= ma20_4h * 0.98)
        bull_reclaim_4h = (c4.iloc[-1] > o4.iloc[-1]) or (to_float(c4.iloc[-1]) >= to_float(c4.iloc[-3]) * 1.02)
        reclaim_vol_4h = float(v4_ratio.iloc[-4:].max())
        obv4_reclaim = to_float(obv4.iloc[-1]) > to_float(obv4.iloc[-12:-2].mean())

        if had_spring_sweep and reclaimed_floor and bull_reclaim_4h and reclaim_vol_4h >= 2.2 and obv4_reclaim and (ppo4_turn_up or ppo4_bull):
            score_d6 = int(np.clip(
                88
                + (4 if funding_pct <= -0.002 else 1)
                + (4 if reclaim_vol_4h >= 3.0 else 2)
                + (3 if oi_ratio >= 25.0 else (1 if oi_ratio >= 18.0 else 0))
                + (2 if ppo4_bull else 0),
                88, 99,
            ))
            candidates_d.append({
                "sub_strategy": "D6 · 와이코프 스프링(신저가 개미털기 후 바닥 재탈환)",
                "score": score_d6,
                "reasons": [
                    f"직전 바닥({prior_floor:.5g}) -{sweep_depth_pct:.1f}% 이탈 개미털기 후 거래량 {reclaim_vol_4h:.1f}배 재탈환",
                    f"OI 집중도 {oi_ratio:.0f}% + 펀딩 {funding_pct:+.4f}% 숏함정 역발상 대시세 초입",
                    f"4H 20MA·OBV 반전 + 4H PPO({ppo4_now:+.2f}%) 상방 턴 확인",
                ],
            })

    if not candidates_d:
        return None
    candidates_d.sort(key=lambda x: x["score"], reverse=True)
    return candidates_d[0]


def analyze_symbol(coin, oi_snapshot_prev=None, market_regime=None):
    symbol = str(coin.get("symbol", "")).upper().strip()
    base_coin = coin.get("base_coin") or symbol.replace("USDT", "")
    if is_stock_or_tradfi_symbol(symbol):
        return None

    regime = market_regime if isinstance(market_regime, dict) else {}
    gate_mode = str(regime.get("gate_mode") or ("FULL_LOCK" if regime.get("gate_locked") else "OPEN"))
    if gate_mode == "FULL_LOCK":
        return None
    counter_trend_only = (gate_mode == "COUNTER_TREND_ONLY")
    btc_1h_pct = to_float(regime.get("btc_1h_pct"), 0.0)
    btc_4h_pct = to_float(regime.get("btc_4h_pct"), 0.0)

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

    # [2차 수학적 안전장치] 최근 24시간(15m 96봉) 고저 변동폭이 0.8% 미만인 스테이블/페깅 토큰 영구 차단
    hi_24_chk = to_float(highs.iloc[-96:].max(), price)
    lo_24_chk = to_float(lows.iloc[-96:].min(), price)
    if lo_24_chk <= 0 or ((hi_24_chk - lo_24_chk) / lo_24_chk * 100.0) < 0.8:
        return None

    if abs(change_24h) < 0.05 and len(df_15m) >= 90:
        p_24h = to_float(opens.iloc[-min(len(df_15m), 96)], price)
        if p_24h > 0:
            change_24h = round((price - p_24h) / p_24h * 100.0, 2)

    oi_q = _evaluate_oi_quadrant(price, oi_m, oi_snapshot_prev)
    if oi_q["block"]:
        return None

    sess_adj, sess_note = _get_session_adjustment()
    now_kst = datetime.now(KST)
    kst_blk = f"{(now_kst.hour // 3) * 3:02d}-{(now_kst.hour // 3) * 3 + 3:02d}"

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

    ppo15_s, ppo15_sig_s, ppo15_hist_s = _calc_ppo(closes, 12, 26, 9)
    ppo15_now = round(to_float(ppo15_s.iloc[-1], 0.0), 2)
    ppo15_sig_now = to_float(ppo15_sig_s.iloc[-1], 0.0)
    ppo15_hist_now = to_float(ppo15_hist_s.iloc[-1], 0.0)
    ppo15_hist_prev = to_float(ppo15_hist_s.iloc[-2], 0.0)
    min_ppo15_6 = to_float(ppo15_s.iloc[-6:].min(), 0.0)
    ppo15_turn_up = ppo15_hist_now > ppo15_hist_prev
    ppo15_bull = (ppo15_now > ppo15_sig_now) and (ppo15_hist_now > 0.0)

    vol_ma20 = vols.rolling(20, min_periods=5).mean().replace(0, np.nan)
    vol_ratios = (vols / vol_ma20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recent_vr = vol_ratios.iloc[-6:]
    vol_mult = round(float(recent_vr.max()), 2) if (not recent_vr.empty and float(recent_vr.max()) > 0.0) else 1.0
    curr_vol_ratio = round(to_float(vol_ratios.iloc[-1], 1.0), 2)

    c_1h = df_1h["close"].astype(float)
    ppo1h_s, ppo1h_sig_s, ppo1h_hist_s = _calc_ppo(c_1h, 12, 26, 9)
    ppo1h_now = to_float(ppo1h_s.iloc[-1], 0.0)
    ppo1h_sig_now = to_float(ppo1h_sig_s.iloc[-1], 0.0)
    ppo1h_hist_now = to_float(ppo1h_hist_s.iloc[-1], 0.0)
    ppo1h_hist_prev = to_float(ppo1h_hist_s.iloc[-2], 0.0)
    ppo1h_bull = (ppo1h_now > ppo1h_sig_now) or (ppo1h_now > 0.0 and ppo1h_hist_now >= ppo1h_hist_prev)

    ma20_1h = to_float(c_1h.rolling(20, min_periods=5).mean().iloc[-1], price)
    ma50_1h = to_float(c_1h.rolling(50, min_periods=10).mean().iloc[-1], price)
    dist_1h_ma20 = round(((price - ma20_1h) / max(ma20_1h, 1e-8)) * 100.0, 2)
    htf_bull = price >= ma20_1h * 0.995
    htf_aligned = price >= ma20_1h >= ma50_1h * 0.992
    htf_status = "1H 정배열지지" if htf_aligned else ("1H 중심선지지" if htf_bull else "1H 눌림목지지")
    oi_ratio = round((oi_m / max(turnover_m, 0.1)) * 100.0, 1)

    extra_meta = {
        "funding_pct": funding_pct,
        "change_24h": round(change_24h, 2),
        "oi_ratio": oi_ratio,
        "btc_1h_pct": btc_1h_pct,
        "kst_block": kst_blk,
    }

    def _get_pm_penalty(tk_name):
        if postmortem is not None and hasattr(postmortem, "get_setup_penalty"):
            try:
                return int(postmortem.get_setup_penalty(symbol, tk_name, rsi6_15m, vol_mult, extra_meta=extra_meta))
            except TypeError:
                try:
                    return int(postmortem.get_setup_penalty(symbol, tk_name, rsi6_15m, vol_mult))
                except Exception:
                    return 0
            except Exception:
                return 0
        return 0

    cached_1d_1w = {"fetched": False, "df_1d": None, "df_1w": None}

    def _get_1d_1w():
        if not cached_1d_1w["fetched"]:
            d1, w1 = _fetch_1d_and_1w(symbol, limit_1d=180, limit_1w=60)
            cached_1d_1w.update({"fetched": True, "df_1d": d1, "df_1w": w1})
        return cached_1d_1w["df_1d"], cached_1d_1w["df_1w"]

    # [1순위] 장기 포지션 (Track D1~D6) — 24h +22% 기급등 추격 차단 + 4H/1D/1W PPO 검증
    if (
        (not counter_trend_only)
        and oi_m >= 0.4
        and turnover_m >= 2.5
        and change_24h < 22.0
        and rsi6_15m < 82.0
        and ma20_disp_pct < 8.5
    ):
        df_4h = fetch_klines(symbol, interval="240", limit=180)
        macro_d = _evaluate_macro_track_d(price, change_24h, turnover_m, oi_m, oi_ratio, funding_pct, df_4h)
        if macro_d is not None:
            df_1d, df_1w = _get_1d_1w()
            mtf_long = _evaluate_weekly_daily_pro_filter(
                price,
                df_1d,
                df_1w,
                mode="LONG",
                track_type="TRACK_D",
                sub_strategy=macro_d.get("sub_strategy", ""),
                funding_pct=funding_pct,
            )
            if mtf_long["allow"]:
                d_score = int(np.clip(
                    macro_d["score"] + mtf_long["score_adj"] + (5 if oi_q["quadrant"] == "Q1" else 0) + max(0, sess_adj // 2),
                    0, 99,
                ))
                d_score = max(0, d_score - _get_pm_penalty("TRACK_D"))
                if d_score >= ELITE_SCORE_THRESHOLD:
                    tps, tp_pcts, tp_labels = _calculate_macro_longterm_tps(price, turnover_m, df_4h, df_1d, df_1w=df_1w)
                    stop_price = round(price * (1.0 - LONGTERM_SL_PCT / 100.0), 6)
                    reasons = [f"장기 전략: {macro_d['sub_strategy']}"] + macro_d["reasons"][:2] + mtf_long["reasons"][:1]
                    if oi_q["note"]:
                        reasons.append(oi_q["note"])
                    htf_long_str = f"{mtf_long['htf_tag']} ({htf_status})" if mtf_long["htf_tag"] else f"4H·1D 매집구조 ({htf_status})"
                    is_d6 = "D6" in str(macro_d.get("sub_strategy", "")).upper()
                    d_title = (
                        "🌀 [장기 포지션 · 와이코프 스프링 50배 역발상 초입]"
                        if is_d6
                        else "🐋 [장기 포지션 · 세력 대시세 매집 초입]"
                    )
                    return {
                        "symbol": symbol,
                        "base_coin": base_coin,
                        "track_type": "TRACK_D",
                        "is_long_term": True,
                        "position_horizon": "장기 포지션 (1개월 이상 홀딩 예상)",
                        "sub_strategy": macro_d["sub_strategy"],
                        "track_title": d_title,
                        "score": d_score,
                        "price": price,
                        "change_24h": round(change_24h, 2),
                        "turnover_m": round(turnover_m, 1),
                        "oi_m": round(oi_m, 1),
                        "oi_ratio": oi_ratio,
                        "funding_pct": funding_pct,
                        "btc_1h_pct": btc_1h_pct,
                        "htf_status": htf_long_str,
                        "rsi6_15m": rsi6_15m,
                        "kdj_j_15m": kdj_j_15m,
                        "ppo_15m": ppo15_now,
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

    # 단기·스윙 과열 추격 차단
    if rsi6_15m >= 78.0 or ma20_disp_pct >= 6.0:
        return None

    obv = (np.sign(closes.diff().fillna(0.0)) * vols).cumsum()
    obv_strong = bool(to_float(obv.iloc[-1]) >= to_float(obv.rolling(10, min_periods=3).mean().iloc[-1]))
    ttm_fired, was_squeezed, cvd_absorption, recent_delta_sum, curr_bar_delta = _calc_ttm_squeeze_and_cvd(df_15m)

    curr_o, curr_c = to_float(opens.iloc[-1], price), to_float(closes.iloc[-1], price)
    curr_h, curr_l = to_float(highs.iloc[-1], price), to_float(lows.iloc[-1], price)
    prev_o, prev_c = to_float(opens.iloc[-2], price), to_float(closes.iloc[-2], price)
    prev_h, prev_l = to_float(highs.iloc[-2], price), to_float(lows.iloc[-2], price)

    swing_high = to_float(highs.iloc[-48:-1].max(), price * 1.03)
    swing_low = to_float(lows.iloc[-48:-1].min(), price * 0.95)
    fib_retracement = (swing_high - price) / max(swing_high - swing_low, price * 0.01)
    pullback_from_high_pct = ((swing_high - price) / max(swing_high, 1e-8)) * 100.0

    supply_score = (10 if oi_ratio >= 18.0 else (7 if oi_ratio >= 8.0 else 4)) + (10 if funding_pct < -0.005 else (7 if funding_pct <= 0.012 else 3))
    base_supply_reasons = []
    if oi_ratio >= 18.0:
        base_supply_reasons.append(f"OI 집중도 최상({oi_ratio:.0f}%)")
    if funding_pct < -0.005:
        base_supply_reasons.append(f"음수 펀딩 숏스퀴즈({funding_pct:+.4f}%)")

    wave_bonus, wave_note = 0, ""
    if wave_engine is not None and hasattr(wave_engine, "evaluate_wave_bonus"):
        try:
            wb, wn = wave_engine.evaluate_wave_bonus(df_15m, df_1h)
            wave_bonus, wave_note = int(wb), str(wn or "")
        except Exception:
            pass

    short_candidates = []

    def _register_candidate(tk, title, tech_sc, tk_reasons):
        tot_sc = int(np.clip(tech_sc + supply_score + oi_q["bonus"] + sess_adj + wave_bonus, 0, 99))
        tot_sc = max(0, tot_sc - _get_pm_penalty(tk))
        if tot_sc >= (ELITE_SCORE_THRESHOLD - 3):
            r_list = list(tk_reasons) + list(base_supply_reasons)
            if oi_q["note"]:
                r_list.append(oi_q["note"])
            if sess_note:
                r_list.append(sess_note)
            if wave_note:
                r_list.append(wave_note)
            short_candidates.append({"track_type": tk, "track_title": title, "score": tot_sc, "reasons": r_list})

    # [Track E] 🧲 유동성 스윕 바닥 반전 (SFP) + 15m PPO 강세 다이버전스 통합
    if len(lows) >= 40:
        span_sfp = min(len(lows) - 3, 96)
        prior_low_24h = to_float(lows.iloc[-span_sfp - 3 : -3].min(), price)
        sweep_win = df_15m.iloc[-3:]
        sweep_min_l = to_float(sweep_win["low"].min(), price)
        max_wick_3 = 0.0
        for _, rw in sweep_win.iterrows():
            r_o, r_c, r_h, r_l = to_float(rw["open"]), to_float(rw["close"]), to_float(rw["high"]), to_float(rw["low"])
            w_ratio = (min(r_o, r_c) - r_l) / max(r_h - r_l, 1e-10)
            if w_ratio > max_wick_3:
                max_wick_3 = w_ratio

        swept_support = sweep_min_l < prior_low_24h * 0.9985
        reclaimed_support = curr_c > prior_low_24h * 1.001 and curr_c >= curr_o
        prior_low_idx = int(np.argmin(lows.iloc[-span_sfp - 3 : -3].values))
        prior_abs_idx = len(lows) - span_sfp - 3 + prior_low_idx
        prior_low_rsi6 = to_float(rsi6_s.iloc[prior_abs_idx], 50.0)
        prior_low_ppo15 = to_float(ppo15_s.iloc[prior_abs_idx], 0.0)
        rsi_bull_div = min_rsi6_6 >= prior_low_rsi6 + 3.0
        ppo_bull_div = (min_ppo15_6 >= prior_low_ppo15 + 0.15) and ppo15_turn_up
        kdj_sharp_turn = min_kdj_j_6 <= 20.0 and kdj_j_15m > kdj_j_prev and rsi6_15m > rsi6_prev

        if swept_support and reclaimed_support and max_wick_3 >= 0.40 and vol_mult >= 2.0 and (ppo_bull_div or rsi_bull_div or kdj_sharp_turn):
            sc_e = (
                35
                + (18 if max_wick_3 >= 0.50 else 13)
                + (15 if vol_mult >= 2.6 else 11)
                + (15 if (ppo_bull_div and (rsi_bull_div or kdj_sharp_turn)) else 10)
                + (10 if (recent_delta_sum > 0 or obv_strong) else 5)
                + (4 if counter_trend_only else 0)
            )
            div_lbl = "PPO·RSI 강세 다이버전스" if ppo_bull_div else ("RSI 강세 다이버전스" if rsi_bull_div else "KDJ 바닥 급반전")
            _register_candidate(
                "TRACK_E",
                "🧲 [Track E · 유동성 스윕(SFP) 바닥 반전 타점]",
                sc_e,
                [
                    f"24H 저점 유동성 스윕 후 지지선 탈환(아래꼬리 {max_wick_3 * 100:.0f}%)",
                    f"바닥 반전 거래량 {vol_mult:.1f}배 폭발 + {div_lbl}",
                ],
            )

    # [Track F] ⚡ BTC 역행 상대강도 대장주 (RS Leader) + 1H PPO 강세 필수
    if len(closes) >= 20 and (-2.4 <= btc_4h_pct <= 0.4) and ppo1h_bull:
        p_4h_ago = to_float(closes.iloc[-17], price)
        coin_4h_pct = ((price - p_4h_ago) / max(p_4h_ago, 1e-8)) * 100.0
        rs_vs_btc = coin_4h_pct - btc_4h_pct
        obv_24h_hi = to_float(obv.iloc[-min(len(obv) - 1, 48):-1].max(), -1e18)
        obv_break_f = to_float(obv.iloc[-1]) >= obv_24h_hi * 0.995 and obv_strong

        if rs_vs_btc >= 4.5 and coin_4h_pct >= 3.0 and htf_aligned and obv_break_f and oi_ratio >= 10.0 and rsi6_15m < 74.0 and ma20_disp_pct < 5.5 and curr_c >= curr_o:
            sc_f = (
                (36 if rs_vs_btc >= 7.0 else 31)
                + 18
                + (15 if obv_break_f else 8)
                + (14 if vol_mult >= 1.8 else 9)
                + (10 if recent_delta_sum > 0 else 5)
                + (4 if counter_trend_only else 0)
            )
            _register_candidate(
                "TRACK_F",
                "⚡ [Track F · BTC 역행 상대강도(RS) 대장주 타점]",
                sc_f,
                [
                    f"BTC 4H({btc_4h_pct:+.1f}%) 대비 상대강도 +{rs_vs_btc:.1f}% 독주 강세",
                    f"1H 정배열·PPO({ppo1h_now:+.2f}%) 강세 + OBV 신고가 + OI {oi_ratio:.0f}%",
                ],
            )

    # [Track A·B·C·G + Track H] — BTC 정상 구간(OPEN)에서만 탐색
    if not counter_trend_only:
        # [Track G] 💥 숏스퀴즈 강제청산 초입 + 15m PPO 상방 턴 필수
        if funding_pct <= -0.020 and oi_ratio >= 18.0 and (ppo15_turn_up or ppo15_bull):
            higher_low_g = to_float(lows.iloc[-6:].min(), price) >= to_float(lows.iloc[-18:-6].min(), price) * 0.998
            short_res_g = to_float(highs.iloc[-13:-1].max(), price)
            break_res_g = curr_c > curr_o and curr_c > ma20_15m_val and curr_c >= short_res_g * 0.996
            if higher_low_g and break_res_g and recent_delta_sum > 0 and curr_bar_delta > 0 and vol_mult >= 1.5:
                sc_g = (
                    (36 if funding_pct <= -0.040 else 32)
                    + (16 if oi_ratio >= 25.0 else 12)
                    + (16 if htf_bull else 8)
                    + (14 if vol_mult >= 2.0 else 10)
                    + (10 if obv_strong else 5)
                )
                _register_candidate(
                    "TRACK_G",
                    "💥 [Track G · 숏스퀴즈 강제청산 초입 타점]",
                    sc_g,
                    [
                        f"깊은 음수 펀딩({funding_pct:+.4f}%) + 고OI({oi_ratio:.0f}%) 숏 쏠림 임계",
                        f"저점 상향 후 20MA·저항 돌파 + PPO({ppo15_now:+.2f}%)·CVD 매수 폭발",
                    ],
                )

        # [Track B] 24시간 스윙 구조 돌파 + 종가 안착 / S-R Flip + 15m PPO 강세 필수
        span_24h = min(len(highs) - 2, 96)
        high_24h = to_float(highs.iloc[-span_24h - 2 : -2].max(), price * 1.05)
        close_confirmed_b = (prev_c >= high_24h * 0.998) and (curr_c >= high_24h * 0.996) and (curr_c >= curr_o)
        had_break_recent = to_float(highs.iloc[-5:-1].max(), 0.0) >= high_24h * 1.001
        sr_flip_retest_b = (
            had_break_recent
            and (min(curr_l, prev_l) <= high_24h * 1.008)
            and (min(curr_l, prev_l) >= high_24h * 0.985)
            and (curr_c >= high_24h * 0.998)
            and (curr_c >= curr_o)
        )
        is_breakout_b = (close_confirmed_b or sr_flip_retest_b) and vol_mult >= 2.0 and change_24h >= 3.0 and (ppo15_bull or ppo15_turn_up)

        if is_breakout_b:
            flip_lbl = "S-R Flip 리테스트 지지" if sr_flip_retest_b else "24H 전고점 종가 안착 돌파"
            sc_b = (
                (35 if vol_mult >= 2.8 else 30)
                + (18 if htf_bull else 0)
                + (14 if obv_strong else 0)
                + (13 if (curr_c > ma20_15m_val and to_float(ma7.iloc[-1]) >= ma20_15m_val) else 0)
                + (8 if 48.0 <= rsi6_15m <= 74.0 else 0)
                + (4 if sr_flip_retest_b else 2)
            )
            _register_candidate(
                "TRACK_B",
                f"🔥 [Track B · 24H 구조 돌파 ({'S-R Flip' if sr_flip_retest_b else '종가 안착'})]",
                sc_b,
                [f"{flip_lbl} + 거래량 {vol_mult:.1f}배 · 15m PPO({ppo15_now:+.2f}%) 확장"],
            )

        # [Track C] 급등 대장주 심층 눌림목 + 거래량 건조(<= 35%) + 5m MSS + 15m PPO 바닥 턴
        if change_24h >= 12.0 and not is_breakout_b:
            if 0.48 <= fib_retracement <= 0.78 and pullback_from_high_pct >= 6.5 and price >= swing_low * 1.015:
                touched_ma20_flush = to_float(lows.iloc[-6:].min(), price) <= ma20_15m_val * 0.995
                had_flush = (min_rsi6_6 <= 38.0) or (min_kdj_j_6 <= 22.0) or touched_ma20_flush
                if had_flush and ppo15_turn_up:
                    win_48_h = highs.iloc[-48:-2]
                    peak_rel_idx = int(np.argmax(win_48_h.values))
                    peak_abs_idx = len(df_15m) - 48 + peak_rel_idx
                    wave1_vols = vols.iloc[max(0, peak_abs_idx - 8) : peak_abs_idx + 1]
                    avg_v_wave1 = max(to_float(wave1_vols.mean(), 1.0), to_float(vols.iloc[-48:-3].nlargest(5).mean(), 1.0) * 0.75)
                    wave2_vols = vols.iloc[peak_abs_idx + 1 : -1] if (peak_abs_idx + 3) < len(vols) else vols.iloc[-4:-1]
                    avg_v_wave2 = to_float(wave2_vols.mean(), avg_v_wave1)
                    vol_dry_ratio = avg_v_wave2 / max(avg_v_wave1, 1e-8)

                    if vol_dry_ratio <= 0.35:
                        curr_lwick = (min(curr_o, curr_c) - curr_l) / max(curr_h - curr_l, 1e-10)
                        prev_lwick = (min(prev_o, prev_c) - prev_l) / max(prev_h - prev_l, 1e-10)
                        max_lwick = max(curr_lwick, prev_lwick)
                        is_bullish_turn = (curr_c > curr_o) or (curr_lwick >= 0.45 and curr_c >= prev_l)
                        is_momentum_turn = (kdj_j_15m > kdj_j_prev) and (rsi6_15m > rsi6_prev) and (rsi6_15m <= 60.0)
                        is_holding_low = curr_c >= to_float(lows.iloc[-4:].min(), curr_l) * 1.003

                        if is_bullish_turn and is_momentum_turn and is_holding_low:
                            mss_ok = _check_5m_mss_trigger(symbol)
                            if mss_ok is True or (mss_ok is None and curr_c > curr_o and curr_c > prev_o):
                                sc_c = (
                                    (34 if 0.52 <= fib_retracement <= 0.72 else 29)
                                    + (18 if price >= ma20_1h * 0.985 else (14 if price >= ma50_1h * 0.985 else 8))
                                    + (14 if obv_strong else 11)
                                    + (14 if (min_rsi6_6 <= 30.0 or max_lwick >= 0.35) else 10)
                                    + (10 if mss_ok else 6)
                                    + (3 if oi_q["quadrant"] == "Q4" else 0)
                                )
                                _register_candidate(
                                    "TRACK_C",
                                    "🚀 [Track C · 대장주 심층 눌림목(5m MSS 전환) 반등 타점]",
                                    sc_c,
                                    [
                                        f"대장주(+{change_24h:.1f}%) 고점대비 -{pullback_from_high_pct:.1f}% · Fib {fib_retracement:.2f} 심층 눌림",
                                        f"5m MSS 반등 + 거래량 건조({vol_dry_ratio * 100:.0f}%) + 15m PPO 상방 턴",
                                    ],
                                )

        # [Track A] TTM Squeeze Fired + 델타(CVD) 흡수 + 15m PPO 상방 확장
        if change_24h < 12.0 and not is_breakout_b and (ppo15_turn_up or ppo15_bull):
            min_bw = to_float(bb_width_pct.iloc[-10:-1].min(), 5.0)
            if ttm_fired or (was_squeezed and cvd_absorption) or (min_bw <= 4.5 and cvd_absorption):
                sc_a = (
                    (35 if ttm_fired else (28 if was_squeezed else 22))
                    + (18 if (vol_mult >= 1.9 and curr_c > curr_o) else (12 if curr_vol_ratio >= 1.3 else 0))
                    + (15 if htf_bull else 0)
                    + (15 if (cvd_absorption and obv_strong) else (10 if obv_strong else 0))
                    + (10 if (kdj_j_15m > kdj_j_prev and 35.0 <= rsi6_15m <= 70.0) else 0)
                )
                r_a = [
                    f"TTM Squeeze(BB<KC) 발사(응축폭 {min_bw:.1f}%) + 15m PPO({ppo15_now:+.2f}%) 상방"
                    if ttm_fired
                    else f"볼린저/켈트너 초응축({min_bw:.1f}%) + 15m PPO({ppo15_now:+.2f}%) 전환"
                ]
                if cvd_absorption:
                    r_a.append("15m 매수 델타(CVD)·OBV 바닥 흡수 확인")
                _register_candidate(
                    "TRACK_A",
                    "💎 [Track A · TTM 스퀴즈 발사 & 델타 흡수 초입]" if ttm_fired else "💎 [Track A · 박스권 초응축 & 델타 매집 초입]",
                    sc_a,
                    r_a,
                )

        # [Track H] 🧬 Top 5 급등 DNA 자동 적응 트랙 + 15m PPO 상방 모멘텀 검증
        if (
            not short_candidates
            and htf_bull
            and curr_c >= ma20_15m_val
            and (-5.0 <= change_24h < 18.0)
            and rsi6_15m < 76.0
            and kdj_j_15m < 85.0
            and ma20_disp_pct < 5.5
            and (ppo15_turn_up or ppo15_bull)
        ):
            dna_f = _load_active_top5_dna_formula()
            if dna_f:
                min_bw_h = to_float(bb_width_pct.iloc[-10:-1].min(), 8.0)
                v_pre_rec = to_float(vols.iloc[-7:-1].mean(), 1.0)
                v_pre_old = to_float(vols.iloc[-25:-7].mean(), v_pre_rec)
                pre_dry_h = v_pre_rec / max(v_pre_old, 1e-8)

                req_lvol = max(1.6, to_float(dna_f.get("launch_vol_min"), 2.0))
                has_launch_trigger = (curr_c > curr_o) and (curr_c > prev_c) and (vol_mult >= req_lvol) and (recent_delta_sum > 0)

                if has_launch_trigger:
                    checks = [
                        ("응축", min_bw_h <= to_float(dna_f.get("bb_width_max"), 10.0)),
                        ("거래량건조", pre_dry_h <= to_float(dna_f.get("vol_dry_max"), 1.0)),
                        ("직전RSI", to_float(dna_f.get("pre_rsi_min"), 25.0) <= rsi6_prev <= to_float(dna_f.get("pre_rsi_max"), 68.0)),
                        ("1H이격", to_float(dna_f.get("ma20_1h_dist_min"), -5.0) <= dist_1h_ma20 <= to_float(dna_f.get("ma20_1h_dist_max"), 6.0)),
                        ("OI비중", oi_ratio >= to_float(dna_f.get("oi_ratio_min"), 8.0)),
                        ("펀딩비", funding_pct <= to_float(dna_f.get("funding_max"), 0.012)),
                    ]
                    matched_names = [nm for nm, ok in checks if ok]
                    match_cnt = len(matched_names)

                    if match_cnt >= 5:
                        sc_h = (
                            (38 if match_cnt == 6 else 33)
                            + (18 if vol_mult >= req_lvol * 1.25 else 13)
                            + (14 if obv_strong else 8)
                            + (13 if htf_aligned else 10)
                            + (10 if (kdj_j_15m > kdj_j_prev and rsi6_15m > rsi6_prev) else 5)
                        )
                        dom_tag = str(dna_f.get("dominant_tag") or "Top 5 급등 DNA")
                        _register_candidate(
                            "TRACK_H",
                            f"🧬 [Track H · Top 5 급등 DNA 자동학습 포착 ({dom_tag})]",
                            sc_h,
                            [
                                f"최근 14일 Top 5 발사직전 DNA {match_cnt}/6 일치 ({', '.join(matched_names[:4])})",
                                f"1H·15m 20MA 안착 + 발사 거래량({vol_mult:.1f}x) + PPO({ppo15_now:+.2f}%) 상방",
                            ],
                        )

    if not short_candidates:
        return None

    # [전문 트레이더 1W(주봉) · 1D(일봉) PPO 최종 관문 검증]
    df_1d, df_1w = _get_1d_1w()
    verified_candidates = []
    for cand in short_candidates:
        mtf_s = _evaluate_weekly_daily_pro_filter(
            price,
            df_1d,
            df_1w,
            mode="SHORT",
            track_type=cand["track_type"],
            funding_pct=funding_pct,
        )
        if not mtf_s["allow"]:
            continue
        final_sc = int(np.clip(cand["score"] + mtf_s["score_adj"], 0, 99))
        if final_sc < ELITE_SCORE_THRESHOLD:
            continue
        merged_reasons = list(cand["reasons"]) + list(mtf_s["reasons"])
        cand["score"] = final_sc
        cand["reasons"] = merged_reasons[:4]
        cand["htf_tag"] = mtf_s["htf_tag"]
        verified_candidates.append(cand)

    if not verified_candidates:
        return None

    verified_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = verified_candidates[0]

    tp_scale = 0.88 if oi_q["squeeze_warn"] else 1.0
    tps, tp_pcts, tp_labels = _calculate_resistance_based_tps(price, df_15m, df_1h, conservative_scale=tp_scale)
    stop_price = round(price * (1.0 - FIXED_SL_PCT / 100.0), 6)
    final_htf_status = f"{best['htf_tag']} · {htf_status}" if best.get("htf_tag") else htf_status

    return {
        "symbol": symbol,
        "base_coin": base_coin,
        "track_type": best["track_type"],
        "is_long_term": False,
        "position_horizon": "단기·스윙 포지션",
        "track_title": best["track_title"],
        "score": best["score"],
        "price": price,
        "change_24h": round(change_24h, 2),
        "turnover_m": round(turnover_m, 1),
        "oi_m": round(oi_m, 1),
        "oi_ratio": oi_ratio,
        "funding_pct": funding_pct,
        "btc_1h_pct": btc_1h_pct,
        "htf_status": final_htf_status,
        "rsi6_15m": rsi6_15m,
        "kdj_j_15m": kdj_j_15m,
        "ppo_15m": ppo15_now,
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
        "reasons": best["reasons"][:4],
        "reason_str": " / ".join(best["reasons"][:4]),
    }
# === END OF FILE ===
