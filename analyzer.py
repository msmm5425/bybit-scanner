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
    from api import get_market_cap_m, fetch_smart_money_ls_ratio
except ImportError:
    def get_market_cap_m(symbol_or_base):
        return 0.0

    def fetch_smart_money_ls_ratio(symbol):
        return {"source": "NONE", "top_ls": 0.0, "retail_ls": 0.0, "divergence": 0.0}

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
_DYNAMIC_TP_CACHE = {"ts": 0, "params": None}


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
    now = time.time()
    if _DYNAMIC_TP_CACHE["params"] is not None and (now - _DYNAMIC_TP_CACHE["ts"]) < 600:
        return _DYNAMIC_TP_CACHE["params"]
    default_params = {"tp1_min_pct": 8.0, "tp2_min_pct": 15.0, "tp3_min_pct": 22.0}
    res_params = default_params
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
                    res_params = {"tp1_min_pct": round(p1, 1), "tp2_min_pct": round(p2, 1), "tp3_min_pct": round(p3, 1)}
        except Exception:
            pass
    _DYNAMIC_TP_CACHE.update({"ts": now, "params": res_params})
    return res_params


def _get_session_adjustment():
    """정규화된 글로벌 유동성 세션 가감점 (-3 ~ +2점)"""
    now_kst = datetime.now(KST)
    hm = now_kst.hour + (now_kst.minute / 60.0)
    if 15.5 <= hm <= 18.5:
        return 2, "런던 유동성 세션(+2점)"
    if hm >= 21.0 or hm <= 0.5:
        return 2, "뉴욕 유동성 세션(+2점)"
    if 11.5 <= hm <= 14.5:
        return -3, ""
    return 0, ""


def _evaluate_oi_quadrant(price, oi_m, oi_snapshot_prev):
    """정규화된 OI × 가격 델타 4분면 (Q1 +2~+4점 / Q2 보수적 TP / Q3 진입 차단)"""
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
        bonus = 4 if doi_pct >= 5.0 else 2
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


def _evaluate_supply_overlay(symbol, oi_m, turnover_m, market_cap_m, ls_info=None):
    """
    [방안 1 · 전 전략 공통 수급 오버레이 (-6 ~ +5점 캡)]
    1) Bybit 단일 스케일 보정 OI/시총(화약고 압축도) & 거래대금/시총(회전율) 가산
    2) Binance · Gate.io · Bitget 통합 고래 포지션(S) vs 개미 계정(T) 롱숏 괴리 가감점
    """
    out = {
        "score_adj": 0,
        "market_cap_m": round(to_float(market_cap_m, 0.0), 2),
        "oi_mc_pct": 0.0,
        "turn_mc_pct": 0.0,
        "ls_source": "NONE",
        "top_ls": 0.0,
        "retail_ls": 0.0,
        "ls_divergence": 0.0,
        "reasons": [],
    }

    adj = 0
    mc_m = to_float(market_cap_m, 0.0)

    # 1) 시총 대비 Bybit OI 압축도 & 회전율 (시총 1M 이상 유효 시총만 적용)
    if mc_m >= 1.0:
        oi_mc_pct = round((to_float(oi_m, 0.0) / mc_m) * 100.0, 1)
        turn_mc_pct = round((to_float(turnover_m, 0.0) / mc_m) * 100.0, 1)
        out["oi_mc_pct"] = oi_mc_pct
        out["turn_mc_pct"] = turn_mc_pct

        if oi_mc_pct >= 15.0:
            adj += 3
            out["reasons"].append(f"Bybit OI/시총 {oi_mc_pct:.1f}% 고압 화약고 압축(+3점)")
        elif oi_mc_pct >= 8.0:
            adj += 2
            out["reasons"].append(f"Bybit OI/시총 {oi_mc_pct:.1f}% 레버리지 축적(+2점)")

        if turn_mc_pct >= 25.0 and oi_mc_pct >= 6.0:
            adj += 1

    # 2) 스마트머니(고래 포지션 S) vs 개미(계정 T) 롱숏 괴리 평가 (Binance · Gate · Bitget 통합)
    ls = ls_info if isinstance(ls_info, dict) else {}
    src = str(ls.get("source") or "NONE").upper()
    s_val = to_float(ls.get("top_ls"), 0.0)
    t_val = to_float(ls.get("retail_ls"), 0.0)
    div = to_float(ls.get("divergence"), 0.0)
    if div <= 0 and s_val > 0 and t_val > 0:
        div = round(s_val / max(t_val, 0.05), 2)

    out["ls_source"] = src
    out["top_ls"] = round(s_val, 2)
    out["retail_ls"] = round(t_val, 2)
    out["ls_divergence"] = round(div, 2)

    src_lbl = {"BINANCE": "바이낸스", "GATE": "Gate", "BITGET": "Bitget"}.get(src, src)
    if src in ("BINANCE", "GATE", "BITGET") and s_val > 0 and t_val > 0:
        if s_val >= 1.50 and t_val <= 1.05 and div >= 1.60:
            adj += 3
            out["reasons"].append(f"{src_lbl} 고래 롱({s_val:.2f}) vs 개미 숏({t_val:.2f}) 괴리 {div:.2f}배(+3점)")
        elif s_val >= 1.30 and div >= 1.35:
            adj += 2
            out["reasons"].append(f"{src_lbl} 스마트머니 롱 우위(고래 {s_val:.2f} / 개미 {t_val:.2f}, +2점)")
        elif (s_val <= 1.05 and t_val >= 2.20) or (s_val < 0.90 and t_val >= 1.80) or (t_val >= 3.00 and div <= 0.65):
            adj -= 5
            out["reasons"].append(f"⚠️ 개미 롱 쏠림 함정 경계(고래 {s_val:.2f} vs 개미 {t_val:.2f}, -5점)")
        elif s_val <= 1.15 and t_val >= 1.80 and div <= 0.65:
            adj -= 3
            out["reasons"].append(f"⚠️ 고래 관망·개미 롱 편중(고래 {s_val:.2f} vs 개미 {t_val:.2f}, -3점)")
    elif src in ("GATE", "BITGET", "BYBIT", "OKX") and t_val > 0:
        if t_val <= 0.75:
            adj += 2
            out["reasons"].append(f"{src_lbl} 리테일 숏 쏠림(L/S {t_val:.2f}, +2점)")
        elif t_val >= 2.80:
            adj -= 4
            out["reasons"].append(f"⚠️ {src_lbl} 리테일 극단 롱 쏠림(L/S {t_val:.2f}, -4점)")
        elif t_val >= 2.10:
            adj -= 2
            out["reasons"].append(f"⚠️ {src_lbl} 리테일 롱 과밀 경계(L/S {t_val:.2f}, -2점)")

    out["score_adj"] = int(np.clip(adj, -6, 5))
    return out


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
    """비율 오실레이터 PPO (12, 26, 9 EMA) — 선형 MACD 가격 압축 착시 제거"""
    s = series.astype(float)
    ema_fast = s.ewm(span=fast, adjust=False, min_periods=min(fast, 4)).mean()
    ema_slow = s.ewm(span=slow, adjust=False, min_periods=min(slow, 6)).mean()
    ppo = ((ema_fast - ema_slow) / ema_slow.replace(0, np.nan)) * 100.0
    ppo = ppo.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sig = ppo.ewm(span=signal, adjust=False, min_periods=min(signal, 3)).mean().fillna(0.0)
    hist = (ppo - sig).fillna(0.0)
    return ppo, sig, hist


def _calc_shifted_vol_ratios(vols, window=20):
    """직전 N봉 평균(shift 1) 대비 거래량 배수 산출 (현재 폭발 봉의 자기 평균 왜곡 제거)"""
    raw_ma = vols.rolling(window, min_periods=5).mean()
    prev_ma = raw_ma.shift(1).fillna(raw_ma).replace(0, np.nan)
    return (vols / prev_ma).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _is_healthy_bull_candle(o, h, l, c, max_uwick=0.45, min_clv=0.0, min_body_pct=0.15):
    """윗꼬리 피뢰침(Shooting Star) 및 1틱 초반 노이즈 양봉 원천 차단"""
    if o <= 0 or c <= o:
        return False
    body_pct = ((c - o) / o) * 100.0
    if body_pct < min_body_pct:
        return False
    rng = max(h - l, 1e-12)
    uwick = (h - max(o, c)) / rng
    clv = ((c - l) - (h - c)) / rng
    return (uwick <= max_uwick) and (clv >= min_clv)


def _max_healthy_bull_vol_ratio(
    opens,
    highs,
    lows,
    closes,
    vol_ratios,
    lookback=2,
    strict_gt=True,
    max_uwick=0.45,
    min_clv=0.0,
):
    """최근 lookback 봉 중 '피뢰침이 아닌 건전한 양봉'의 거래량 배수 최대값만 인정"""
    if len(vol_ratios) == 0:
        return 0.0
    o_s = opens.iloc[-lookback:].astype(float)
    h_s = highs.iloc[-lookback:].astype(float)
    l_s = lows.iloc[-lookback:].astype(float)
    c_s = closes.iloc[-lookback:].astype(float)
    vr_s = vol_ratios.iloc[-lookback:].astype(float)

    rng = (h_s - l_s).replace(0, np.nan)
    uwick = ((h_s - np.maximum(o_s, c_s)) / rng).fillna(0.0)
    clv = (((c_s - l_s) - (h_s - c_s)) / rng).fillna(0.0)
    dir_ok = (c_s > o_s) if strict_gt else (c_s >= o_s)
    mask = dir_ok & (uwick <= max_uwick) & (clv >= min_clv)

    bull_vals = vr_s[mask]
    if bull_vals.empty:
        return 0.0
    return round(float(bull_vals.max()), 2)


def _resample_1d_to_1w(df_1d):
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
    """[전문 트레이더 주봉(1W) · 일봉(1D) 탑다운 + PPO + 상단 일봉 20MA 저항벽(Overhead Clean Air) 검증]"""
    out = {"allow": True, "score_adj": 0, "htf_tag": "", "reasons": [], "block_reason": ""}
    is_spring_d6 = "D6" in str(sub_strategy or "").upper()

    has_1d = df_1d is not None and not df_1d.empty and len(df_1d) >= 14
    dist_1d_ma20 = 0.0
    ma20_1d_falling = False
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

        ma20_1d_s = c1d.rolling(20, min_periods=7).mean()
        ma20_1d = to_float(ma20_1d_s.iloc[-1], price)
        ma20_1d_prev3 = to_float(ma20_1d_s.iloc[-min(len(ma20_1d_s), 4)], ma20_1d)
        ma20_1d_falling = ma20_1d < ma20_1d_prev3 * 0.998
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

        out["score_adj"] = int(min(6, out["score_adj"]))
        return out

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

    if has_1d and tk in ("TRACK_A", "TRACK_B", "TRACK_F", "TRACK_G", "TRACK_H"):
        if (-9.5 <= dist_1d_ma20 <= -2.0) and ma20_1d_falling:
            out["score_adj"] -= 3
            if "역배열" not in out["htf_tag"]:
                out["reasons"].append(f"⚠️ 상단 1D 20MA 저항벽({abs(dist_1d_ma20):.1f}% 위) 근접 경계")

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
    c_last, o_last, h_last, l_last = to_float(c5.iloc[-1]), to_float(o5.iloc[-1]), to_float(h5.iloc[-1]), to_float(l5.iloc[-1])
    healthy_5m = _is_healthy_bull_candle(o_last, h_last, l_last, c_last, max_uwick=0.45, min_clv=0.0, min_body_pct=0.10)
    mss_break = c_last > prev_swing_h and healthy_5m

    bull_engulf = (
        healthy_5m
        and to_float(c5.iloc[-2]) < to_float(o5.iloc[-2])
        and c_last >= to_float(o5.iloc[-2])
        and o_last <= to_float(c5.iloc[-2]) * 1.003
    )

    return bool((higher_low and mss_break) or bull_engulf)


def _calculate_resistance_based_tps(price, df_15m, df_1h, conservative_scale=1.0):
    """차트 실제 저항선 + 일간 자율학습 장세 변동성 보조 계수(0.92~1.12) 결합"""
    highs_15 = df_15m["high"].astype(float)
    lows_15 = df_15m["low"].astype(float)
    highs_1h = df_1h["high"].astype(float)
    lows_1h = df_1h["low"].astype(float)

    dyn_tp = get_dynamic_tp_params()
    mkt_vol_mult = float(np.clip(to_float(dyn_tp.get("tp1_min_pct"), 8.0) / 8.0, 0.92, 1.12))

    atr_1h_pct = to_float(((highs_1h - lows_1h) / df_1h["close"].replace(0, np.nan)).tail(24).mean(), 0.025) * 100.0
    vol_scale = float(np.clip((atr_1h_pct / 2.2) * mkt_vol_mult, 0.82, 1.48))

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


def _calculate_macro_longterm_tps(price, turnover_m, oi_m, df_4h, df_1d, df_1w=None, market_cap_m=0.0):
    """[방안 1 보완] 실제 시가총액(market_cap_m) 우선 Tier 분류 (미수신 시 거래대금·OI 프록시 자동 폴백)"""
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

    mc_m = to_float(market_cap_m, 0.0)
    if mc_m >= 1.0:
        if mc_m < 35.0:
            tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier1 초소형확장", 1.32, 145.0, 250.0
        elif mc_m < 180.0:
            tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier2 계단식확장", 1.12, 115.0, 190.0
        else:
            tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier3 중대형추세확장", 0.95, 90.0, 145.0
    else:
        size_proxy = max(to_float(turnover_m, 0.0), to_float(oi_m, 0.0) * 4.5)
        if size_proxy < 12.0:
            tier_lbl, tier_mult, tp3_min_cap, tp3_max_cap = "Tier1 초대형확장", 1.32, 145.0, 250.0
        elif size_proxy < 45.0:
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
    """[장기 포지션 Track D1~D6 정밀 보완 + 영점 캘리브레이션(86~91점)]"""
    if df_4h is None or df_4h.empty or len(df_4h) < 55:
        return None
    if change_24h >= 22.0:
        return None

    c4 = df_4h["close"].astype(float)
    h4 = df_4h["high"].astype(float)
    l4 = df_4h["low"].astype(float)
    o4 = df_4h["open"].astype(float)
    v4 = df_4h["volume"].astype(float).fillna(0.0)

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

    v4_ratio = _calc_shifted_vol_ratios(v4, window=20)
    bull_v4_2 = _max_healthy_bull_vol_ratio(o4, h4, l4, c4, v4_ratio, lookback=2, strict_gt=True, max_uwick=0.45, min_clv=0.0)
    bull_v4_3 = _max_healthy_bull_vol_ratio(o4, h4, l4, c4, v4_ratio, lookback=3, strict_gt=False, max_uwick=0.48, min_clv=-0.05)

    curr_o4, curr_h4, curr_l4, curr_c4 = to_float(o4.iloc[-1]), to_float(h4.iloc[-1]), to_float(l4.iloc[-1]), to_float(c4.iloc[-1])
    prev_o4, prev_h4, prev_l4, prev_c4 = to_float(o4.iloc[-2]), to_float(h4.iloc[-2]), to_float(l4.iloc[-2]), to_float(c4.iloc[-2])
    curr_4h_healthy_bull = _is_healthy_bull_candle(curr_o4, curr_h4, curr_l4, curr_c4, max_uwick=0.45, min_clv=0.0, min_body_pct=0.20)
    prev_4h_healthy_bull = _is_healthy_bull_candle(prev_o4, prev_h4, prev_l4, prev_c4, max_uwick=0.45, min_clv=0.0, min_body_pct=0.25)
    turn_4h_ok = curr_4h_healthy_bull or (prev_4h_healthy_bull and curr_c4 >= curr_o4 * 0.996 and curr_c4 >= prev_c4 * 0.992)

    ma20_4h_s = c4.rolling(20, min_periods=5).mean()
    ma50_4h_s = c4.rolling(50, min_periods=10).mean()
    ma20_4h = to_float(ma20_4h_s.iloc[-1], price)
    ma20_4h_prev3 = to_float(ma20_4h_s.iloc[-4], ma20_4h)
    ma50_4h = to_float(ma50_4h_s.iloc[-1], price)

    candidates_d = []

    # D1: 스텔스 OI·OBV 선행 매집
    hi_10d = to_float(h4.iloc[-60:-1].max(), price * 1.05)
    obv_hi_10d = to_float(obv4.iloc[-60:-1].max(), 0.0)
    v4_win12 = v4.iloc[-12:]
    buy_v4_12 = float(v4_win12[c4.iloc[-12:] >= o4.iloc[-12:]].sum())
    sell_v4_12 = float(v4_win12[c4.iloc[-12:] < o4.iloc[-12:]].sum())
    if -3.0 <= change_24h <= 11.0 and oi_m >= 1.2 and oi_ratio >= 28.0 and funding_pct <= 0.008 and buy_v4_12 >= sell_v4_12:
        if (hi_10d * 0.86 <= price < hi_10d * 0.995) and to_float(obv4.iloc[-1]) > obv_hi_10d and price >= ma20_4h * 0.985 and (ppo4_bull or ppo4_turn_up):
            score_d1 = int(np.clip(86 + min(2, int((oi_ratio - 28) / 8)) + (2 if funding_pct < 0 else 0) + (1 if ppo4_zero_up else 0), 86, 91))
            candidates_d.append({
                "sub_strategy": "D1 · 스텔스 OI·OBV 선행 매집",
                "score": score_d1,
                "vol_metric": max(bull_v4_2, 1.0),
                "reasons": [
                    f"10일 저항선 압착 중 4H OBV 최고치 선행 돌파 + PPO({ppo4_now:+.2f}%) 상방",
                    f"스텔스 OI {oi_m:.1f}M(비중 {oi_ratio:.0f}%) · 최근 2일 매수 거래량 우위",
                    f"펀딩비 {funding_pct:+.4f}% 저과열 바닥 매집 구조",
                ],
            })

    # D2: 양봉/흡수형 매집봉(최소 3봉 전) + 매집봉 후 대량 투매 음봉 차단 + 거래량 건조 눌림
    recent_30 = df_4h.iloc[-30:-3]
    vr_30 = v4_ratio.iloc[-30:-3]
    if not vr_30.empty:
        r30_o = recent_30["open"].astype(float)
        r30_h = recent_30["high"].astype(float)
        r30_l = recent_30["low"].astype(float)
        r30_c = recent_30["close"].astype(float)
        r30_rng = (r30_h - r30_l).replace(0, np.nan)
        r30_lwick = ((np.minimum(r30_o, r30_c) - r30_l) / r30_rng).fillna(0.0)
        r30_uwick = ((r30_h - np.maximum(r30_o, r30_c)) / r30_rng).fillna(0.0)
        valid_pump_mask = (((r30_c >= r30_o) & (r30_uwick <= 0.68)) | (r30_lwick >= 0.45)) & (vr_30 >= 3.8)

        if valid_pump_mask.any():
            valid_vr = vr_30[valid_pump_mask]
            pump_idx_label = valid_vr.idxmax()
            pump_pos_abs = df_4h.index.get_loc(pump_idx_label)
            pump_vr_val = float(valid_vr.loc[pump_idx_label])
            pump_row = df_4h.iloc[pump_pos_abs]
            pump_vol = to_float(pump_row["volume"], 1.0)
            pump_open = to_float(pump_row["open"], price)
            pump_high = to_float(pump_row["high"], price)
            pump_mid = (pump_open + pump_high) * 0.5
            pump_span_pct = ((pump_high - pump_open) / max(pump_open, 1e-8)) * 100.0

            after_df = df_4h.iloc[pump_pos_abs + 1 : -1]
            if len(after_df) >= 2:
                after_red = after_df[after_df["close"].astype(float) < after_df["open"].astype(float)]
                max_red_vol = to_float(after_red["volume"].max(), 0.0) if not after_red.empty else 0.0
                no_heavy_dump = (pump_vol > 0) and (max_red_vol / pump_vol <= 0.65)
                held_pump_base = to_float(after_df["close"].min(), price) >= pump_open * 0.96
                dry_vol_4h = to_float(v4.iloc[-3:-1].mean(), pump_vol)
                is_dry_up = (pump_vol > 0) and (dry_vol_4h / pump_vol <= 0.32)
                is_turn_up = turn_4h_ok and price >= max(pump_open * 0.99, ma20_4h * 0.985) and price <= pump_high * 1.02

                if pump_span_pct >= 2.0 and no_heavy_dump and held_pump_base and is_dry_up and is_turn_up and (ppo4_turn_up or ppo4_zero_up):
                    score_d2 = int(np.clip(86 + (2 if price >= pump_mid else 1) + (2 if oi_ratio >= 15.0 else 0) + (1 if ppo4_zero_up else 0), 86, 91))
                    candidates_d.append({
                        "sub_strategy": "D2 · 매집봉 출현 후 거래량 건조 눌림",
                        "score": score_d2,
                        "vol_metric": round(pump_vr_val, 2),
                        "reasons": [
                            f"최근 5일 내 4H 양봉 매집봉(거래량 {pump_vr_val:.1f}배) 출현·시가 방어",
                            "매집봉 후 투매 이탈 없이 조정 거래량 68%+ 건조",
                            f"4H 20MA 위 건전 양봉 반등 + PPO({ppo4_now:+.2f}%) 상방 턴",
                        ],
                    })

    # D3: 4H 30일 장기 박스권 실제 첫 돌파
    if len(df_4h) >= 120:
        hi_30d = to_float(h4.iloc[:-2].max(), price)
        lo_30d = to_float(l4.iloc[:-2].min(), price * 0.8)
        range_30d_pct = (hi_30d - lo_30d) / max(lo_30d, 1e-8) * 100.0
        was_inside_box = to_float(c4.iloc[-8:-2].max(), hi_30d) < hi_30d * 0.999
        if (
            was_inside_box
            and hi_30d * 1.001 <= price <= hi_30d * 1.055
            and price <= ma20_4h * 1.15
            and turn_4h_ok
            and bull_v4_2 >= 2.4
            and 5.0 <= range_30d_pct <= 55.0
            and ppo4_bull
        ):
            score_d3 = int(np.clip(87 + min(2, int(bull_v4_2 - 1.8)) + (1 if price >= ma50_4h else 0) + (1 if ppo4_zero_up else 0), 87, 91))
            candidates_d.append({
                "sub_strategy": "D3 · 4H 30일 장기 박스권 첫 돌파",
                "score": score_d3,
                "vol_metric": bull_v4_2,
                "reasons": [
                    f"4H 30일 장기 박스권 첫 돌파 안착 + PPO({ppo4_now:+.2f}%) {'0선 돌파' if ppo4_zero_up else '강세확장'}",
                    f"4H 돌파 건전 양봉 거래량 평균 대비 {bull_v4_2:.1f}배 폭발",
                    f"장기 박스권 응축폭 {range_30d_pct:.1f}% 소화 후 1파동 개시",
                ],
            })

    # D4: 4H 장기 하락추세 탈출 & 50MA 첫 돌파
    if len(df_4h) >= 85:
        hi_30d_d4 = to_float(h4.iloc[:-1].max(), price)
        lo_recent_30 = to_float(l4.iloc[-30:].min(), price)
        dd_30d_pct = ((hi_30d_d4 - lo_recent_30) / max(hi_30d_d4, 1e-8)) * 100.0
        higher_low_4h = to_float(l4.iloc[-12:].min(), price) >= to_float(l4.iloc[-36:-12].min(), price) * 0.995
        prior_18_c = c4.iloc[-20:-2]
        prior_18_ma50 = ma50_4h_s.iloc[-20:-2]
        below_ma50_ratio = float((prior_18_c < prior_18_ma50).mean()) if len(prior_18_c) >= 10 else 0.0
        fresh_ma50_break = (
            ma50_4h > 0
            and below_ma50_ratio >= 0.70
            and price > ma50_4h * 1.002
            and price <= ma50_4h * 1.08
            and turn_4h_ok
        )
        ma20_turning = ma20_4h > ma20_4h_prev3 and ma20_4h >= ma50_4h * 0.94
        obv_14d_hi = to_float(obv4.iloc[-84:-1].max(), -1e18)
        obv_14d_break = to_float(obv4.iloc[-1]) >= obv_14d_hi

        if dd_30d_pct >= 25.0 and higher_low_4h and fresh_ma50_break and bull_v4_2 >= 2.2 and ma20_turning and obv_14d_break and ppo4_bull:
            score_d4 = int(np.clip(87 + min(2, int(bull_v4_2 - 1.5)) + (1 if ma20_4h >= ma50_4h else 0) + (1 if ppo4_zero_up else 0), 87, 91))
            candidates_d.append({
                "sub_strategy": "D4 · 4H 장기 하락추세 탈출 & 50MA 돌파",
                "score": score_d4,
                "vol_metric": bull_v4_2,
                "reasons": [
                    f"30일 고점 대비 -{dd_30d_pct:.0f}% 조정 후 50MA 하단 횡보 탈출·이중바닥 완성",
                    f"4H 50MA 건전 양봉 거래량 {bull_v4_2:.1f}배 돌파 + PPO({ppo4_now:+.2f}%) {'0선 안착' if ppo4_zero_up else '골든크로스'}",
                    "4H OBV 14일 최고치 선행 돌파(장기 추세 1파동 초입)",
                ],
            })

    # D5: 4H 매크로 PPO·RSI14·OBV 3중 강세 다이버전스
    if len(df_4h) >= 65:
        lo_14d = to_float(l4.iloc[-min(len(l4), 84):].min(), price)
        in_bottom_zone = price <= lo_14d * 1.12
        prev_win = l4.iloc[-42:-12]
        rec_win = l4.iloc[-12:]
        p_idx = int(np.argmin(prev_win.values))
        r_idx = int(np.argmin(rec_win.values))
        p_abs = len(df_4h) - 42 + p_idx
        r_abs = len(df_4h) - 12 + r_idx

        if (r_abs - p_abs) >= 6:
            l_prev_v = to_float(l4.iloc[p_abs])
            l_rec_v = to_float(l4.iloc[r_abs])
            mid_high_v = to_float(h4.iloc[p_abs + 1 : r_abs].max(), 0.0)
            has_two_swings = mid_high_v >= max(l_prev_v, l_rec_v) * 1.05

            rsi_prev_v = to_float(rsi14_4h.iloc[p_abs], 50.0)
            rsi_rec_v = to_float(rsi14_4h.iloc[r_abs], 50.0)
            ppo_prev_v = to_float(ppo4.iloc[p_abs], 0.0)
            ppo_rec_v = to_float(ppo4.iloc[r_abs], 0.0)
            obv_prev_v = to_float(obv4.iloc[p_abs], 0.0)
            obv_curr_v = to_float(obv4.iloc[-1], 0.0)

            ppo_bull_div_4h = (ppo_rec_v >= ppo_prev_v + 0.35) and ppo4_turn_up
            bull_div_4h = has_two_swings and (l_rec_v <= l_prev_v * 1.03) and (rsi_rec_v >= rsi_prev_v + 4.0) and ppo_bull_div_4h and (obv_curr_v > obv_prev_v)
            body_4h_pct = ((curr_c4 - curr_o4) / max(curr_o4, 1e-8)) * 100.0
            curr_v4_r = to_float(v4_ratio.iloc[-1], 0.0)
            strong_bull_4h = _is_healthy_bull_candle(curr_o4, curr_h4, curr_l4, curr_c4, max_uwick=0.35, min_clv=0.25, min_body_pct=1.5) and curr_v4_r >= 1.2

            if in_bottom_zone and bull_div_4h and strong_bull_4h and oi_ratio >= 20.0 and funding_pct <= 0.005:
                score_d5 = int(np.clip(87 + min(2, int((rsi_rec_v - rsi_prev_v) / 4.0)) + (2 if funding_pct < 0 else 1), 87, 91))
                candidates_d.append({
                    "sub_strategy": "D5 · 4H 매크로 PPO·RSI 강세 다이버전스 바닥 매집",
                    "score": score_d5,
                    "vol_metric": round(curr_v4_r, 2),
                    "reasons": [
                        f"14일 이중바닥 4H PPO({ppo_prev_v:+.1f}%→{ppo_rec_v:+.1f}%)·RSI14·OBV 3중 강세 다이버전스",
                        f"고OI 집중도({oi_ratio:.0f}%) + 저과열 펀딩({funding_pct:+.4f}%) 세력 바닥 흡수",
                        f"4H 건전 장대양봉(+{body_4h_pct:.1f}% · 거래량 {curr_v4_r:.1f}배) 턴어라운드 개시",
                    ],
                })

    # D6: 와이코프 스프링
    if len(df_4h) >= 100 and turnover_m >= 3.0 and oi_m >= 0.8 and oi_ratio >= 15.0 and funding_pct <= 0.006:
        prior_floor = to_float(l4.iloc[:-24].min(), price)
        prior_high_30 = to_float(h4.iloc[:-12].max(), price)
        spring_low = to_float(l4.iloc[-24:].min(), price)
        sweep_depth_pct = ((prior_floor - spring_low) / max(prior_floor, 1e-8)) * 100.0

        had_spring_sweep = (2.5 <= sweep_depth_pct <= 35.0) and (prior_high_30 >= prior_floor * 1.15)
        reclaimed_floor = (prior_floor * 1.003 <= price <= prior_floor * 1.10) and (price >= ma20_4h * 0.98)
        fresh_reclaim_6 = (to_float(l4.iloc[-6:].min(), price) <= prior_floor * 1.02) or (to_float(c4.iloc[-6:-1].min(), price) <= prior_floor * 1.01)
        obv4_reclaim = to_float(obv4.iloc[-1]) > to_float(obv4.iloc[-12:-2].mean())

        if had_spring_sweep and reclaimed_floor and fresh_reclaim_6 and turn_4h_ok and bull_v4_3 >= 2.2 and obv4_reclaim and (ppo4_turn_up or ppo4_bull):
            score_d6 = int(np.clip(
                86
                + (2 if funding_pct <= -0.002 else 0)
                + (2 if bull_v4_3 >= 3.0 else 1)
                + (1 if oi_ratio >= 25.0 else 0),
                86, 91,
            ))
            candidates_d.append({
                "sub_strategy": "D6 · 와이코프 스프링(신저가 개미털기 후 바닥 재탈환)",
                "score": score_d6,
                "vol_metric": bull_v4_3,
                "reasons": [
                    f"직전 바닥({prior_floor:.5g}) -{sweep_depth_pct:.1f}% 개미털기 후 건전 양봉 {bull_v4_3:.1f}배 재탈환",
                    f"OI 집중도 {oi_ratio:.0f}% + 펀딩 {funding_pct:+.4f}% 숏함정 역발상 대시세 초입",
                    f"4H 20MA·OBV 반전 + 4H PPO({ppo4_now:+.2f}%) 상방 턴 확인",
                ],
            })

    if not candidates_d:
        return None
    candidates_d.sort(key=lambda x: x["score"], reverse=True)
    return candidates_d[0]


def analyze_symbol(
    coin,
    oi_snapshot_prev=None,
    market_regime=None,
    scan_long_term=True,
    min_score_short=None,
    min_score_long=None,
):
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

    short_cut = int(min_score_short) if min_score_short is not None else ELITE_SCORE_THRESHOLD
    long_cut = int(min_score_long) if min_score_long is not None else ELITE_SCORE_THRESHOLD

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

    # 최근 24시간(15m 96봉) 고저 변동폭이 0.8% 미만인 스테이블/페깅 토큰 차단
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

    # 로컬 시총 캐시 조회 (미등록 시 온디맨드 조회 포함)
    market_cap_m = get_market_cap_m(symbol)

    # 최종 관문 통과 정예 후보에만 1회 호출되는 스마트머니/개미 L/S 로컬 캐시
    cached_overlay = {"fetched": False, "data": None}

    def _get_supply_overlay():
        if not cached_overlay["fetched"]:
            ls_info = fetch_smart_money_ls_ratio(symbol)
            cached_overlay["data"] = _evaluate_supply_overlay(symbol, oi_m, turnover_m, market_cap_m, ls_info=ls_info)
            cached_overlay["fetched"] = True
        return cached_overlay["data"]

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

    curr_o, curr_c = to_float(opens.iloc[-1], price), to_float(closes.iloc[-1], price)
    curr_h, curr_l = to_float(highs.iloc[-1], price), to_float(lows.iloc[-1], price)
    prev_o, prev_c = to_float(opens.iloc[-2], price), to_float(closes.iloc[-2], price)
    prev_h, prev_l = to_float(highs.iloc[-2], price), to_float(lows.iloc[-2], price)

    is_curr_healthy_bull = _is_healthy_bull_candle(curr_o, curr_h, curr_l, curr_c, max_uwick=0.45, min_clv=0.0, min_body_pct=0.15)
    is_prev_healthy_bull = _is_healthy_bull_candle(prev_o, prev_h, prev_l, prev_c, max_uwick=0.45, min_clv=0.0, min_body_pct=0.20)
    curr_uwick_ratio = (curr_h - max(curr_o, curr_c)) / max(curr_h - curr_l, 1e-12)
    bull_trigger_ok = is_curr_healthy_bull or (is_prev_healthy_bull and curr_c >= curr_o and curr_uwick_ratio <= 0.45 and curr_c >= prev_c * 0.997)

    vol_ratios = _calc_shifted_vol_ratios(vols, window=20)
    bull_vol_2 = _max_healthy_bull_vol_ratio(opens, highs, lows, closes, vol_ratios, lookback=2, strict_gt=True, max_uwick=0.45, min_clv=0.0)
    bull_vol_5 = _max_healthy_bull_vol_ratio(opens, highs, lows, closes, vol_ratios, lookback=5, strict_gt=True, max_uwick=0.45, min_clv=0.0)
    curr_bull_vol = round(to_float(vol_ratios.iloc[-1], 0.0), 2) if is_curr_healthy_bull else 0.0

    v16 = vols.iloc[-16:]
    buy_v16 = float(v16[closes.iloc[-16:] >= opens.iloc[-16:]].sum())
    sell_v16 = float(v16[closes.iloc[-16:] < opens.iloc[-16:]].sum())
    buy_dom_4h = buy_v16 / max(sell_v16, 1e-8)

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

    def _get_pm_penalty(tk_name, vol_val):
        if postmortem is not None and hasattr(postmortem, "get_setup_penalty"):
            try:
                return int(postmortem.get_setup_penalty(symbol, tk_name, rsi6_15m, vol_val, extra_meta=extra_meta))
            except TypeError:
                try:
                    return int(postmortem.get_setup_penalty(symbol, tk_name, rsi6_15m, vol_val))
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

    # =========================================================================
    # [1순위] 장기 포지션 (Track D1~D6) + 공통 수급 오버레이(OI/시총 · 고래/개미 L/S)
    # =========================================================================
    if (
        scan_long_term
        and (not counter_trend_only)
        and oi_m >= 0.4
        and turnover_m >= 2.5
        and change_24h < 22.0
        and rsi6_15m < 75.0
        and kdj_j_15m < 92.0
        and ma20_disp_pct < 7.0
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
                ov = _get_supply_overlay()
                d_vol_m = to_float(macro_d.get("vol_metric"), max(bull_vol_2, 1.0))
                raw_d = (
                    macro_d["score"]
                    + mtf_long["score_adj"]
                    + ov["score_adj"]
                    + (3 if oi_q["quadrant"] == "Q1" else 0)
                    + (1 if sess_adj > 0 else 0)
                )
                d_score = int(np.clip(raw_d - _get_pm_penalty("TRACK_D", d_vol_m), 0, 99))
                if d_score >= long_cut:
                    tps, tp_pcts, tp_labels = _calculate_macro_longterm_tps(
                        price, turnover_m, oi_m, df_4h, df_1d, df_1w=df_1w, market_cap_m=market_cap_m
                    )
                    stop_price = round(price * (1.0 - LONGTERM_SL_PCT / 100.0), 6)
                    reasons = (
                        [f"장기 전략: {macro_d['sub_strategy']}"]
                        + macro_d["reasons"][:2]
                        + ov["reasons"][:1]
                        + mtf_long["reasons"][:1]
                    )
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
                        "market_cap_m": ov["market_cap_m"],
                        "oi_mc_pct": ov["oi_mc_pct"],
                        "turn_mc_pct": ov["turn_mc_pct"],
                        "ls_source": ov["ls_source"],
                        "top_ls": ov["top_ls"],
                        "retail_ls": ov["retail_ls"],
                        "ls_divergence": ov["ls_divergence"],
                        "funding_pct": funding_pct,
                        "btc_1h_pct": btc_1h_pct,
                        "htf_status": htf_long_str,
                        "rsi6_15m": rsi6_15m,
                        "kdj_j_15m": kdj_j_15m,
                        "ppo_15m": ppo15_now,
                        "vol_mult": d_vol_m,
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

    swing_high_ref = to_float(highs.iloc[-min(len(highs) - 1, 48):-1].max(), price * 1.03)

    supply_score = (6 if oi_ratio >= 20.0 else (5 if oi_ratio >= 9.0 else 3)) + (6 if funding_pct < -0.005 else (4 if funding_pct <= 0.012 else 2))
    base_supply_reasons = []
    if oi_ratio >= 20.0:
        base_supply_reasons.append(f"OI 집중도 최상({oi_ratio:.0f}%)")
    if funding_pct < -0.005:
        base_supply_reasons.append(f"음수 펀딩 숏스퀴즈({funding_pct:+.4f}%)")

    wave_bonus, wave_note = 0, ""
    if wave_engine is not None and hasattr(wave_engine, "evaluate_wave_bonus"):
        try:
            wb, wn = wave_engine.evaluate_wave_bonus(df_15m, df_1h)
            wave_bonus, wave_note = int(np.clip(int(wb), -3, 2)), str(wn or "")
        except Exception:
            pass

    short_candidates = []

    def _register_candidate(tk, title, tech_sc, tk_reasons, vol_metric):
        capped_tech = min(83, int(tech_sc))
        raw_sc = capped_tech + supply_score + oi_q["bonus"] + sess_adj + wave_bonus - _get_pm_penalty(tk, vol_metric)
        if raw_sc >= (short_cut - 6):
            r_list = list(tk_reasons) + list(base_supply_reasons)
            if oi_q["note"]:
                r_list.append(oi_q["note"])
            if sess_note:
                r_list.append(sess_note)
            if wave_note:
                r_list.append(wave_note)
            short_candidates.append({
                "track_type": tk,
                "track_title": title,
                "raw_score": int(raw_sc),
                "reasons": r_list,
                "vol_metric": round(to_float(vol_metric, bull_vol_2), 2),
            })

    # [Track E] 🧲 유동성 스윕(SFP) 바닥 반전
    if len(lows) >= 40:
        span_sfp = min(len(lows) - 3, 96)
        prior_low_slice = lows.iloc[-span_sfp - 3 : -3]
        prior_low_24h = to_float(prior_low_slice.min(), price)
        sweep_win = df_15m.iloc[-3:]
        sweep_min_l = to_float(sweep_win["low"].min(), price)
        sweep_depth_pct = ((prior_low_24h - sweep_min_l) / max(prior_low_24h, 1e-8)) * 100.0

        max_wick_3 = 0.0
        for _, rw in sweep_win.iterrows():
            r_o, r_c, r_h, r_l = to_float(rw["open"]), to_float(rw["close"]), to_float(rw["high"]), to_float(rw["low"])
            w_ratio = (min(r_o, r_c) - r_l) / max(r_h - r_l, 1e-10)
            if w_ratio > max_wick_3:
                max_wick_3 = w_ratio

        swept_support = (0.15 <= sweep_depth_pct <= 4.5)
        reclaimed_support = (
            (prior_low_24h * 1.001 <= curr_c <= prior_low_24h * 1.035)
            and (curr_c <= sweep_min_l * 1.055)
            and bull_trigger_ok
        )

        prior_low_idx = int(np.argmin(prior_low_slice.values))
        prior_abs_idx = len(lows) - span_sfp - 3 + prior_low_idx
        has_div_gap = (len(lows) - 1 - prior_abs_idx) >= 8
        prior_low_rsi6 = to_float(rsi6_s.iloc[prior_abs_idx], 50.0)
        prior_low_ppo15 = to_float(ppo15_s.iloc[prior_abs_idx], 0.0)
        rsi_bull_div = has_div_gap and (min_rsi6_6 >= prior_low_rsi6 + 3.0)
        ppo_bull_div = has_div_gap and (min_ppo15_6 >= prior_low_ppo15 + 0.15) and ppo15_turn_up
        kdj_sharp_turn = min_kdj_j_6 <= 20.0 and kdj_j_15m > kdj_j_prev and rsi6_15m > rsi6_prev

        if swept_support and reclaimed_support and max_wick_3 >= 0.40 and bull_vol_2 >= 2.0 and (ppo_bull_div or rsi_bull_div or kdj_sharp_turn):
            sc_e = (
                44
                + (14 if max_wick_3 >= 0.50 else 10)
                + (13 if bull_vol_2 >= 2.6 else 10)
                + (11 if (ppo_bull_div and (rsi_bull_div or kdj_sharp_turn)) else 8)
                + (6 if (recent_delta_sum > 0 or obv_strong) else 3)
                + (2 if counter_trend_only else 0)
            )
            div_lbl = "PPO·RSI 강세 다이버전스" if ppo_bull_div else ("RSI 강세 다이버전스" if rsi_bull_div else "KDJ 바닥 급반전")
            _register_candidate(
                "TRACK_E",
                "🧲 [Track E · 유동성 스윕(SFP) 바닥 반전 타점]",
                sc_e,
                [
                    f"24H 저점 -{sweep_depth_pct:.1f}% 스윕 후 지지선 탈환(아래꼬리 {max_wick_3 * 100:.0f}%)",
                    f"최근 2봉 건전 양봉 거래량 {bull_vol_2:.1f}배 폭발 + {div_lbl}",
                ],
                vol_metric=bull_vol_2,
            )

    # [Track F] ⚡ BTC 역행 상대강도 대장주
    btc_weak_for_f = (-2.4 <= btc_4h_pct <= 0.4) or (btc_1h_pct <= -0.4)
    if len(closes) >= 20 and btc_weak_for_f and ppo1h_bull and change_24h <= 28.0:
        p_4h_ago = to_float(closes.iloc[-17], price)
        coin_4h_pct = ((price - p_4h_ago) / max(p_4h_ago, 1e-8)) * 100.0
        rs_vs_btc = coin_4h_pct - btc_4h_pct
        hi_4h_f = to_float(highs.iloc[-16:].max(), price)
        holding_high_f = price >= hi_4h_f * 0.972
        obv_24h_hi = to_float(obv.iloc[-min(len(obv) - 1, 48):-1].max(), -1e18)
        obv_break_f = to_float(obv.iloc[-1]) >= obv_24h_hi * 0.995 and obv_strong

        if (
            rs_vs_btc >= 4.5
            and coin_4h_pct >= 3.0
            and holding_high_f
            and htf_aligned
            and obv_break_f
            and oi_ratio >= 10.0
            and rsi6_15m < 74.0
            and ma20_disp_pct < 5.5
            and bull_trigger_ok
            and buy_dom_4h >= 1.15
            and bull_vol_2 >= 1.5
        ):
            sc_f = (
                (35 if rs_vs_btc >= 7.0 else 32)
                + 16
                + (13 if obv_break_f else 8)
                + (12 if bull_vol_2 >= 2.0 else 10)
                + (6 if recent_delta_sum > 0 else 3)
                + (2 if counter_trend_only else 0)
            )
            _register_candidate(
                "TRACK_F",
                "⚡ [Track F · BTC 역행 상대강도(RS) 대장주 타점]",
                sc_f,
                [
                    f"BTC 4H({btc_4h_pct:+.1f}%) 대비 상대강도 +{rs_vs_btc:.1f}% 독주 · 4H 고점권 유지",
                    f"1H 정배열·PPO({ppo1h_now:+.2f}%) + 매수우위 {buy_dom_4h:.1f}x · 양봉 거래량 {bull_vol_2:.1f}배",
                ],
                vol_metric=bull_vol_2,
            )

    # [Track A·B·C·G + Track H] — BTC 정상 구간(OPEN)에서만 탐색
    if not counter_trend_only:
        # [Track G] 💥 숏스퀴즈 강제청산 초입
        if (
            (-6.0 <= change_24h <= 18.0)
            and htf_bull
            and funding_pct <= -0.020
            and oi_ratio >= 18.0
            and (ppo15_turn_up or ppo15_bull)
        ):
            higher_low_g = to_float(lows.iloc[-6:].min(), price) >= to_float(lows.iloc[-18:-6].min(), price) * 0.998
            short_res_g = to_float(highs.iloc[-25:-1].max(), price)
            break_res_g = (
                bull_trigger_ok
                and curr_c > ma20_15m_val
                and (short_res_g * 1.000 <= curr_c <= short_res_g * 1.035)
            )
            if higher_low_g and break_res_g and recent_delta_sum > 0 and curr_bar_delta > 0 and bull_vol_2 >= 1.6:
                sc_g = (
                    (37 if funding_pct <= -0.040 else 34)
                    + (16 if oi_ratio >= 25.0 else 13)
                    + (14 if htf_aligned else 11)
                    + (12 if bull_vol_2 >= 2.2 else 9)
                    + (8 if obv_strong else 5)
                )
                _register_candidate(
                    "TRACK_G",
                    "💥 [Track G · 숏스퀴즈 강제청산 초입 타점]",
                    sc_g,
                    [
                        f"깊은 음수 펀딩({funding_pct:+.4f}%) + 고OI({oi_ratio:.0f}%) 숏 쏠림 임계",
                        f"6H 저항 실제 돌파 건전 양봉 {bull_vol_2:.1f}배 + PPO({ppo15_now:+.2f}%)·CVD 폭발",
                    ],
                    vol_metric=bull_vol_2,
                )

        # [Track B] 🔥 24시간 스윙 구조 실제 돌파 / S-R Flip
        span_24h = min(len(highs) - 2, 96)
        high_24h = to_float(highs.iloc[-span_24h - 2 : -2].max(), price * 1.05)
        b_candle_healthy = _is_healthy_bull_candle(curr_o, curr_h, curr_l, curr_c, max_uwick=0.40, min_clv=0.05, min_body_pct=0.15)

        close_confirmed_b = (
            (prev_c >= high_24h * 1.000)
            and (high_24h * 1.001 <= curr_c <= high_24h * 1.032)
            and b_candle_healthy
            and (bull_vol_2 >= 2.0)
        )

        had_break_recent = to_float(highs.iloc[-5:-1].max(), 0.0) >= high_24h * 1.002
        retest_red_vr = to_float(vol_ratios.iloc[-2], 0.0) if (prev_c < prev_o) else 0.0
        retest_dry_ok = (retest_red_vr <= bull_vol_5 * 0.85) or (curr_bull_vol >= 1.2)
        sr_flip_retest_b = (
            had_break_recent
            and (high_24h * 0.988 <= min(curr_l, prev_l) <= high_24h * 1.008)
            and (high_24h * 1.001 <= curr_c <= high_24h * 1.032)
            and b_candle_healthy
            and (bull_vol_5 >= 2.0)
            and retest_dry_ok
            and (bull_vol_2 >= 1.2)
        )
        is_breakout_b = (close_confirmed_b or sr_flip_retest_b) and (3.0 <= change_24h <= 28.0) and (ppo15_bull or ppo15_turn_up)

        if is_breakout_b:
            b_vol_show = bull_vol_2 if close_confirmed_b else max(bull_vol_2, bull_vol_5)
            flip_lbl = "S-R Flip 리테스트 지지(눌림건조·반등양봉)" if sr_flip_retest_b else "24H 전고점 종가 안착 돌파"
            sc_b = (
                (35 if b_vol_show >= 2.8 else 31)
                + (16 if htf_aligned else (13 if htf_bull else 5))
                + (13 if obv_strong else 6)
                + (11 if (curr_c > ma20_15m_val and to_float(ma7.iloc[-1]) >= ma20_15m_val) else 5)
                + (8 if 48.0 <= rsi6_15m <= 74.0 else 4)
                + (4 if sr_flip_retest_b else 2)
            )
            _register_candidate(
                "TRACK_B",
                f"🔥 [Track B · 24H 구조 돌파 ({'S-R Flip' if sr_flip_retest_b else '종가 안착'})]",
                sc_b,
                [f"{flip_lbl} + 건전 양봉 거래량 {b_vol_show:.1f}배 · 15m PPO({ppo15_now:+.2f}%) 확장"],
                vol_metric=b_vol_show,
            )

        # [Track C] 🚀 대장주 심층 눌림목 + 5m MSS + 저점 대비 SL 보호(<= +5.2%)
        if change_24h >= 12.0 and not is_breakout_b and price >= ma50_1h * 0.985:
            win_len_c = min(len(df_15m) - 2, 48)
            win_h_c = highs.iloc[-win_len_c - 1 : -1]
            peak_rel_c = int(np.argmax(win_h_c.values))
            peak_abs_c = len(df_15m) - win_len_c - 1 + peak_rel_c
            bars_since_peak = (len(df_15m) - 1) - peak_abs_c

            if bars_since_peak >= 4 and peak_abs_c >= 6:
                pre_low_slice = lows.iloc[max(0, peak_abs_c - 36) : peak_abs_c]
                if not pre_low_slice.empty:
                    c_swing_high = to_float(highs.iloc[peak_abs_c], price * 1.05)
                    c_swing_low = to_float(pre_low_slice.min(), price * 0.90)
                    wave1_gain_pct = ((c_swing_high - c_swing_low) / max(c_swing_low, 1e-8)) * 100.0
                    fib_ret_c = (c_swing_high - price) / max(c_swing_high - c_swing_low, price * 0.01)
                    pb_from_high_c = ((c_swing_high - price) / max(c_swing_high, 1e-8)) * 100.0

                    if wave1_gain_pct >= 10.0 and 0.48 <= fib_ret_c <= 0.78 and pb_from_high_c >= 6.5 and price >= c_swing_low * 1.015:
                        touched_ma20_flush = to_float(lows.iloc[-6:].min(), price) <= ma20_15m_val * 0.995
                        had_flush = (min_rsi6_6 <= 38.0) or (min_kdj_j_6 <= 22.0) or touched_ma20_flush

                        if had_flush and ppo15_turn_up:
                            wave1_vols = vols.iloc[max(0, peak_abs_c - 8) : peak_abs_c + 1]
                            avg_v_wave1 = max(to_float(wave1_vols.mean(), 1.0), to_float(vols.iloc[max(0, peak_abs_c - 20) : peak_abs_c + 1].nlargest(4).mean(), 1.0) * 0.72)
                            wave2_vols = vols.iloc[peak_abs_c + 1 : -1]
                            avg_v_wave2 = to_float(wave2_vols.mean(), avg_v_wave1)
                            vol_dry_ratio = avg_v_wave2 / max(avg_v_wave1, 1e-8)

                            curr_vol_v = to_float(vols.iloc[-1], 0.0)
                            curr_lwick = (min(curr_o, curr_c) - curr_l) / max(curr_h - curr_l, 1e-10)
                            prev_lwick = (min(prev_o, prev_c) - prev_l) / max(prev_h - prev_l, 1e-10)
                            max_lwick = max(curr_lwick, prev_lwick)

                            recent_low_4 = to_float(lows.iloc[-4:].min(), curr_l)
                            rebound_vol_wake = (curr_c > curr_o) and (curr_uwick_ratio <= 0.45) and (curr_vol_v >= avg_v_wave2 * 1.15 or bull_vol_2 >= 1.0)
                            is_bullish_turn = bull_trigger_ok or (curr_lwick >= 0.45 and curr_c >= prev_l and curr_uwick_ratio <= 0.35)
                            is_momentum_turn = (kdj_j_15m > kdj_j_prev) and (rsi6_15m > rsi6_prev) and (rsi6_15m <= 60.0)
                            is_holding_low = (recent_low_4 * 1.003 <= curr_c <= recent_low_4 * 1.052)

                            if vol_dry_ratio <= 0.35 and rebound_vol_wake and is_bullish_turn and is_momentum_turn and is_holding_low:
                                mss_ok = _check_5m_mss_trigger(symbol)
                                if mss_ok is True or (mss_ok is None and is_curr_healthy_bull and curr_c > prev_o):
                                    sc_c = (
                                        (35 if 0.52 <= fib_ret_c <= 0.72 else 31)
                                        + (16 if price >= ma20_1h * 0.985 else 13)
                                        + (13 if obv_strong else 10)
                                        + (13 if (min_rsi6_6 <= 30.0 or max_lwick >= 0.35) else 10)
                                        + (10 if mss_ok else 6)
                                        + (2 if oi_q["quadrant"] == "Q4" else 0)
                                    )
                                    _register_candidate(
                                        "TRACK_C",
                                        "🚀 [Track C · 대장주 심층 눌림목(5m MSS 전환) 반등 타점]",
                                        sc_c,
                                        [
                                            f"대장주 1파(+{wave1_gain_pct:.0f}%) 후 -{pb_from_high_c:.1f}% · Fib {fib_ret_c:.2f} 심층 눌림",
                                            f"5m MSS 반등 + 거래량 건조({vol_dry_ratio * 100:.0f}%) 후 건전 양봉 회복 + PPO 턴",
                                        ],
                                        vol_metric=max(bull_vol_2, 1.0),
                                    )

        # [Track A] 💎 TTM 스퀴즈 발사 & 델타 흡수
        if (
            (-5.0 <= change_24h < 12.0)
            and htf_bull
            and not is_breakout_b
            and (ppo15_turn_up or ppo15_bull)
            and bull_trigger_ok
            and bull_vol_2 >= 1.8
        ):
            min_bw = to_float(bb_width_pct.iloc[-10:-1].min(), 5.0)
            box_hi_10 = to_float(highs.iloc[-10:-1].max(), price)
            breaking_squeeze_box = curr_c >= box_hi_10 * 0.996

            if breaking_squeeze_box and (ttm_fired or (was_squeezed and cvd_absorption) or (min_bw <= 4.5 and cvd_absorption)):
                sc_a = (
                    (33 if ttm_fired else (28 if was_squeezed else 24))
                    + (16 if bull_vol_2 >= 2.3 else 13)
                    + (15 if htf_aligned else 12)
                    + (14 if (cvd_absorption and obv_strong) else (9 if obv_strong else 4))
                    + (8 if (kdj_j_15m > kdj_j_prev and 35.0 <= rsi6_15m <= 70.0) else 4)
                )
                r_a = [
                    f"TTM Squeeze(BB<KC) 상단 돌파(응축폭 {min_bw:.1f}%) + 건전 양봉 {bull_vol_2:.1f}배"
                    if ttm_fired
                    else f"박스권 초응축({min_bw:.1f}%) 상단 돌파 + 건전 양봉 {bull_vol_2:.1f}배"
                ]
                if cvd_absorption:
                    r_a.append(f"15m 매수 델타(CVD)·OBV 흡수 + PPO({ppo15_now:+.2f}%) 상방")
                _register_candidate(
                    "TRACK_A",
                    "💎 [Track A · TTM 스퀴즈 발사 & 델타 흡수 초입]" if ttm_fired else "💎 [Track A · 박스권 초응축 & 델타 매집 초입]",
                    sc_a,
                    r_a,
                    vol_metric=bull_vol_2,
                )

        # [Track H] 🧬 Top 5 급등 DNA 자동 적응 트랙
        if (
            not short_candidates
            and htf_bull
            and curr_c >= ma20_15m_val
            and (-5.0 <= change_24h < 12.0)
            and oi_ratio >= 8.0
            and funding_pct <= 0.010
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

                short_hi_12 = to_float(highs.iloc[-13:-1].max(), price)
                req_lvol = max(1.6, to_float(dna_f.get("launch_vol_min"), 2.0))
                has_launch_trigger = (
                    bull_trigger_ok
                    and (curr_c > prev_c)
                    and (curr_c >= short_hi_12 * 0.985)
                    and (bull_vol_2 >= req_lvol)
                    and (recent_delta_sum > 0)
                )

                if has_launch_trigger:
                    chart_checks = [
                        ("응축", min_bw_h <= min(9.5, to_float(dna_f.get("bb_width_max"), 9.5))),
                        ("거래량건조", pre_dry_h <= min(0.95, to_float(dna_f.get("vol_dry_max"), 0.95))),
                        ("직전RSI", to_float(dna_f.get("pre_rsi_min"), 25.0) <= rsi6_prev <= to_float(dna_f.get("pre_rsi_max"), 68.0)),
                        ("1H이격", to_float(dna_f.get("ma20_1h_dist_min"), -4.5) <= dist_1h_ma20 <= to_float(dna_f.get("ma20_1h_dist_max"), 6.5)),
                    ]
                    matched_names = [nm for nm, ok in chart_checks if ok]
                    match_cnt = len(matched_names)

                    if match_cnt >= 3:
                        sc_h = (
                            (35 if match_cnt == 4 else 31)
                            + (16 if bull_vol_2 >= req_lvol * 1.25 else 12)
                            + (13 if obv_strong else 8)
                            + (13 if htf_aligned else 10)
                            + (8 if (kdj_j_15m > kdj_j_prev and rsi6_15m > rsi6_prev) else 4)
                        )
                        dom_tag = str(dna_f.get("dominant_tag") or "Top 5 급등 DNA")
                        _register_candidate(
                            "TRACK_H",
                            f"🧬 [Track H · Top 5 급등 DNA 자동학습 포착 ({dom_tag})]",
                            sc_h,
                            [
                                f"최근 14일 Top 5 차트 DNA {match_cnt + 1}/5 일치 ({', '.join(matched_names)}·발사거래량)",
                                f"단기 상단 돌파 + 건전 양봉 거래량({bull_vol_2:.1f}x) + PPO({ppo15_now:+.2f}%) 상방",
                            ],
                            vol_metric=bull_vol_2,
                        )

    if not short_candidates:
        return None

    # [최종 정예 관문] 1W(주봉) · 1D(일봉) PPO + 공통 수급 오버레이(OI/시총 · 고래/개미 L/S) 결합
    df_1d, df_1w = _get_1d_1w()
    ov_short = _get_supply_overlay()
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
        final_sc = int(np.clip(cand["raw_score"] + mtf_s["score_adj"] + ov_short["score_adj"], 0, 99))
        if final_sc < short_cut:
            continue
        merged_reasons = list(cand["reasons"][:2]) + list(ov_short["reasons"][:1]) + list(mtf_s["reasons"]) + list(cand["reasons"][2:])
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
        "market_cap_m": ov_short["market_cap_m"],
        "oi_mc_pct": ov_short["oi_mc_pct"],
        "turn_mc_pct": ov_short["turn_mc_pct"],
        "ls_source": ov_short["ls_source"],
        "top_ls": ov_short["top_ls"],
        "retail_ls": ov_short["retail_ls"],
        "ls_divergence": ov_short["ls_divergence"],
        "funding_pct": funding_pct,
        "btc_1h_pct": btc_1h_pct,
        "htf_status": final_htf_status,
        "rsi6_15m": rsi6_15m,
        "kdj_j_15m": kdj_j_15m,
        "ppo_15m": ppo15_now,
        "vol_mult": best.get("vol_metric", bull_vol_2),
        "swing_high_line": swing_high_ref,
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
