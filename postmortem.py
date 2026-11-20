import os
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

try:
    import config as cfg
    STATE_FILE = getattr(cfg, "STATE_FILE", "bot_state.json")
    HISTORY_FILE = getattr(cfg, "HISTORY_FILE", "analyzer_history.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
    is_tradfi_ticker = getattr(cfg, "is_tradfi_ticker", lambda x: False)
    is_heavy_major_coin = getattr(cfg, "is_heavy_major_coin", lambda x: False)
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception:
    cfg = None
    STATE_FILE = "bot_state.json"
    HISTORY_FILE = "analyzer_history.json"
    KST = timezone(timedelta(hours=9))
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}
    is_tradfi_ticker = lambda x: False
    is_heavy_major_coin = lambda x: False
    FIXED_SL_PCT = 6

try:
    import api as api_mod
except Exception:
    api_mod = None

from wave_engine import (
    to_float,
    fmt_price,
    calc_rsi,
    calc_kdj_j,
    calc_atr,
    calc_obv_and_cvd,
    extract_three_wave_structure,
)

BYBIT_HOSTS = (
    "https://api.bybit-tr.com",
    "https://api.bybit.kz",
    "https://api.bybit.nl",
    "https://api.bytick.com",
    "https://api.bybit.com",
)


def normalize_funding_pct(raw_val):
    v = to_float(raw_val, 0.0)
    if abs(v) > 0.15:
        return v / 100.0 if abs(v) <= 2.5 else v
    return v


def load_json_file(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[JSON 로드 경고] {path}: {e}")
    return default_val


def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[파일 저장 오류] {path}: {e}")


def _get_bybit_whitelist():
    if api_mod is not None and hasattr(api_mod, "get_valid_bybit_linear_symbols"):
        try:
            syms, _ = api_mod.get_valid_bybit_linear_symbols()
            if syms:
                return syms
        except Exception:
            pass
    return set()


def fetch_klines(symbol, interval="5", limit=300):
    """오직 Bybit V5 공식 캔들만 조회 (게이트아이오·비트겟 폴백 완전 제거)"""
    if api_mod is not None and hasattr(api_mod, "fetch_klines"):
        df = api_mod.fetch_klines(symbol, str(interval), limit=int(limit))
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return pd.DataFrame()


def enrich_ticker_metrics(candidates):
    valid_syms = _get_bybit_whitelist()
    ticker_map = {}

    for host in BYBIT_HOSTS:
        try:
            r = requests.get(f"{host}/v5/market/tickers", params={"category": "linear"}, headers=BROWSER_HEADERS, timeout=5)
            if r.status_code == 200:
                items = r.json().get("result", {}).get("list", [])
                if items:
                    for it in items:
                        s = str(it.get("symbol", "")).upper()
                        ticker_map[s] = {
                            "turnover_m": to_float(it.get("turnover24h")) / 1e6,
                            "oi_m": to_float(it.get("openInterestValue")) / 1e6,
                            "funding_pct": to_float(it.get("fundingRate")) * 100.0,
                        }
                    break
        except requests.RequestException:
            pass

    filtered_candidates = []
    for c in candidates:
        sym = str(c.get("symbol", "")).upper()
        base = sym[:-4] if sym.endswith("USDT") else sym
        if not base.isascii() or is_tradfi_ticker(base):
            continue
        if valid_syms and sym not in valid_syms:
            continue
        if ticker_map and sym not in ticker_map:
            continue

        ext = ticker_map.get(sym, {})
        t_m = to_float(c.get("turnover_m") or ext.get("turnover_m"), 0.0)
        o_m = to_float(c.get("oi_m") or ext.get("oi_m"), 0.0)
        f_p = normalize_funding_pct(c.get("funding_pct") if c.get("funding_pct") is not None else ext.get("funding_pct", 0.0))

        c["turnover_m"] = t_m
        c["oi_m"] = o_m
        c["funding_pct"] = f_p
        filtered_candidates.append(c)

    return filtered_candidates


def fetch_top_gainer_candidates():
    """오직 바이비트 V5 선물 상장 종목 중에서만 Top 5 후보군 추출"""
    if api_mod is not None and hasattr(api_mod, "get_scanner_candidates"):
        try:
            res = api_mod.get_scanner_candidates()
            cands = res[0] if isinstance(res, tuple) else res
            if isinstance(cands, list) and len(cands) > 0:
                return enrich_ticker_metrics(cands)
        except Exception as e:
            print(f"[api.get_scanner_candidates 오류] {e}")

    return []


def extract_ts_ms(df, pos, now_ts, interval_min=5):
    fallback_ms = int((now_ts - max(0, (len(df) - 1 - pos)) * interval_min * 60) * 1000)
    if "timestamp" in df.columns:
        val = df["timestamp"].iloc[pos]
        if isinstance(val, pd.Timestamp):
            return int(val.timestamp() * 1000)
        try:
            num = float(val)
            if not np.isnan(num) and num > 0:
                return int(num * 1000) if num < 1e11 else int(num)
        except Exception:
            pass
    return fallback_ms


def slice_df_window_ms(df, start_ms, end_ms, now_ts, interval_min=5):
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        try:
            ts_s = pd.to_numeric(df["timestamp"], errors="coerce")
            if ts_s.notna().any():
                if ts_s.max() < 1e11:
                    ts_s = ts_s * 1000.0
                sub = df[(ts_s >= start_ms) & (ts_s <= end_ms)]
                if not sub.empty:
                    return sub
        except Exception:
            pass
    n = len(df)
    end_bars_ago = max(0, int((now_ts * 1000 - end_ms) / (interval_min * 60000)))
    start_bars_ago = max(end_bars_ago + 1, int((now_ts * 1000 - start_ms) / (interval_min * 60000)) + 1)
    idx_start = max(0, n - start_bars_ago)
    idx_end = max(idx_start + 1, n - end_bars_ago)
    return df.iloc[idx_start:idx_end]


def reverse_engineer_top_coin(coin, now_ts):
    symbol = str(coin.get("symbol", "UNKNOWN")).upper()
    base_check = symbol[:-4] if symbol.endswith("USDT") else symbol
    if not base_check.isascii() or is_tradfi_ticker(base_check):
        return None

    valid_syms = _get_bybit_whitelist()
    if valid_syms and symbol not in valid_syms:
        return None

    try:
        df_5m = fetch_klines(symbol, "5", limit=300)
        if df_5m is None or df_5m.empty or len(df_5m) < 25:
            return None

        df_1h = fetch_klines(symbol, "60", limit=75)
        df_5m = df_5m.reset_index(drop=True)

        for p in (5, 10, 20, 50, 112):
            df_5m[f"ma{p}"] = df_5m["close"].rolling(p, min_periods=3).mean()
            df_5m[f"ema{p}"] = df_5m["close"].ewm(span=p, adjust=False).mean()

        df_5m["rsi6"] = calc_rsi(df_5m["close"], 6)
        df_5m["kdj_j"] = calc_kdj_j(df_5m, 9)
        df_5m["vol_ma20"] = df_5m["volume"].rolling(20, min_periods=3).mean().shift(1).bfill()
        df_5m["vol_mult"] = (df_5m["volume"] / df_5m["vol_ma20"].replace(0, np.nan)).fillna(1.0)
        atr_5m = calc_atr(df_5m, 14)

        ribbon_max = df_5m[["ema5", "ema20", "ema50"]].max(axis=1)
        ribbon_min = df_5m[["ema5", "ema20", "ema50"]].min(axis=1)
        df_5m["ribbon_spread"] = (ribbon_max - ribbon_min) / df_5m["close"].replace(0, np.nan) * 100.0

        obv, cvd, _ = calc_obv_and_cvd(df_5m)
        df_5m["obv"] = obv
        df_5m["cvd"] = cvd

        start_idx = min(20, max(3, len(df_5m) // 8))
        search_df = df_5m.iloc[start_idx:-2].copy()
        if search_df.empty:
            search_df = df_5m.iloc[:-1].copy()

        overall_low = to_float(search_df["low"].min(), to_float(df_5m["close"].iloc[-1]))
        overall_high = to_float(search_df["high"].max(), to_float(df_5m["close"].iloc[-1]))
        price_ceiling_for_ignition = overall_low + (overall_high - overall_low) * 0.55

        ign_candidates = search_df[
            (search_df["close"] > search_df["open"])
            & (search_df["vol_mult"] >= 2.2)
            & (search_df["close"] <= price_ceiling_for_ignition * 1.05)
        ]
        if ign_candidates.empty:
            ign_candidates = search_df[(search_df["close"] > search_df["open"]) & (search_df["vol_mult"] >= 1.6)]
        if ign_candidates.empty:
            ign_idx = search_df["vol_mult"].idxmax()
        else:
            ign_idx = ign_candidates.index[0]

        ign_pos = int(df_5m.index.get_loc(ign_idx))
        ign_row = df_5m.iloc[ign_pos]
        ign_price = to_float(ign_row["close"])
        ign_open = to_float(ign_row["open"], ign_price)
        ign_high = to_float(ign_row["high"], ign_price)
        ign_low = to_float(ign_row["low"], ign_price)
        ign_vol = to_float(ign_row["volume"], 1.0)
        ign_vol_mult = to_float(ign_row["vol_mult"], 2.5)

        ign_ts_ms = extract_ts_ms(df_5m, ign_pos, now_ts, interval_min=5)
        ign_dt = datetime.fromtimestamp(ign_ts_ms / 1000.0, tz=KST)
        ign_time_str = ign_dt.strftime("%m/%d %H:%M KST")
        ign_minute = ign_dt.minute

        post_df = df_5m.iloc[ign_pos:]
        peak_pos_rel = post_df["high"].idxmax()
        peak_pos = int(df_5m.index.get_loc(peak_pos_rel))
        peak_high = to_float(post_df["high"].max(), ign_price)
        pump_from_ign_pct = max(1.0, (peak_high - ign_price) / max(ign_price, 1e-8) * 100.0)
        hours_to_peak = max(0.2, round((peak_pos - ign_pos) * 5.0 / 60.0, 1))

        pre_start = max(0, ign_pos - 24)
        pre_df = df_5m.iloc[pre_start:ign_pos]
        if pre_df.empty:
            pre_df = df_5m.iloc[:max(1, ign_pos)]
        pre_high = to_float(pre_df["high"].max(), ign_price)
        pre_low = to_float(pre_df["low"].min(), ign_price)
        pre_box_pct = max(1.0, (pre_high - pre_low) / max(pre_low, 1e-8) * 100.0)

        wave_h = max(ign_high - pre_low, ign_price * 0.055)
        waves = extract_three_wave_structure(df_5m, ign_pos, peak_pos, ign_price, pre_low, wave_h)

        pre5_df = df_5m.iloc[max(0, ign_pos - 6):ign_pos]
        if pre5_df.empty:
            pre5_df = pre_df
        vol_ma20_at_ign = to_float(ign_row["vol_ma20"], 1.0)
        pre5_min_vol_ratio = to_float(pre5_df["volume"].min(), vol_ma20_at_ign) / max(vol_ma20_at_ign, 1e-8)
        pre5_min_ribbon = to_float(pre5_df["ribbon_spread"].min(), 2.5)

        ma5_i = to_float(ign_row["ma5"], ign_price)
        ma20_i = to_float(ign_row["ma20"], ign_price)
        ma50_i = to_float(ign_row["ma50"], ign_price)
        ma112_i = to_float(ign_row["ma112"], ign_price)

        full_aligned = ma5_i > ma20_i > ma50_i > ma112_i
        short_aligned = ma5_i > ma20_i > ma50_i

        min_body_floor = pre5_df["close"] * 0.0015
        body_s = np.maximum((pre5_df["close"] - pre5_df["open"]).abs(), min_body_floor)
        l_wick = (pre5_df[["open", "close"]].min(axis=1) - pre5_df["low"]).clip(lower=0.0)
        spring_wick_ratio = min(15.0, to_float((l_wick / body_s).max(), 0.0))
        cond_spring = (not short_aligned) and (spring_wick_ratio >= 1.4) and (ign_price > ma20_i)

        obv_up = to_float(pre_df["obv"].iloc[-1], 0) > to_float(pre_df["obv"].iloc[:10].mean(), 0) if len(pre_df) >= 4 else True
        cvd_up = to_float(pre_df["cvd"].iloc[-1], 0) > to_float(pre_df["cvd"].iloc[:10].mean(), 0) if len(pre_df) >= 4 else True

        bb_std = df_5m["close"].rolling(20, min_periods=5).std()
        bb_u = df_5m["ma20"] + 2.0 * bb_std
        bb_l = df_5m["ma20"] - 2.0 * bb_std
        kc_u = df_5m["ma20"] + 1.5 * atr_5m
        kc_l = df_5m["ma20"] - 1.5 * atr_5m
        ttm_sqz = bool(((bb_u < kc_u) & (bb_l > kc_l)).iloc[max(0, ign_pos - 8):max(1, ign_pos)].any())

        after_4 = df_5m.iloc[ign_pos + 1: min(len(df_5m), ign_pos + 5)]
        if not after_4.empty:
            pullback_low = to_float(after_4["low"].min(), ign_price)
            pullback_min_vol = to_float(after_4["volume"].min(), ign_vol)
            dryup_ratio = pullback_min_vol / max(ign_vol, 1e-8)
        else:
            pullback_low = ign_low
            dryup_ratio = 0.25

        sr_err_pct = abs(pullback_low - pre_high) / max(pre_high, 1e-8) * 100.0
        impulse_len = max(ign_high - pre_high, ign_price * 0.004)
        bull_flag_line = ign_high - impulse_len * 0.382
        candle_mid_fvg = (ign_open + ign_high) / 2.0

        is_sr_flip = sr_err_pct <= 1.50 and dryup_ratio <= 0.40
        is_bull_flag = (pullback_low >= bull_flag_line) and (dryup_ratio <= 0.45)
        is_fvg_50_hold = (pullback_low >= candle_mid_fvg * 0.998) and (dryup_ratio <= 0.45)
        is_pocket_pivot = (pre5_min_vol_ratio <= 0.42) and (ign_vol_mult >= 2.8)

        pre_peak_df = df_5m.iloc[ign_pos: peak_pos + 1]
        min_after_ign = to_float(pre_peak_df["low"].min(), ign_low)
        mae_pct = max(1.5, (ign_price - min_after_ign) / max(ign_price, 1e-8) * 100.0)

        if df_1h is not None and not df_1h.empty and len(df_1h) >= 3:
            ma20_1h = to_float(df_1h["close"].rolling(20, min_periods=2).mean().iloc[-1], ign_price)
            htf_aligned = to_float(df_5m["close"].iloc[-1]) >= ma20_1h * 0.99
        else:
            htf_aligned = to_float(df_5m["close"].iloc[-1]) >= ma112_i * 0.99

        turnover_m = to_float(coin.get("turnover_m"))
        oi_m = to_float(coin.get("oi_m"))
        funding_pct = normalize_funding_pct(coin.get("funding_pct", 0.0))
        change_24h = to_float(coin.get("change_24h"), to_float(coin.get("price24hPcnt")) * 100.0)
        oi_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0 if turnover_m > 0 else 0.0

        alphas = []
        if peak_pos < len(df_5m) - 3 and pump_from_ign_pct >= 30.0:
            after_peak_df = df_5m.iloc[peak_pos:]
            pb_low_after_peak = to_float(after_peak_df["low"].min(), peak_high)
            fib_ret = (peak_high - pb_low_after_peak) / max(peak_high - ign_price, 1e-8)
            curr_close = to_float(df_5m["close"].iloc[-1], pb_low_after_peak)
            if 0.35 <= fib_ret <= 0.65 and curr_close >= pb_low_after_peak * 1.03:
                alphas.append(f"대장주 피보나치 {int(round(fib_ret*100))}% 눌림목 지지 후 2파동 반등(Track C)")

        if is_bull_flag:
            alphas.append(f"High-Tight Bull Flag 공중눌림(돌파봉 상단 38.2% 사수·거래량 {int(round(dryup_ratio*100))}% 건조)")
        if is_pocket_pivot:
            alphas.append(f"포켓피벗(Pocket Pivot) 초동 직격(발화전 거래량 {int(round(pre5_min_vol_ratio*100))}% 극건조 후 {ign_vol_mult:.1f}배 폭발)")
        if is_fvg_50_hold and not is_bull_flag:
            alphas.append("장대양봉 중심값(50% Equilibrium/FVG) 칼지지 반등")
        if cond_spring:
            alphas.append(f"유동성 스윕(Spring 아랫꼬리 {spring_wick_ratio:.1f}배 스탑헌팅 후 MA20 탈환)")
        if cvd_up:
            alphas.append("CVD 강세/흡수 다이버전스(선행 시장가 매수 누적)")
        if ttm_sqz:
            alphas.append("TTM 켈트너 스퀴즈 폭발(BB⊂KC 변동성 압축 후 상단 분출)")
        if pre5_min_ribbon <= 2.2:
            alphas.append(f"이평선 리본 초밀집 확산(5·20·50 이격 {pre5_min_ribbon:.1f}% 수렴 후 분출)")
        if ign_minute in (0, 5, 10, 15):
            alphas.append(f"정각 오프닝 드라이브({ign_dt.strftime('%H:%M')} 새 1시간봉 시동 수급)")
        elif ign_minute in (45, 50, 55):
            alphas.append(f"정각 마감 전 종가관리 피벗({ign_dt.strftime('%H:%M')} 선취매 수급)")
        if htf_aligned:
            alphas.append("상위프레임(1H 중심선) 구조적 지지 반등")
        if is_sr_flip:
            alphas.append(f"돌파후 박스상단 S/R Flip 리테스트(오차 {sr_err_pct:.2f}%·거래량 {int(round(dryup_ratio*100))}% 건조)")
        if funding_pct <= 0.015:
            alphas.append(f"숏스퀴즈/청정 수급(펀딩 {funding_pct:+.4f}%·OI비중 {int(round(oi_ratio))}%)")

        if is_sr_flip:
            post_pattern_str = f"전고점 S/R Flip 칼지지(오차 {sr_err_pct:.2f}%) (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"
        elif is_bull_flag:
            post_pattern_str = f"High-Tight Bull Flag 공중눌림(상단 38.2% 지지) (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"
        elif is_fvg_50_hold:
            post_pattern_str = f"장대양봉 50%(FVG) 중심선 지지 반등 (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"
        else:
            post_pattern_str = f"단기 자율 눌림 후 반등 (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"

        if full_aligned:
            ma_str = "[✅ 완전 정배열(5>20>50>112)]"
        elif short_aligned:
            ma_str = "[✅ 단기 정배열(5>20>50)]"
        else:
            ma_str = "[⚠️ 비정배열(역배열/혼조 돌파)]"

        obv_str = "[✅ 우상향 선행매집]" if (obv_up or cvd_up) else "[➖ 중립/횡보]"

        return {
            "symbol": symbol,
            "base": symbol.replace("USDT", ""),
            "price": to_float(df_5m["close"].iloc[-1]),
            "change_24h": change_24h,
            "turnover_m": turnover_m,
            "oi_m": oi_m,
            "oi_ratio": oi_ratio,
            "funding_pct": funding_pct,
            "ign_time_str": ign_time_str,
            "ign_price": ign_price,
            "peak_high": peak_high,
            "pump_from_ign_pct": pump_from_ign_pct,
            "waves": waves,
            "hours_to_peak": hours_to_peak,
            "ma_str": ma_str,
            "full_aligned": full_aligned,
            "short_aligned": short_aligned,
            "cond_spring": cond_spring,
            "pre_box_pct": pre_box_pct,
            "obv_str": obv_str,
            "obv_up": obv_up,
            "cvd_up": cvd_up,
            "ign_vol_mult": ign_vol_mult,
            "pre5_min_vol_ratio": pre5_min_vol_ratio,
            "pre5_min_ribbon": pre5_min_ribbon,
            "dryup_ratio": dryup_ratio,
            "is_sr_flip": is_sr_flip,
            "is_bull_flag": is_bull_flag,
            "is_fvg_50_hold": is_fvg_50_hold,
            "is_pocket_pivot": is_pocket_pivot,
            "ttm_sqz": ttm_sqz,
            "htf_aligned": htf_aligned,
            "post_pattern_str": post_pattern_str,
            "alphas": alphas,
            "mae_pct": mae_pct,
            "support_line": pre_low,
            "swing_high_line": ign_high,
        }
    except Exception as e:
        print(f"[종목 역추적 오류] {symbol}: {type(e).__name__} - {e}")
        return None


def analyze_scanner_coins_and_postmortem(now_ts):
    valid_syms = _get_bybit_whitelist()
    state = load_json_file(STATE_FILE, {"tracked": {}, "sl_history": {}, "signal_episodes": {}})
    if not isinstance(state, dict):
        state = {"tracked": {}, "sl_history": {}, "signal_episodes": {}}

    tracked = state.get("tracked", {})
    if not isinstance(tracked, dict):
        tracked = {}

    sl_history = state.get("sl_history", {})
    if not isinstance(sl_history, dict):
        sl_history = {}

    tp_win_count = 0
    be_protect_count = 0
    active_count = 0
    sl_count_48h = 0
    counted_sl_keys = set()

    scanner_wave_stats = {
        "w1_gains": [], "w1_fibs": [], "w1_ma20_exts": [], "w1_ma50_exts": [], "w1_rsis": [], "w1_kdjs": [], "w1_holds": [],
        "w2_gains": [], "w2_fibs": [], "w2_ma20_exts": [], "w2_ma50_exts": [], "w2_rsis": [], "w2_kdjs": [],
        "w3_gains": [], "w3_fibs": [], "w3_ma20_exts": [], "w3_ma50_exts": [], "w3_rsis": [], "w3_kdjs": [],
        "coin_wave_samples": [],
    }

    for sym in list(tracked.keys()):
        t_item = tracked.get(sym)
        if not isinstance(t_item, dict):
            continue
        sym_u = str(sym).upper()
        base_c = sym_u[:-4] if sym_u.endswith("USDT") else sym_u
        if not base_c.isascii() or is_tradfi_ticker(base_c):
            continue
        # 바이비트 미상장 종목($UNITAS, $BNC 등)은 성적표 및 분석에서 즉시 제외
        if valid_syms and sym_u not in valid_syms:
            continue

        c_at = int(t_item.get("created_at", 0))
        if (now_ts - c_at) > 172800:
            continue

        sl_key = f"{sym}_{c_at}"
        is_sl_item = bool(t_item.get("sl_alerted") or (sl_key in sl_history))

        stop_p_val = to_float(t_item.get("initial_stop_price") or t_item.get("stop_price"))
        entry_low = to_float(t_item.get("entry_low"))
        entry_high = to_float(t_item.get("entry_high"))
        entry_p = to_float(t_item.get("entry_price"))
        if entry_p <= 0:
            entry_p = (entry_low + entry_high) / 2.0 if entry_low > 0 else stop_p_val

        item_sl_pct = float(FIXED_SL_PCT)

        tps_list = t_item.get("tps") or []
        tp1_val = to_float(tps_list[0]) if len(tps_list) >= 1 else entry_p * 1.085

        if entry_p > 0:
            df_15m_chk = fetch_klines(sym_u, "15", limit=180)
            if not df_15m_chk.empty:
                for p in (5, 10, 20, 50, 112):
                    df_15m_chk[f"ma{p}"] = df_15m_chk["close"].rolling(p, min_periods=2).mean()
                df_15m_chk["rsi6"] = calc_rsi(df_15m_chk["close"], 6)
                df_15m_chk["kdj_j"] = calc_kdj_j(df_15m_chk, 9)
                df_15m_chk["vol_ma20"] = df_15m_chk["volume"].rolling(20, min_periods=3).mean().shift(1).bfill()
                df_15m_chk["vol_mult"] = (df_15m_chk["volume"] / df_15m_chk["vol_ma20"].replace(0, np.nan)).fillna(1.0)

                end_eval_ms = int(t_item.get("sl_at") or now_ts) * 1000 if is_sl_item else now_ts * 1000
                win_15m = slice_df_window_ms(df_15m_chk, c_at * 1000, end_eval_ms, now_ts, interval_min=15)
                if not win_15m.empty:
                    ign_pos_15 = int(df_15m_chk.index.get_loc(win_15m.index[0]))
                    peak_idx_15 = win_15m["high"].idxmax()
                    peak_pos_15 = int(df_15m_chk.index.get_loc(peak_idx_15))
                    max_h_15m = to_float(win_15m["high"].max(), entry_p)
                    fav_pct = round((max_h_15m - entry_p) / entry_p * 100.0, 2)
                    t_item["max_favorable_pct"] = max(to_float(t_item.get("max_favorable_pct")), max(0.0, fav_pct))

                    if not is_sl_item:
                        if max_h_15m >= tp1_val:
                            t_item["tp1_hit"] = True
                        elif fav_pct >= item_sl_pct:
                            t_item["be_armed"] = True

                    if fav_pct >= 5.0 and peak_pos_15 >= ign_pos_15 and not is_heavy_major_coin(base_c):
                        sup_p = entry_p * 0.960
                        w_h = max(to_float(t_item.get("swing_high_line"), entry_p * 1.02) - sup_p, entry_p * 0.055)
                        w_res = extract_three_wave_structure(df_15m_chk, ign_pos_15, peak_pos_15, entry_p, sup_p, w_h)
                        t_item["wave_raw"] = w_res

                        scanner_wave_stats["w1_gains"].append(w_res["w1"]["gain_pct"])
                        scanner_wave_stats["w1_fibs"].append(w_res["w1"]["fib_mult"])
                        scanner_wave_stats["w1_ma20_exts"].append(w_res["w1"]["ma20_ext_pct"])
                        scanner_wave_stats["w1_ma50_exts"].append(w_res["w1"]["ma50_ext_pct"])
                        scanner_wave_stats["w1_rsis"].append(w_res["w1"]["rsi6"])
                        scanner_wave_stats["w1_kdjs"].append(w_res["w1"]["kdj_j"])
                        scanner_wave_stats["w1_holds"].append(w_res["w1"]["hold_tag"])

                        if fav_pct >= 10.0:
                            scanner_wave_stats["w2_gains"].append(w_res["w2"]["gain_pct"])
                            scanner_wave_stats["w2_fibs"].append(w_res["w2"]["fib_mult"])
                            scanner_wave_stats["w2_ma20_exts"].append(w_res["w2"]["ma20_ext_pct"])
                            scanner_wave_stats["w2_ma50_exts"].append(w_res["w2"]["ma50_ext_pct"])
                            scanner_wave_stats["w2_rsis"].append(w_res["w2"]["rsi6"])
                            scanner_wave_stats["w2_kdjs"].append(w_res["w2"]["kdj_j"])

                        if fav_pct >= 14.0:
                            scanner_wave_stats["w3_gains"].append(w_res["w3"]["gain_pct"])
                            scanner_wave_stats["w3_fibs"].append(w_res["w3"]["fib_mult"])
                            scanner_wave_stats["w3_ma20_exts"].append(w_res["w3"]["ma20_ext_pct"])
                            scanner_wave_stats["w3_ma50_exts"].append(w_res["w3"]["ma50_ext_pct"])
                            scanner_wave_stats["w3_rsis"].append(w_res["w3"]["rsi6"])
                            scanner_wave_stats["w3_kdjs"].append(w_res["w3"]["kdj_j"])

                        scanner_wave_stats["coin_wave_samples"].append({
                            "base": base_c,
                            "max_fav": fav_pct,
                            "w1": w_res["w1"],
                            "w2": w_res["w2"],
                            "w3": w_res["w3"],
                        })

        fav_now = to_float(t_item.get("max_favorable_pct"), 0.0)

        if is_sl_item and fav_now < 7.5:
            sl_count_48h += 1
            counted_sl_keys.add(sl_key)
        elif (t_item.get("tp1_hit") or t_item.get("tp2_hit") or t_item.get("tp3_completed")) and fav_now >= 7.0:
            tp_win_count += 1
        elif t_item.get("be_armed") and fav_now >= item_sl_pct:
            be_protect_count += 1
        elif is_sl_item:
            sl_count_48h += 1
            counted_sl_keys.add(sl_key)
        else:
            active_count += 1

        if is_sl_item and sl_key not in sl_history:
            sl_history[sl_key] = {
                "symbol": sym_u,
                "track_type": t_item.get("track_type", "TRACK_A"),
                "score": int(t_item.get("score", 85)),
                "entry_price": entry_p,
                "stop_price": stop_p_val,
                "sl_trigger_price": to_float(t_item.get("sl_trigger_price"), stop_p_val),
                "sl_pct": FIXED_SL_PCT,
                "created_at": c_at,
                "sl_at": int(t_item.get("sl_at", 0)),
                "hold_minutes": int(t_item.get("hold_minutes", 0)),
                "max_favorable_pct": fav_now,
                "vol_mult": to_float(t_item.get("vol_mult", 1.0)),
                "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
                "oi_m": to_float(t_item.get("oi_m", 0.0)),
                "turnover_m": to_float(t_item.get("turnover_m", 0.0)),
                "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
                "reasons": t_item.get("reasons", []),
            }

    recent_sl_list = []
    for k_sl, v_sl in sl_history.items():
        if isinstance(v_sl, dict) and (now_ts - int(v_sl.get("created_at", now_ts))) <= 172800:
            sym_sl = str(v_sl.get("symbol", "")).upper()
            if valid_syms and sym_sl not in valid_syms:
                continue
            recent_sl_list.append(v_sl)
            if k_sl not in counted_sl_keys:
                sl_count_48h += 1
                counted_sl_keys.add(k_sl)

    recent_sl_list.sort(key=lambda x: int(x.get("sl_at") or x.get("created_at") or 0), reverse=True)

    other_lines = [
        "━━━━━━━━━━━━━━━━━━",
        "🔬 [내 스캐너 포착 '다른 코인들'의 1차·2차·3차 고점 형성 로데이터 심층 분석]",
    ]
    samples = sorted(scanner_wave_stats["coin_wave_samples"], key=lambda x: x["max_fav"], reverse=True)
    if not samples:
        other_lines.append("• 최근 48h 포착 종목의 파동 로데이터를 누적 중입니다.")
    else:
        w1_g = float(np.median(scanner_wave_stats["w1_gains"]))
        w1_f = float(np.median(scanner_wave_stats["w1_fibs"]))
        w1_e20 = float(np.median(scanner_wave_stats["w1_ma20_exts"]))
        w1_r = float(np.median(scanner_wave_stats["w1_rsis"]))
        w1_k = float(np.median(scanner_wave_stats["w1_kdjs"]))

        w2_g = float(np.median(scanner_wave_stats["w2_gains"])) if scanner_wave_stats["w2_gains"] else w1_g * 1.65
        w2_f = float(np.median(scanner_wave_stats["w2_fibs"])) if scanner_wave_stats["w2_fibs"] else w1_f * 1.45
        w2_e20 = float(np.median(scanner_wave_stats["w2_ma20_exts"])) if scanner_wave_stats["w2_ma20_exts"] else w1_e20 * 1.6
        w2_r = float(np.median(scanner_wave_stats["w2_rsis"])) if scanner_wave_stats["w2_rsis"] else 78.0

        w3_g = float(np.median(scanner_wave_stats["w3_gains"])) if scanner_wave_stats["w3_gains"] else w2_g * 1.45
        w3_f = float(np.median(scanner_wave_stats["w3_fibs"])) if scanner_wave_stats["w3_fibs"] else w2_f * 1.30
        w3_e20 = float(np.median(scanner_wave_stats["w3_ma20_exts"])) if scanner_wave_stats["w3_ma20_exts"] else w2_e20 * 1.4
        w3_r = float(np.median(scanner_wave_stats["w3_rsis"])) if scanner_wave_stats["w3_rsis"] else 84.0

        other_lines.extend([
            f"• 분석 표본: 최근 48h 스캐너 포착 상승 파동 {len(samples)}종목 전수 역추적",
            f"  - [1차: 첫 조정 형성 지점]: 중앙값 +{w1_g:.1f}% | Fib {w1_f:.2f}배 | 15m MA20이격 +{w1_e20:.1f}% | RSI6 {w1_r:.0f}·KDJ_J {w1_k:.0f}",
            f"  - [2차: 최종고점 전 중간조정 고점]: 중앙값 +{w2_g:.1f}% | Fib {w2_f:.2f}배 | 15m MA20이격 +{w2_e20:.1f}% | RSI6 {w2_r:.0f}",
            f"  - [3차: 최종 클라이맥스 고점]: 중앙값 +{w3_g:.1f}% | Fib {w3_f:.2f}배 | 15m MA20이격 +{w3_e20:.1f}% | RSI6 {w3_r:.0f}",
            "• [주요 포착 코인별 1·2·3차 파동 실측 로데이터 상위 3선]:",
        ])
        for s_item in samples[:3]:
            sw1, sw2, sw3 = s_item["w1"], s_item["w2"], s_item["w3"]
            other_lines.append(
                f"  - ${s_item['base']}: 1차첫조정 +{sw1['gain_pct']:.1f}%(Fib {sw1['fib_mult']:.2f}·MA20+{sw1['ma20_ext_pct']:.1f}%·{sw1['hold_tag']} 지지) "
                f"→ 2차고점 +{sw2['gain_pct']:.1f}%(Fib {sw2['fib_mult']:.2f}) "
                f"→ 3차최종 +{sw3['gain_pct']:.1f}%(Fib {sw3['fib_mult']:.2f}·RSI {sw3['rsi6']:.0f})"
            )

    total_eval = tp_win_count + be_protect_count + sl_count_48h + active_count

    lines = [
        "\n".join(other_lines),
        "━━━━━━━━━━━━━━━━━━",
        "🚨 [내 스캐너 알림 종목 오답 노트 (최근 48h 손절 심층 역추적 & 첫 저항 로데이터)]",
    ]

    if total_eval == 0 and not recent_sl_list:
        lines.append("• 최근 48시간 내 손절 이탈 종목이 없습니다. (방어 필터 클린 유지 중)")
        return "\n".join(lines), [], scanner_wave_stats

    decided = tp_win_count + be_protect_count + sl_count_48h
    win_rate_str = f"{((tp_win_count + be_protect_count) / decided * 100.0):.1f}%" if decided > 0 else "집계중"
    lines.append(
        f"• 최근 48h 스캐너 성적표: 총 {total_eval}건 포착 | 익절·본전방어 {tp_win_count + be_protect_count}건(익절 {tp_win_count}·1R방어 {be_protect_count}) | 손절 {sl_count_48h}건 | 순항중 {active_count}건 (확정방어승률 {win_rate_str})"
    )

    if not recent_sl_list:
        lines.append(f"• ✅ 손절(SL) 발생 0건: 현재 -{FIXED_SL_PCT}% 고정 손절 및 1R(+{FIXED_SL_PCT}%) 본전 방어가 완벽히 작동하고 있습니다.")
        return "\n".join(lines), [], scanner_wave_stats

    df_btc_15m = fetch_klines("BTCUSDT", "15", limit=200)
    cause_counter = {
        "BTC_DUMP": 0,
        "STOP_HUNT_REBOUND": 0,
        "PROFIT_GIVEBACK": 0,
        "FAKEOUT_WHIPSAW": 0,
        "LOW_OI_DRAIN": 0,
    }
    prescription_notes = []

    track_label_map = {
        "TRACK_A": "Track A(선취매)",
        "TRACK_B": "Track B(돌파)",
        "TRACK_C": "Track C(대장주눌림)",
    }

    for idx, sl_item in enumerate(recent_sl_list[:5], 1):
        sym = sl_item.get("symbol", "UNKNOWN")
        base = sym.replace("USDT", "")
        t_type = track_label_map.get(sl_item.get("track_type", "TRACK_A"), "Track A(선취매)")
        entry_p = to_float(sl_item.get("entry_price"))
        stop_p = to_float(sl_item.get("stop_price"))
        sl_pct = int(FIXED_SL_PCT)

        created_at = int(sl_item.get("created_at", now_ts - 3600))
        sl_at = int(sl_item.get("sl_at", 0))
        max_fav = to_float(sl_item.get("max_favorable_pct", 0.0))
        rsi_entry = to_float(sl_item.get("rsi6_15m", 50.0))

        age_hours = (now_ts - created_at) / 3600.0
        iv_str = "5" if age_hours <= 14.0 else "15"
        iv_min = 5 if iv_str == "5" else 15
        df_coin = fetch_klines(sym, iv_str, limit=200)

        rebound_after_sl_pct = 0.0
        peak_fib_at_sl = 1.4
        peak_ma20_ext_at_sl = 1.8
        peak_rsi_at_sl = 60.0
        resist_ma_tag = "단기매물대"

        if df_coin is not None and not df_coin.empty and entry_p > 0:
            df_coin["ma20"] = df_coin["close"].rolling(20, min_periods=5).mean()
            df_coin["ma50"] = df_coin["close"].rolling(50, min_periods=10).mean()
            df_coin["ma112"] = df_coin["close"].rolling(112, min_periods=20).mean()
            df_coin["rsi6"] = calc_rsi(df_coin["close"], 6)
            after_entry_df = slice_df_window_ms(df_coin, created_at * 1000, now_ts * 1000, now_ts, interval_min=iv_min)
            if not after_entry_df.empty:
                if (sl_at <= 0 or sl_at >= now_ts - 120) and stop_p > 0:
                    breach_rows = after_entry_df[after_entry_df["low"] <= stop_p]
                    if not breach_rows.empty:
                        first_b_pos = df_coin.index.get_loc(breach_rows.index[0])
                        sl_at = int(extract_ts_ms(df_coin, first_b_pos, now_ts, interval_min=iv_min) // 1000)

                if sl_at <= created_at:
                    sl_at = min(now_ts, created_at + 1800)

                hold_win = slice_df_window_ms(df_coin, created_at * 1000, (sl_at + iv_min * 60) * 1000, now_ts, interval_min=iv_min)
                if not hold_win.empty:
                    pk_idx = hold_win["high"].idxmax()
                    actual_high = to_float(hold_win.loc[pk_idx, "high"], entry_p)
                    calc_fav = round((actual_high - entry_p) / entry_p * 100.0, 2)
                    max_fav = max(max_fav, max(0.0, calc_fav))

                    ma20_pk = to_float(df_coin.loc[pk_idx, "ma20"], entry_p)
                    ma50_pk = to_float(df_coin.loc[pk_idx, "ma50"], entry_p)
                    ma112_pk = to_float(df_coin.loc[pk_idx, "ma112"], entry_p)
                    peak_ma20_ext_at_sl = round(max(0.1, (actual_high - ma20_pk) / max(ma20_pk, 1e-8) * 100.0), 1)
                    peak_rsi_at_sl = round(to_float(df_coin.loc[pk_idx, "rsi6"], 60.0), 1)
                    peak_fib_at_sl = round(max(1.0, (actual_high - entry_p * 0.960) / (entry_p * 0.055)), 2)

                    if abs(actual_high - ma112_pk) / max(ma112_pk, 1e-8) <= 0.015:
                        resist_ma_tag = "MA112 장기이평 저항"
                    elif abs(actual_high - ma50_pk) / max(ma50_pk, 1e-8) <= 0.015:
                        resist_ma_tag = "MA50 중기이평 저항"
                    else:
                        resist_ma_tag = f"MA20 +{peak_ma20_ext_at_sl}% 이격저항"

                after_sl_win = slice_df_window_ms(df_coin, sl_at * 1000, (sl_at + 7200) * 1000, now_ts, interval_min=iv_min)
                if not after_sl_win.empty and stop_p > 0:
                    post_sl_high = to_float(after_sl_win["high"].max(), stop_p)
                    rebound_after_sl_pct = (post_sl_high - stop_p) / max(stop_p, 1e-8) * 100.0

        if sl_at <= created_at:
            sl_at = now_ts
        hold_min = max(5, int((sl_at - created_at) // 60))

        btc_drop_pct = 0.0
        if df_btc_15m is not None and not df_btc_15m.empty:
            btc_win = slice_df_window_ms(df_btc_15m, (sl_at - 3600) * 1000, (sl_at + 900) * 1000, now_ts, interval_min=15)
            if not btc_win.empty:
                b_high = to_float(btc_win["high"].max(), 1.0)
                b_low = to_float(btc_win["low"].min(), 1.0)
                btc_drop_pct = (b_low - b_high) / max(b_high, 1e-8) * 100.0

        if max_fav >= sl_pct:
            cause_tag = f"유형3 [1R(+{sl_pct}%) 도달(최고 +{max_fav:.1f}%) 후 반락]"
            cause_detail = f"첫 조정 고점(Fib {peak_fib_at_sl}배·{resist_ma_tag}·RSI {peak_rsi_at_sl})에서 꺾임 (1R 본전방어 및 TP1 학습 반영)"
            cause_counter["PROFIT_GIVEBACK"] += 1
        elif btc_drop_pct <= -0.80:
            cause_tag = f"유형1 [BTC 도미노 급락({btc_drop_pct:.2f}%)]"
            cause_detail = f"반락 전 첫 고점 +{max_fav:.1f}%(Fib {peak_fib_at_sl}배·{resist_ma_tag}) 형성 후 BTC 투매 연동 하락"
            cause_counter["BTC_DUMP"] += 1
        elif rebound_after_sl_pct >= 4.0:
            cause_tag = f"유형2 [스탑헌팅 후 +{rebound_after_sl_pct:.1f}% V자 재상승]"
            cause_detail = f"아랫꼬리 털기 후 급반등 (신규 -{FIXED_SL_PCT}% 고정 손절 적용 시 휩소 방어 가능)"
            cause_counter["STOP_HUNT_REBOUND"] += 1
        elif hold_min <= 45 and max_fav < 1.2:
            cause_tag = f"유형4 [돌파 직후 윗꼬리 휩소(Fakeout·{hold_min}분 만에 이탈)]"
            cause_detail = f"15m RSI({rsi_entry:.1f}) 단기 고점권에서 후속 매수세 실종으로 즉시 반락"
            cause_counter["FAKEOUT_WHIPSAW"] += 1
        else:
            cause_tag = "유형5 [박스 하단 수급 이탈 및 추세 둔화]"
            cause_detail = f"초동 반등(+{max_fav:.1f}%·{resist_ma_tag}) 후 후속 거래량 부족으로 지지선 이탈"
            cause_counter["LOW_OI_DRAIN"] += 1

        sl_dt_str = datetime.fromtimestamp(sl_at, tz=KST).strftime("%m/%d %H:%M")
        lines.append(
            f"  {idx}) ❌ ${base} ({t_type} | SL -{sl_pct}% | {sl_dt_str} 이탈 | 보유 {hold_min}분 | 첫고점 +{max_fav:.1f}%)"
        )
        lines.append(f"     - [패인 & 저항로데이터]: {cause_tag} — {cause_detail}")

    if cause_counter["BTC_DUMP"] > 0:
        prescription_notes.append(
            "• [BTC 급락 방어]: BTC 15m(-0.7%↓) 및 1H(-1.2%↓) 투매 시 컷오프 85점→90점 상향 & 상대강도(24h +2.5%↑) 디커플링 필터 유지"
        )
    if cause_counter["STOP_HUNT_REBOUND"] > 0:
        prescription_notes.append(
            f"• [스탑헌팅 방어]: 급등 전 아랫꼬리 휩소 견디기 위해 FIXED_SL_PCT = {FIXED_SL_PCT}% 고정 손절 및 4h V자 재진입 유지"
        )
    if cause_counter["PROFIT_GIVEBACK"] > 0:
        prescription_notes.append(
            f"• [1R 본전 방어 & 1차 조정지점 학습]: +{FIXED_SL_PCT}% 도달 시 본전 방어 고정 & 첫 조정 저항 지점(Fib·MA20 이격)에 맞춰 TP1 조율"
        )
    if cause_counter["FAKEOUT_WHIPSAW"] > 0:
        prescription_notes.append(
            "• [윗꼬리 휩소 방어]: 돌파 직후 꺾인 Fakeout 방지 — PULLBACK_DRYUP_MAX를 0.20 이하로 유지 권장"
        )
    if cause_counter["LOW_OI_DRAIN"] > 0:
        prescription_notes.append(
            "• [수급 둔화 & 장기이평 천장 방어]: 머리 위 5m/15m MA112 저항 및 과이격 차단, $PEPE 등 무거운 준메이저·대형 밈코인 배제"
        )

    if prescription_notes:
        lines.append("• 🛠️ [오답 노트 기반 스캐너 업데이트 처방전]:")
        lines.extend([f"  {note}" for note in prescription_notes])

    return "\n".join(lines), prescription_notes, scanner_wave_stats


def calibrate_3wave_tp_system(analyzed_top5, scanner_wave_stats):
    t5_w1_gain = [x["waves"]["w1"]["gain_pct"] for x in analyzed_top5]
    t5_w1_fib = [x["waves"]["w1"]["fib_mult"] for x in analyzed_top5]
    t5_w1_ext = [x["waves"]["w1"]["ma20_ext_pct"] for x in analyzed_top5]
    t5_w1_rsi = [x["waves"]["w1"]["rsi6"] for x in analyzed_top5]

    t5_w2_gain = [x["waves"]["w2"]["gain_pct"] for x in analyzed_top5]
    t5_w2_fib = [x["waves"]["w2"]["fib_mult"] for x in analyzed_top5]
    t5_w2_ext = [x["waves"]["w2"]["ma20_ext_pct"] for x in analyzed_top5]
    t5_w2_rsi = [x["waves"]["w2"]["rsi6"] for x in analyzed_top5]

    t5_w3_gain = [x["waves"]["w3"]["gain_pct"] for x in analyzed_top5]
    t5_w3_fib = [x["waves"]["w3"]["fib_mult"] for x in analyzed_top5]
    t5_w3_ext = [x["waves"]["w3"]["ma20_ext_pct"] for x in analyzed_top5]
    t5_w3_rsi = [x["waves"]["w3"]["rsi6"] for x in analyzed_top5]

    t5_g1 = float(np.median(t5_w1_gain)) if t5_w1_gain else 9.0
    t5_g2 = float(np.median(t5_w2_gain)) if t5_w2_gain else 16.0
    t5_g3 = float(np.median(t5_w3_gain)) if t5_w3_gain else 25.0

    t5_f1 = float(np.median(t5_w1_fib)) if t5_w1_fib else 1.80
    t5_f2 = float(np.median(t5_w2_fib)) if t5_w2_fib else 2.80
    t5_f3 = float(np.median(t5_w3_fib)) if t5_w3_fib else 4.00

    t5_e1 = float(np.median(t5_w1_ext)) if t5_w1_ext else 5.0
    t5_e2 = float(np.median(t5_w2_ext)) if t5_w2_ext else 9.0
    t5_e3 = float(np.median(t5_w3_ext)) if t5_w3_ext else 13.5

    pm_w1_gain = scanner_wave_stats.get("w1_gains", [])
    pm_w1_fib = scanner_wave_stats.get("w1_fibs", [])
    pm_w1_ext = scanner_wave_stats.get("w1_ma20_exts", [])

    pm_w2_gain = scanner_wave_stats.get("w2_gains", [])
    pm_w2_fib = scanner_wave_stats.get("w2_fibs", [])
    pm_w2_ext = scanner_wave_stats.get("w2_ma20_exts", [])

    pm_w3_gain = scanner_wave_stats.get("w3_gains", [])
    pm_w3_fib = scanner_wave_stats.get("w3_fibs", [])
    pm_w3_ext = scanner_wave_stats.get("w3_ma20_exts", [])

    def _blend_with_anchor(t5_v, pm_arr, anchor_v):
        if len(pm_arr) >= 3:
            pm_v = float(np.median(pm_arr))
            return (t5_v * 0.25) + (pm_v * 0.50) + (anchor_v * 0.25)
        elif len(pm_arr) >= 1:
            pm_v = float(np.median(pm_arr))
            return (t5_v * 0.30) + (pm_v * 0.40) + (anchor_v * 0.30)
        else:
            return (t5_v * 0.40) + (anchor_v * 0.60)

    raw_tp1 = _blend_with_anchor(t5_g1, pm_w1_gain, 8.5)
    tp1_pct = int(np.clip(round(raw_tp1), 8, 10))
    tp1_fib = round(float(np.clip(_blend_with_anchor(t5_f1, pm_w1_fib, 1.75), 1.55, 2.10)), 2)
    tp1_ext = round(float(np.clip(_blend_with_anchor(t5_e1, pm_w1_ext, 5.0), 3.5, 8.0)), 1)
    tp1_rsi = round(float(np.median(t5_w1_rsi + scanner_wave_stats.get("w1_rsis", []))) if t5_w1_rsi else 82.0, 1)

    raw_tp2 = _blend_with_anchor(t5_g2, pm_w2_gain, 15.5)
    tp2_pct = int(np.clip(round(max(tp1_pct + 6, raw_tp2)), 14, 18))
    tp2_fib = round(float(np.clip(max(tp1_fib + 0.65, _blend_with_anchor(t5_f2, pm_w2_fib, 2.75)), 2.30, 3.20)), 2)
    tp2_ext = round(float(np.clip(max(tp1_ext + 2.5, _blend_with_anchor(t5_e2, pm_w2_ext, 8.5)), 6.5, 13.0)), 1)
    tp2_rsi = round(float(np.median(t5_w2_rsi + scanner_wave_stats.get("w2_rsis", []))) if t5_w2_rsi else 86.0, 1)

    raw_tp3 = _blend_with_anchor(t5_g3, pm_w3_gain, 24.0)
    tp3_pct = int(np.clip(round(max(tp2_pct + 7, raw_tp3)), 21, 28))
    tp3_fib = round(float(np.clip(max(tp2_fib + 0.75, _blend_with_anchor(t5_f3, pm_w3_fib, 3.85)), 3.20, 4.30)), 2)
    tp3_ext = round(float(np.clip(max(tp2_ext + 3.5, _blend_with_anchor(t5_e3, pm_w3_ext, 13.0)), 10.0, 18.0)), 1)
    tp3_rsi = round(float(np.median(t5_w3_rsi + scanner_wave_stats.get("w3_rsis", []))) if t5_w3_rsi else 82.0, 1)

    return {
        "tp1_min_pct": float(tp1_pct),
        "tp2_min_pct": float(tp2_pct),
        "tp3_min_pct": float(tp3_pct),
        "tp1_fib_mult": tp1_fib,
        "tp2_fib_mult": tp2_fib,
        "tp3_fib_mult": tp3_fib,
        "tp1_ma20_ext_pct": tp1_ext,
        "tp2_ma20_ext_pct": tp2_ext,
        "tp3_ma20_ext_pct": tp3_ext,
        "tp1_rsi6": tp1_rsi,
        "tp2_rsi6": tp2_rsi,
        "tp3_rsi6": tp3_rsi,
        "top5_w1_med": round(t5_g1, 1),
        "top5_w2_med": round(t5_g2, 1),
        "top5_w3_med": round(t5_g3, 1),
        "pm_w1_med": round(float(np.median(pm_w1_gain)) if pm_w1_gain else tp1_pct, 1),
        "pm_w2_med": round(float(np.median(pm_w2_gain)) if pm_w2_gain else tp2_pct, 1),
        "pm_w3_med": round(float(np.median(pm_w3_gain)) if pm_w3_gain else tp3_pct, 1),
        "pm_sample_cnt": len(pm_w1_gain),
    }


def _extract_flat_cohorts(obj):
    results = []
    if isinstance(obj, list):
        for el in obj:
            results.extend(_extract_flat_cohorts(el))
    elif isinstance(obj, dict):
        c_list = obj.get("coins") or obj.get("items") or obj.get("records") or obj.get("top5")
        if (
            str(obj.get("date_str", "")) in ("cohorts", "snapshots", "history")
            and isinstance(c_list, list)
            and c_list
            and isinstance(c_list[0], dict)
            and ("symbol" not in c_list[0])
        ):
            results.extend(_extract_flat_cohorts(c_list))
            return results

        for key in ("cohorts", "snapshots", "history", "data"):
            if isinstance(obj.get(key), list):
                results.extend(_extract_flat_cohorts(obj[key]))

        if isinstance(c_list, list) and c_list and isinstance(c_list[0], dict) and ("symbol" in c_list[0]):
            ts = int(to_float(obj.get("timestamp") or obj.get("captured_ts") or obj.get("created_at") or obj.get("ts"), 0))
            date_s = str(obj.get("date_str") or obj.get("captured_at") or obj.get("date") or obj.get("cohort_id") or "")
            if (not date_s or date_s in ("cohorts", "snapshots")) and ts > 0:
                date_s = datetime.fromtimestamp(ts, tz=KST).strftime("%Y-%m-%d %H:%M")
            results.append({
                "date_str": date_s or "이전 포착분",
                "timestamp": ts if ts > 0 else int(time.time()) - 172800,
                "coins": c_list,
            })
    return results


def normalize_history_data(raw_data):
    flat = _extract_flat_cohorts(raw_data)
    dedup = {}
    for item in flat:
        k = item.get("date_str", "")
        dedup[k] = item
    res = list(dedup.values())
    res.sort(key=lambda x: int(x.get("timestamp", 0)))
    return res


def evaluate_48h_history(history_list, now_ts, tp_tuning):
    valid_syms = _get_bybit_whitelist()
    today_prefix = datetime.now(KST).strftime("%Y-%m-%d")
    target_snap = None
    for snap in reversed(history_list):
        if not isinstance(snap, dict):
            continue
        age_h = (now_ts - int(snap.get("timestamp", 0))) / 3600.0
        if 20.0 <= age_h <= 96.0 and snap.get("coins"):
            target_snap = snap
            break
    if not target_snap:
        for snap in reversed(history_list):
            if not str(snap.get("date_str", "")).startswith(today_prefix) and snap.get("coins"):
                target_snap = snap
                break

    lines = [
        "━━━━━━━━━━━━━━━━━━",
        "⏱️ [48시간 경과 동일 상위종목 실전 익절(TP1~TP3) 사후 검증]",
    ]
    if not target_snap or not target_snap.get("coins"):
        lines.append("• 오늘 첫 정상 스냅샷이 저장되었습니다. (다음 리포트부터 실전 익절 도달률이 자동 검증됩니다)")
        return "\n".join(lines)

    snap_date = target_snap.get("date_str", "이전 포착분")
    lines[1] = f"⏱️ [48시간 경과 동일 상위종목({snap_date} 포착분) 실전 익절(TP1~TP3) 사후 검증]"

    t1_thr = tp_tuning.get("tp1_min_pct", 8.5)
    t2_thr = tp_tuning.get("tp2_min_pct", 16.0)
    t3_thr = tp_tuning.get("tp3_min_pct", 24.0)

    added_rows = 0
    for item in target_snap.get("coins", []):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", "")).upper()
        if not sym:
            continue
        if valid_syms and sym not in valid_syms:
            continue
        base = sym.replace("USDT", "")
        if not base.isascii() or is_tradfi_ticker(base):
            continue
        ign_p = to_float(item.get("ign_price") or item.get("ignition_price") or item.get("entry_price") or item.get("price"))
        ign_ts = int(item.get("timestamp_ms", int(target_snap.get("timestamp", now_ts - 172800)) * 1000))

        df_1h = fetch_klines(sym, "60", limit=200)
        if df_1h is None or df_1h.empty or ign_p <= 0:
            continue
        post_1h = slice_df_window_ms(df_1h, ign_ts, now_ts * 1000, now_ts, interval_min=60)
        max_h = to_float(post_1h["high"].max() if not post_1h.empty else df_1h["high"].max(), ign_p)
        max_gain_pct = max(0.0, (max_h - ign_p) / ign_p * 100.0)

        if max_gain_pct >= t3_thr:
            badge = "🎯 [3차 TP 전량익절 졸업]"
        elif max_gain_pct >= t2_thr:
            badge = "✅ [2차 TP 돌파]"
        elif max_gain_pct >= t1_thr:
            badge = "✅ [1차 TP 달성]"
        else:
            badge = "⏳ [보합/눌림 진행]"

        lines.append(
            f"• ${base}: 발화가 {fmt_price(ign_p)} → 48h 최고 {fmt_price(max_h)} (+{max_gain_pct:.1f}%) {badge}"
        )
        added_rows += 1

    if added_rows == 0:
        lines.append("• 오늘 첫 정상 스냅샷이 저장되었습니다. (다음 리포트부터 실전 익절 도달률이 자동 검증됩니다)")

    return "\n".join(lines)
