import os, json, time, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import config as cfg
    STATE_FILE = getattr(cfg, "STATE_FILE", "bot_state.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
except Exception:
    cfg = None
    STATE_FILE = "bot_state.json"
    KST = timezone(timedelta(hours=9))
    FIXED_SL_PCT = 6
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}

try:
    import postmortem
except Exception:
    postmortem = None

from api import (
    get_scanner_candidates,
    fetch_klines,
    get_valid_bybit_linear_symbols,
    is_stock_or_tradfi_symbol,
    clean_url,
)

SCHEME = "ht" + "tps://"
HISTORY_FILE = "analyzer_history.json"
DNA_PROFILE_FILE = "top5_dna_profile.json"
MARKET_CAP_CACHE_FILE = "market_cap_cache.json"
DNA_RETENTION_SEC = 14 * 86400
DNA_VERSION = "v3_pure_chart"
STATE_CANDIDATE_FILES = list(dict.fromkeys([STATE_FILE, "bot_state.json", "scanner_state.json"]))

# [백그라운드 자동 학습 전용 모드] 일간 텔레그램 알림 OFF
SEND_DAILY_TELEGRAM = False


def to_float(val, default=0.0):
    try:
        v = float(val)
        return default if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return default


def fmt_price(price):
    p = to_float(price, 0.0)
    if p <= 0:
        return "0.0000"
    return f"{p:,.2f}" if p >= 1000 else (f"{p:.3f}" if p >= 10 else (f"{p:.4f}" if p >= 1 else (f"{p:.5f}" if p >= 0.01 else f"{p:.6f}")))


def send_telegram_report(text):
    if not SEND_DAILY_TELEGRAM:
        print("[일간 백그라운드 모드] 텔레그램 발송 생략 (Top 5 DNA · 시가총액 캐시 자동 학습 완료)")
        return False
    token = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or getattr(cfg, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or getattr(cfg, "TELEGRAM_CHAT_ID", "") or "").strip()
    if not text or not token or not chat_id:
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendMessage")
    for i in range(0, len(text), 3800):
        chunk = text[i : i + 3800]
        for _ in range(3):
            try:
                r = requests.post(url, json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}, timeout=12)
                if r.status_code == 200:
                    time.sleep(0.6)
                    break
            except Exception:
                pass
            time.sleep(1.0)
    return True


def load_json_safe(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, type(default_val)):
                    return d
        except Exception:
            pass
    return default_val


def load_history_safe():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, list):
                    return {"records": d, "learned_tp_params": {}, "dynamic_tp_tuning": {}}
                if isinstance(d, dict):
                    recs = d.get("records") or d.get("history") or d.get("items") or []
                    d["records"] = recs if isinstance(recs, list) else []
                    return d
        except Exception:
            pass
    return {"records": [], "learned_tp_params": {}, "dynamic_tp_tuning": {}}


def save_json_safe(path, data):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        pass


def update_market_cap_cache(now_ts):
    """
    [방안 1 · 시가총액 캐시 일간 자동 갱신]
    - CoinGecko 상위 1,250개(250개 × 5페이지) 시가총액($M)을 수집해 market_cap_cache.json에 저장
    - 동일 티커 충돌 시 시가총액이 더 큰 본진 코인 우선 반영, API 일시 제한 시 기존 캐시 보존
    """
    existing = load_json_safe(MARKET_CAP_CACHE_FILE, {"updated_at": 0, "count": 0, "caps": {}})
    caps = dict(existing.get("caps") or {})
    fresh_caps = {}
    fetched_pages = 0

    for page in range(1, 6):
        try:
            url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/coins/markets")
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 250,
                "page": page,
                "sparkline": "false",
            }
            resp = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=10)
            if resp.status_code == 429:
                time.sleep(5.0)
                resp = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=10)
            if resp.status_code != 200:
                break
            rows = resp.json()
            if not isinstance(rows, list) or not rows:
                break
            fetched_pages += 1
            for it in rows:
                sym_b = str(it.get("symbol") or "").upper().strip()
                mc_usd = to_float(it.get("market_cap"), 0.0)
                if not sym_b or not sym_b.isascii() or mc_usd <= 0:
                    continue
                if is_stock_or_tradfi_symbol(sym_b):
                    continue
                mc_m = round(mc_usd / 1e6, 3)
                if sym_b not in fresh_caps or mc_m > fresh_caps[sym_b]:
                    fresh_caps[sym_b] = mc_m
            if page < 5:
                time.sleep(1.8)
        except Exception as e:
            print(f"[시총 캐시 수집 page {page} 예외] {e}")
            break

    if fresh_caps:
        caps.update(fresh_caps)
        save_json_safe(
            MARKET_CAP_CACHE_FILE,
            {
                "updated_at": now_ts,
                "fetched_pages": fetched_pages,
                "count": len(caps),
                "caps": caps,
            },
        )
    return len(caps), len(fresh_caps)


def load_merged_state():
    merged = {"tracked": {}, "alert_history": {}, "sl_history": {}, "signal_episodes": {}}
    for path in STATE_CANDIDATE_FILES:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if not isinstance(d, dict):
                continue
            for k in ("tracked", "alert_history", "sl_history", "signal_episodes"):
                sub = d.get(k)
                if isinstance(sub, dict):
                    for sk, sv in sub.items():
                        sym_chk = sv.get("symbol", sk) if isinstance(sv, dict) else sk
                        if is_stock_or_tradfi_symbol(sym_chk):
                            continue
                        if sk not in merged[k] or (isinstance(sv, dict) and to_float(sv.get("max_favorable_pct")) > to_float((merged[k].get(sk) or {}).get("max_favorable_pct"))):
                            merged[k][sk] = sv
        except Exception:
            pass
    return merged


def get_recent_scanner_caught_map(state, now_ts, lookback_sec=172800):
    caught = {}
    for ep in (state.get("signal_episodes") or {}).values():
        if not isinstance(ep, dict):
            continue
        sym = str(ep.get("symbol", "")).upper()
        c_at = int(ep.get("created_at", 0))
        if sym and c_at > 0 and (now_ts - c_at) <= lookback_sec:
            if sym not in caught or c_at > caught[sym]["created_at"]:
                caught[sym] = {"track_type": str(ep.get("track_type", "TRACK_A")), "created_at": c_at}
    for sym, tv in (state.get("tracked") or {}).items():
        if not isinstance(tv, dict):
            continue
        s_up = str(sym).upper()
        c_at = int(tv.get("created_at", 0))
        if s_up and c_at > 0 and (now_ts - c_at) <= lookback_sec:
            if s_up not in caught or c_at > caught[s_up]["created_at"]:
                caught[s_up] = {"track_type": str(tv.get("track_type", "TRACK_A")), "created_at": c_at}
    return caught


def calc_rsi(series, period=6):
    delta = series.diff()
    up = delta.clip(lower=0.0).ewm(com=period - 1, adjust=True, min_periods=period).mean()
    down = (-1.0 * delta.clip(upper=0.0)).ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = up / down.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def _find_launchpad_index(df_15m):
    n = len(df_15m)
    if n < 35:
        return max(20, n - 4)

    closes = df_15m["close"].astype(float).values
    opens = df_15m["open"].astype(float).values
    highs = df_15m["high"].astype(float).values
    lows = df_15m["low"].astype(float).values
    vols = df_15m["volume"].astype(float).fillna(0.0).values

    vol_ma20 = pd.Series(vols).rolling(20, min_periods=5).mean().shift(1).replace(0, np.nan).values

    search_start = max(22, n - 108)
    peak_idx = search_start + int(np.argmax(highs[search_start:]))
    pre_peak_start = max(22, peak_idx - 72)
    if peak_idx <= pre_peak_start + 3:
        pre_peak_start = max(22, n - 96)
        peak_idx = n - 1

    trough_idx = pre_peak_start + int(np.argmin(lows[pre_peak_start : peak_idx + 1]))

    for i in range(max(22, trough_idx), min(peak_idx + 1, n - 1)):
        v_ma = vol_ma20[i] if not np.isnan(vol_ma20[i]) else np.nanmean(vols[max(0, i - 20) : i])
        v_ratio = vols[i] / max(v_ma, 1e-8) if v_ma and v_ma > 0 else 1.0
        bar_gain = ((closes[i] - opens[i]) / max(opens[i], 1e-8)) * 100.0
        gain_2bar = ((closes[min(i + 1, n - 1)] - opens[i]) / max(opens[i], 1e-8)) * 100.0
        rng = max(highs[i] - lows[i], 1e-12)
        uwick = (highs[i] - max(opens[i], closes[i])) / rng
        if v_ratio >= 1.8 and bar_gain >= 1.0 and gain_2bar >= 2.2 and uwick <= 0.45:
            return i

    return min(max(22, trough_idx + 1), n - 2)


def analyze_top_gainer_with_dna(coin, caught_map):
    sym = coin["symbol"]
    base = coin.get("base_coin") or sym.replace("USDT", "")
    price = to_float(coin.get("price"))
    chg_24h = to_float(coin.get("change_24h"))
    turn_m = to_float(coin.get("turnover_m"))
    oi_m = to_float(coin.get("oi_m"))
    oi_ratio = round((oi_m / max(turn_m, 0.1)) * 100.0, 1)
    fund_pct = to_float(coin.get("funding_pct"))

    df_15m = fetch_klines(sym, interval="15", limit=200)
    df_1h = fetch_klines(sym, interval="60", limit=72)

    vol_spike, rsi15, squeeze_pct = 1.5, 55.0, 4.5
    pattern_note = "수급 집중 돌파"
    htf_trend = "1H 정배열 확산"
    dna = None

    if not df_15m.empty and len(df_15m) >= 35:
        closes = df_15m["close"].astype(float)
        opens = df_15m["open"].astype(float)
        vols = df_15m["volume"].astype(float).fillna(0.0)

        live_p = to_float(closes.iloc[-1], 0.0)
        if live_p > 0 and (price <= 0 or 0.50 <= (live_p / price) <= 2.00):
            price = live_p

        rsi6_s = calc_rsi(closes, 6)
        rsi15 = round(to_float(rsi6_s.iloc[-1], 55.0), 1)

        ma20_s = closes.rolling(20, min_periods=5).mean()
        std20_s = closes.rolling(20, min_periods=5).std().fillna(0.0)
        bb_width_s = ((std20_s * 4.0) / ma20_s.replace(0, np.nan) * 100.0).fillna(6.0)

        l_idx = _find_launchpad_index(df_15m)
        pre_idx = max(20, l_idx - 1)
        l_ts = to_float(df_15m["timestamp"].iloc[l_idx], 0.0)
        l_price = to_float(opens.iloc[l_idx], price)

        pre_bb_width = round(to_float(bb_width_s.iloc[max(0, pre_idx - 8) : pre_idx + 1].min(), 6.0), 2)
        pre_rsi6 = round(to_float(rsi6_s.iloc[pre_idx], 50.0), 1)

        v_pre_recent = to_float(vols.iloc[max(0, pre_idx - 5) : pre_idx + 1].mean(), 1.0)
        v_pre_base = to_float(vols.iloc[max(0, pre_idx - 23) : max(1, pre_idx - 5)].mean(), v_pre_recent)
        pre_vol_dry_ratio = round(float(np.clip(v_pre_recent / max(v_pre_base, 1e-8), 0.05, 3.0)), 2)

        v_base_20 = to_float(vols.iloc[max(0, l_idx - 20) : l_idx].mean(), 1.0)
        l_win = df_15m.iloc[l_idx : min(len(df_15m), l_idx + 2)]
        l_bull_vols = []
        for _, rw in l_win.iterrows():
            ro, rh, rl, rc, rv = (
                to_float(rw["open"]), to_float(rw["high"]),
                to_float(rw["low"]), to_float(rw["close"]), to_float(rw["volume"]),
            )
            uwick = (rh - max(ro, rc)) / max(rh - rl, 1e-12)
            if rc > ro and uwick <= 0.45:
                l_bull_vols.append(rv)
        v_launch_max = max(l_bull_vols) if l_bull_vols else to_float(vols.iloc[l_idx], v_base_20)
        launch_vol_mult = round(float(np.clip(v_launch_max / max(v_base_20, 1e-8), 0.5, 25.0)), 2)

        pre_1h_ma20_dist = 0.0
        if not df_1h.empty and len(df_1h) >= 20:
            c1h = df_1h["close"].astype(float)
            ma20_1h_s = c1h.rolling(20, min_periods=5).mean()
            ma20_1h_now = to_float(ma20_1h_s.iloc[-1], price)
            htf_trend = "1H 정배열 돌파·확산" if price >= ma20_1h_now else "1H 급등 후 중심선 눌림"

            if "timestamp" in df_1h.columns and l_ts > 0:
                before_1h = df_1h[df_1h["timestamp"] <= l_ts]
                if len(before_1h) >= 20:
                    ma20_at_launch = to_float(before_1h["close"].astype(float).rolling(20, min_periods=5).mean().iloc[-1], l_price)
                    if ma20_at_launch > 0:
                        pre_1h_ma20_dist = round(((l_price - ma20_at_launch) / ma20_at_launch) * 100.0, 2)

        squeeze_pct = pre_bb_width
        vol_spike = launch_vol_mult

        if pre_bb_width <= 5.5 and launch_vol_mult >= 2.2:
            pattern_tag = "초응축 돌파 폭발"
            pattern_note = f"발사직전 초응축({pre_bb_width:.1f}%) 후 양봉 거래량 {launch_vol_mult:.1f}배 폭발"
        elif pre_rsi6 <= 38.0 or pre_1h_ma20_dist <= -2.0:
            pattern_tag = "과매도 바닥 급반전"
            pattern_note = f"과매도 바닥(RSI {pre_rsi6:.1f}·1H이격 {pre_1h_ma20_dist:+.1f}%) 후 거래량 {launch_vol_mult:.1f}배 반전"
        elif pre_vol_dry_ratio <= 0.65:
            pattern_tag = "거래량 건조 후 발사"
            pattern_note = f"발사직전 거래량 건조({pre_vol_dry_ratio * 100:.0f}%) 후 {launch_vol_mult:.1f}배 폭발"
        else:
            pattern_tag = "1H 지지 추세가속"
            pattern_note = f"1H 추세선(이격 {pre_1h_ma20_dist:+.1f}%) 지지 + 발사 거래량 {launch_vol_mult:.1f}배 가속"

        caught_info = caught_map.get(sym)
        dna = {
            "dna_version": DNA_VERSION,
            "symbol": sym,
            "timestamp": int(time.time()),
            "change_24h": round(chg_24h, 2),
            "caught_by_scanner": bool(caught_info is not None),
            "caught_track": caught_info["track_type"] if caught_info else "MISSED",
            "pre_bb_width": pre_bb_width,
            "pre_vol_dry_ratio": pre_vol_dry_ratio,
            "launch_vol_mult": launch_vol_mult,
            "pre_rsi6": pre_rsi6,
            "pre_1h_ma20_dist": pre_1h_ma20_dist,
            "pattern_tag": pattern_tag,
        }

    return {
        "symbol": sym, "base": base, "price": price, "change_24h": chg_24h,
        "turnover_m": turn_m, "oi_m": oi_m, "oi_ratio": oi_ratio, "funding_pct": fund_pct,
        "vol_spike": vol_spike, "rsi15": rsi15, "squeeze_pct": squeeze_pct,
        "pattern_note": pattern_note, "htf_trend": htf_trend, "dna": dna,
    }


def update_top5_dna_profile(analyzed_top5, now_ts):
    profile = load_json_safe(DNA_PROFILE_FILE, {"samples": [], "active_dna_formula": {}, "updated_at": 0})
    samples = [
        s for s in (profile.get("samples") or [])
        if isinstance(s, dict)
        and (now_ts - int(to_float(s.get("timestamp"), 0))) <= DNA_RETENTION_SEC
        and not is_stock_or_tradfi_symbol(s.get("symbol", ""))
    ]

    for item in analyzed_top5:
        dna = item.get("dna")
        if not isinstance(dna, dict):
            continue
        sym = dna["symbol"]
        dup_recent = any(
            str(s.get("symbol", "")).upper() == sym and (now_ts - int(to_float(s.get("timestamp"), 0))) < 21600
            for s in samples
        )
        if not dup_recent:
            samples.insert(0, dna)

    samples = samples[:120]

    weighted_pool = []
    missed_cnt = 0
    for s in samples:
        weighted_pool.append(s)
        if not s.get("caught_by_scanner"):
            weighted_pool.append(s)
            missed_cnt += 1

    formula = {
        "enabled": False,
        "dna_version": DNA_VERSION,
        "sample_count": len(samples),
        "missed_count": missed_cnt,
        "updated_at": now_ts,
    }
    if len(samples) >= 5 and weighted_pool:
        bb_vals = [to_float(x.get("pre_bb_width"), 6.0) for x in weighted_pool]
        dry_vals = [to_float(x.get("pre_vol_dry_ratio"), 0.8) for x in weighted_pool]
        lvol_vals = [to_float(x.get("launch_vol_mult"), 2.0) for x in weighted_pool]
        rsi_vals = [to_float(x.get("pre_rsi6"), 50.0) for x in weighted_pool]
        dist_vals = [to_float(x.get("pre_1h_ma20_dist"), 0.0) for x in weighted_pool]
        tags = [str(x.get("pattern_tag") or "초응축 돌파 폭발") for x in weighted_pool]
        dominant_tag = max(set(tags), key=tags.count)

        formula = {
            "enabled": True,
            "dna_version": DNA_VERSION,
            "sample_count": len(samples),
            "missed_count": missed_cnt,
            "dominant_tag": dominant_tag,
            "bb_width_max": round(float(np.clip(np.percentile(bb_vals, 72), 4.0, 9.5)), 2),
            "vol_dry_max": round(float(np.clip(np.percentile(dry_vals, 72), 0.50, 0.95)), 2),
            "launch_vol_min": round(float(np.clip(np.percentile(lvol_vals, 35), 1.8, 3.8)), 2),
            "pre_rsi_min": round(float(np.clip(np.percentile(rsi_vals, 15), 22.0, 48.0)), 1),
            "pre_rsi_max": round(float(np.clip(np.percentile(rsi_vals, 82), 55.0, 68.0)), 1),
            "ma20_1h_dist_min": round(float(np.clip(np.percentile(dist_vals, 15), -4.5, 0.5)), 2),
            "ma20_1h_dist_max": round(float(np.clip(np.percentile(dist_vals, 82), 2.0, 6.5)), 2),
            "updated_at": now_ts,
        }

    profile["samples"] = samples
    profile["active_dna_formula"] = formula
    profile["updated_at"] = now_ts
    save_json_safe(DNA_PROFILE_FILE, profile)
    return formula


def verify_48h_history(history_data, current_map, now_ts):
    records = history_data.get("records", [])
    valid_recs = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_ts = int(rec.get("timestamp", 0))
        age_h = (now_ts - rec_ts) / 3600.0
        sym = str(rec.get("symbol", "")).upper()
        entry_p = to_float(rec.get("price"), 0.0)
        if 0 <= age_h <= 96 and sym and entry_p > 0 and not is_stock_or_tradfi_symbol(sym):
            valid_recs.append(rec)

    def _display_priority(rec):
        age_h = (now_ts - int(rec.get("timestamp", 0))) / 3600.0
        if 6.0 <= age_h <= 54.0:
            return (0, -int(rec.get("timestamp", 0)))
        if 0.2 <= age_h < 6.0:
            return (1, -int(rec.get("timestamp", 0)))
        return (2, -int(rec.get("timestamp", 0)))

    valid_recs.sort(key=_display_priority)
    verified_lines, seen_syms = [], set()
    for rec in valid_recs:
        rec_ts = int(rec.get("timestamp", 0))
        age_h = (now_ts - rec_ts) / 3600.0
        sym = str(rec.get("symbol", "")).upper()
        entry_p = to_float(rec.get("price"), 0.0)

        curr_p = to_float(current_map.get(sym, {}).get("price"), entry_p)
        max_h = max(to_float(rec.get("max_high_48h"), entry_p), curr_p)

        if 0.2 <= age_h <= 54.0 and sym not in seen_syms and len(seen_syms) < 6:
            df_1h = fetch_klines(sym, interval="60", limit=48)
            if not df_1h.empty and "timestamp" in df_1h.columns:
                live_1h_c = to_float(df_1h["close"].iloc[-1], 0.0)
                if live_1h_c > 0 and 0.40 <= (live_1h_c / entry_p) <= 4.00:
                    curr_p = live_1h_c
                post_df = df_1h[df_1h["timestamp"] >= rec_ts * 1000]
                if not post_df.empty:
                    max_h = max(max_h, to_float(post_df["high"].max(), entry_p), curr_p)

        rec["max_high_48h"] = max_h
        max_gain = round((max_h - entry_p) / entry_p * 100.0, 1)
        curr_gain = round((curr_p - entry_p) / entry_p * 100.0, 1)
        rec["max_gain_pct"] = max_gain

        if age_h >= 0.2 and sym not in seen_syms and len(verified_lines) < 6:
            seen_syms.add(sym)
            base = sym.replace("USDT", "")
            status = "🚀추가급등" if max_gain >= 12.0 else ("✅상승유지" if curr_gain >= 0 else "📉차익조정")
            verified_lines.append(
                f"• ${base} ({age_h:.1f}h 경과 | {status}): 기준 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (+{max_gain:.1f}%) | 현재 {curr_gain:+.1f}%"
            )

    valid_recs.sort(key=lambda x: int(x.get("timestamp", 0)), reverse=True)
    history_data["records"] = valid_recs[:50]
    return verified_lines


def main():
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    # [방안 1] CoinGecko 상위 1,250개 시가총액 캐시 일간 자동 갱신
    mc_total, mc_fresh = update_market_cap_cache(now_ts)

    state = load_merged_state()
    caught_map = get_recent_scanner_caught_map(state, now_ts)

    candidates, _, _ = get_scanner_candidates()
    valid_syms, _ = get_valid_bybit_linear_symbols()
    if valid_syms:
        candidates = [c for c in candidates if c.get("symbol") in valid_syms and not is_stock_or_tradfi_symbol(c.get("symbol", ""))]

    history_data = load_history_safe()
    current_map = {c["symbol"]: c for c in candidates}
    verified_48h_lines = verify_48h_history(history_data, current_map, now_ts)

    top5 = sorted(candidates, key=lambda x: to_float(x.get("change_24h")), reverse=True)[:5]
    analyzed_top5 = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        f_map = {ex.submit(analyze_top_gainer_with_dna, c, caught_map): idx for idx, c in enumerate(top5)}
        temp_list = [None] * len(top5)
        for fut in as_completed(f_map):
            idx = f_map[fut]
            try:
                temp_list[idx] = fut.result()
            except Exception:
                pass
        analyzed_top5 = [x for x in temp_list if x is not None]

    dna_formula = update_top5_dna_profile(analyzed_top5, now_ts)

    for info in analyzed_top5:
        already_recent = any(
            isinstance(r, dict)
            and str(r.get("symbol", "")).upper() == info["symbol"]
            and (now_ts - int(r.get("timestamp", 0))) < 21600
            for r in history_data["records"]
        )
        if not already_recent:
            history_data["records"].insert(0, {
                "symbol": info["symbol"], "price": info["price"], "change_24h": info["change_24h"],
                "max_high_48h": info["price"], "max_gain_pct": 0.0, "timestamp": now_ts,
            })

    gains = [to_float(x.get("max_gain_pct")) for x in history_data.get("records", []) if to_float(x.get("max_gain_pct")) >= 6.0]
    p1_l, p2_l, p3_l = (
        (float(np.clip(np.percentile(gains, 35), 7.5, 11.0)),
         float(np.clip(np.percentile(gains, 65), 14.0, 20.0)),
         float(np.clip(np.percentile(gains, 85), 22.0, 32.0)))
        if len(gains) >= 5 else (8.0, 15.0, 22.0)
    )
    tp_obj = {
        "tp1_min_pct": round(p1_l, 1), "tp2_min_pct": round(p2_l, 1),
        "tp3_min_pct": round(p3_l, 1), "updated_at": now_ts,
    }
    history_data["learned_tp_params"] = tp_obj
    history_data["dynamic_tp_tuning"] = tp_obj
    save_json_safe(HISTORY_FILE, history_data)

    log_lines = [
        f"🧬 [Bybit Top 5 백그라운드 DNA & 시총 캐시 학습 완료] ({now_str})",
        f"• 시가총액 캐시: 총 {mc_total}종목 보관 (금회 갱신 {mc_fresh}종목) | 48h 검증 {len(verified_48h_lines)}건",
    ]
    for idx, it in enumerate(analyzed_top5, 1):
        dna = it.get("dna") or {}
        c_tag = f"✅포착({dna.get('caught_track')})" if dna.get("caught_by_scanner") else "❌미포착 사각지대"
        log_lines.append(
            f"  {idx}. ${it['base']} ({it['change_24h']:+.1f}%) [{c_tag}] | "
            f"응축 {dna.get('pre_bb_width', 0)}% · 건조 {dna.get('pre_vol_dry_ratio', 0)} · "
            f"발사양봉 {dna.get('launch_vol_mult', 0)}x · 직전RSI {dna.get('pre_rsi6', 0)} · 1H이격 {dna.get('pre_1h_ma20_dist', 0):+}%"
        )
    if dna_formula.get("enabled"):
        log_lines.append(
            f"• 🧬 Track H 순수 5D 차트 DNA ({DNA_VERSION} | 14일 표본 {dna_formula['sample_count']}건 / 미포착 {dna_formula['missed_count']}건): "
            f"주력 [{dna_formula['dominant_tag']}] | 응축 <= {dna_formula['bb_width_max']}% · 건조 <= {dna_formula['vol_dry_max']} · "
            f"발사양봉 >= {dna_formula['launch_vol_min']}x · 직전RSI {dna_formula['pre_rsi_min']}~{dna_formula['pre_rsi_max']}"
        )

    full_log = "\n".join(log_lines)
    print(full_log)
    send_telegram_report(full_log)


if __name__ == "__main__":
    main()
# === END OF FILE ===
