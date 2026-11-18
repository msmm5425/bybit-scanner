import os, sys, json, time, subprocess, requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from config import STATE_FILE, KST, to_float, is_tradfi_ticker
from api import (
    get_scanner_candidates,
    fetch_verified_1m_klines,
    fetch_klines,
    get_valid_bybit_linear_symbols,
)

try:
    from api import is_stock_or_tradfi_symbol, clean_url
except ImportError:
    import re
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

    def clean_url(raw_url):
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", str(raw_url or "").strip())
        return s.replace("[", "").replace("]", "")

from analyzer import analyze_symbol, get_dynamic_tp_params
from notifier import (
    build_briefing_message,
    build_x_post_message,
    build_x_intent_reply_markup,
    send_telegram_single,
    fmt_price,
    generate_signal_chart_bytes,
    send_telegram_photo_with_caption,
)

try:
    import postmortem
except Exception:
    postmortem = None

SCHEME = "ht" + "tps://"
RE_ALERT_COOLDOWN_SEC = 86400
SL_RECOVERY_COOLDOWN_SEC = 14400
HISTORY_RETENTION_SEC = 259200
EPISODE_RETENTION_SEC = 604800
MAX_NEW_ALERTS_PER_SCAN = getattr(cfg, "MAX_NEW_ALERTS_PER_SCAN", 3)
DEFAULT_FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
LOCAL_RUNTIME_CACHE = "/tmp/scanner_runtime_state.json"
ALL_STATE_FILES = list(dict.fromkeys([STATE_FILE, "bot_state.json", "scanner_state.json", LOCAL_RUNTIME_CACHE]))

ALL_LOCK_FLAGS = (
    "tp1_hit", "tp2_hit", "tp3_hit",
    "tp1_alerted", "tp2_alerted", "tp3_alerted",
    "tp3_completed", "trailing_closed", "sl_alerted", "closed_alerted",
)


def send_telegram_reliable(text, reply_markup=None):
    if not text:
        return False
    token = (getattr(cfg, "TELEGRAM_BOT_TOKEN", "") or os.getenv("TELEGRAM_TOKEN", "") or os.getenv("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = (getattr(cfg, "TELEGRAM_CHAT_ID", "") or os.getenv("TELEGRAM_CHAT_ID", "") or os.getenv("CHAT_ID", "")).strip()
    if not token or not chat_id:
        return send_telegram_single(text, reply_markup=reply_markup)

    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendMessage")
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if reply_markup:
        payload["reply_markup"] = reply_markup

    for _ in range(3):
        try:
            r = requests.post(url, json=payload, timeout=12)
            if r.status_code == 200:
                time.sleep(0.6)
                return True
            if r.status_code == 429:
                wait_s = int(r.json().get("parameters", {}).get("retry_after", 3))
                time.sleep(wait_s + 0.5)
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _read_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, dict) else {}
        except Exception:
            pass
    return {}


def sanitize_flags_by_mfe(item):
    if not isinstance(item, dict):
        return item
    entry_p = to_float(item.get("entry_price"))
    max_h = max(to_float(item.get("max_high_price")), entry_p)
    fav = to_float(item.get("max_favorable_pct"))
    if entry_p > 0 and max_h > entry_p:
        fav = max(fav, round((max_h - entry_p) / entry_p * 100.0, 2))
        item["max_favorable_pct"] = fav

    tps = item.get("tps") or []
    tp1_req = ((to_float(tps[0]) - entry_p) / entry_p * 100.0 - 0.2) if (len(tps) > 0 and entry_p > 0 and to_float(tps[0]) > entry_p) else 6.8
    tp2_req = ((to_float(tps[1]) - entry_p) / entry_p * 100.0 - 0.2) if (len(tps) > 1 and entry_p > 0 and to_float(tps[1]) > entry_p) else 13.0
    tp3_req = ((to_float(tps[2]) - entry_p) / entry_p * 100.0 - 0.2) if (len(tps) > 2 and entry_p > 0 and to_float(tps[2]) > entry_p) else 19.5

    if fav >= max(6.5, tp1_req) or item.get("tp1_alerted"):
        item["tp1_hit"] = True
    if fav >= max(12.5, tp2_req) or item.get("tp2_alerted"):
        item["tp2_hit"] = True
    if fav >= max(19.0, tp3_req) or item.get("tp3_alerted"):
        item["tp3_hit"] = True
    return item


def _merge_state_dicts(base_state, incoming_state):
    if not isinstance(incoming_state, dict):
        return base_state
    for s, ts_v in (incoming_state.get("alert_history") or {}).items():
        if not is_stock_or_tradfi_symbol(s):
            base_state["alert_history"][s] = max(int(base_state["alert_history"].get(s, 0)), int(to_float(ts_v, 0)))
    for k, v in (incoming_state.get("sl_history") or {}).items():
        if isinstance(v, dict) and not is_stock_or_tradfi_symbol(v.get("symbol", "")):
            base_state["sl_history"][k] = v
    for ek, ev in (incoming_state.get("signal_episodes") or {}).items():
        if not isinstance(ev, dict) or is_stock_or_tradfi_symbol(ev.get("symbol", "")):
            continue
        ev = sanitize_flags_by_mfe(dict(ev))
        sym_e = str(ev.get("symbol", "")).upper()
        c_e = int(ev.get("created_at", 0))
        if sym_e and c_e > 0:
            base_state["alert_history"][sym_e] = max(int(base_state["alert_history"].get(sym_e, 0)), c_e)
        p = base_state["signal_episodes"].get(ek, {})
        if not p:
            base_state["signal_episodes"][ek] = ev
        else:
            p["max_high_price"] = max(to_float(p.get("max_high_price")), to_float(ev.get("max_high_price")))
            p["max_favorable_pct"] = max(to_float(p.get("max_favorable_pct")), to_float(ev.get("max_favorable_pct")))
            p["stop_price"] = max(to_float(p.get("stop_price")), to_float(ev.get("stop_price")))
            for fg in ALL_LOCK_FLAGS:
                if ev.get(fg) or p.get(fg):
                    p[fg] = True
            base_state["signal_episodes"][ek] = sanitize_flags_by_mfe(p)
    for s, tv in (incoming_state.get("tracked") or {}).items():
        if not isinstance(tv, dict) or is_stock_or_tradfi_symbol(s):
            continue
        tv = sanitize_flags_by_mfe(dict(tv))
        c_t = int(tv.get("created_at", 0))
        if c_t > 0:
            base_state["alert_history"][s] = max(int(base_state["alert_history"].get(s, 0)), c_t)
        pt = base_state["tracked"].get(s)
        if not pt or c_t > int(pt.get("created_at", 0)):
            base_state["tracked"][s] = tv
        elif c_t == int(pt.get("created_at", 0)):
            pt["max_high_price"] = max(to_float(pt.get("max_high_price")), to_float(tv.get("max_high_price")))
            pt["max_favorable_pct"] = max(to_float(pt.get("max_favorable_pct")), to_float(tv.get("max_favorable_pct")))
            pt["stop_price"] = max(to_float(pt.get("stop_price")), to_float(tv.get("stop_price")))
            for fg in ALL_LOCK_FLAGS:
                if tv.get(fg) or pt.get(fg):
                    pt[fg] = True
            base_state["tracked"][s] = sanitize_flags_by_mfe(pt)
    return base_state


def load_state():
    merged = {"tracked": {}, "alert_history": {}, "sl_history": {}, "signal_episodes": {}}
    for path in ALL_STATE_FILES:
        _merge_state_dicts(merged, _read_json(path))
    return merged


def save_state(state):
    for path in ALL_STATE_FILES:
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            pass


def sync_episode_ledger(signal_episodes, symbol, t_item, sl_history=None):
    c_at = int(t_item.get("created_at", 0))
    if c_at <= 0 or is_stock_or_tradfi_symbol(symbol):
        return
    ep_key = f"{symbol}_{c_at}"
    prev = signal_episodes.get(ep_key, {})
    max_h = max(to_float(prev.get("max_high_price")), to_float(t_item.get("max_high_price")), to_float(t_item.get("entry_price")))
    max_fav = max(to_float(prev.get("max_favorable_pct")), to_float(t_item.get("max_favorable_pct")))
    t_item["max_high_price"] = max_h
    t_item["max_favorable_pct"] = max_fav
    for fg in ALL_LOCK_FLAGS:
        if prev.get(fg) or t_item.get(fg):
            t_item[fg] = True
    sanitize_flags_by_mfe(t_item)

    signal_episodes[ep_key] = {
        "symbol": symbol, "created_at": c_at, "track_type": t_item.get("track_type", "TRACK_A"),
        "is_long_term": bool(t_item.get("is_long_term", False)),
        "score": int(t_item.get("score", 85)), "entry_price": to_float(t_item.get("entry_price")),
        "initial_stop_price": to_float(t_item.get("initial_stop_price") or t_item.get("stop_price")),
        "stop_price": to_float(t_item.get("stop_price")), "sl_pct": to_float(t_item.get("sl_pct", DEFAULT_FIXED_SL_PCT)),
        "tps": t_item.get("tps", []), "tp_pcts": t_item.get("tp_pcts", []),
        "max_high_price": t_item["max_high_price"], "max_favorable_pct": t_item["max_favorable_pct"],
        "realized_weighted_pct": to_float(t_item.get("realized_weighted_pct", 0.0)),
        "vol_mult": to_float(t_item.get("vol_mult", 1.0)), "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
        "oi_m": to_float(t_item.get("oi_m", 0.0)), "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
        "hold_minutes": int(t_item.get("hold_minutes", 0)),
        **{fg: bool(t_item.get(fg, False)) for fg in ALL_LOCK_FLAGS},
        "sl_at": int(t_item.get("sl_at", 0)),
    }


def evaluate_single_tracked_1m(symbol, t_item, now_ts, tp_params):
    if t_item.get("tp3_completed") or t_item.get("sl_alerted") or t_item.get("closed_alerted"):
        return None
    entry_ref = to_float(t_item.get("entry_price"))
    if entry_ref <= 0:
        return None

    is_long = bool(t_item.get("is_long_term") or t_item.get("track_type") == "TRACK_D")
    sl_pct_use = int(round(to_float(t_item.get("sl_pct"), 12 if is_long else DEFAULT_FIXED_SL_PCT)))
    trail_drop_pct = 8 if is_long else 3
    trail_mult = 1.0 - (trail_drop_pct / 100.0)
    horizon_tag = "[장기 포지션] " if is_long else ""

    stop_p = to_float(t_item.get("stop_price"))
    init_stop_p = to_float(t_item.get("initial_stop_price"), default=entry_ref * (1.0 - sl_pct_use / 100.0))
    created_at = int(t_item.get("created_at", now_ts))
    created_ms = created_at * 1000
    prev_max_h = to_float(t_item.get("max_high_price"), default=entry_ref)
    prev_peak_ms = int(to_float(t_item.get("peak_ts_ms"), default=created_ms))

    hist_min_low, hist_min_close = entry_ref, entry_ref
    if (now_ts - created_at) >= 300:
        df_15m = fetch_klines(symbol, "15", limit=150)
        if not df_15m.empty and "timestamp" in df_15m.columns:
            post_15m = df_15m[df_15m["timestamp"] >= (created_ms - 900000)]
            if not post_15m.empty:
                p_idx = post_15m["high"].idxmax()
                h_15m = to_float(post_15m.loc[p_idx, "high"], default=entry_ref)
                if 0.40 <= (h_15m / entry_ref) <= 6.00 and h_15m >= prev_max_h:
                    prev_max_h = h_15m
                    prev_peak_ms = int(to_float(post_15m.loc[p_idx, "timestamp"], default=created_ms))
                hist_min_low = min(hist_min_low, to_float(post_15m["low"].min(), default=entry_ref))
                hist_min_close = min(hist_min_close, to_float(post_15m["close"].min(), default=entry_ref))

    df_1m = fetch_verified_1m_klines(symbol, entry_ref=entry_ref, limit=60)
    post_1m = df_1m[df_1m["timestamp"] >= created_ms].copy() if not df_1m.empty else df_1m

    base = symbol.replace("USDT", "")
    if not post_1m.empty:
        w_idx = post_1m["high"].idxmax()
        w_high = to_float(post_1m.loc[w_idx, "high"], default=entry_ref)
        w_ts = int(to_float(post_1m.loc[w_idx, "timestamp"], default=created_ms))
        max_high = max(w_high, prev_max_h, entry_ref)
        peak_ts_ms = w_ts if w_high >= prev_max_h else prev_peak_ms
        last_close = to_float(post_1m["close"].iloc[-1], default=entry_ref)
        last_ts_ms = int(to_float(post_1m["timestamp"].iloc[-1], default=now_ts * 1000))
        min_1m_low = to_float(post_1m["low"].min(), default=entry_ref)
        min_1m_close = to_float(post_1m["close"].min(), default=entry_ref)
    else:
        max_high, peak_ts_ms = max(prev_max_h, entry_ref), prev_peak_ms
        last_close, last_ts_ms = entry_ref, now_ts * 1000
        min_1m_low, min_1m_close = hist_min_low, hist_min_close

    max_fav = round((max_high - entry_ref) / entry_ref * 100.0, 2)
    t_item["max_high_price"] = max_high
    t_item["peak_ts_ms"] = peak_ts_ms
    t_item["max_favorable_pct"] = max_fav
    hold_min = max(1, int((now_ts - created_at) // 60))
    t_item["hold_minutes"] = hold_min

    tps = t_item.get("tps") or []
    tp1_p = to_float(tps[0]) if len(tps) > 0 and to_float(tps[0]) > 0 else entry_ref * (1.0 + tp_params["tp1_min_pct"] / 100.0)
    tp2_p = to_float(tps[1]) if len(tps) > 1 and to_float(tps[1]) > 0 else entry_ref * (1.0 + tp_params["tp2_min_pct"] / 100.0)
    tp3_p = to_float(tps[2]) if len(tps) > 2 and to_float(tps[2]) > 0 else entry_ref * (1.0 + tp_params["tp3_min_pct"] / 100.0)

    g1 = round(max(6.5, (tp1_p - entry_ref) / entry_ref * 100.0), 1)
    g2 = round(max(12.5, (tp2_p - entry_ref) / entry_ref * 100.0), 1)
    g3 = round(max(19.5, (tp3_p - entry_ref) / entry_ref * 100.0), 1)

    def1_p = max(entry_ref * 1.01, tp1_p * 0.95)
    def2_p = max(def1_p, tp2_p * 0.95)
    msgs = []

    # [3차 돌파]: 30% 익절 확정(누적 60%) + 잔여 40% 자동 트레일링(단기 -3% / 장기 -8%)
    if max_high >= tp3_p or t_item.get("tp3_hit"):
        for fg in ("tp1_hit", "tp2_hit", "tp3_hit", "tp1_alerted", "tp2_alerted"):
            t_item[fg] = True
        trail3_p = max(def2_p, tp3_p * trail_mult, max_high * trail_mult)
        stop_p = max(stop_p, trail3_p)
        t_item["stop_price"] = stop_p

        if not t_item.get("tp3_alerted"):
            t_item["tp3_alerted"] = True
            w1, w3 = round(g1 * 0.30, 2), round(g3 * 0.30, 2)
            tr_pct = round((stop_p - entry_ref) / entry_ref * 100.0, 1)
            msgs.append(
                f"🚀 {horizon_tag}[3차 목표가(TP3) 돌파! 30% 익절 & 잔여 40% 고점 -{trail_drop_pct}% 트레일링 가동] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 📊 [누적 60% 분할 익절]: 1차(30%) +{g1:.1f}%(+{w1:.2f}%) + 3차(30%) +{g3:.1f}%(+{w3:.2f}%)\n"
                f"• 🎯 남은 40% 물량은 고점 대비 -{trail_drop_pct}% 자동 트레일링 스탑({fmt_price(stop_p)}, +{tr_pct:.1f}%)으로 끝까지 수익을 추적합니다!"
            )

        if last_ts_ms > peak_ts_ms and last_close <= stop_p and not t_item.get("closed_alerted"):
            t_item["tp3_completed"] = True
            t_item["trailing_closed"] = True
            t_item["closed_alerted"] = True
            t_item["tp3_at"] = now_ts
            exit_g = round(max(g2 * 0.9, (stop_p - entry_ref) / entry_ref * 100.0), 1)
            w1, w3, w_rem = round(g1 * 0.30, 2), round(g3 * 0.30, 2), round(exit_g * 0.40, 2)
            total_cum = round(w1 + w3 + w_rem, 2)
            t_item["realized_weighted_pct"] = total_cum
            msgs.append(
                f"🎯 {horizon_tag}[3차 돌파 후 잔여 40% 트레일링 스탑 최종 익절 완료] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 최고가 {fmt_price(max_high)} (+{max_fav:.1f}%) → 마감가 {fmt_price(stop_p)} (+{exit_g:.1f}%)\n"
                f"• 💰 [최종 합산 확정 수익률]: +{total_cum:.2f}% 달성 (비중 100% 합산)"
            )
        return {"messages": msgs, "sl_record": None}

    # [2차 돌파]: 70% 홀딩 + 2차지점 -5% 고정 방어선 (트레일링 스탑 없음)
    if max_high >= tp2_p or t_item.get("tp2_hit"):
        for fg in ("tp1_hit", "tp2_hit", "tp1_alerted"):
            t_item[fg] = True
        stop_p = max(stop_p, def2_p)
        t_item["stop_price"] = stop_p

        if not t_item.get("tp2_alerted"):
            t_item["tp2_alerted"] = True
            def2_pct = round((stop_p - entry_ref) / entry_ref * 100.0, 1)
            msgs.append(
                f"✅ {horizon_tag}[2차 구간(TP2) 돌파 & 2차 방어선(-5%) 상향 고정] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 🛡️ [2차 방어선 설정]: 2차 지점 대비 -5% 아래인 {fmt_price(stop_p)} USDT (+{def2_pct:.1f}%)로 방어선을 고정하고 잔여 70% 물량으로 3차 목표가(+{g3:.1f}%)를 홀딩합니다."
            )

        if last_ts_ms > peak_ts_ms and last_close <= stop_p and not t_item.get("closed_alerted"):
            t_item["tp3_completed"] = True
            t_item["trailing_closed"] = True
            t_item["closed_alerted"] = True
            t_item["tp3_at"] = now_ts
            def2_g = round(max(1.0, (stop_p - entry_ref) / entry_ref * 100.0), 1)
            total_cum = round(g1 * 0.30 + def2_g * 0.70, 2)
            t_item["realized_weighted_pct"] = total_cum
            msgs.append(
                f"🔒 {horizon_tag}[2차 방어선(-5%) 수익 마감 완료] ${base} ({symbol})\n"
                f"• 진입 후 최고 +{max_fav:.1f}% 도달 후 2차 방어선({fmt_price(stop_p)}) 이탈로 잔여 70% 물량을 익절 마감합니다.\n"
                f"• 💰 [최종 합산 확정 수익률]: +{total_cum:.2f}% 달성 (비중 100% 합산)"
            )
        return {"messages": msgs, "sl_record": None}

    # [1차 돌파]: 30% 분할 익절 + 1차지점 -5% 고정 방어선
    if max_high >= tp1_p or t_item.get("tp1_hit"):
        t_item["tp1_hit"] = True
        stop_p = max(stop_p, def1_p)
        t_item["stop_price"] = stop_p

        if not t_item.get("tp1_alerted"):
            t_item["tp1_alerted"] = True
            w1 = round(g1 * 0.30, 2)
            def1_pct = round((stop_p - entry_ref) / entry_ref * 100.0, 1)
            msgs.append(
                f"✅ {horizon_tag}[1차 목표가(TP1) 돌파! 30% 익절 & 1차 방어선(-5%) 가동] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 📊 1차 분할 익절(비중 30%): +{g1:.1f}% 확정 (기여 +{w1:.2f}%)\n"
                f"• 🛡️ [1차 방어선 설정]: 남은 70% 물량은 1차 지점 대비 -5% 아래인 {fmt_price(stop_p)} USDT (+{def1_pct:.1f}%)를 방어선으로 설정하여 2·3차 파동을 홀딩합니다."
            )

        if last_ts_ms > peak_ts_ms and last_close <= stop_p and not t_item.get("closed_alerted"):
            t_item["tp3_completed"] = True
            t_item["trailing_closed"] = True
            t_item["closed_alerted"] = True
            t_item["tp3_at"] = now_ts
            def1_g = round(max(0.5, (stop_p - entry_ref) / entry_ref * 100.0), 1)
            total_cum = round(g1 * 0.30 + def1_g * 0.70, 2)
            t_item["realized_weighted_pct"] = total_cum
            msgs.append(
                f"🔒 {horizon_tag}[1차 방어선(-5%) 수익 마감 완료] ${base} ({symbol})\n"
                f"• 진입 후 최고 +{max_fav:.1f}% 도달 후 1차 방어선({fmt_price(stop_p)}) 이탈로 잔여 70% 물량을 안전하게 익절 마감합니다.\n"
                f"• 💰 [최종 합산 확정 수익률]: +{total_cum:.2f}% 달성 (비중 100% 합산)"
            )
        return {"messages": msgs, "sl_record": None}

    # [1차 도달 전]: 손절선 이탈 감시 (단기 -6% / 장기 -12%)
    min_low = min(hist_min_low, min_1m_low)
    min_close = min(hist_min_close, min_1m_close)
    if min_low <= init_stop_p and (min_close <= init_stop_p or last_close <= init_stop_p):
        trig = min(min_close, last_close)
        t_item["sl_alerted"] = True
        t_item["closed_alerted"] = True
        t_item["sl_at"] = now_ts
        sl_payload = {
            "symbol": symbol, "track_type": t_item.get("track_type", "TRACK_A"),
            "score": int(t_item.get("score", 85)), "entry_price": entry_ref,
            "stop_price": init_stop_p, "sl_trigger_price": trig,
            "sl_pct": sl_pct_use, "created_at": created_at, "sl_at": now_ts,
            "hold_minutes": hold_min, "max_high_price": max_high, "max_favorable_pct": max_fav,
            "vol_mult": to_float(t_item.get("vol_mult", 1.0)), "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
            "oi_m": to_float(t_item.get("oi_m", 0.0)), "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
            "reasons": t_item.get("reasons", []),
        }
        if postmortem is not None and hasattr(postmortem, "record_sl_postmortem"):
            try:
                postmortem.record_sl_postmortem(sl_payload)
            except Exception:
                pass
        msgs.append(
            f"⚠️ {horizon_tag}[손절 기준가 이탈] ${base} ({symbol})\n"
            f"• 이탈 확인가: {fmt_price(trig)} USDT (손절선 {fmt_price(init_stop_p)}, -{sl_pct_use}% 이탈)\n"
            f"• 진입 후 최고 상승폭: +{max_fav:.1f}% (보유 {hold_min}분)"
        )
        return {"messages": msgs, "sl_record": (f"{symbol}_{created_at}", sl_payload)}

    return {"messages": msgs, "sl_record": None}


def _send_signal_notifications(res, now_str):
    send_telegram_reliable(build_briefing_message(res, now_str))
    x_msg = build_x_post_message(res)
    markup = build_x_intent_reply_markup(x_msg)
    base = res.get("base_coin") or res.get("base") or ""
    sym = res.get("symbol", "")
    chart_bytes = generate_signal_chart_bytes(res)
    if not (chart_bytes and send_telegram_photo_with_caption(chart_bytes, caption_text="", reply_markup=markup)):
        send_telegram_reliable(f"📲 ${base} ({sym}) X 포스팅 바로가기", reply_markup=markup)


def _git_sync_state():
    """로컬 메모리와 원격 상태를 합집합 병합한 뒤 GitHub에 영구 커밋·푸시 (중복 알림 원천 차단)"""
    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "GitHub Actions"], check=False)
        for _ in range(3):
            subprocess.run(["git", "fetch", "origin", "main"], check=False)
            loc_dict = load_state()
            for rf in ("bot_state.json", "scanner_state.json"):
                show_res = subprocess.run(["git", "show", f"origin/main:{rf}"], capture_output=True, text=True, check=False)
                if show_res.returncode == 0 and show_res.stdout:
                    try:
                        loc_dict = _merge_state_dicts(loc_dict, json.loads(show_res.stdout))
                    except Exception:
                        pass
            save_state(loc_dict)
            subprocess.run(["git", "reset", "--soft", "origin/main"], check=False)
            with open(LOCAL_RUNTIME_CACHE, "r", encoding="utf-8") as s:
                raw_txt = s.read()
            for target_f in (STATE_FILE, "bot_state.json", "scanner_state.json"):
                with open(target_f, "w", encoding="utf-8") as d:
                    d.write(raw_txt)
                subprocess.run(["git", "add", target_f], check=False)
            if subprocess.run(["git", "diff", "--staged", "--quiet"], check=False).returncode != 0:
                subprocess.run(["git", "commit", "-m", "chore: update scanner state [skip ci]"], check=False)
                if subprocess.run(["git", "push", "origin", "HEAD:main"], check=False).returncode == 0:
                    break
            else:
                break
            time.sleep(1.5)
    except Exception:
        pass


def main():
    start_t = time.time()
    state = load_state()
    tracked, alert_history = state["tracked"], state["alert_history"]
    sl_history, signal_episodes = state["sl_history"], state["signal_episodes"]
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    tp_params = get_dynamic_tp_params()
    valid_syms, _ = get_valid_bybit_linear_symbols()
    candidates, _, rwa_blocked = get_scanner_candidates()
    sent_count = 0

    for h_sym in list(alert_history.keys()):
        if is_stock_or_tradfi_symbol(h_sym) or (now_ts - int(alert_history.get(h_sym, 0)) > 172800):
            del alert_history[h_sym]
    for sl_k in list(sl_history.keys()):
        sl_v = sl_history.get(sl_k) or {}
        if is_stock_or_tradfi_symbol(sl_v.get("symbol", "")) or (now_ts - int(sl_v.get("sl_at", 0)) > HISTORY_RETENTION_SEC):
            del sl_history[sl_k]
    for ep_k in list(signal_episodes.keys()):
        ep = signal_episodes.get(ep_k) or {}
        sym_ep = str(ep.get("symbol", "")).upper()
        c_at = int(ep.get("created_at", 0))
        if is_stock_or_tradfi_symbol(sym_ep) or (now_ts - c_at > EPISODE_RETENTION_SEC):
            del signal_episodes[ep_k]
            continue
        signal_episodes[ep_k] = sanitize_flags_by_mfe(ep)
        if sym_ep and c_at > int(alert_history.get(sym_ep, 0)):
            alert_history[sym_ep] = c_at

    active_syms = []
    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        t_item = tracked[sym]
        c_at = int(t_item.get("created_at", 0))
        age_sec = now_ts - c_at
        if (valid_syms and sym not in valid_syms) or is_stock_or_tradfi_symbol(sym) or is_tradfi_ticker(base_c) or base_c in rwa_blocked:
            del tracked[sym]
            continue
        sync_episode_ledger(signal_episodes, sym, t_item, sl_history)
        if c_at > int(alert_history.get(sym, 0)):
            alert_history[sym] = c_at
        is_done = bool(t_item.get("sl_alerted") or t_item.get("tp3_completed") or t_item.get("closed_alerted"))
        done_ts = int(t_item.get("sl_at") or t_item.get("tp3_at") or c_at)
        max_age = 1209600 if t_item.get("is_long_term") else 432000
        if (is_done and (now_ts - done_ts) > 86400) or age_sec > max_age:
            del tracked[sym]
            continue
        if not is_done:
            active_syms.append(sym)

    if active_syms:
        pending_msgs = []
        with ThreadPoolExecutor(max_workers=8) as ex:
            f_map = {ex.submit(evaluate_single_tracked_1m, s, tracked[s], now_ts, tp_params): s for s in active_syms}
            for fut in as_completed(f_map):
                s = f_map[fut]
                try:
                    out = fut.result()
                    if out and out.get("sl_record"):
                        k, v = out["sl_record"]
                        sl_history[k] = v
                    sync_episode_ledger(signal_episodes, s, tracked[s], sl_history)
                    for m in (out or {}).get("messages", []):
                        pending_msgs.append(m)
                except Exception as e:
                    print(f"[1분봉 감시 예외] {s}: {e}")
        save_state(state)
        if pending_msgs:
            _git_sync_state()
            for m in pending_msgs:
                send_telegram_reliable(m)
                sent_count += 1

    if not candidates:
        save_state(state)
        return

    def _was_stopped_recently(sym_check):
        pt = tracked.get(sym_check)
        if pt and pt.get("sl_alerted"):
            return True
        return any(
            isinstance(v, dict) and v.get("symbol") == sym_check and (now_ts - int(v.get("sl_at", 0))) < RE_ALERT_COOLDOWN_SEC
            for v in sl_history.values()
        )

    to_analyze = []
    for c in candidates:
        sym = c["symbol"]
        if (valid_syms and sym not in valid_syms) or is_stock_or_tradfi_symbol(sym):
            continue
        prev_t = tracked.get(sym)
        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed") and not prev_t.get("closed_alerted"):
            continue
        cd = SL_RECOVERY_COOLDOWN_SEC if _was_stopped_recently(sym) else RE_ALERT_COOLDOWN_SEC
        if (now_ts - int(alert_history.get(sym, 0))) < cd:
            continue
        to_analyze.append(c)

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        f_map = {ex.submit(analyze_symbol, c): c["symbol"] for c in to_analyze}
        for fut in as_completed(f_map):
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception:
                pass

    results.sort(key=lambda x: x["score"], reverse=True)
    new_sent = 0
    for res in results:
        if new_sent >= MAX_NEW_ALERTS_PER_SCAN:
            break
        sym = res["symbol"]
        if is_stock_or_tradfi_symbol(sym):
            continue
        prev_t = tracked.get(sym)
        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed") and not prev_t.get("closed_alerted"):
            continue

        elapsed_alert = now_ts - int(alert_history.get(sym, 0))
        was_sl = _was_stopped_recently(sym)
        if was_sl and elapsed_alert < RE_ALERT_COOLDOWN_SEC:
            if elapsed_alert < SL_RECOVERY_COOLDOWN_SEC or res["score"] < 88:
                continue
            res["track_title"] = "🔄 [스탑헌팅 후 V자 탈환 재진입 타점]"
            if "스탑헌팅 바닥 확인 후 V자 재탈환" not in res["reasons"]:
                res["reasons"].insert(0, "스탑헌팅 바닥 확인 후 V자 재탈환")
            res["reason_str"] = " / ".join(res["reasons"])
        elif elapsed_alert < RE_ALERT_COOLDOWN_SEC:
            continue

        tracked[sym] = {
            "created_at": now_ts, "track_type": res["track_type"],
            "is_long_term": bool(res.get("is_long_term", False)),
            "score": res["score"], "entry_price": res["price"], "swing_high_line": res["swing_high_line"],
            "entry_low": res["entry_low"], "entry_high": res["entry_high"],
            "initial_stop_price": res["stop_price"], "stop_price": res["stop_price"],
            "sl_pct": res["sl_pct"], "tps": res["tps"], "tp_pcts": res["tp_pcts"],
            "vol_mult": res.get("vol_mult", 1.0), "rsi6_15m": res.get("rsi6_15m", 50.0),
            "oi_m": res.get("oi_m", 0.0), "turnover_m": res.get("turnover_m", 0.0),
            "funding_pct": res.get("funding_pct", 0.0), "reasons": res.get("reasons", []),
            "max_high_price": res["price"], "peak_ts_ms": now_ts * 1000, "max_favorable_pct": 0.0,
        }
        alert_history[sym] = now_ts
        sync_episode_ledger(signal_episodes, sym, tracked[sym], sl_history)
        save_state(state)
        _git_sync_state()
        _send_signal_notifications(res, now_str)
        new_sent += 1
        sent_count += 1

    save_state(state)
    print(f"[{now_str}] 스캔 완료 ({time.time() - start_t:.1f}초) - 전송 {sent_count}건")


def run_loop_mode(iterations=9, sleep_sec=295):
    for i in range(1, iterations + 1):
        print(f"=== [회차 {i}/{iterations}] 5분 주기 스캔 시작 ===")
        _git_sync_state()
        try:
            main()
        except Exception as e:
            print(f"[스캔 루프 예외] {e}")
        _git_sync_state()
        if i < iterations:
            time.sleep(sleep_sec)


if __name__ == "__main__":
    if "--loop" in sys.argv:
        run_loop_mode()
    else:
        main()
# === END OF FILE ===
