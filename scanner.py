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
        "KOSPI", "LITE", "ANTA", "TSLA", "NVDA", "AAPL", "MSFT", "GOOG", "AMZN",
        "META", "NFLX", "MSTR", "COIN", "SPY", "QQQ", "XAU", "XAG", "WTI", "BRENT", "SPX", "NDX",
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
MAX_NEW_ALERTS_PER_SCAN = 2
DEFAULT_FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
DEFAULT_LONGTERM_SL_PCT = 20
BTC_DUMP_DROP_15M_PCT = to_float(getattr(cfg, "BTC_DUMP_DROP_15M_PCT", -0.70), -0.70)
BTC_DUMP_DROP_1H_PCT = to_float(getattr(cfg, "BTC_DUMP_DROP_1H_PCT", -1.50), -1.50)
BTC_CRASH_1H_PCT = -2.50
BTC_CRASH_15M_PCT = -1.50

LOCAL_RUNTIME_CACHE = "/tmp/scanner_runtime_state.json"
ALL_STATE_FILES = list(dict.fromkeys([STATE_FILE, "bot_state.json", "scanner_state.json", LOCAL_RUNTIME_CACHE]))
POSTMORTEM_DB_FILE = getattr(postmortem, "POSTMORTEM_DB_FILE", "postmortem_db.json") if postmortem else "postmortem_db.json"

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


def check_btc_market_regime():
    """
    [BTC 3단계 마켓 레짐 게이트 · A안 반영]
    - OPEN: 정상 구간 → 전체 트랙(A~H, D1~D6) 탐색
    - COUNTER_TREND_ONLY: 완만 조정(15m <= -0.7% 또는 1H <= -1.5%, 단 1H > -2.5%) → Track E(SFP)·Track F(RS)만 예외 허용
    - FULL_LOCK: 폭락 구간(1H <= -2.5% 또는 15m <= -1.5%) → 모든 신규 진입 100% 전면 차단
    """
    regime = {
        "gate_mode": "OPEN",
        "gate_locked": False,
        "btc_15m_pct": 0.0,
        "btc_1h_pct": 0.0,
        "btc_4h_pct": 0.0,
        "reason": "정상",
    }
    try:
        df15 = fetch_klines("BTCUSDT", interval="15", limit=20)
        df1h = fetch_klines("BTCUSDT", interval="60", limit=12)

        if not df15.empty and len(df15) >= 2:
            c_now = to_float(df15["close"].iloc[-1])
            o_now = to_float(df15["open"].iloc[-1], c_now)
            c_prev = to_float(df15["close"].iloc[-2], c_now)
            if o_now > 0 and c_prev > 0:
                regime["btc_15m_pct"] = round(min((c_now - o_now) / o_now * 100.0, (c_now - c_prev) / c_prev * 100.0), 2)

        if not df1h.empty and len(df1h) >= 2:
            c_now_h = to_float(df1h["close"].iloc[-1])
            o_now_h = to_float(df1h["open"].iloc[-1], c_now_h)
            c_prev_h = to_float(df1h["close"].iloc[-2], c_now_h)
            if o_now_h > 0 and c_prev_h > 0:
                regime["btc_1h_pct"] = round(min((c_now_h - o_now_h) / o_now_h * 100.0, (c_now_h - c_prev_h) / c_prev_h * 100.0), 2)
            if len(df1h) >= 5:
                c_4h_ago = to_float(df1h["close"].iloc[-5], c_now_h)
                if c_4h_ago > 0:
                    regime["btc_4h_pct"] = round(((c_now_h - c_4h_ago) / c_4h_ago) * 100.0, 2)

        if regime["btc_1h_pct"] <= BTC_CRASH_1H_PCT or regime["btc_15m_pct"] <= BTC_CRASH_15M_PCT:
            regime["gate_mode"] = "FULL_LOCK"
            regime["gate_locked"] = True
            regime["reason"] = f"BTC 폭락 전면 차단 (15m {regime['btc_15m_pct']:+.2f}%, 1H {regime['btc_1h_pct']:+.2f}%)"
        elif regime["btc_15m_pct"] <= BTC_DUMP_DROP_15M_PCT or regime["btc_1h_pct"] <= BTC_DUMP_DROP_1H_PCT:
            regime["gate_mode"] = "COUNTER_TREND_ONLY"
            regime["gate_locked"] = False
            regime["reason"] = f"BTC 완만 조정 · A안 역추세(Track E·F) 전용 모드 (15m {regime['btc_15m_pct']:+.2f}%, 1H {regime['btc_1h_pct']:+.2f}%)"
    except Exception:
        pass
    return regime


def _get_oi_ref_and_update_snapshot(oi_snapshot, candidates, now_ts):
    ref_map = {}
    for sym in list(oi_snapshot.keys()):
        if is_stock_or_tradfi_symbol(sym):
            del oi_snapshot[sym]
            continue
        hist = oi_snapshot.get(sym)
        if not isinstance(hist, list):
            del oi_snapshot[sym]
            continue
        hist = [h for h in hist if isinstance(h, list) and len(h) == 3 and (now_ts - int(h[0])) <= 7200]
        if hist:
            oi_snapshot[sym] = hist[-4:]
        else:
            del oi_snapshot[sym]

    for c in candidates:
        sym = c.get("symbol", "")
        p = to_float(c.get("price"))
        oi = to_float(c.get("oi_m"))
        if not sym or p <= 0 or oi <= 0:
            continue

        hist = oi_snapshot.get(sym, [])
        best_ref = None
        for h in hist:
            age = now_ts - int(h[0])
            if 600 <= age <= 5400:
                best_ref = {"ts": int(h[0]), "price": to_float(h[1]), "oi_m": to_float(h[2])}
                break
        if best_ref:
            ref_map[sym] = best_ref

        if not hist or (now_ts - int(hist[-1][0])) >= 240:
            hist.append([now_ts, round(p, 6), round(oi, 3)])
            oi_snapshot[sym] = hist[-4:]

    return ref_map


def sanitize_flags_by_mfe(item):
    """
    [버그 수정] 단순 윗꼬리 MFE(fav)만으로 tp2_hit을 강제 승격하지 않도록 수정!
    - tp2_hit은 오직 15분봉(장기 1H) 확정 종가 돌파(tp2_alerted / tp2_hit) 또는 TP3 돌파 시에만 유지
    """
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
    tp3_req = ((to_float(tps[2]) - entry_p) / entry_p * 100.0 - 0.2) if (len(tps) > 2 and entry_p > 0 and to_float(tps[2]) > entry_p) else 19.5

    if fav >= max(6.5, tp1_req) or item.get("tp1_alerted"):
        item["tp1_hit"] = True
    if item.get("tp2_alerted") or item.get("tp3_alerted") or item.get("tp3_hit"):
        item["tp2_hit"] = True
    if fav >= max(19.0, tp3_req) or item.get("tp3_alerted"):
        item["tp3_hit"] = True
        item["tp2_hit"] = True
        item["tp1_hit"] = True
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
    for s, h_list in (incoming_state.get("oi_snapshot") or {}).items():
        if not is_stock_or_tradfi_symbol(s) and isinstance(h_list, list):
            cur_list = base_state.setdefault("oi_snapshot", {}).get(s, [])
            by_ts = {int(x[0]): x for x in cur_list + h_list if isinstance(x, list) and len(x) == 3}
            base_state["oi_snapshot"][s] = [by_ts[t] for t in sorted(by_ts.keys())][-4:]
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
            p["tp2_armed_ts_ms"] = max(int(to_float(p.get("tp2_armed_ts_ms"), 0)), int(to_float(ev.get("tp2_armed_ts_ms"), 0)))
            p["tp3_armed_ts_ms"] = max(int(to_float(p.get("tp3_armed_ts_ms"), 0)), int(to_float(ev.get("tp3_armed_ts_ms"), 0)))
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
            pt["tp2_armed_ts_ms"] = max(int(to_float(pt.get("tp2_armed_ts_ms"), 0)), int(to_float(tv.get("tp2_armed_ts_ms"), 0)))
            pt["tp3_armed_ts_ms"] = max(int(to_float(pt.get("tp3_armed_ts_ms"), 0)), int(to_float(tv.get("tp3_armed_ts_ms"), 0)))
            for fg in ALL_LOCK_FLAGS:
                if tv.get(fg) or pt.get(fg):
                    pt[fg] = True
            base_state["tracked"][s] = sanitize_flags_by_mfe(pt)

        cur_t = base_state["tracked"].get(s)
        if isinstance(cur_t, dict):
            ep_ref = base_state["signal_episodes"].get(f"{s}_{int(cur_t.get('created_at', 0))}")
            if isinstance(ep_ref, dict):
                for fg in ALL_LOCK_FLAGS:
                    if ep_ref.get(fg) or cur_t.get(fg):
                        cur_t[fg] = True
                        ep_ref[fg] = True
                for tk_ms in ("tp2_armed_ts_ms", "tp3_armed_ts_ms"):
                    m_ts = max(int(to_float(cur_t.get(tk_ms), 0)), int(to_float(ep_ref.get(tk_ms), 0)))
                    if m_ts > 0:
                        cur_t[tk_ms] = m_ts
                        ep_ref[tk_ms] = m_ts
    return base_state


def load_state():
    merged = {"tracked": {}, "alert_history": {}, "sl_history": {}, "signal_episodes": {}, "oi_snapshot": {}}
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
    for tk_ms in ("tp2_armed_ts_ms", "tp3_armed_ts_ms"):
        m_ts = max(int(to_float(prev.get(tk_ms), 0)), int(to_float(t_item.get(tk_ms), 0)))
        if m_ts > 0:
            t_item[tk_ms] = m_ts
    for fg in ALL_LOCK_FLAGS:
        if prev.get(fg) or t_item.get(fg):
            t_item[fg] = True
    sanitize_flags_by_mfe(t_item)

    signal_episodes[ep_key] = {
        "symbol": symbol, "created_at": c_at, "track_type": t_item.get("track_type", "TRACK_A"),
        "is_long_term": bool(t_item.get("is_long_term", False)),
        "score": int(t_item.get("score", 90)), "entry_price": to_float(t_item.get("entry_price")),
        "initial_stop_price": to_float(t_item.get("initial_stop_price") or t_item.get("stop_price")),
        "stop_price": to_float(t_item.get("stop_price")), "sl_pct": to_float(t_item.get("sl_pct", DEFAULT_FIXED_SL_PCT)),
        "tps": t_item.get("tps", []), "tp_pcts": t_item.get("tp_pcts", []),
        "max_high_price": t_item["max_high_price"], "max_favorable_pct": t_item["max_favorable_pct"],
        "realized_weighted_pct": to_float(t_item.get("realized_weighted_pct", 0.0)),
        "vol_mult": to_float(t_item.get("vol_mult", 1.0)), "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
        "oi_m": to_float(t_item.get("oi_m", 0.0)), "oi_ratio": to_float(t_item.get("oi_ratio", 0.0)),
        "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
        "change_24h": to_float(t_item.get("change_24h", 0.0)),
        "btc_1h_pct": to_float(t_item.get("btc_1h_pct", 0.0)),
        "hold_minutes": int(t_item.get("hold_minutes", 0)),
        "tp2_armed_ts_ms": int(to_float(t_item.get("tp2_armed_ts_ms"), 0)),
        "tp3_armed_ts_ms": int(to_float(t_item.get("tp3_armed_ts_ms"), 0)),
        **{fg: bool(t_item.get(fg, False)) for fg in ALL_LOCK_FLAGS},
        "sl_at": int(t_item.get("sl_at", 0)),
    }


def _record_pm_win(symbol, t_item, now_ts, max_fav):
    if postmortem is not None and hasattr(postmortem, "record_win_postmortem"):
        try:
            postmortem.record_win_postmortem({
                "symbol": symbol,
                "track_type": t_item.get("track_type", "TRACK_A"),
                "score": int(t_item.get("score", 90)),
                "created_at": int(t_item.get("created_at", now_ts)),
                "win_at": now_ts,
                "max_favorable_pct": max_fav,
                "vol_mult": to_float(t_item.get("vol_mult", 1.0)),
                "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
                "oi_ratio": to_float(t_item.get("oi_ratio", 0.0)),
                "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
                "change_24h": to_float(t_item.get("change_24h", 0.0)),
                "btc_1h_pct": to_float(t_item.get("btc_1h_pct", 0.0)),
            })
        except Exception:
            pass


def evaluate_single_tracked_1m(symbol, t_item, now_ts, tp_params):
    """
    [방어선 캔들 마감 확정 엔진]
    1) 진입 시각(created_ms) 이후 캔들만 엄격 집계 (진입 전 15분봉 고점·저점 오염 원천 차단)
    2) TP2 방어선 상향: 단기 '15분봉' / 장기 '1시간봉' 마감 종가(close >= TP2) 확정 시에만 상향
    3) 동시 마감 원천 차단: 방어선을 상향한 회차에서는 절대 즉시 마감하지 않음
    4) 방어선 이탈 마감(TP1 방어선 & TP3 트레일링): 상향 이후 마감된 '15분봉'(장기 '1시간봉') 종가 이탈 시 마감
    """
    if t_item.get("tp3_completed") or t_item.get("sl_alerted") or t_item.get("closed_alerted"):
        return None
    entry_ref = to_float(t_item.get("entry_price"))
    if entry_ref <= 0:
        return None

    is_long = bool(t_item.get("is_long_term") or t_item.get("track_type") == "TRACK_D")
    sl_pct_use = int(round(to_float(t_item.get("sl_pct"), DEFAULT_LONGTERM_SL_PCT if is_long else DEFAULT_FIXED_SL_PCT)))
    trail_drop_pct = 10 if is_long else 4
    trail_mult = 1.0 - (trail_drop_pct / 100.0)
    horizon_tag = "[장기 포지션] " if is_long else ""

    # 단기·스윙 = 15분봉 종가 확정 / 장기(Track D) = 1시간봉 종가 확정
    confirm_tf = "60" if is_long else "15"
    confirm_tf_ms = 3600000 if is_long else 900000
    confirm_tf_lbl = "1H" if is_long else "15m"

    stop_p = to_float(t_item.get("stop_price"))
    init_stop_p = to_float(t_item.get("initial_stop_price"), default=entry_ref * (1.0 - sl_pct_use / 100.0))
    created_at = int(t_item.get("created_at", now_ts))
    created_ms = created_at * 1000
    now_ms = now_ts * 1000
    prev_max_h = to_float(t_item.get("max_high_price"), default=entry_ref)
    prev_peak_ms = int(to_float(t_item.get("peak_ts_ms"), default=created_ms))

    hist_min_low, hist_min_close = None, None
    hist_last_close, hist_last_ts_ms = 0.0, 0

    # 15분봉 조회 (고점·저점은 반드시 진입 이후 개시된 캔들 timestamp >= created_ms 만 사용!)
    df_15m = fetch_klines(symbol, "15", limit=150) if (now_ts - created_at) >= 180 else None
    if df_15m is not None and not df_15m.empty and "timestamp" in df_15m.columns:
        post_15m = df_15m[df_15m["timestamp"] >= created_ms]
        if not post_15m.empty:
            p_idx = post_15m["high"].idxmax()
            h_15m = to_float(post_15m.loc[p_idx, "high"], default=entry_ref)
            if 0.40 <= (h_15m / entry_ref) <= 15.0 and h_15m >= prev_max_h:
                prev_max_h = h_15m
                prev_peak_ms = int(to_float(post_15m.loc[p_idx, "timestamp"], default=created_ms))
            hist_min_low = to_float(post_15m["low"].min(), default=0.0)
            hist_min_close = to_float(post_15m["close"].min(), default=0.0)
            hist_last_close = to_float(post_15m["close"].iloc[-1], default=0.0)
            hist_last_ts_ms = int(to_float(post_15m["timestamp"].iloc[-1], default=0))

    # 확정 마감 캔들 데이터프레임 준비 (단기=15m 재사용 / 장기=1H 조회)
    df_confirm = fetch_klines(symbol, "60", limit=150) if is_long else df_15m
    closed_post_tf = None
    if df_confirm is not None and not df_confirm.empty and "timestamp" in df_confirm.columns:
        close_ts_series = df_confirm["timestamp"].astype("int64") + confirm_tf_ms
        # 진입 후 최소 60초 이상 경과 시점에 마감 완료된 캔들만 추출 (미마감 현재 캔들 제외)
        mask_closed = (close_ts_series >= (created_ms + 60000)) & (close_ts_series <= now_ms)
        closed_post_tf = df_confirm[mask_closed].copy()
        if not closed_post_tf.empty:
            closed_post_tf["close_ts_ms"] = close_ts_series[mask_closed].values

    # 실시간 1분봉 조회 (TP1/TP3 터치 및 손절 SL 실시간 감시용)
    df_1m = fetch_verified_1m_klines(symbol, entry_ref=entry_ref, limit=60)
    if df_1m.empty:
        time.sleep(0.4)
        df_1m = fetch_verified_1m_klines(symbol, entry_ref=entry_ref, limit=30)
    post_1m = df_1m[df_1m["timestamp"] >= created_ms].copy() if not df_1m.empty else df_1m

    base = symbol.replace("USDT", "")
    has_live_price = False
    last_close, last_ts_ms = 0.0, 0
    min_1m_low, min_1m_close = None, None

    if not post_1m.empty:
        w_idx = post_1m["high"].idxmax()
        w_high = to_float(post_1m.loc[w_idx, "high"], default=entry_ref)
        w_ts = int(to_float(post_1m.loc[w_idx, "timestamp"], default=created_ms))
        max_high = max(w_high, prev_max_h, entry_ref)
        peak_ts_ms = w_ts if w_high >= prev_max_h else prev_peak_ms
        last_close = to_float(post_1m["close"].iloc[-1], default=0.0)
        last_ts_ms = int(to_float(post_1m["timestamp"].iloc[-1], default=0))
        min_1m_low = to_float(post_1m["low"].min(), default=0.0)
        min_1m_close = to_float(post_1m["close"].min(), default=0.0)
        if last_close > 0 and last_ts_ms > 0:
            has_live_price = True
    else:
        max_high, peak_ts_ms = max(prev_max_h, entry_ref), prev_peak_ms
        if hist_last_close > 0 and hist_last_ts_ms > 0:
            has_live_price = True
            last_close, last_ts_ms = hist_last_close, hist_last_ts_ms

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
    g2 = round(max(12.0, (tp2_p - entry_ref) / entry_ref * 100.0), 1)
    g3 = round(max(19.5, (tp3_p - entry_ref) / entry_ref * 100.0), 1)
    msgs = []

    # 최신 마감 완료 캔들(15m / 장기 1H) 정보 추출
    has_closed_bar = closed_post_tf is not None and not closed_post_tf.empty
    last_closed_c = to_float(closed_post_tf["close"].iloc[-1], 0.0) if has_closed_bar else 0.0
    last_closed_ts_ms = int(to_float(closed_post_tf["close_ts_ms"].iloc[-1], 0)) if has_closed_bar else 0

    # =========================================================================
    # [3차 돌파]: 50% 익절 확정 + 방어선 TP2 상향 + 잔여 50% 트레일링(이후 15m/1H 종가 이탈 시 마감)
    # =========================================================================
    if max_high >= tp3_p or t_item.get("tp3_hit"):
        for fg in ("tp1_hit", "tp2_hit", "tp3_hit", "tp1_alerted", "tp2_alerted"):
            t_item[fg] = True
        trail3_p = max(tp2_p, max_high * trail_mult)
        stop_p = max(stop_p, trail3_p)
        t_item["stop_price"] = stop_p

        just_armed_tp3 = False
        if not t_item.get("tp3_alerted"):
            t_item["tp3_alerted"] = True
            t_item["tp3_armed_ts_ms"] = max(peak_ts_ms, now_ms)
            just_armed_tp3 = True
            _record_pm_win(symbol, t_item, now_ts, max_fav)
            w3 = round(g3 * 0.50, 2)
            tr_pct = round((stop_p - entry_ref) / entry_ref * 100.0, 1)
            msgs.append(
                f"🚀 {horizon_tag}[3차 목표가(TP3) 돌파! 50% 익절 & 방어선 TP2 상향 + 고점 -{trail_drop_pct}% 트레일링 가동] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 📊 [3차 분할 익절(비중 50%)]: +{g3:.1f}% 확정 (기여 +{w3:.2f}%)\n"
                f"• 🎯 남은 50% 물량은 바닥 방어선 TP2({fmt_price(tp2_p)}, +{g2:.1f}%) + 고점 대비 -{trail_drop_pct}% 자동 트레일링({fmt_price(stop_p)}, +{tr_pct:.1f}% · {confirm_tf_lbl} 종가 이탈 시 마감)으로 추적합니다!"
            )

        tp3_armed_ms = int(to_float(t_item.get("tp3_armed_ts_ms"), default=peak_ts_ms))
        if (
            (not just_armed_tp3)
            and has_closed_bar
            and last_closed_c > 0
            and last_closed_ts_ms > max(peak_ts_ms, tp3_armed_ms)
            and last_closed_c <= stop_p
            and not t_item.get("closed_alerted")
        ):
            t_item["tp3_completed"] = True
            t_item["trailing_closed"] = True
            t_item["closed_alerted"] = True
            t_item["tp3_at"] = now_ts
            exit_g = round(max(g2, (stop_p - entry_ref) / entry_ref * 100.0), 1)
            w3, w_rem = round(g3 * 0.50, 2), round(exit_g * 0.50, 2)
            total_cum = round(w3 + w_rem, 2)
            t_item["realized_weighted_pct"] = total_cum
            msgs.append(
                f"🎯 {horizon_tag}[3차 돌파 후 잔여 50% 트레일링 스탑({confirm_tf_lbl} 종가) 최종 익절 완료] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 최고가 {fmt_price(max_high)} (+{max_fav:.1f}%) → {confirm_tf_lbl} 종가 마감 {fmt_price(last_closed_c)} (트레일링선 {fmt_price(stop_p)})\n"
                f"• 📊 [최종 익절 내역]: 3차(50%) +{g3:.1f}%(+{w3:.2f}%) + 잔여(50%) +{exit_g:.1f}%(+{w_rem:.2f}%)\n"
                f"• 💰 [최종 합산 확정 수익률]: +{total_cum:.2f}% 달성 (비중 100% 합산)"
            )
        return {"messages": msgs, "sl_record": None}

    # =========================================================================
    # [2차 돌파]: 단기 15m / 장기 1H 마감 종가가 TP2 위 안착 시에만 방어선 TP1 상향 고정!
    # =========================================================================
    tp2_confirmed = bool(t_item.get("tp2_hit") or t_item.get("tp2_alerted"))
    tp2_confirm_close_ts = int(to_float(t_item.get("tp2_armed_ts_ms"), 0))

    if (not tp2_confirmed) and has_closed_bar:
        tp2_bars = closed_post_tf[closed_post_tf["close"].astype(float) >= tp2_p]
        if not tp2_bars.empty:
            tp2_confirmed = True
            tp2_confirm_close_ts = int(to_float(tp2_bars["close_ts_ms"].iloc[0], now_ms))

    if tp2_confirmed:
        for fg in ("tp1_hit", "tp2_hit", "tp1_alerted"):
            t_item[fg] = True
        stop_p = max(stop_p, tp1_p)
        t_item["stop_price"] = stop_p

        just_armed_tp2 = False
        if not t_item.get("tp2_alerted"):
            t_item["tp2_alerted"] = True
            t_item["tp2_armed_ts_ms"] = tp2_confirm_close_ts if tp2_confirm_close_ts > 0 else now_ms
            just_armed_tp2 = True
            _record_pm_win(symbol, t_item, now_ts, max_fav)
            msgs.append(
                f"✅ {horizon_tag}[2차 구간(TP2) {confirm_tf_lbl} 종가 안착 돌파! 100% 홀딩 & 방어선 TP1 상향 고정] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 🛡️ [방어선 상향 고정]: {confirm_tf_lbl} 종가 돌파 확인 완료! 추가 익절 없이 100% 전량 홀딩하며 방어선을 TP1({fmt_price(tp1_p)}, +{g1:.1f}%)로 상향 고정합니다. (이탈 판정: 이후 {confirm_tf_lbl} 종가 기준)"
            )

        # 방어선 상향 당회차 즉시 마감 원천 차단 + 상향 이후 마감된 15m(장기 1H) 종가가 TP1 이하일 때만 익절 마감
        tp2_armed_ms = int(to_float(t_item.get("tp2_armed_ts_ms"), default=peak_ts_ms))
        if (
            (not just_armed_tp2)
            and has_closed_bar
            and last_closed_c > 0
            and last_closed_ts_ms > tp2_armed_ms
            and last_closed_c <= stop_p
            and not t_item.get("closed_alerted")
        ):
            t_item["tp3_completed"] = True
            t_item["trailing_closed"] = True
            t_item["closed_alerted"] = True
            t_item["tp3_at"] = now_ts
            exit_g = round(max(g1, (stop_p - entry_ref) / entry_ref * 100.0), 1)
            t_item["realized_weighted_pct"] = exit_g
            msgs.append(
                f"🔒 {horizon_tag}[TP1 방어선 {confirm_tf_lbl} 종가 이탈 · 100% 전량 수익 마감 완료] ${base} ({symbol})\n"
                f"• 진입 후 최고 +{max_fav:.1f}% 도달 후 방어선 TP1({fmt_price(stop_p)})을 {confirm_tf_lbl} 종가({fmt_price(last_closed_c)})로 이탈하여 전량(100%) 익절 마감합니다.\n"
                f"• 💰 [최종 합산 확정 수익률]: +{exit_g:.2f}% 달성 (100% 전량 익절)"
            )
        return {"messages": msgs, "sl_record": None}

    # [1차 돌파]: 익절·방어선 이동 없이 100% 홀딩 유지
    if max_high >= tp1_p or t_item.get("tp1_hit"):
        t_item["tp1_hit"] = True
        if not t_item.get("tp1_alerted"):
            t_item["tp1_alerted"] = True
            msgs.append(
                f"✅ {horizon_tag}[1차 목표가(TP1) 돌파! 익절 없이 100% 홀딩 유지] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 📈 분할 익절 없이 100% 전량 홀딩하며 2차 목표가({fmt_price(tp2_p)}, +{g2:.1f}% · {confirm_tf_lbl} 종가 안착 시 방어선 상향) 돌파를 추적합니다."
            )

    # [2차 확정 전]: 고정 손절선 이탈 감시 (단기 -6% / 장기 -20%)
    valid_lows = [v for v in (hist_min_low, min_1m_low) if v is not None and v > 0]
    valid_closes = [v for v in (hist_min_close, min_1m_close) if v is not None and v > 0]
    if has_live_price and valid_lows and valid_closes:
        min_low, min_close = min(valid_lows), min(valid_closes)
        if min_low <= init_stop_p and (min_close <= init_stop_p or (last_close > 0 and last_close <= init_stop_p)):
            trig = min(min_close, last_close) if last_close > 0 else min_close
            t_item["sl_alerted"] = True
            t_item["closed_alerted"] = True
            t_item["sl_at"] = now_ts
            oi_r_val = to_float(t_item.get("oi_ratio"), 0.0)
            if oi_r_val <= 0 and to_float(t_item.get("turnover_m"), 0.0) > 0:
                oi_r_val = round(to_float(t_item.get("oi_m"), 0.0) / to_float(t_item.get("turnover_m"), 1.0) * 100.0, 1)
            sl_payload = {
                "symbol": symbol, "track_type": t_item.get("track_type", "TRACK_A"),
                "score": int(t_item.get("score", 90)), "entry_price": entry_ref,
                "stop_price": init_stop_p, "sl_trigger_price": trig,
                "sl_pct": sl_pct_use, "created_at": created_at, "sl_at": now_ts,
                "hold_minutes": hold_min, "max_high_price": max_high, "max_favorable_pct": max_fav,
                "vol_mult": to_float(t_item.get("vol_mult", 1.0)), "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
                "oi_m": to_float(t_item.get("oi_m", 0.0)), "oi_ratio": oi_r_val,
                "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
                "change_24h": to_float(t_item.get("change_24h", 0.0)),
                "btc_1h_pct": to_float(t_item.get("btc_1h_pct", 0.0)),
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
    is_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "GitHub Actions"], check=False)
        for attempt in range(4):
            fetch_r = subprocess.run(["git", "fetch", "origin", "main"], check=False)
            if fetch_r.returncode != 0 and is_ci:
                time.sleep(2.0)
                continue

            loc_dict = load_state()
            for rf in ("bot_state.json", "scanner_state.json"):
                show_res = subprocess.run(["git", "show", f"origin/main:{rf}"], capture_output=True, text=True, check=False)
                if show_res.returncode == 0 and show_res.stdout:
                    try:
                        loc_dict = _merge_state_dicts(loc_dict, json.loads(show_res.stdout))
                    except Exception:
                        pass
            save_state(loc_dict)

            pm_raw = None
            if os.path.exists(POSTMORTEM_DB_FILE):
                try:
                    with open(POSTMORTEM_DB_FILE, "r", encoding="utf-8") as pf:
                        pm_raw = pf.read()
                except Exception:
                    pass

            subprocess.run(["git", "reset", "--soft", "origin/main"], check=False)
            with open(LOCAL_RUNTIME_CACHE, "r", encoding="utf-8") as s:
                raw_txt = s.read()
            for target_f in (STATE_FILE, "bot_state.json", "scanner_state.json"):
                with open(target_f, "w", encoding="utf-8") as d:
                    d.write(raw_txt)
                subprocess.run(["git", "add", target_f], check=False)
            if pm_raw:
                with open(POSTMORTEM_DB_FILE, "w", encoding="utf-8") as pf:
                    pf.write(pm_raw)
                subprocess.run(["git", "add", POSTMORTEM_DB_FILE], check=False)

            if subprocess.run(["git", "diff", "--staged", "--quiet"], check=False).returncode != 0:
                subprocess.run(["git", "commit", "-m", "chore: update scanner state [skip ci]"], check=False)
                if subprocess.run(["git", "push", "origin", "HEAD:main"], check=False).returncode == 0:
                    return True
            else:
                return True
            time.sleep(2.0 + attempt)
    except Exception as e:
        print(f"[Git 동기화 예외] {e}")
    return not is_ci


def _ensure_synced_before_alert():
    if _git_sync_state():
        return True
    time.sleep(2.0)
    return _git_sync_state()


def main():
    start_t = time.time()
    state = load_state()
    tracked, alert_history = state["tracked"], state["alert_history"]
    sl_history, signal_episodes = state["sl_history"], state["signal_episodes"]
    oi_snapshot = state.setdefault("oi_snapshot", {})
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

    # [1단계] 기존 보유 포지션 감시 (보유 암호화폐 보호 + BTC 게이트와 무관하게 항상 실행)
    active_syms = []
    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        t_item = tracked[sym]
        c_at = int(t_item.get("created_at", 0))
        age_sec = now_ts - c_at
        if is_stock_or_tradfi_symbol(sym) or is_tradfi_ticker(base_c) or base_c in rwa_blocked:
            del tracked[sym]
            continue
        sync_episode_ledger(signal_episodes, sym, t_item, sl_history)
        if c_at > int(alert_history.get(sym, 0)):
            alert_history[sym] = c_at
        is_done = bool(t_item.get("sl_alerted") or t_item.get("tp3_completed") or t_item.get("closed_alerted"))
        done_ts = int(t_item.get("sl_at") or t_item.get("tp3_at") or c_at)
        max_age = 2592000 if t_item.get("is_long_term") else 432000
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
            if _ensure_synced_before_alert():
                for m in pending_msgs:
                    send_telegram_reliable(m)
                    sent_count += 1
            else:
                print(f"[중복방지 보류] Git 동기화 실패로 추적 알림 {len(pending_msgs)}건 발송 보류")

    # [오답노트 v2.0] 손절 후 2시간+ 경과 종목 자동 사후 부검 (스캔당 최대 4건 경량 처리)
    if postmortem is not None and hasattr(postmortem, "run_post_sl_autopsy"):
        try:
            autopsied = postmortem.run_post_sl_autopsy(fetch_klines, max_checks=4)
            if autopsied:
                print(f"[사후 부검 완료] {autopsied}건 분석 반영")
        except Exception:
            pass

    if not candidates:
        save_state(state)
        return

    oi_ref_map = _get_oi_ref_and_update_snapshot(oi_snapshot, candidates, now_ts)

    # [2단계] BTC 3단계 레짐 체크 (A안: 폭락 시 전면 차단 / 완만 조정 시 Track E·F만 예외 허용)
    market_regime = check_btc_market_regime()
    if market_regime["gate_mode"] == "FULL_LOCK":
        save_state(state)
        print(f"[{now_str}] 🛑 [BTC 폭락 전면 차단] {market_regime['reason']} (기존 감시 {sent_count}건 완료)")
        return
    if market_regime["gate_mode"] == "COUNTER_TREND_ONLY":
        print(f"[{now_str}] ⚠️ {market_regime['reason']}")

    def _recent_sl_count(sym_check):
        return sum(
            1 for v in sl_history.values()
            if isinstance(v, dict) and v.get("symbol") == sym_check and (now_ts - int(v.get("sl_at", 0))) < RE_ALERT_COOLDOWN_SEC
        )

    def _was_stopped_recently(sym_check):
        pt = tracked.get(sym_check)
        if pt and pt.get("sl_alerted") and (now_ts - int(pt.get("sl_at") or pt.get("created_at") or 0)) < RE_ALERT_COOLDOWN_SEC:
            return True
        return _recent_sl_count(sym_check) > 0

    to_analyze = []
    for c in candidates:
        sym = c["symbol"]
        if (valid_syms and sym not in valid_syms) or is_stock_or_tradfi_symbol(sym):
            continue
        prev_t = tracked.get(sym)
        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed") and not prev_t.get("closed_alerted"):
            continue
        sl_cnt = _recent_sl_count(sym)
        cd = SL_RECOVERY_COOLDOWN_SEC if (_was_stopped_recently(sym) and sl_cnt < 2) else RE_ALERT_COOLDOWN_SEC
        if (now_ts - int(alert_history.get(sym, 0))) < cd:
            continue
        to_analyze.append(c)

    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        f_map = {
            ex.submit(analyze_symbol, c, oi_ref_map.get(c["symbol"]), market_regime): c["symbol"]
            for c in to_analyze
        }
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
        sl_cnt = _recent_sl_count(sym)

        if was_sl and elapsed_alert < RE_ALERT_COOLDOWN_SEC:
            if sl_cnt >= 2 or elapsed_alert < SL_RECOVERY_COOLDOWN_SEC or res["score"] < 92 or to_float(res.get("rsi6_15m"), 50.0) >= 68.0:
                continue
            res["track_title"] = "🔄 [스탑헌팅 후 V자 탈환 정예 재진입]"
            if "스탑헌팅 바닥 확인 후 비과열 V자 재탈환" not in res["reasons"]:
                res["reasons"].insert(0, "스탑헌팅 바닥 확인 후 비과열 V자 재탈환")
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
            "oi_m": res.get("oi_m", 0.0), "oi_ratio": res.get("oi_ratio", 0.0),
            "turnover_m": res.get("turnover_m", 0.0), "change_24h": res.get("change_24h", 0.0),
            "funding_pct": res.get("funding_pct", 0.0), "btc_1h_pct": res.get("btc_1h_pct", 0.0),
            "reasons": res.get("reasons", []),
            "max_high_price": res["price"], "peak_ts_ms": now_ts * 1000, "max_favorable_pct": 0.0,
        }
        alert_history[sym] = now_ts
        sync_episode_ledger(signal_episodes, sym, tracked[sym], sl_history)
        save_state(state)

        if _ensure_synced_before_alert():
            _send_signal_notifications(res, now_str)
            new_sent += 1
            sent_count += 1
        else:
            print(f"[중복방지 보류] Git 동기화 실패로 {sym} 신규 알림 발송 보류")

    save_state(state)
    print(f"[{now_str}] 스캔 완료 ({time.time() - start_t:.1f}초 | BTC 15m {market_regime['btc_15m_pct']:+.2f}% / 4H {market_regime['btc_4h_pct']:+.2f}%) - 전송 {sent_count}건")


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
