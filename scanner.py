import os, sys, json, time, copy, tempfile, subprocess, requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from config import (
    STATE_FILE,
    KST,
    SCORE_THRESHOLD,
    REENTRY_SCORE_THRESHOLD,
    FIXED_SL_PCT,
    LONGTERM_SL_PCT,
    ROUNDTRIP_FEE_SLIPPAGE_PCT,
    BTC_DUMP_DROP_15M_PCT,
    BTC_DUMP_DROP_1H_PCT,
    BTC_CRASH_15M_PCT,
    BTC_CRASH_1H_PCT,
    POSTMORTEM_DB_FILE,
    to_float,
    is_tradfi_ticker,
)
from api import (
    get_scanner_candidates,
    fetch_verified_1m_klines,
    fetch_klines,
    fetch_klines_window,
    get_valid_bybit_linear_symbols,
    is_stock_or_tradfi_symbol,
    clean_url,
)
from analyzer import analyze_symbol, get_dynamic_tp_params
from notifier import (
    build_briefing_message,
    build_x_post_message,
    build_x_intent_reply_markup,
    build_x_hook_line,
    send_telegram_single,
    fmt_price,
    generate_signal_chart_bytes,
    send_telegram_photo_with_caption,
)

try:
    from notifier import build_x_hook_pair
except Exception:
    def build_x_hook_pair(res):
        return build_x_hook_line(res), ""

try:
    from notifier import (
        generate_profit_replay_video_bytes,
        send_telegram_video_with_caption,
        get_last_video_error,
    )
except Exception as _imp_e:
    print(f"[경고] notifier 영상 함수 import 실패: {_imp_e}")
    generate_profit_replay_video_bytes = None
    send_telegram_video_with_caption = None
    get_last_video_error = lambda: f"notifier 영상 함수 import 실패: {_imp_e}"

try:
    from notifier import generate_profit_replay_videos_bundle
except Exception:
    generate_profit_replay_videos_bundle = None

try:
    import postmortem
except Exception:
    postmortem = None

SCHEME = "ht" + "tps://"
RE_ALERT_COOLDOWN_SEC = 86400
SL_RECOVERY_COOLDOWN_SEC = 14400
TP_CLOSE_REENTRY_COOLDOWN_SEC = 21600
HISTORY_RETENTION_SEC = 259200

# 원장·추적 보존 기간: 단기 원장 35일(추적 안전상한 90일) / 장기(Track D) 원장·추적 모두 365일(1년)
EPISODE_RETENTION_SEC = 35 * 86400
LONGTERM_EPISODE_RETENTION_SEC = 365 * 86400
SHORT_TRACK_MAX_AGE_SEC = 90 * 86400
LONGTERM_TRACK_MAX_AGE_SEC = 365 * 86400

MAX_NEW_ALERTS_PER_SCAN = 2

# config.py 단일 원천 기준 (신규 88점 / 손절 후 재진입 90점 · 소프트 캡 없음)
ELITE_SCORE_THRESHOLD = int(SCORE_THRESHOLD)
REENTRY_CUT_SCORE = int(REENTRY_SCORE_THRESHOLD)
DEFAULT_FIXED_SL_PCT = int(FIXED_SL_PCT)
DEFAULT_LONGTERM_SL_PCT = int(LONGTERM_SL_PCT)
FEE_SLIPPAGE_PCT = float(ROUNDTRIP_FEE_SLIPPAGE_PCT)

# Windows / Linux OS 공용 임시 폴더 사용 (Git 저장 파일은 STATE_FILE 단일화)
_TMP_DIR = tempfile.gettempdir()
LOCAL_RUNTIME_CACHE = os.path.join(_TMP_DIR, "scanner_runtime_state.json")
GIT_STATE_STAGE_TMP = os.path.join(_TMP_DIR, "scanner_git_state_clean.json")
GIT_PM_STAGE_TMP = os.path.join(_TMP_DIR, "postmortem_git_merged.json")
LEGACY_DUPLICATE_STATE_FILE = "scanner_state.json"
READ_STATE_FILES = list(dict.fromkeys([STATE_FILE, LEGACY_DUPLICATE_STATE_FILE, LOCAL_RUNTIME_CACHE]))

ALL_LOCK_FLAGS = (
    "tp1_hit", "tp2_hit", "tp3_hit",
    "tp1_alerted", "tp2_alerted", "tp3_alerted",
    "tp3_completed", "trailing_closed", "sl_alerted", "expired_closed", "closed_alerted",
)


def send_telegram_reliable(text, reply_markup=None):
    """텔레그램 메시지 재시도 전송 (429 레이트 리밋 대기 + HTTP 오류 진단 로그 출력)"""
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

    for attempt in range(3):
        try:
            r = requests.post(url, json=payload, timeout=12)
            if r.status_code == 200:
                time.sleep(0.6)
                return True
            if r.status_code == 429:
                wait_s = int(to_float((r.json().get("parameters") or {}).get("retry_after"), 3))
                time.sleep(wait_s + 0.5)
                continue
            print(f"[Telegram sendMessage 오류 {attempt + 1}/3] HTTP {r.status_code}: {r.text[:160]}")
        except Exception as e:
            print(f"[Telegram sendMessage 예외 {attempt + 1}/3] {e}")
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


def _write_json_atomic(path, obj):
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def check_btc_market_regime():
    """BTC 3단계 마켓 레짐 게이트 (BTCUSDT 1H limit=24 정상 수집)"""
    regime = {
        "gate_mode": "OPEN",
        "gate_locked": False,
        "btc_15m_pct": 0.0,
        "btc_1h_pct": 0.0,
        "btc_4h_pct": 0.0,
        "btc_data_ok": False,
        "reason": "정상",
    }
    try:
        df15 = fetch_klines("BTCUSDT", interval="15", limit=24)
        df1h = fetch_klines("BTCUSDT", interval="60", limit=24)

        ok_15, ok_1h = False, False
        if not df15.empty and len(df15) >= 2:
            c_now = to_float(df15["close"].iloc[-1])
            o_now = to_float(df15["open"].iloc[-1], c_now)
            c_prev = to_float(df15["close"].iloc[-2], c_now)
            if o_now > 0 and c_prev > 0:
                regime["btc_15m_pct"] = round(min((c_now - o_now) / o_now * 100.0, (c_now - c_prev) / c_prev * 100.0), 2)
                ok_15 = True

        if not df1h.empty and len(df1h) >= 2:
            c_now_h = to_float(df1h["close"].iloc[-1])
            o_now_h = to_float(df1h["open"].iloc[-1], c_now_h)
            c_prev_h = to_float(df1h["close"].iloc[-2], c_now_h)
            if o_now_h > 0 and c_prev_h > 0:
                regime["btc_1h_pct"] = round(min((c_now_h - o_now_h) / o_now_h * 100.0, (c_now_h - c_prev_h) / c_prev_h * 100.0), 2)
                ok_1h = True
            if len(df1h) >= 5:
                c_4h_ago = to_float(df1h["close"].iloc[-5], c_now_h)
                if c_4h_ago > 0:
                    regime["btc_4h_pct"] = round(((c_now_h - c_4h_ago) / c_4h_ago) * 100.0, 2)

        regime["btc_data_ok"] = bool(ok_15 and ok_1h)
        if not regime["btc_data_ok"]:
            print(f"[경고] BTC 레짐 캔들 부분 수신 (15m={ok_15}, 1H={ok_1h})")

        if regime["btc_1h_pct"] <= BTC_CRASH_1H_PCT or regime["btc_15m_pct"] <= BTC_CRASH_15M_PCT:
            regime["gate_mode"] = "FULL_LOCK"
            regime["gate_locked"] = True
            regime["reason"] = f"BTC 폭락 전면 차단 (15m {regime['btc_15m_pct']:+.2f}%, 1H {regime['btc_1h_pct']:+.2f}%)"
        elif regime["btc_15m_pct"] <= BTC_DUMP_DROP_15M_PCT or regime["btc_1h_pct"] <= BTC_DUMP_DROP_1H_PCT:
            regime["gate_mode"] = "COUNTER_TREND_ONLY"
            regime["gate_locked"] = False
            regime["reason"] = f"BTC 완만 조정 · 역추세(Track E·F) + 장기(Track D) 전용 모드 (15m {regime['btc_15m_pct']:+.2f}%, 1H {regime['btc_1h_pct']:+.2f}%)"
    except Exception as e:
        print(f"[BTC 레짐 조회 예외] {e}")
    return regime


def _get_oi_ref_and_update_snapshot(oi_snapshot, candidates, now_ts):
    """종목별 OI 스냅샷 8개 보관 (임시 캐시 전용, 4시간 30분 연속 유지)"""
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
            oi_snapshot[sym] = hist[-8:]
        else:
            del oi_snapshot[sym]

    for c in candidates:
        sym = c.get("symbol", "")
        p = to_float(c.get("price"))
        oi = to_float(c.get("oi_m"))
        if not sym or p <= 0 or oi <= 0:
            continue

        hist = oi_snapshot.get(sym, [])
        ideal_pool = []
        fallback_pool = []
        for h in hist:
            age = now_ts - int(h[0])
            if 900 <= age <= 2700:
                ideal_pool.append((abs(age - 1800), h))
            elif 600 <= age <= 5400:
                fallback_pool.append((abs(age - 1800), h))

        chosen = None
        if ideal_pool:
            ideal_pool.sort(key=lambda x: x[0])
            chosen = ideal_pool[0][1]
        elif fallback_pool:
            fallback_pool.sort(key=lambda x: x[0])
            chosen = fallback_pool[0][1]

        if chosen is not None:
            ref_map[sym] = {"ts": int(chosen[0]), "price": to_float(chosen[1]), "oi_m": to_float(chosen[2])}

        if not hist or (now_ts - int(hist[-1][0])) >= 240:
            hist.append([now_ts, round(p, 6), round(oi, 3)])
            oi_snapshot[sym] = hist[-8:]

    return ref_map


def sanitize_flags_by_mfe(item):
    """단순 윗꼬리 MFE로 tp2_hit·tp3_hit 강제 승격 방지 (확정 종가 플래그 유지)"""
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

    if fav >= max(6.5, tp1_req) or item.get("tp1_alerted"):
        item["tp1_hit"] = True
    if not item.get("sl_alerted"):
        if item.get("tp2_alerted") or item.get("tp3_alerted"):
            item["tp2_hit"] = True
            item["tp1_hit"] = True
        if item.get("tp3_alerted"):
            item["tp3_hit"] = True
    return item


def _record_pruned_tracked(state, sym, created_at, now_ts):
    """의도적으로 삭제한 tracked 심볼의 (created_at) 툼스톤 기록 (원격 병합 부활 방지)"""
    pruned = state.setdefault("_pruned_tracked", {})
    prev_c = int(to_float((pruned.get(sym) or {}).get("created_at"), 0))
    pruned[sym] = {"created_at": max(prev_c, int(to_float(created_at, 0))), "pruned_at": int(now_ts)}


def _apply_state_cleanup(state, now_ts, rwa_blocked=None):
    """
    - 장기 포지션(Track D): tracked 최대 추적 365일(1년) + 원장 보존 마감일 기준 365일(1년)
    - 단기 포지션: tracked 안전상한 90일 + 원장 보존 마감일 기준 35일
    - tracked에 살아 있는 에피소드 키는 기간과 무관하게 signal_episodes 선삭제 원천 차단
    """
    rwa_set = set(rwa_blocked or ())
    tracked = state.setdefault("tracked", {})
    alert_history = state.setdefault("alert_history", {})
    sl_history = state.setdefault("sl_history", {})
    signal_episodes = state.setdefault("signal_episodes", {})
    pruned = state.setdefault("_pruned_tracked", {})

    for ps in list(pruned.keys()):
        p_info = pruned.get(ps)
        if not isinstance(p_info, dict) or (now_ts - int(to_float(p_info.get("pruned_at"), 0))) > 172800:
            del pruned[ps]

    for h_sym in list(alert_history.keys()):
        if is_stock_or_tradfi_symbol(h_sym) or (now_ts - int(to_float(alert_history.get(h_sym), 0)) > 172800):
            del alert_history[h_sym]

    for sl_k in list(sl_history.keys()):
        sl_v = sl_history.get(sl_k) or {}
        if is_stock_or_tradfi_symbol(sl_v.get("symbol", "")) or (now_ts - int(to_float(sl_v.get("sl_at"), 0)) > HISTORY_RETENTION_SEC):
            del sl_history[sl_k]

    for sym in list(tracked.keys()):
        t_item = tracked.get(sym)
        if not isinstance(t_item, dict):
            del tracked[sym]
            continue
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        c_at = int(to_float(t_item.get("created_at"), 0))
        is_long_t = bool(t_item.get("is_long_term") or t_item.get("track_type") == "TRACK_D")

        p_rec = pruned.get(sym)
        if isinstance(p_rec, dict) and c_at <= int(to_float(p_rec.get("created_at"), 0)):
            del tracked[sym]
            continue

        if is_stock_or_tradfi_symbol(sym) or is_tradfi_ticker(base_c) or (rwa_set and base_c in rwa_set):
            _record_pruned_tracked(state, sym, c_at, now_ts)
            del tracked[sym]
            continue

        is_done = bool(
            t_item.get("sl_alerted")
            or t_item.get("tp3_completed")
            or t_item.get("expired_closed")
            or t_item.get("closed_alerted")
        )
        done_ts = int(to_float(t_item.get("sl_at") or t_item.get("tp3_at") or t_item.get("expired_at") or c_at, c_at))
        if is_done and (now_ts - done_ts) > 86400:
            _record_pruned_tracked(state, sym, c_at, now_ts)
            del tracked[sym]
            continue

        track_max_age = LONGTERM_TRACK_MAX_AGE_SEC if is_long_t else SHORT_TRACK_MAX_AGE_SEC
        if c_at > 0 and (now_ts - c_at) > track_max_age:
            _record_pruned_tracked(state, sym, c_at, now_ts)
            del tracked[sym]
            continue

    active_tracked_ep_keys = set()
    for s_k, t_v in tracked.items():
        if isinstance(t_v, dict):
            c_v = int(to_float(t_v.get("created_at"), 0))
            if c_v > 0:
                active_tracked_ep_keys.add(f"{s_k}_{c_v}")

    for ep_k in list(signal_episodes.keys()):
        ep = signal_episodes.get(ep_k) or {}
        sym_ep = str(ep.get("symbol", "")).upper()
        c_at = int(to_float(ep.get("created_at"), 0))
        if is_stock_or_tradfi_symbol(sym_ep):
            del signal_episodes[ep_k]
            continue
        if ep_k not in active_tracked_ep_keys:
            is_long_ep = bool(ep.get("is_long_term") or ep.get("track_type") == "TRACK_D")
            ret_limit = LONGTERM_EPISODE_RETENTION_SEC if is_long_ep else EPISODE_RETENTION_SEC
            anchor_ts = max(
                c_at,
                int(to_float(ep.get("sl_at"), 0)),
                int(to_float(ep.get("tp3_at"), 0)),
                int(to_float(ep.get("expired_at"), 0)),
            )
            if anchor_ts > 0 and (now_ts - anchor_ts) > ret_limit:
                del signal_episodes[ep_k]
                continue
        signal_episodes[ep_k] = sanitize_flags_by_mfe(ep)
        if sym_ep and c_at > int(to_float(alert_history.get(sym_ep), 0)) and (now_ts - c_at) <= 172800:
            alert_history[sym_ep] = c_at


def _merge_state_dicts(base_state, incoming_state):
    """상태 딕셔너리 단방향 합집합 병합 (_pruned_tracked 툼스톤 존중)"""
    if not isinstance(incoming_state, dict):
        return base_state
    pruned = base_state.setdefault("_pruned_tracked", {})
    for ps, pv in (incoming_state.get("_pruned_tracked") or {}).items():
        if isinstance(pv, dict):
            prev_p = pruned.get(ps) or {}
            if int(to_float(pv.get("created_at"), 0)) >= int(to_float(prev_p.get("created_at"), 0)):
                pruned[ps] = dict(pv)

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
            base_state["oi_snapshot"][s] = [by_ts[t] for t in sorted(by_ts.keys())][-8:]
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
            if to_float(ev.get("exit_price"), 0.0) > 0:
                p["exit_price"] = to_float(ev.get("exit_price"))
            if abs(to_float(ev.get("realized_weighted_pct"), 0.0)) > 0 and abs(to_float(p.get("realized_weighted_pct"), 0.0)) == 0:
                p["realized_weighted_pct"] = to_float(ev.get("realized_weighted_pct"))
            for str_k in ("x_hook_ko", "x_hook_jp", "sub_strategy"):
                if ev.get(str_k) and not p.get(str_k):
                    p[str_k] = ev.get(str_k)
            for tk_ts in ("tp2_armed_ts_ms", "tp3_armed_ts_ms", "tp3_at", "sl_at", "expired_at", "last_checked_ts"):
                p[tk_ts] = max(int(to_float(p.get(tk_ts), 0)), int(to_float(ev.get(tk_ts), 0)))
            for fg in ALL_LOCK_FLAGS:
                if ev.get(fg) or p.get(fg):
                    p[fg] = True
            base_state["signal_episodes"][ek] = sanitize_flags_by_mfe(p)
    for s, tv in (incoming_state.get("tracked") or {}).items():
        if not isinstance(tv, dict) or is_stock_or_tradfi_symbol(s):
            continue
        tv = sanitize_flags_by_mfe(dict(tv))
        c_t = int(tv.get("created_at", 0))
        p_rec = pruned.get(s)
        if isinstance(p_rec, dict) and c_t <= int(to_float(p_rec.get("created_at"), 0)):
            continue
        if c_t > 0:
            base_state["alert_history"][s] = max(int(base_state["alert_history"].get(s, 0)), c_t)
        pt = base_state["tracked"].get(s)
        if not pt or c_t > int(pt.get("created_at", 0)):
            base_state["tracked"][s] = tv
        elif c_t == int(pt.get("created_at", 0)):
            pt["max_high_price"] = max(to_float(pt.get("max_high_price")), to_float(tv.get("max_high_price")))
            pt["max_favorable_pct"] = max(to_float(pt.get("max_favorable_pct")), to_float(tv.get("max_favorable_pct")))
            pt["stop_price"] = max(to_float(pt.get("stop_price")), to_float(tv.get("stop_price")))
            if to_float(tv.get("exit_price"), 0.0) > 0:
                pt["exit_price"] = to_float(tv.get("exit_price"))
            if abs(to_float(tv.get("realized_weighted_pct"), 0.0)) > 0 and abs(to_float(pt.get("realized_weighted_pct"), 0.0)) == 0:
                pt["realized_weighted_pct"] = to_float(tv.get("realized_weighted_pct"))
            for str_k in ("x_hook_ko", "x_hook_jp", "sub_strategy"):
                if tv.get(str_k) and not pt.get(str_k):
                    pt[str_k] = tv.get(str_k)
            for tk_ts in ("tp2_armed_ts_ms", "tp3_armed_ts_ms", "tp3_at", "sl_at", "expired_at", "last_checked_ts"):
                pt[tk_ts] = max(int(to_float(pt.get(tk_ts), 0)), int(to_float(tv.get(tk_ts), 0)))
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
                for str_k in ("x_hook_ko", "x_hook_jp", "sub_strategy"):
                    if cur_t.get(str_k) and not ep_ref.get(str_k):
                        ep_ref[str_k] = cur_t.get(str_k)
                    elif ep_ref.get(str_k) and not cur_t.get(str_k):
                        cur_t[str_k] = ep_ref.get(str_k)
                for tk_ts in ("tp2_armed_ts_ms", "tp3_armed_ts_ms", "tp3_at", "sl_at", "expired_at", "last_checked_ts"):
                    m_ts = max(int(to_float(cur_t.get(tk_ts), 0)), int(to_float(ep_ref.get(tk_ts), 0)))
                    if m_ts > 0:
                        cur_t[tk_ts] = m_ts
                        ep_ref[tk_ts] = m_ts
    return base_state


def _build_git_clean_state(state):
    """Git 저장용 경량 상태 생성 (휘발성 oi_snapshot·런타임 툼스톤 제외)"""
    return {
        "tracked": state.get("tracked") or {},
        "alert_history": state.get("alert_history") or {},
        "sl_history": state.get("sl_history") or {},
        "signal_episodes": state.get("signal_episodes") or {},
    }


def load_state():
    merged = {
        "tracked": {}, "alert_history": {}, "sl_history": {},
        "signal_episodes": {}, "oi_snapshot": {}, "_pruned_tracked": {},
    }
    _merge_state_dicts(merged, _read_json(LOCAL_RUNTIME_CACHE))
    for path in READ_STATE_FILES:
        if path != LOCAL_RUNTIME_CACHE:
            _merge_state_dicts(merged, _read_json(path))
    return merged


def save_state(state):
    """전체 런타임 상태는 OS 임시 캐시에, Git용 단일 상태(bot_state.json)는 경량본으로 저장"""
    _write_json_atomic(LOCAL_RUNTIME_CACHE, state)
    _write_json_atomic(STATE_FILE, _build_git_clean_state(state))


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
    for tk_ts in ("tp2_armed_ts_ms", "tp3_armed_ts_ms", "tp3_at", "sl_at", "expired_at", "last_checked_ts"):
        m_ts = max(int(to_float(prev.get(tk_ts), 0)), int(to_float(t_item.get(tk_ts), 0)))
        if m_ts > 0:
            t_item[tk_ts] = m_ts
    for fg in ALL_LOCK_FLAGS:
        if prev.get(fg) or t_item.get(fg):
            t_item[fg] = True
    sanitize_flags_by_mfe(t_item)

    realized_v = to_float(t_item.get("realized_weighted_pct"), 0.0)
    if realized_v == 0.0 and abs(to_float(prev.get("realized_weighted_pct"), 0.0)) > 0:
        realized_v = to_float(prev.get("realized_weighted_pct"), 0.0)
        t_item["realized_weighted_pct"] = realized_v

    signal_episodes[ep_key] = {
        "symbol": symbol, "created_at": c_at, "track_type": t_item.get("track_type", "TRACK_A"),
        "is_long_term": bool(t_item.get("is_long_term", False)),
        "sub_strategy": str(t_item.get("sub_strategy") or prev.get("sub_strategy") or ""),
        "x_hook_ko": str(t_item.get("x_hook_ko") or prev.get("x_hook_ko") or ""),
        "x_hook_jp": str(t_item.get("x_hook_jp") or prev.get("x_hook_jp") or ""),
        "score": int(t_item.get("score", ELITE_SCORE_THRESHOLD)), "entry_price": to_float(t_item.get("entry_price")),
        "initial_stop_price": to_float(t_item.get("initial_stop_price") or t_item.get("stop_price")),
        "stop_price": to_float(t_item.get("stop_price")), "sl_pct": to_float(t_item.get("sl_pct", DEFAULT_FIXED_SL_PCT)),
        "exit_price": to_float(t_item.get("exit_price") or prev.get("exit_price"), 0.0),
        "tps": t_item.get("tps", []), "tp_pcts": t_item.get("tp_pcts", []),
        "max_high_price": t_item["max_high_price"], "max_favorable_pct": t_item["max_favorable_pct"],
        "realized_weighted_pct": round(realized_v, 2),
        "vol_mult": to_float(t_item.get("vol_mult", 1.0)), "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
        "oi_m": to_float(t_item.get("oi_m", 0.0)), "oi_ratio": to_float(t_item.get("oi_ratio", 0.0)),
        "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
        "change_24h": to_float(t_item.get("change_24h", 0.0)),
        "btc_1h_pct": to_float(t_item.get("btc_1h_pct", 0.0)),
        "hold_minutes": int(t_item.get("hold_minutes", 0)),
        "tp2_armed_ts_ms": int(to_float(t_item.get("tp2_armed_ts_ms"), 0)),
        "tp3_armed_ts_ms": int(to_float(t_item.get("tp3_armed_ts_ms"), 0)),
        "tp3_at": int(to_float(t_item.get("tp3_at"), 0)),
        "sl_at": int(to_float(t_item.get("sl_at"), 0)),
        "expired_at": int(to_float(t_item.get("expired_at"), 0)),
        "last_checked_ts": int(to_float(t_item.get("last_checked_ts"), 0)),
        **{fg: bool(t_item.get(fg, False)) for fg in ALL_LOCK_FLAGS},
    }


def _record_pm_win(symbol, t_item, now_ts, max_fav):
    if postmortem is not None and hasattr(postmortem, "record_win_postmortem"):
        try:
            postmortem.record_win_postmortem({
                "symbol": symbol,
                "track_type": t_item.get("track_type", "TRACK_A"),
                "score": int(t_item.get("score", ELITE_SCORE_THRESHOLD)),
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


def _filter_valid_price_df(df_in, entry_ref_val):
    """[가격 교차검증] 거래소 간 티커·액면 단위(1000X 등) 불일치 캔들 원천 제거"""
    if df_in is None or df_in.empty or entry_ref_val <= 0:
        return df_in
    try:
        c_s = df_in["close"].astype(float)
        h_s = df_in["high"].astype(float)
        l_s = df_in["low"].astype(float)
        ratio_c = c_s / entry_ref_val
        ratio_l = l_s / entry_ref_val
        ratio_h = h_s / entry_ref_val
        mask = (
            (ratio_c >= 0.40) & (ratio_c <= 15.0)
            & (ratio_l >= 0.30) & (ratio_h <= 18.0)
        )
        return df_in[mask].copy()
    except Exception:
        return df_in


def _fetch_post_candles_no_gap(symbol, interval, created_ms, last_checked_ts, now_ms, fast_limit_sec, max_bars):
    """보유 시간이 150봉 창구를 넘어도 fetch_klines_window 페이지네이션으로 누락 없이 추적"""
    age_sec = max(0.0, (now_ms - created_ms) / 1000.0)
    since_last_sec = max(0.0, (now_ms / 1000.0) - last_checked_ts) if last_checked_ts > 0 else age_sec
    if age_sec <= fast_limit_sec:
        return fetch_klines(symbol, interval, limit=150)

    if last_checked_ts > 0 and since_last_sec <= fast_limit_sec:
        return fetch_klines(symbol, interval, limit=150)

    start_ms = created_ms if last_checked_ts <= 0 else max(created_ms, (last_checked_ts - 7200) * 1000)
    df_win = fetch_klines_window(symbol, interval=interval, start_ms=start_ms, end_ms=now_ms, max_bars=max_bars)
    if df_win is not None and not df_win.empty:
        return df_win
    return fetch_klines(symbol, interval, limit=150)


def evaluate_single_tracked_1m(symbol, t_item, now_ts, tp_params):
    """
    [기존 전량 홀딩·고정 손절 철학 유지 + 15m/1H 가격 교차검증 + TP2/TP3 전수 확정봉 순회]
    - 단기(SL -6% · 15m/1m 기준, 방어선 15m 확정봉 전수 순회)
    - 장기(Track D: SL -20% · 1H 확정봉 단독 판정, 방어선 1H 확정봉 전수 순회)
    """
    if t_item.get("tp3_completed") or t_item.get("sl_alerted") or t_item.get("expired_closed") or t_item.get("closed_alerted"):
        return None
    entry_ref = to_float(t_item.get("entry_price"))
    if entry_ref <= 0:
        return None

    is_long = bool(t_item.get("is_long_term") or t_item.get("track_type") == "TRACK_D")
    sl_pct_use = int(round(to_float(t_item.get("sl_pct"), DEFAULT_LONGTERM_SL_PCT if is_long else DEFAULT_FIXED_SL_PCT)))
    trail_drop_pct = 10 if is_long else 4
    trail_mult = 1.0 - (trail_drop_pct / 100.0)
    horizon_tag = "[장기 포지션] " if is_long else ""

    confirm_tf_ms = 3600000 if is_long else 900000
    confirm_tf_lbl = "1H" if is_long else "15m"

    stop_p = to_float(t_item.get("stop_price"))
    init_stop_p = to_float(t_item.get("initial_stop_price"), default=entry_ref * (1.0 - sl_pct_use / 100.0))
    created_at = int(t_item.get("created_at", now_ts))
    last_checked_ts = int(to_float(t_item.get("last_checked_ts"), 0))
    created_ms = created_at * 1000
    now_ms = now_ts * 1000
    prev_max_h = to_float(t_item.get("max_high_price"), default=entry_ref)
    prev_peak_ms = int(to_float(t_item.get("peak_ts_ms"), default=created_ms))

    hist_min_low, hist_min_close = None, None
    hist_last_close, hist_last_ts_ms = 0.0, 0

    if is_long:
        df_confirm = _fetch_post_candles_no_gap(
            symbol, "60", created_ms, last_checked_ts, now_ms,
            fast_limit_sec=5 * 86400, max_bars=750,
        )
        df_confirm = _filter_valid_price_df(df_confirm, entry_ref)
        if df_confirm is not None and not df_confirm.empty and "timestamp" in df_confirm.columns:
            post_1h = df_confirm[df_confirm["timestamp"] >= created_ms]
            if not post_1h.empty:
                p_idx_h = post_1h["high"].idxmax()
                h_1h = to_float(post_1h.loc[p_idx_h, "high"], default=entry_ref)
                if 0.40 <= (h_1h / entry_ref) <= 15.0 and h_1h >= prev_max_h:
                    prev_max_h = h_1h
                    prev_peak_ms = int(to_float(post_1h.loc[p_idx_h, "timestamp"], default=created_ms))
                hist_last_close = to_float(post_1h["close"].iloc[-1], default=0.0)
                hist_last_ts_ms = int(to_float(post_1h["timestamp"].iloc[-1], default=0))
    else:
        df_15m = None
        if (now_ts - created_at) >= 180:
            df_15m = _fetch_post_candles_no_gap(
                symbol, "15", created_ms, last_checked_ts, now_ms,
                fast_limit_sec=30 * 3600, max_bars=500,
            )
            df_15m = _filter_valid_price_df(df_15m, entry_ref)
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
        df_confirm = df_15m

    closed_post_tf = None
    if df_confirm is not None and not df_confirm.empty and "timestamp" in df_confirm.columns:
        close_ts_series = df_confirm["timestamp"].astype("int64") + confirm_tf_ms
        mask_closed = (close_ts_series >= (created_ms + 60000)) & (close_ts_series <= now_ms)
        closed_post_tf = df_confirm[mask_closed].copy()
        if not closed_post_tf.empty:
            closed_post_tf["close_ts_ms"] = close_ts_series[mask_closed].values

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
    t_item["last_checked_ts"] = now_ts

    tps = t_item.get("tps") or []
    tp1_p = to_float(tps[0]) if len(tps) > 0 and to_float(tps[0]) > 0 else entry_ref * (1.0 + tp_params["tp1_min_pct"] / 100.0)
    tp2_p = to_float(tps[1]) if len(tps) > 1 and to_float(tps[1]) > 0 else entry_ref * (1.0 + tp_params["tp2_min_pct"] / 100.0)
    tp3_p = to_float(tps[2]) if len(tps) > 2 and to_float(tps[2]) > 0 else entry_ref * (1.0 + tp_params["tp3_min_pct"] / 100.0)

    g1 = round(max(6.5, (tp1_p - entry_ref) / entry_ref * 100.0), 1)
    g2 = round(max(12.0, (tp2_p - entry_ref) / entry_ref * 100.0), 1)
    g3 = round(max(19.5, (tp3_p - entry_ref) / entry_ref * 100.0), 1)
    msgs = []

    has_closed_bar = closed_post_tf is not None and not closed_post_tf.empty
    last_closed_c = to_float(closed_post_tf["close"].iloc[-1], 0.0) if has_closed_bar else 0.0

    def _make_video_payload(close_type_str, exit_price_val, total_cum_val):
        return {
            "symbol": symbol,
            "base_coin": base,
            "track_type": t_item.get("track_type", "TRACK_A"),
            "is_long_term": is_long,
            "sub_strategy": str(t_item.get("sub_strategy") or ""),
            "x_hook_ko": str(t_item.get("x_hook_ko") or ""),
            "x_hook_jp": str(t_item.get("x_hook_jp") or ""),
            "reasons": t_item.get("reasons") or [],
            "close_type": close_type_str,
            "entry_price": entry_ref,
            "initial_stop_price": init_stop_p,
            "sl_pct": sl_pct_use,
            "tps": [tp1_p, tp2_p, tp3_p],
            "max_high_price": max_high,
            "max_favorable_pct": max_fav,
            "exit_price": exit_price_val,
            "realized_weighted_pct": total_cum_val,
            "created_at": created_at,
            "closed_at": now_ts,
            "hold_minutes": hold_min,
        }

    # [3차 달성]: 50% 익절 확정 + 잔여 50% 트레일링 최종 익절 완료 시에만 영상 제작
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
                f"🚀 {horizon_tag}[3차 목표가(TP3) 달성!] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 📊 [3차 분할 익절(비중 50%)]: +{g3:.1f}% 확정 (기여 +{w3:.2f}%)\n"
                f"• 🎯 남은 50% 물량은 바닥 방어선 TP2({fmt_price(tp2_p)}, +{g2:.1f}%) + 고점 대비 -{trail_drop_pct}% 자동 트레일링({fmt_price(stop_p)}, +{tr_pct:.1f}% · {confirm_tf_lbl} 종가 이탈 시 마감)으로 추적합니다!"
            )

        video_payload = None
        tp3_armed_ms = int(to_float(t_item.get("tp3_armed_ts_ms"), default=peak_ts_ms))
        if (not just_armed_tp3) and has_closed_bar and not t_item.get("closed_alerted"):
            tp3_after = closed_post_tf[closed_post_tf["close_ts_ms"] > max(peak_ts_ms, tp3_armed_ms)]
            tp3_breach = tp3_after[
                (tp3_after["close"].astype(float) > 0)
                & (tp3_after["close"].astype(float) <= stop_p)
            ]
            if not tp3_breach.empty:
                breach_c = to_float(tp3_breach["close"].iloc[0], last_closed_c)
                t_item["tp3_completed"] = True
                t_item["trailing_closed"] = True
                t_item["closed_alerted"] = True
                t_item["tp3_at"] = now_ts
                t_item["exit_price"] = breach_c
                exit_g = round((breach_c - entry_ref) / entry_ref * 100.0, 2)
                w3 = round(g3 * 0.50, 2)
                w_rem = round(exit_g * 0.50, 2)
                total_cum = round((w3 + w_rem) - FEE_SLIPPAGE_PCT, 2)
                t_item["realized_weighted_pct"] = total_cum
                msgs.append(
                    f"🎯 {horizon_tag}[3차 달성 후 잔여 50% 트레일링 스탑({confirm_tf_lbl} 종가) 최종 익절 완료] ${base} ({symbol})\n"
                    f"• 진입가 {fmt_price(entry_ref)} → 최고가 {fmt_price(max_high)} (+{max_fav:.1f}%) → {confirm_tf_lbl} 종가 마감 {fmt_price(breach_c)} (트레일링선 {fmt_price(stop_p)})\n"
                    f"• 📊 [최종 익절 내역]: 3차(50%) +{g3:.1f}%(+{w3:.2f}%) + 잔여(50%) {exit_g:+.1f}%({w_rem:+.2f}%)\n"
                    f"• 💰 [최종 합산 확정 수익률]: {total_cum:+.2f}% 달성 (비중 100% 합산 · 비용반영)"
                )
                if total_cum > 0:
                    video_payload = _make_video_payload("TP3_TRAILING", breach_c, total_cum)
        return {"messages": msgs, "sl_record": None, "video_payload": video_payload}

    # [2차 달성]: TP1 방어선 100% 전량 수익 익절 완료 시에만 영상 제작
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
                f"✅ {horizon_tag}[2차 목표가(TP2) 달성!] \n"
                f"${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 🛡️ [방어선 상향 ]: {confirm_tf_lbl} 종가 돌파 확인 완료! 100% 전량 홀딩하며 방어선을 TP1({fmt_price(tp1_p)}, +{g1:.1f}%)로 상향 합니다. (이탈 판정: 이후 {confirm_tf_lbl} 종가 기준)"
            )

        video_payload = None
        tp2_armed_ms = int(to_float(t_item.get("tp2_armed_ts_ms"), default=peak_ts_ms))
        if (not just_armed_tp2) and has_closed_bar and not t_item.get("closed_alerted"):
            tp2_after = closed_post_tf[closed_post_tf["close_ts_ms"] > tp2_armed_ms]
            tp2_breach = tp2_after[
                (tp2_after["close"].astype(float) > 0)
                & (tp2_after["close"].astype(float) <= stop_p)
            ]
            if not tp2_breach.empty:
                breach_c = to_float(tp2_breach["close"].iloc[0], last_closed_c)
                t_item["tp3_completed"] = True
                t_item["trailing_closed"] = True
                t_item["closed_alerted"] = True
                t_item["tp3_at"] = now_ts
                t_item["exit_price"] = breach_c
                exit_g = round(((breach_c - entry_ref) / entry_ref * 100.0) - FEE_SLIPPAGE_PCT, 2)
                t_item["realized_weighted_pct"] = exit_g
                msgs.append(
                    f"🔒 {horizon_tag}[ 100% 전량 수익 익절 완료] ${base} ({symbol})\n"
                    f"• 진입 후 최고 +{max_fav:.1f}% 도달 후 방어선 TP1({fmt_price(stop_p)})을 {confirm_tf_lbl} 종가({fmt_price(breach_c)})로 이탈하여 익절 마감합니다.\n"
                    f"• 💰 [최종 합산 확정 수익률]: {exit_g:+.2f}% 달성 (100% 전량 익절 · 비용반영)"
                )
                if exit_g > 0:
                    video_payload = _make_video_payload("TP1_DEFENSE", breach_c, exit_g)
        return {"messages": msgs, "sl_record": None, "video_payload": video_payload}

    # [1차 달성]: 익절·방어선 이동 없이 100% 홀딩 유지 (영상 제작 안 함)
    if max_high >= tp1_p or t_item.get("tp1_hit"):
        t_item["tp1_hit"] = True
        if not t_item.get("tp1_alerted"):
            t_item["tp1_alerted"] = True
            msgs.append(
                f"✅ {horizon_tag}[1차 목표가(TP1) 달성! ] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 📈  100% 전량 홀딩하며 2차 목표가({fmt_price(tp2_p)}) 돌파를 기대합니다."
            )

    # [2차 확정 전]: 고정 손절선 이탈 감시
    sl_triggered = False
    trig = 0.0
    if is_long:
        if has_closed_bar:
            sl_1h_bars = closed_post_tf[closed_post_tf["close"].astype(float) <= init_stop_p]
            if not sl_1h_bars.empty:
                sl_triggered = True
                trig = to_float(sl_1h_bars["close"].iloc[0], last_closed_c)
    else:
        valid_lows = [v for v in (hist_min_low, min_1m_low) if v is not None and v > 0]
        valid_closes = [v for v in (hist_min_close, min_1m_close) if v is not None and v > 0]
        if has_live_price and valid_lows and valid_closes:
            min_low, min_close = min(valid_lows), min(valid_closes)
            if min_low <= init_stop_p and (min_close <= init_stop_p or (last_close > 0 and last_close <= init_stop_p)):
                sl_triggered = True
                trig = min(min_close, last_close) if last_close > 0 else min_close

    if sl_triggered and trig > 0:
        actual_sl_pct = round(((trig - entry_ref) / entry_ref * 100.0) - FEE_SLIPPAGE_PCT, 2)
        t_item["sl_alerted"] = True
        t_item["closed_alerted"] = True
        t_item["sl_at"] = now_ts
        t_item["exit_price"] = trig
        t_item["realized_weighted_pct"] = actual_sl_pct
        oi_r_val = to_float(t_item.get("oi_ratio"), 0.0)
        if oi_r_val <= 0 and to_float(t_item.get("turnover_m"), 0.0) > 0:
            oi_r_val = round(to_float(t_item.get("oi_m"), 0.0) / to_float(t_item.get("turnover_m"), 1.0) * 100.0, 1)
        sl_payload = {
            "symbol": symbol, "track_type": t_item.get("track_type", "TRACK_A"),
            "score": int(t_item.get("score", ELITE_SCORE_THRESHOLD)), "entry_price": entry_ref,
            "stop_price": init_stop_p, "sl_trigger_price": trig,
            "sl_pct": sl_pct_use, "actual_sl_pct": actual_sl_pct,
            "created_at": created_at, "sl_at": now_ts,
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
        sl_basis_txt = "1H 종가 이탈" if is_long else "이탈 확인가"
        msgs.append(
            f"⚠️ {horizon_tag}[손절 기준가 이탈] ${base} ({symbol})\n"
            f"• {sl_basis_txt}: {fmt_price(trig)} USDT (손절선 {fmt_price(init_stop_p)}, 실정산 {actual_sl_pct:+.2f}%)\n"
            f"• 진입 후 최고 상승폭: +{max_fav:.1f}% (보유 {hold_min}분)"
        )
        return {"messages": msgs, "sl_record": (f"{symbol}_{created_at}", sl_payload), "video_payload": None}

    return {"messages": msgs, "sl_record": None, "video_payload": None}


def _send_signal_notifications(res, now_str):
    send_telegram_reliable(build_briefing_message(res, now_str))
    x_msg = build_x_post_message(res)
    markup = build_x_intent_reply_markup(x_msg)
    base = res.get("base_coin") or res.get("base") or ""
    sym = res.get("symbol", "")
    chart_bytes = generate_signal_chart_bytes(res)
    if not (chart_bytes and send_telegram_photo_with_caption(chart_bytes, caption_text="", reply_markup=markup)):
        send_telegram_reliable(f"📲 ${base} ({sym}) X 포스팅 바로가기", reply_markup=markup)


def _send_video_compat(video_bytes, caption_text, filename, width=960, height=1280, duration=30):
    """구/신 notifier.py send_telegram_video_with_caption 시그니처 모두 안전 호환"""
    if not callable(send_telegram_video_with_caption):
        return False
    try:
        return send_telegram_video_with_caption(
            video_bytes,
            caption_text=caption_text,
            reply_markup=None,
            filename=filename,
            width=width,
            height=height,
            duration=duration,
        )
    except TypeError:
        return send_telegram_video_with_caption(
            video_bytes,
            caption_text=caption_text,
            reply_markup=None,
            filename=filename,
        )


def _send_profit_replay_video(vp, notify_error=True):
    """
    전량 수익 마감 완료 시 세로형(3:4 · 960x1280) 리플레이 영상 2종 동시 발송
    1) 클린 세로형 영상 (무음 · 요약 카드)
    2) 한글자막 + 일본어 여성 아나운서(1.1배속) 음성 세로형 영상
    """
    if not vp:
        return False
    sym = str(vp.get("symbol", "")).upper()
    base = vp.get("base_coin") or sym.replace("USDT", "")
    tot_pct = to_float(vp.get("realized_weighted_pct"))
    max_fav = to_float(vp.get("max_favorable_pct"))

    if not callable(send_telegram_video_with_caption):
        err_msg = get_last_video_error() if callable(get_last_video_error) else "notifier 영상 모듈 미로드"
        print(f"[리플레이 영상 불가] {sym}: {err_msg}")
        if notify_error:
            send_telegram_reliable(f"⚠️ [${base} 리플레이 영상 제작 실패 진단]: {err_msg}")
        return False

    try:
        # 신버전 듀얼 세로형 번들 생성 우선 사용
        if callable(generate_profit_replay_videos_bundle):
            bundle = generate_profit_replay_videos_bundle(vp)
            if not isinstance(bundle, dict) or (not bundle.get("clean") and not bundle.get("voiced")):
                err_msg = get_last_video_error() if callable(get_last_video_error) else "세로형 영상 번들 생성 실패"
                print(f"[리플레이 영상 번들 실패] {sym}: {err_msg}")
                if notify_error:
                    send_telegram_reliable(f"⚠️ [${base} 리플레이 영상 제작 실패 진단]: {err_msg}")
                return False

            w_px = int(to_float(bundle.get("width"), 960))
            h_px = int(to_float(bundle.get("height"), 1280))
            dur_clean = int(round(to_float(bundle.get("clean_duration"), 30)))
            dur_voiced = int(round(to_float(bundle.get("voiced_duration"), dur_clean)))
            sent_any = False

            if bundle.get("clean"):
                cap_clean = f"🎬 ${base} 최종 익절 리플레이 [클린 세로형] | 합산 +{tot_pct:.2f}% (최고 +{max_fav:.1f}%)"
                ok_c = _send_video_compat(
                    bundle["clean"],
                    caption_text=cap_clean,
                    filename=f"{base.lower()}_replay_clean_vertical.mp4",
                    width=w_px,
                    height=h_px,
                    duration=dur_clean,
                )
                if ok_c:
                    sent_any = True
                elif notify_error:
                    err_msg = get_last_video_error() if callable(get_last_video_error) else "클린 영상 전송 실패"
                    send_telegram_reliable(f"⚠️ [${base} 클린 세로형 영상 전송 실패]: {err_msg}")

            if bundle.get("voiced"):
                cap_voiced = f"🎙️ ${base} 최종 익절 리플레이 [한글자막·일본어음성] | 합산 +{tot_pct:.2f}% (최고 +{max_fav:.1f}%)"
                ok_v = _send_video_compat(
                    bundle["voiced"],
                    caption_text=cap_voiced,
                    filename=f"{base.lower()}_replay_sub_jp_vertical.mp4",
                    width=w_px,
                    height=h_px,
                    duration=dur_voiced,
                )
                if ok_v:
                    sent_any = True
                elif notify_error:
                    err_msg = get_last_video_error() if callable(get_last_video_error) else "자막/음성 영상 전송 실패"
                    send_telegram_reliable(f"⚠️ [${base} 자막·일본어음성 영상 전송 실패]: {err_msg}")
            elif notify_error and bundle.get("voiced_error"):
                send_telegram_reliable(f"⚠️ [${base} 자막·일본어음성 영상 생성 진단]: {bundle.get('voiced_error')}")

            return sent_any

        # 구버전 notifier.py 호환 폴백 (단일 영상)
        if not callable(generate_profit_replay_video_bytes):
            return False
        video_bytes = generate_profit_replay_video_bytes(vp)
        if not video_bytes:
            err_msg = get_last_video_error() if callable(get_last_video_error) else "영상 생성 결과 비어 있음"
            print(f"[리플레이 영상 생성 실패] {sym}: {err_msg}")
            if notify_error:
                send_telegram_reliable(f"⚠️ [${base} 리플레이 영상 제작 실패 진단]: {err_msg}")
            return False
        caption = f"🎬 ${base} 최종 익절 30초 리플레이 | 합산 +{tot_pct:.2f}% (최고 +{max_fav:.1f}%)"
        fname = f"{base.lower()}_profit_replay.mp4"
        ok = _send_video_compat(video_bytes, caption_text=caption, filename=fname, width=1280, height=720, duration=30)
        if not ok and notify_error:
            err_msg = get_last_video_error() if callable(get_last_video_error) else "텔레그램 전송 실패"
            send_telegram_reliable(f"⚠️ [${base} 리플레이 영상 전송 실패 진단]: {err_msg}")
        return ok
    except Exception as e:
        print(f"[리플레이 영상 발송 예외] {sym}: {e}")
        if notify_error:
            send_telegram_reliable(f"⚠️ [${base} 리플레이 영상 예외 진단]: {str(e)[:160]}")
        return False


def run_latest_profit_video_test():
    """[--test-video] 저장된 원장에서 가장 최근 수익 마감 종목을 찾아 세로형 리플레이 영상 2종을 즉시 테스트 발송"""
    state = load_state()
    episodes = state.get("signal_episodes") or {}
    tracked = state.get("tracked") or {}
    now_ts = int(time.time())

    pool = []
    for ep in episodes.values():
        if isinstance(ep, dict) and ep.get("symbol") and not is_stock_or_tradfi_symbol(ep.get("symbol")):
            pool.append(ep)
    for sym, tv in tracked.items():
        if isinstance(tv, dict) and not is_stock_or_tradfi_symbol(sym):
            item = dict(tv)
            item["symbol"] = sym
            pool.append(item)

    profit_closed = [
        x for x in pool
        if (x.get("tp3_completed") or x.get("trailing_closed") or x.get("closed_alerted"))
        and not x.get("sl_alerted")
        and not x.get("expired_closed")
        and to_float(x.get("entry_price")) > 0
    ]

    if profit_closed:
        profit_closed.sort(key=lambda x: int(to_float(x.get("tp3_at") or x.get("created_at"), 0)), reverse=True)
        target = profit_closed[0]
    else:
        positive = [x for x in pool if to_float(x.get("entry_price")) > 0 and to_float(x.get("max_favorable_pct")) > 0]
        if not positive:
            send_telegram_reliable("⚠️ [리플레이 영상 테스트] 저장된 익절/수익 종목 기록이 없습니다.")
            return
        positive.sort(key=lambda x: int(to_float(x.get("created_at"), 0)), reverse=True)
        target = positive[0]

    sym = str(target.get("symbol", "")).upper()
    base = sym.replace("USDT", "")
    entry_p = to_float(target.get("entry_price"))
    tps = target.get("tps") or [entry_p * 1.08, entry_p * 1.15, entry_p * 1.22]
    created_at = int(to_float(target.get("created_at"), now_ts - 7200))
    closed_at = int(to_float(target.get("tp3_at"), 0))
    if closed_at <= created_at:
        closed_at = min(now_ts, created_at + max(1800, int(to_float(target.get("hold_minutes"), 120)) * 60))

    close_type = "TP3_TRAILING" if target.get("tp3_hit") else "TP1_DEFENSE"
    saved_exit_p = to_float(target.get("exit_price"), 0.0)
    exit_p = saved_exit_p if saved_exit_p > 0 else to_float(target.get("stop_price"), to_float(tps[0], entry_p * 1.08))
    realized_pct = to_float(target.get("realized_weighted_pct"), round((exit_p - entry_p) / entry_p * 100.0, 2))
    vp = {
        "symbol": sym,
        "base_coin": base,
        "track_type": target.get("track_type", "TRACK_A"),
        "is_long_term": bool(target.get("is_long_term") or target.get("track_type") == "TRACK_D"),
        "sub_strategy": str(target.get("sub_strategy") or ""),
        "x_hook_ko": str(target.get("x_hook_ko") or ""),
        "x_hook_jp": str(target.get("x_hook_jp") or ""),
        "reasons": target.get("reasons") or [],
        "close_type": close_type,
        "entry_price": entry_p,
        "initial_stop_price": to_float(target.get("initial_stop_price"), entry_p * 0.94),
        "sl_pct": int(round(to_float(target.get("sl_pct"), 6))),
        "tps": tps,
        "max_high_price": max(to_float(target.get("max_high_price"), entry_p), exit_p),
        "max_favorable_pct": to_float(target.get("max_favorable_pct"), realized_pct),
        "exit_price": exit_p,
        "realized_weighted_pct": realized_pct,
        "created_at": created_at,
        "closed_at": closed_at,
        "hold_minutes": max(1, int((closed_at - created_at) // 60)),
    }
    print(f"[리플레이 영상 수동 테스트 시작] 대상: {sym} (진입 {entry_p} → 마감 {exit_p}, +{realized_pct:.2f}%)")
    ok = _send_profit_replay_video(vp, notify_error=True)
    print(f"[리플레이 영상 수동 테스트 결과] {'성공' if ok else '실패'}")


def _git_sync_state():
    """
    [Git .py 롤백 차단 + 삭제 심볼 부활 방지]
    - 병합 완료된 JSON에 _apply_state_cleanup() 재적용 → OS 임시 폴더 보관
    - git reset --hard origin/main으로 최신 코드 100% 보존 → bot_state.json·postmortem_db.json만 커밋·푸시
    """
    is_ci = os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    if not is_ci:
        return True

    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "GitHub Actions"], check=False)
        for attempt in range(4):
            fetch_r = subprocess.run(["git", "fetch", "origin", "main"], check=False)
            if fetch_r.returncode != 0:
                time.sleep(2.0)
                continue

            now_sync = int(time.time())
            loc_dict = load_state()
            for rf in (STATE_FILE, LEGACY_DUPLICATE_STATE_FILE):
                show_res = subprocess.run(["git", "show", f"origin/main:{rf}"], capture_output=True, text=True, check=False)
                if show_res.returncode == 0 and show_res.stdout:
                    try:
                        loc_dict = _merge_state_dicts(loc_dict, json.loads(show_res.stdout))
                    except Exception:
                        pass
            _apply_state_cleanup(loc_dict, now_sync)
            _write_json_atomic(LOCAL_RUNTIME_CACHE, loc_dict)
            _write_json_atomic(GIT_STATE_STAGE_TMP, _build_git_clean_state(loc_dict))

            pm_merged = None
            local_pm = _read_json(POSTMORTEM_DB_FILE)
            remote_pm = {}
            show_pm = subprocess.run(["git", "show", f"origin/main:{POSTMORTEM_DB_FILE}"], capture_output=True, text=True, check=False)
            if show_pm.returncode == 0 and show_pm.stdout:
                try:
                    parsed_rpm = json.loads(show_pm.stdout)
                    if isinstance(parsed_rpm, dict):
                        remote_pm = parsed_rpm
                except Exception:
                    pass
            if local_pm or remote_pm:
                if postmortem is not None and hasattr(postmortem, "merge_postmortem_dbs"):
                    try:
                        pm_merged = postmortem.merge_postmortem_dbs(local_pm, remote_pm)
                    except Exception:
                        pm_merged = local_pm or remote_pm
                else:
                    pm_merged = local_pm or remote_pm
                _write_json_atomic(GIT_PM_STAGE_TMP, pm_merged)

            reset_r = subprocess.run(["git", "reset", "--hard", "origin/main"], check=False)
            if reset_r.returncode != 0:
                time.sleep(2.0)
                continue

            with open(GIT_STATE_STAGE_TMP, "r", encoding="utf-8") as s:
                clean_state_txt = s.read()
            with open(STATE_FILE, "w", encoding="utf-8") as d:
                d.write(clean_state_txt)
            subprocess.run(["git", "add", STATE_FILE], check=False)

            if os.path.exists(LEGACY_DUPLICATE_STATE_FILE) and LEGACY_DUPLICATE_STATE_FILE != STATE_FILE:
                subprocess.run(["git", "rm", "-f", "--ignore-unmatch", LEGACY_DUPLICATE_STATE_FILE], check=False)

            if pm_merged and os.path.exists(GIT_PM_STAGE_TMP):
                with open(GIT_PM_STAGE_TMP, "r", encoding="utf-8") as ps:
                    pm_txt = ps.read()
                with open(POSTMORTEM_DB_FILE, "w", encoding="utf-8") as pd_f:
                    pd_f.write(pm_txt)
                subprocess.run(["git", "add", POSTMORTEM_DB_FILE], check=False)
                if postmortem is not None and hasattr(postmortem, "invalidate_cache"):
                    postmortem.invalidate_cache()

            if subprocess.run(["git", "diff", "--staged", "--quiet"], check=False).returncode != 0:
                subprocess.run(["git", "commit", "-m", "chore: update scanner state [skip ci]"], check=False)
                if subprocess.run(["git", "push", "origin", "HEAD:main"], check=False).returncode == 0:
                    return True
            else:
                return True
            time.sleep(2.0 + attempt)
    except Exception as e:
        print(f"[Git 동기화 예외] {e}")
    return False


def _ensure_synced_before_alert():
    if _git_sync_state():
        return True
    time.sleep(2.0)
    return _git_sync_state()


def main(scan_long_term=True):
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

    _apply_state_cleanup(state, now_ts, rwa_blocked=rwa_blocked)

    # [1단계] 기존 보유 포지션 감시 (Git 동기화 실패 시 알림 증발 방지 롤백 스냅샷 확보)
    active_syms = []
    for sym in list(tracked.keys()):
        t_item = tracked[sym]
        c_at = int(t_item.get("created_at", 0))
        sync_episode_ledger(signal_episodes, sym, t_item, sl_history)
        if c_at > int(alert_history.get(sym, 0)) and (now_ts - c_at) <= 172800:
            alert_history[sym] = c_at
        is_done = bool(t_item.get("sl_alerted") or t_item.get("tp3_completed") or t_item.get("expired_closed") or t_item.get("closed_alerted"))
        if not is_done:
            active_syms.append(sym)

    if active_syms:
        pre_eval_tracked = {s: copy.deepcopy(tracked[s]) for s in active_syms}
        pre_eval_episodes = {}
        for s in active_syms:
            ek_s = f"{s}_{int(to_float(tracked[s].get('created_at'), 0))}"
            if ek_s in signal_episodes:
                pre_eval_episodes[ek_s] = copy.deepcopy(signal_episodes[ek_s])

        pending_msgs = []
        pending_videos = []
        emitted_syms = []
        added_sl_keys = []

        with ThreadPoolExecutor(max_workers=8) as ex:
            f_map = {
                ex.submit(evaluate_single_tracked_1m, s, tracked[s], now_ts, tp_params): s
                for s in active_syms
            }
            for fut in as_completed(f_map):
                s = f_map[fut]
                try:
                    out = fut.result()
                    if out and out.get("sl_record"):
                        k, v = out["sl_record"]
                        sl_history[k] = v
                        added_sl_keys.append(k)
                    sync_episode_ledger(signal_episodes, s, tracked[s], sl_history)
                    out_msgs = (out or {}).get("messages", [])
                    if out_msgs:
                        emitted_syms.append(s)
                        pending_msgs.extend(out_msgs)
                    if out and out.get("video_payload"):
                        pending_videos.append(out["video_payload"])
                except Exception as e:
                    print(f"[1분봉 감시 예외] {s}: {e}")
        save_state(state)
        if pending_msgs:
            if _ensure_synced_before_alert():
                for m in pending_msgs:
                    send_telegram_reliable(m)
                    sent_count += 1
                for vp in pending_videos:
                    if _send_profit_replay_video(vp, notify_error=True):
                        sent_count += 1
            else:
                for s_rb in emitted_syms:
                    if s_rb in pre_eval_tracked:
                        tracked[s_rb] = pre_eval_tracked[s_rb]
                        ek_rb = f"{s_rb}_{int(to_float(tracked[s_rb].get('created_at'), 0))}"
                        if ek_rb in pre_eval_episodes:
                            signal_episodes[ek_rb] = pre_eval_episodes[ek_rb]
                        else:
                            signal_episodes.pop(ek_rb, None)
                for sl_k_rb in added_sl_keys:
                    sl_history.pop(sl_k_rb, None)
                save_state(state)
                print(f"[중복방지 보류·롤백완료] Git 동기화 실패로 추적 알림 {len(pending_msgs)}건 원복 (다음 회차 자동 재시도)")

    # [오답노트 v3] 손절 후 2시간+ 경과 종목 자동 사후 부검 (25시간 커버)
    if postmortem is not None and hasattr(postmortem, "run_post_sl_autopsy"):
        try:
            autopsied = postmortem.run_post_sl_autopsy(fetch_klines, max_checks=4)
            if autopsied:
                print(f"[사후 부검 완료] {autopsied}건 분석 반영")
        except Exception:
            pass

    if not candidates:
        save_state(state)
        print(f"[{now_str}] ⚠️ [경고] 후보군 수집 0건 (API 제한/네트워크 확인 필요 | 기존 감시 {sent_count}건 완료)")
        return

    oi_ref_map = _get_oi_ref_and_update_snapshot(oi_snapshot, candidates, now_ts)

    # [2단계] BTC 3단계 레짐 체크
    market_regime = check_btc_market_regime()
    if market_regime["gate_mode"] == "FULL_LOCK":
        save_state(state)
        print(f"[{now_str}] 🛑 [BTC 폭락 전면 차단] {market_regime['reason']} (기존 감시 {sent_count}건 완료)")
        return
    if market_regime["gate_mode"] == "COUNTER_TREND_ONLY":
        print(f"[{now_str}] ⚠️ {market_regime['reason']}")

    active_short_cnt = 0
    active_long_cnt = 0
    for s_k, t_v in tracked.items():
        if not isinstance(t_v, dict) or is_stock_or_tradfi_symbol(s_k):
            continue
        if t_v.get("sl_alerted") or t_v.get("tp3_completed") or t_v.get("expired_closed") or t_v.get("closed_alerted"):
            continue
        if t_v.get("is_long_term") or t_v.get("track_type") == "TRACK_D":
            active_long_cnt += 1
        else:
            active_short_cnt += 1

    min_score_short = ELITE_SCORE_THRESHOLD
    min_score_long = ELITE_SCORE_THRESHOLD

    def _recent_sl_count(sym_check):
        return sum(
            1 for v in sl_history.values()
            if isinstance(v, dict) and str(v.get("symbol", "")).upper() == sym_check and (now_ts - int(to_float(v.get("sl_at"), 0))) < RE_ALERT_COOLDOWN_SEC
        )

    def _latest_sl_at(sym_check):
        latest_ts = 0
        pt = tracked.get(sym_check)
        if isinstance(pt, dict) and pt.get("sl_alerted"):
            ts_pt = int(to_float(pt.get("sl_at") or pt.get("created_at"), 0))
            if ts_pt > latest_ts:
                latest_ts = ts_pt
        for v in sl_history.values():
            if isinstance(v, dict) and str(v.get("symbol", "")).upper() == sym_check:
                ts_v = int(to_float(v.get("sl_at"), 0))
                if ts_v > latest_ts:
                    latest_ts = ts_v
        return latest_ts

    def _closed_profit_recently(sym_check):
        pt = tracked.get(sym_check)
        if pt and (pt.get("tp3_completed") or pt.get("trailing_closed") or pt.get("closed_alerted")) and not pt.get("sl_alerted") and not pt.get("expired_closed"):
            close_ts = int(to_float(pt.get("tp3_at"), 0))
            if close_ts > 0 and (now_ts - close_ts) < TP_CLOSE_REENTRY_COOLDOWN_SEC:
                return True
        for ep in signal_episodes.values():
            if isinstance(ep, dict) and str(ep.get("symbol", "")).upper() == sym_check:
                if (ep.get("tp3_completed") or ep.get("trailing_closed") or ep.get("closed_alerted")) and not ep.get("sl_alerted") and not ep.get("expired_closed"):
                    close_ts = int(to_float(ep.get("tp3_at"), 0))
                    if close_ts > 0 and (now_ts - close_ts) < TP_CLOSE_REENTRY_COOLDOWN_SEC:
                        return True
        return False

    to_analyze = []
    for c in candidates:
        sym = c["symbol"]
        if (valid_syms and sym not in valid_syms) or is_stock_or_tradfi_symbol(sym):
            continue
        prev_t = tracked.get(sym)
        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed") and not prev_t.get("expired_closed") and not prev_t.get("closed_alerted"):
            continue
        if _closed_profit_recently(sym):
            continue

        latest_sl_ts = _latest_sl_at(sym)
        was_sl_24h = latest_sl_ts > 0 and (now_ts - latest_sl_ts) < RE_ALERT_COOLDOWN_SEC
        sl_cnt = _recent_sl_count(sym)
        if was_sl_24h:
            if sl_cnt >= 2 or (now_ts - latest_sl_ts) < SL_RECOVERY_COOLDOWN_SEC:
                continue
        else:
            if (now_ts - int(to_float(alert_history.get(sym), 0))) < RE_ALERT_COOLDOWN_SEC:
                continue
        to_analyze.append(c)

    results = []
    analyze_err_cnt = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        f_map = {
            ex.submit(
                analyze_symbol,
                c,
                oi_ref_map.get(c["symbol"]),
                market_regime,
                scan_long_term,
                min_score_short,
                min_score_long,
            ): c["symbol"]
            for c in to_analyze
        }
        for fut in as_completed(f_map):
            sym_a = f_map[fut]
            try:
                r = fut.result()
                if r:
                    results.append(r)
            except Exception as e:
                analyze_err_cnt += 1
                if analyze_err_cnt <= 3:
                    print(f"[분석 예외] {sym_a}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    new_sent = 0
    for res in results:
        if new_sent >= MAX_NEW_ALERTS_PER_SCAN:
            break
        sym = res["symbol"]
        if is_stock_or_tradfi_symbol(sym):
            continue
        prev_t = tracked.get(sym)
        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed") and not prev_t.get("expired_closed") and not prev_t.get("closed_alerted"):
            continue
        if _closed_profit_recently(sym):
            continue

        is_res_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
        if int(res.get("score", 0)) < ELITE_SCORE_THRESHOLD:
            continue

        latest_sl_ts = _latest_sl_at(sym)
        elapsed_since_sl = (now_ts - latest_sl_ts) if latest_sl_ts > 0 else (RE_ALERT_COOLDOWN_SEC + 1)
        was_sl_24h = latest_sl_ts > 0 and elapsed_since_sl < RE_ALERT_COOLDOWN_SEC
        elapsed_alert = now_ts - int(to_float(alert_history.get(sym), 0))
        sl_cnt = _recent_sl_count(sym)

        if was_sl_24h:
            if (
                sl_cnt >= 2
                or elapsed_since_sl < SL_RECOVERY_COOLDOWN_SEC
                or res["score"] < REENTRY_CUT_SCORE
                or to_float(res.get("rsi6_15m"), 50.0) >= 68.0
            ):
                continue
            orig_title = str(res.get("track_title", "🚀 [타점 포착]"))
            if "재진입" not in orig_title:
                res["track_title"] = f"{orig_title} (🔄재진입)"
            if "스탑헌팅 바닥 확인 후 비과열 V자 재탈환" not in res["reasons"]:
                res["reasons"].insert(0, "스탑헌팅 바닥 확인 후 비과열 V자 재탈환")
            res["reason_str"] = " / ".join(res["reasons"])
        elif elapsed_alert < RE_ALERT_COOLDOWN_SEC:
            continue

        prev_tracked_snap = copy.deepcopy(tracked.get(sym))
        prev_alert_ts = alert_history.get(sym)
        new_ep_key = f"{sym}_{now_ts}"
        prev_ep_snap = copy.deepcopy(signal_episodes.get(new_ep_key))

        # X 양식 간략 포착근거(한글·일본어)를 원장에 영구 보존 (익절 자막·음성 영상 100% 일치)
        hook_ko, hook_jp = build_x_hook_pair(res)

        tracked[sym] = {
            "created_at": now_ts, "last_checked_ts": now_ts, "track_type": res["track_type"],
            "is_long_term": is_res_long,
            "sub_strategy": str(res.get("sub_strategy") or ""),
            "x_hook_ko": hook_ko,
            "x_hook_jp": hook_jp,
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
            if is_res_long:
                active_long_cnt += 1
            else:
                active_short_cnt += 1
        else:
            if prev_tracked_snap is None:
                tracked.pop(sym, None)
            else:
                tracked[sym] = prev_tracked_snap
            if prev_alert_ts is None:
                alert_history.pop(sym, None)
            else:
                alert_history[sym] = prev_alert_ts
            if prev_ep_snap is None:
                signal_episodes.pop(new_ep_key, None)
            else:
                signal_episodes[new_ep_key] = prev_ep_snap
            save_state(state)
            print(f"[중복방지 보류·롤백완료] Git 동기화 실패로 {sym} 신규 알림 원복 (다음 회차 재평가)")

    save_state(state)
    lt_lbl = "장기ON" if scan_long_term else "단기전용"
    top_info = f"최고 ${results[0].get('base_coin', results[0]['symbol'])} {results[0]['score']}점({results[0]['track_type']})" if results else "통과없음"
    err_info = f" · 예외{analyze_err_cnt}" if analyze_err_cnt > 0 else ""
    print(
        f"[{now_str}] 스캔 완료 ({time.time() - start_t:.1f}초 | {lt_lbl} | 게이트:{market_regime['gate_mode']} | "
        f"퍼널: 후보{len(candidates)}→분석{len(to_analyze)}→통과{len(results)}({top_info}{err_info}) | "
        f"보유 단기{active_short_cnt}·장기{active_long_cnt} · 컷{ELITE_SCORE_THRESHOLD}/{REENTRY_CUT_SCORE} | "
        f"BTC 15m {market_regime['btc_15m_pct']:+.2f}% / 1H {market_regime['btc_1h_pct']:+.2f}% / 4H {market_regime['btc_4h_pct']:+.2f}%) - 전송 {sent_count}건"
    )


def run_loop_mode(max_iterations=70, sleep_sec=285, max_budget_sec=16200):
    """4시간 30분(270분) 무공백 릴레이 루프"""
    loop_start = time.time()
    budget_min = max_budget_sec / 60.0

    for i in range(1, max_iterations + 1):
        elapsed = time.time() - loop_start
        if i > 1 and elapsed >= max_budget_sec:
            print(f"[릴레이 예산 도달] {elapsed / 60.0:.1f}분 / {budget_min:.0f}분 경과 — 대기 중인 다음 슬롯으로 바통 터치합니다.")
            break

        scan_lt = (i % 3 == 1)
        print(
            f"=== [회차 {i}/{max_iterations} | 경과 {elapsed / 60.0:.1f}분 / {budget_min:.0f}분] "
            f"5분 주기 스캔 시작 ({'단기+장기' if scan_lt else '단기전용'}) ==="
        )
        _git_sync_state()
        try:
            main(scan_long_term=scan_lt)
        except Exception as e:
            print(f"[스캔 루프 예외] {e}")
        _git_sync_state()

        if i < max_iterations:
            rem_budget = max_budget_sec - (time.time() - loop_start)
            max_wait_allowed = rem_budget - 45.0
            if max_wait_allowed < 60.0:
                print(f"[릴레이 정상 마무리] 잔여 시간 {rem_budget:.0f}초 — 대기 중인 다음 슬롯으로 바통 터치합니다.")
                break
            time.sleep(min(sleep_sec, max_wait_allowed))


if __name__ == "__main__":
    if "--test-video" in sys.argv:
        run_latest_profit_video_test()
    elif "--loop" in sys.argv:
        run_loop_mode()
    else:
        main(scan_long_term=True)
# === END OF FILE ===
