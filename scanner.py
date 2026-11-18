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
    clean_url,
)
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

SCHEME = "ht" + "tps://"
RE_ALERT_COOLDOWN_SEC = 43200
SL_RECOVERY_COOLDOWN_SEC = getattr(cfg, "SL_RECOVERY_COOLDOWN_SEC", 14400)
HISTORY_RETENTION_SEC = 259200
EPISODE_RETENTION_SEC = 864000
MAX_NEW_ALERTS_PER_SCAN = getattr(cfg, "MAX_NEW_ALERTS_PER_SCAN", 3)
DEFAULT_FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
LOCAL_RUNTIME_CACHE = "/tmp/scanner_runtime_state.json"

LOCK_FLAGS = (
    "be_armed", "tp1_hit", "tp2_hit",
    "tp1_alerted", "tp2_alerted", "tp3_alerted",
    "tp3_completed", "sl_alerted", "closed_alerted",
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


def load_state():
    merged = {"tracked": {}, "alert_history": {}, "sl_history": {}, "signal_episodes": {}}
    for src in (_read_json(STATE_FILE), _read_json(f"{STATE_FILE}.tmp"), _read_json(LOCAL_RUNTIME_CACHE)):
        for s, ts_v in (src.get("alert_history") or {}).items():
            merged["alert_history"][s] = max(int(merged["alert_history"].get(s, 0)), int(to_float(ts_v, 0)))
        for k, v in (src.get("sl_history") or {}).items():
            if isinstance(v, dict):
                merged["sl_history"][k] = v
        for ek, ev in (src.get("signal_episodes") or {}).items():
            if not isinstance(ev, dict):
                continue
            p = merged["signal_episodes"].get(ek, {})
            if not p:
                merged["signal_episodes"][ek] = dict(ev)
            else:
                for fg in LOCK_FLAGS:
                    if ev.get(fg) or p.get(fg):
                        p[fg] = True
                p["max_high_price"] = max(to_float(p.get("max_high_price")), to_float(ev.get("max_high_price")))
                p["max_favorable_pct"] = max(to_float(p.get("max_favorable_pct")), to_float(ev.get("max_favorable_pct")))
        for s, tv in (src.get("tracked") or {}).items():
            if not isinstance(tv, dict):
                continue
            pt = merged["tracked"].get(s)
            if not pt or int(tv.get("created_at", 0)) > int(pt.get("created_at", 0)):
                merged["tracked"][s] = dict(tv)
            elif int(tv.get("created_at", 0)) == int(pt.get("created_at", 0)):
                for fg in LOCK_FLAGS:
                    if tv.get(fg) or pt.get(fg):
                        pt[fg] = True
                pt["max_high_price"] = max(to_float(pt.get("max_high_price")), to_float(tv.get("max_high_price")))
                pt["max_favorable_pct"] = max(to_float(pt.get("max_favorable_pct")), to_float(tv.get("max_favorable_pct")))
                pt["stop_price"] = max(to_float(pt.get("stop_price")), to_float(tv.get("stop_price")))
    return merged


def save_state(state):
    for path in (STATE_FILE, LOCAL_RUNTIME_CACHE):
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
    if c_at <= 0:
        return
    ep_key = f"{symbol}_{c_at}"
    prev = signal_episodes.get(ep_key, {})
    max_h = max(to_float(prev.get("max_high_price")), to_float(t_item.get("max_high_price")), to_float(t_item.get("entry_price")))
    max_fav = max(to_float(prev.get("max_favorable_pct")), to_float(t_item.get("max_favorable_pct")))
    t_item["max_high_price"] = max_h
    t_item["max_favorable_pct"] = max_fav

    for fg in LOCK_FLAGS:
        if prev.get(fg) or t_item.get(fg):
            t_item[fg] = True
    if isinstance(sl_history, dict) and ep_key in sl_history:
        t_item["sl_alerted"] = True
        t_item["closed_alerted"] = True

    signal_episodes[ep_key] = {
        "symbol": symbol,
        "created_at": c_at,
        "track_type": t_item.get("track_type", "TRACK_A"),
        "score": int(t_item.get("score", 85)),
        "entry_price": to_float(t_item.get("entry_price")),
        "initial_stop_price": to_float(t_item.get("initial_stop_price") or t_item.get("stop_price")),
        "stop_price": to_float(t_item.get("stop_price")),
        "sl_pct": to_float(t_item.get("sl_pct", DEFAULT_FIXED_SL_PCT)),
        "tps": t_item.get("tps", []),
        "max_high_price": max_h,
        "max_favorable_pct": max_fav,
        "vol_mult": to_float(t_item.get("vol_mult", 1.0)),
        "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
        "oi_m": to_float(t_item.get("oi_m", 0.0)),
        "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
        "hold_minutes": int(t_item.get("hold_minutes", 0)),
        **{fg: bool(t_item.get(fg, False)) for fg in LOCK_FLAGS},
        "sl_at": int(t_item.get("sl_at", 0)),
    }


def evaluate_single_tracked_1m(symbol, t_item, now_ts, tp_params):
    if t_item.get("tp3_completed") or t_item.get("sl_alerted") or t_item.get("closed_alerted"):
        return None
    entry_ref = to_float(t_item.get("entry_price"))
    if entry_ref <= 0:
        return None

    stop_p = to_float(t_item.get("stop_price"))
    init_stop_p = to_float(t_item.get("initial_stop_price"), default=stop_p)
    created_at = int(t_item.get("created_at", now_ts))
    created_ms = created_at * 1000
    prev_max_h = to_float(t_item.get("max_high_price"), default=entry_ref)
    prev_peak_ms = int(to_float(t_item.get("peak_ts_ms"), default=created_ms))

    # [핵심 복구] 포착 후 10분 이상 경과한 모든 종목은 15분봉(최대 48시간) 전 구간을 매번 소급 스캔하여
    # 스캐너 중단/지연 중에 터치한 익절가(TP1·TP2·TP3)와 손절가를 100% 잡아냅니다.
    hist_min_low = entry_ref
    hist_min_close = entry_ref
    if (now_ts - created_at) >= 600:
        df_15m = fetch_klines(symbol, "15", limit=192)
        if not df_15m.empty and "timestamp" in df_15m.columns:
            post_15m = df_15m[df_15m["timestamp"] >= created_ms]
            if not post_15m.empty:
                p_idx = post_15m["high"].idxmax()
                h_15m = to_float(post_15m.loc[p_idx, "high"], default=entry_ref)
                if 0.40 <= (h_15m / entry_ref) <= 2.60 and h_15m >= prev_max_h:
                    prev_max_h = h_15m
                    prev_peak_ms = int(to_float(post_15m.loc[p_idx, "timestamp"], default=created_ms))
                hist_min_low = min(hist_min_low, to_float(post_15m["low"].min(), default=entry_ref))
                hist_min_close = min(hist_min_close, to_float(post_15m["close"].min(), default=entry_ref))

    df_1m = fetch_verified_1m_klines(symbol, entry_ref=entry_ref, limit=60)
    if df_1m.empty:
        return None
    post_1m = df_1m[df_1m["timestamp"] >= created_ms].copy()
    if post_1m.empty:
        return None

    base = symbol.replace("USDT", "")
    w_idx = post_1m["high"].idxmax()
    w_high = to_float(post_1m.loc[w_idx, "high"], default=entry_ref)
    w_ts = int(to_float(post_1m.loc[w_idx, "timestamp"], default=created_ms))
    max_high = max(w_high, prev_max_h, entry_ref)
    peak_ts_ms = w_ts if w_high >= prev_max_h else prev_peak_ms
    max_fav = round((max_high - entry_ref) / entry_ref * 100.0, 2)

    t_item["max_high_price"] = max_high
    t_item["peak_ts_ms"] = peak_ts_ms
    t_item["max_favorable_pct"] = max_fav
    last_close = to_float(post_1m["close"].iloc[-1], default=entry_ref)
    last_ts_ms = int(to_float(post_1m["timestamp"].iloc[-1], default=now_ts * 1000))
    hold_min = max(1, int((now_ts - created_at) // 60))
    t_item["hold_minutes"] = hold_min

    tps = t_item.get("tps") or []
    tp1_p = to_float(tps[0]) if len(tps) > 0 and to_float(tps[0]) > 0 else entry_ref * (1.0 + tp_params["tp1_min_pct"] / 100.0)
    tp2_p = to_float(tps[1]) if len(tps) > 1 and to_float(tps[1]) > 0 else entry_ref * (1.0 + tp_params["tp2_min_pct"] / 100.0)
    tp3_p = to_float(tps[2]) if len(tps) > 2 and to_float(tps[2]) > 0 else entry_ref * (1.0 + tp_params["tp3_min_pct"] / 100.0)

    g1 = max(7.0, (tp1_p - entry_ref) / entry_ref * 100.0)
    g2 = max(13.0, (tp2_p - entry_ref) / entry_ref * 100.0)
    g3 = max(20.0, (tp3_p - entry_ref) / entry_ref * 100.0)
    msgs = []

    if max_high >= tp3_p:
        already = bool(t_item.get("tp3_alerted") or t_item.get("closed_alerted"))
        for fg in ("be_armed", "tp1_hit", "tp2_hit", "tp1_alerted", "tp2_alerted", "tp3_alerted", "tp3_completed", "closed_alerted"):
            t_item[fg] = True
        t_item["tp3_at"] = now_ts
        if not already:
            cum = int(round(g1)) + int(round(g2)) + int(round(max(g3, max_fav)))
            msgs.append(
                f"🎯 [3차 최종 목표가(TP3) 전량 익절 완료] ${base} ({symbol})\n"
                f"• 진입 기준가: {fmt_price(entry_ref)} → 도달 최고가: {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 💰 [최종 확정 누적 수익률]: +{cum}%달성 (1~3차)"
            )
        return {"messages": msgs, "sl_record": None}

    if max_high >= tp2_p:
        already = bool(t_item.get("tp2_alerted") or t_item.get("tp2_hit"))
        for fg in ("be_armed", "tp1_hit", "tp2_hit", "tp1_alerted", "tp2_alerted"):
            t_item[fg] = True
        stop_p = max(stop_p, entry_ref * (1.0 + (g2 * 0.85) / 100.0), max_high * 0.965)
        t_item["stop_price"] = stop_p
        if not already:
            l_pct = round((stop_p - entry_ref) / entry_ref * 100.0, 1)
            msgs.append(
                f"✅ [2차 목표가(TP2) 돌파 & 3차 고점 트레일링 가동] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 2차 분할 익절(30%) 확정! 남은 물량은 트레일링 보호선({fmt_price(stop_p)}, +{l_pct:.1f}%)으로 3차 고점을 추적합니다."
            )
    elif max_high >= tp1_p:
        already = bool(t_item.get("tp1_alerted") or t_item.get("tp1_hit"))
        for fg in ("be_armed", "tp1_hit", "tp1_alerted"):
            t_item[fg] = True
        stop_p = max(stop_p, entry_ref * 1.006)
        t_item["stop_price"] = stop_p
        if not already:
            msgs.append(
                f"✅ [1차 목표가(TP1) 달성] ${base} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav:.1f}%)\n"
                f"• 1차 분할 익절(40%) 확정! 남은 물량은 보호선({fmt_price(stop_p)})으로 홀딩합니다."
            )

    if max_fav >= float(DEFAULT_FIXED_SL_PCT) and not t_item.get("be_armed") and not t_item.get("tp1_hit"):
        t_item["be_armed"] = True
        stop_p = max(stop_p, entry_ref * 1.004)
        t_item["stop_price"] = stop_p
        msgs.append(
            f"🛡️ [본전 방어(Breakeven·1R) 자동 발동] ${base} ({symbol})\n"
            f"• 최고 +{max_fav:.1f}% 도달 확인 → 손절선을 진입가 위({fmt_price(stop_p)}, +0.4%)로 상향 고정합니다."
        )

    if t_item.get("tp1_hit") or t_item.get("tp2_hit") or t_item.get("be_armed"):
        if last_ts_ms > peak_ts_ms and last_close <= stop_p and not t_item.get("closed_alerted"):
            t_item["tp3_completed"] = True
            t_item["closed_alerted"] = True
            t_item["tp3_at"] = now_ts
            lock_g = max(0.4, (stop_p - entry_ref) / entry_ref * 100.0)
            cum = (int(round(g1)) + int(round(g2)) + int(round(lock_g))) if t_item.get("tp2_hit") else (
                (int(round(g1)) + int(round(lock_g))) if t_item.get("tp1_hit") else max(1, int(round(lock_g)))
            )
            msgs.append(
                f"🔒 [트레일링/방어선 수익 마감 완료] ${base} ({symbol})\n"
                f"• 진입 후 최고 +{max_fav:.1f}% 도달 후 방어선 이탈로 잔여 물량을 최종 익절 마감합니다.\n"
                f"• 💰 [최종 확정 누적 수익률]: +{cum}%달성 (1~3차)"
            )
        return {"messages": msgs, "sl_record": None}

    min_low = min(hist_min_low, to_float(post_1m["low"].min(), default=entry_ref))
    min_close = min(hist_min_close, to_float(post_1m["close"].min(), default=entry_ref))
    if min_low <= init_stop_p and (min_close <= init_stop_p or last_close <= init_stop_p):
        trig = min(min_close, last_close)
        t_item["sl_alerted"] = True
        t_item["closed_alerted"] = True
        t_item["sl_at"] = now_ts
        sl_rec = (f"{symbol}_{created_at}", {
            "symbol": symbol, "track_type": t_item.get("track_type", "TRACK_A"),
            "score": int(t_item.get("score", 85)), "entry_price": entry_ref,
            "stop_price": init_stop_p, "sl_trigger_price": trig,
            "sl_pct": DEFAULT_FIXED_SL_PCT, "created_at": created_at, "sl_at": now_ts,
            "hold_minutes": hold_min, "max_high_price": max_high, "max_favorable_pct": max_fav,
            "vol_mult": to_float(t_item.get("vol_mult", 1.0)), "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
            "oi_m": to_float(t_item.get("oi_m", 0.0)), "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
            "reasons": t_item.get("reasons", []),
        })
        msgs.append(
            f"⚠️ [손절 기준가 이탈] ${base} ({symbol})\n"
            f"• 이탈 확인가: {fmt_price(trig)} USDT (고정 손절선 {fmt_price(init_stop_p)}, -{DEFAULT_FIXED_SL_PCT}% 이탈)\n"
            f"• 진입 후 최고 상승폭: +{max_fav:.1f}% (보유 {hold_min}분)"
        )
        return {"messages": msgs, "sl_record": sl_rec}

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

    for ep in signal_episodes.values():
        if isinstance(ep, dict) and ep.get("symbol"):
            alert_history[ep["symbol"]] = max(int(alert_history.get(ep["symbol"], 0)), int(ep.get("created_at", 0)))
    for s, t in tracked.items():
        if isinstance(t, dict):
            alert_history[s] = max(int(alert_history.get(s, 0)), int(t.get("created_at", 0)))

    active_syms = []
    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        t_item = tracked[sym]
        if (valid_syms and sym not in valid_syms) or is_tradfi_ticker(base_c) or base_c in rwa_blocked:
            del tracked[sym]
            continue
        sync_episode_ledger(signal_episodes, sym, t_item, sl_history)
        if not t_item.get("sl_alerted") and not t_item.get("tp3_completed"):
            active_syms.append(sym)

    if active_syms:
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
                        send_telegram_reliable(m)
                        sent_count += 1
                except Exception as e:
                    print(f"[1분봉 감시 예외] {s}: {e}")
        save_state(state)

    if not candidates:
        print("[알림] 후보군 데이터 없음.")
        save_state(state)
        return

    to_analyze = []
    for c in candidates:
        sym = c["symbol"]
        if valid_syms and sym not in valid_syms:
            continue
        prev_t = tracked.get(sym)
        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed"):
            continue
        cd = SL_RECOVERY_COOLDOWN_SEC if (prev_t and prev_t.get("sl_alerted")) else RE_ALERT_COOLDOWN_SEC
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
        if (now_ts - int(alert_history.get(sym, 0))) < RE_ALERT_COOLDOWN_SEC:
            continue
        tracked[sym] = {
            "created_at": now_ts, "track_type": res["track_type"], "score": res["score"],
            "entry_price": res["price"], "swing_high_line": res["swing_high_line"],
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
        _send_signal_notifications(res, now_str)
        new_sent += 1
        sent_count += 1

    save_state(state)
    print(f"[{now_str}] 스캔 완료 ({time.time() - start_t:.1f}초) - 전송 {sent_count}건")


def _git_sync_state():
    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "GitHub Actions"], check=False)
        if not os.path.exists(STATE_FILE):
            return
        for _ in range(3):
            subprocess.run(["git", "fetch", "origin", "main"], check=False)
            subprocess.run(["git", "reset", "--soft", "origin/main"], check=False)
            if os.path.exists(LOCAL_RUNTIME_CACHE):
                with open(LOCAL_RUNTIME_CACHE, "r", encoding="utf-8") as s, open(STATE_FILE, "w", encoding="utf-8") as d:
                    d.write(s.read())
            subprocess.run(["git", "add", STATE_FILE], check=False)
            if subprocess.run(["git", "diff", "--staged", "--quiet"], check=False).returncode != 0:
                subprocess.run(["git", "commit", "-m", "chore: update scanner state [skip ci]"], check=False)
                if subprocess.run(["git", "push", "origin", "HEAD:main"], check=False).returncode == 0:
                    break
            else:
                break
            time.sleep(2)
    except Exception:
        pass


def run_loop_mode(iterations=9, sleep_sec=295):
    for i in range(1, iterations + 1):
        print(f"=== [회차 {i}/{iterations}] 5분 주기 스캔 시작 ===")
        subprocess.run(["git", "fetch", "origin", "main"], check=False)
        subprocess.run(["git", "checkout", "origin/main", "--", STATE_FILE], check=False)
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
