import os, json, time, subprocess, requests
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    import config as cfg
    STATE_FILE = getattr(cfg, "STATE_FILE", "bot_state.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception:
    cfg = None
    STATE_FILE = "bot_state.json"
    KST = timezone(timedelta(hours=9))
    FIXED_SL_PCT = 6

from api import get_scanner_candidates, get_valid_bybit_linear_symbols, fetch_klines, is_stock_or_tradfi_symbol, clean_url

SCHEME = "ht" + "tps://"
STATE_CANDIDATE_FILES = list(dict.fromkeys([STATE_FILE, "bot_state.json", "scanner_state.json"]))


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


def calc_weighted_realized_pct(e, c):
    """
    신규 익절 구조 반영:
    - 1차 돌파: 30% 익절 + 잔여 70% (1차지점 -5% 방어선)
    - 2차 돌파: 잔여 70% 홀딩 + (2차지점 -5% 방어선)
    - 3차 돌파: 30% 익절 + 잔여 40% (고점 대비 -3% 자동 트레일링 스탑)
    """
    saved_w = to_float(e.get("realized_weighted_pct"), 0.0)
    if saved_w > 0:
        return round(saved_w, 2)

    entry_p = to_float(e.get("entry_price"))
    tps = e.get("tps") or []
    tp1_p = to_float(tps[0]) if (len(tps) > 0 and to_float(tps[0]) > entry_p) else entry_p * 1.08
    tp2_p = to_float(tps[1]) if (len(tps) > 1 and to_float(tps[1]) > entry_p) else entry_p * 1.15
    tp3_p = to_float(tps[2]) if (len(tps) > 2 and to_float(tps[2]) > entry_p) else entry_p * 1.22

    g1 = round((tp1_p - entry_ref) / entry_ref * 100.0, 1) if (entry_ref := max(entry_p, 1e-8)) > 0 else 8.0
    g3 = round((tp3_p - entry_ref) / entry_ref * 100.0, 1) if entry_ref > 0 else 22.0
    def1_g = max(1.0, round(((tp1_p * 0.95) - entry_ref) / entry_ref * 100.0, 1))
    def2_g = max(4.0, round(((tp2_p * 0.95) - entry_ref) / entry_ref * 100.0, 1))

    if c["tp3"]:
        trail_exit_g = max(def2_g, round(c["fav"] - 3.0, 1))
        return round(g1 * 0.30 + g3 * 0.30 + trail_exit_g * 0.40, 2)
    if c["tp2"]:
        rem_g = def2_g if c["trailing_closed"] else c["fav"]
        return round(g1 * 0.30 + rem_g * 0.70, 2)
    if c["tp1"]:
        rem_g = def1_g if c["trailing_closed"] else c["fav"]
        return round(g1 * 0.30 + rem_g * 0.70, 2)
    return 0.0


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


def save_merged_state(state):
    for path in STATE_CANDIDATE_FILES:
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            pass


def git_push_saved_files(extra_files=None):
    target_files = list(STATE_CANDIDATE_FILES)
    if extra_files:
        for ef in extra_files:
            if ef not in target_files:
                target_files.append(ef)
    try:
        subprocess.run(["git", "config", "--local", "user.email", "actions@github.com"], check=False)
        subprocess.run(["git", "config", "--local", "user.name", "GitHub Actions"], check=False)
        existing = [f for f in target_files if os.path.exists(f)]
        if not existing:
            return
        raw_backups = {f: open(f, "r", encoding="utf-8").read() for f in existing}
        for _ in range(3):
            subprocess.run(["git", "fetch", "origin", "main"], check=False)
            subprocess.run(["git", "reset", "--soft", "origin/main"], check=False)
            for f, content in raw_backups.items():
                with open(f, "w", encoding="utf-8") as fp:
                    fp.write(content)
                subprocess.run(["git", "add", f], check=False)
            if subprocess.run(["git", "diff", "--staged", "--quiet"], check=False).returncode != 0:
                subprocess.run(["git", "commit", "-m", "chore: sync backfilled state & history [skip ci]"], check=False)
                if subprocess.run(["git", "push", "origin", "HEAD:main"], check=False).returncode == 0:
                    break
            else:
                break
            time.sleep(2)
    except Exception:
        pass


def ensure_missing_icnt_and_backfill(episodes, tracked, now_ts):
    for ep_k, e in list(episodes.items()):
        if not isinstance(e, dict):
            continue
        sym = str(e.get("symbol", "")).upper()
        if is_stock_or_tradfi_symbol(sym):
            del episodes[ep_k]
            continue
        entry_p = to_float(e.get("entry_price"))
        c_at = int(e.get("created_at", 0))
        fav = to_float(e.get("max_favorable_pct"))
        if sym and entry_p > 0 and c_at > 0 and (now_ts - c_at) <= 259200 and fav < 19.5:
            df_15m = fetch_klines(sym, "15", limit=120)
            if not df_15m.empty and "timestamp" in df_15m.columns:
                post_df = df_15m[df_15m["timestamp"] >= (c_at * 1000 - 900000)]
                if not post_df.empty:
                    h_max = to_float(post_df["high"].max(), entry_p)
                    if 0.40 <= (h_max / entry_p) <= 2.80 and h_max > to_float(e.get("max_high_price"), entry_p):
                        e["max_high_price"] = h_max
                        e["max_favorable_pct"] = round((h_max - entry_p) / entry_p * 100.0, 2)
                        episodes[ep_k] = e


def classify_episode_by_mfe(e):
    fav = to_float(e.get("max_favorable_pct"))
    entry_p = to_float(e.get("entry_price"))
    max_h = to_float(e.get("max_high_price"))
    if entry_p > 0 and max_h > entry_p:
        fav = max(fav, round((max_h - entry_p) / entry_p * 100.0, 2))
    tps = e.get("tps") or []
    tp1_req = ((to_float(tps[0]) - entry_p) / entry_p * 100.0 - 0.3) if (len(tps) > 0 and entry_p > 0 and to_float(tps[0]) > entry_p) else 6.8
    tp2_req = ((to_float(tps[1]) - entry_p) / entry_p * 100.0 - 0.3) if (len(tps) > 1 and entry_p > 0 and to_float(tps[1]) > entry_p) else 13.0
    tp3_req = ((to_float(tps[2]) - entry_p) / entry_p * 100.0 - 0.3) if (len(tps) > 2 and entry_p > 0 and to_float(tps[2]) > entry_p) else 19.5

    tp1 = fav >= max(6.5, tp1_req)
    tp2 = fav >= max(12.5, tp2_req)
    tp3 = fav >= max(19.0, tp3_req)
    is_sl = bool(e.get("sl_alerted")) and not tp1
    be_save = False
    trailing_closed = bool(e.get("tp3_completed") or e.get("closed_alerted")) and tp1 and not is_sl
    return {"fav": fav, "tp1": tp1, "tp2": tp2, "tp3": tp3, "be_save": be_save, "sl": is_sl, "trailing_closed": trailing_closed}


def send_telegram_weekly(text):
    token = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or getattr(cfg, "TELEGRAM_BOT_TOKEN", "") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or getattr(cfg, "TELEGRAM_CHAT_ID", "") or "").strip()
    if not text or not token or not chat_id:
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendMessage")
    for i in range(0, len(text), 3800):
        chunk = text[i : i + 3800]
        for _ in range(3):
            try:
                r = requests.post(url, json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}, timeout=15)
                if r.status_code == 200:
                    time.sleep(0.6)
                    break
            except Exception:
                pass
            time.sleep(1.0)
    return True


def main():
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    week_ago_ts = now_ts - (7 * 86400)

    state = load_merged_state()
    episodes = state.get("signal_episodes") or {}
    tracked = state.get("tracked") or {}

    for sym, t_item in list(tracked.items()):
        if is_stock_or_tradfi_symbol(sym):
            del tracked[sym]
            continue
        if isinstance(t_item, dict):
            c_at = int(t_item.get("created_at", 0))
            ep_k = f"{sym}_{c_at}"
            if c_at > 0 and ep_k not in episodes:
                episodes[ep_k] = dict(t_item, symbol=sym)

    ensure_missing_icnt_and_backfill(episodes, tracked, now_ts)
    state["signal_episodes"] = episodes
    state["tracked"] = tracked
    save_merged_state(state)

    candidates, _, _ = get_scanner_candidates()

    weekly_eps = []
    for ep in episodes.values():
        if not isinstance(ep, dict):
            continue
        c_at = int(ep.get("created_at", 0))
        sym = str(ep.get("symbol", "")).upper()
        if c_at >= week_ago_ts and not is_stock_or_tradfi_symbol(sym):
            weekly_eps.append((ep, classify_episode_by_mfe(ep)))

    total_signals = len(weekly_eps)
    tp1_hits = sum(1 for _, c in weekly_eps if c["tp1"])
    tp2_hits = sum(1 for _, c in weekly_eps if c["tp2"])
    tp3_hits = sum(1 for _, c in weekly_eps if c["tp3"])
    trail_closes = sum(1 for _, c in weekly_eps if c["trailing_closed"])
    sl_hits = sum(1 for _, c in weekly_eps if c["sl"])
    active_cnt = max(0, total_signals - (tp1_hits + sl_hits))

    closed_cnt = tp1_hits + sl_hits
    win_rate = round((tp1_hits / closed_cnt) * 100.0, 1) if closed_cnt > 0 else 0.0

    fav_list = [c["fav"] for _, c in weekly_eps if c["fav"] > 0]
    avg_mfe = round(float(np.mean(fav_list)), 1) if fav_list else 0.0
    med_mfe = round(float(np.median(fav_list)), 1) if fav_list else 0.0

    w1_rate = round(tp1_hits / max(total_signals, 1) * 100.0, 1)
    w2_retention = round(tp2_hits / max(tp1_hits, 1) * 100.0, 1) if tp1_hits > 0 else 0.0
    w3_retention = round(tp3_hits / max(tp2_hits, 1) * 100.0, 1) if tp2_hits > 0 else 0.0

    track_stats = {"TRACK_A": [0, 0, []], "TRACK_B": [0, 0, []], "TRACK_C": [0, 0, []]}
    for e, c in weekly_eps:
        tk = str(e.get("track_type", "TRACK_A"))
        if tk not in track_stats:
            track_stats[tk] = [0, 0, []]
        track_stats[tk][0] += 1
        if c["tp1"]:
            track_stats[tk][1] += 1
        track_stats[tk][2].append(c["fav"])

    best_5 = sorted(weekly_eps, key=lambda x: x[1]["fav"], reverse=True)[:5]
    sl_eps = [(e, c) for e, c in weekly_eps if c["sl"]]

    lines = [
        f"📅 [Bybit 멀티프레임 스캐너 주간 종합 & 3파동 심층 리포트] ({now_str})",
        "",
        "📊 [1. 최근 7일 실전 시그널 종합 성적]",
        f"• 주간 총 포착 종목: {total_signals}건 (마감 {closed_cnt}건 / 실시간 추적 중 {active_cnt}건)",
        f"• 실전 익절 승률: {win_rate:.1f}% (익절 {tp1_hits}건 vs 손절 {sl_hits}건)",
        f"• 단계별 실제 달성: 1차익절(TP1·30%) {tp1_hits}건 | 2차돌파(방어선상향) {tp2_hits}건 | 3차돌파(TP3·30%+트레일링) {tp3_hits}건 | 방어선/트레일링마감 {trail_closes}건",
        f"• 포착 종목 평균 최고 상승폭(MFE): +{avg_mfe:.1f}% (중앙값 +{med_mfe:.1f}%)",
        "",
        "🧭 [2. 트랙별(Track A·B·C) 성과 및 평균 상승폭]",
    ]

    track_names = {
        "TRACK_A": "Track A (눌림목/스퀴즈)",
        "TRACK_B": "Track B (수급 돌파/리테스트)",
        "TRACK_C": "Track C (대장주 Fib 되돌림)",
    }
    best_track_label = "Track A (눌림목/스퀴즈)"
    best_track_rate = -1.0
    for tk, label in track_names.items():
        tot_t, win_t, mfes = track_stats.get(tk, [0, 0, []])
        rate_t = round((win_t / tot_t) * 100.0, 1) if tot_t > 0 else 0.0
        avg_t = round(float(np.mean(mfes)), 1) if mfes else 0.0
        if tot_t > 0 and rate_t > best_track_rate:
            best_track_rate = rate_t
            best_track_label = label
        lines.append(f"• {label}: {tot_t}건 중 {win_t}건 수익 (승률 {rate_t:.1f}% | 평균 최고 +{avg_t:.1f}%)")

    lines.extend([
        "",
        "🌊 [3. 3파동(TP1→TP2→TP3) 실제 도달률 & 꺾임 구간 심층 해부]",
        f"• 1파동(TP1) 돌파율: 전체 포착 중 {w1_rate:.1f}% ({tp1_hits}/{total_signals}건)",
        f"• 2파동(TP2) 연장 유지력: TP1 돌파 종목 중 {w2_retention:.1f}%가 2차 구간 안착 ({tp2_hits}/{max(tp1_hits, 1)}건)",
        f"• 3파동(TP3) 클라이맥스 도달률: TP2 돌파 종목 중 {w3_retention:.1f}% 3차 돌파 ({tp3_hits}/{max(tp2_hits, 1)}건)",
        "\n🏆 [4. 주간 베스트 수익률 Top 5 타점 (신규 30%·30%·40% 비중 합산 수익)]",
    ])

    if best_5:
        for idx, (b, c) in enumerate(best_5, 1):
            sym = b.get("symbol", "")
            base = sym.replace("USDT", "")
            entry_p = to_float(b.get("entry_price"))
            max_h = to_float(b.get("max_high_price"))
            fav_pct = c["fav"]
            w_pct = calc_weighted_realized_pct(b, c)
            stage = "3차돌파·트레일링" if c["tp3"] else ("2차방어선확보" if c["tp2"] else ("1차익절·방어선" if c["tp1"] else "추적중"))
            lines.append(
                f"{idx}. ${base} ({sym}): 진입 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (최고 +{fav_pct:.1f}% | 비중합산 +{w_pct:.2f}% | {stage})"
            )
    else:
        lines.append("• 집계된 주간 타점 기록이 아직 없습니다.")

    lines.append(f"\n📓 [5. 주간 손절(SL) 오답노트 원인 해부 (총 {len(sl_eps)}건)]")
    if sl_eps:
        for s, c in sorted(sl_eps, key=lambda x: int(x[0].get("created_at", 0)), reverse=True)[:5]:
            base = str(s.get("symbol", "")).replace("USDT", "")
            fav = c["fav"]
            sc = int(s.get("score", 85))
            tk = str(s.get("track_type", "TRACK_A"))
            cause = "돌파 직후 매수세 실종(가짜 돌파)" if fav < 1.5 else f"+{fav:.1f}% 반등 후 차익 매물 출회"
            lines.append(f"  ↳ ⚠️ ${base} ({tk}·{sc}점): 최고 +{fav:.1f}% 도달 후 -{FIXED_SL_PCT}% 이탈 ({cause})")
    else:
        lines.append("• 최근 7일 내 손절 마감된 종목이 없습니다.")

    lines.extend([
        "",
        "⚙️ [6. 차주 매매 전략 & 운용 가이드]",
        f"• 주력 우위 트랙: 현재 장세에서는 [{best_track_label}] 타점의 승률이 가장 우수합니다.",
        "• 방어선 원칙: 1차 돌파 시(30% 익절 + 1차-5% 방어선), 2차 돌파 시(홀딩 + 2차-5% 방어선), 3차 돌파 시(30% 익절 + 잔여 40% 고점 -3% 자동 트레일링).",
    ])

    if candidates:
        top_mkt = candidates[:3]
        mkt_str = ", ".join(f"${c['base_coin']}({c['change_24h']:+.1f}%)" for c in top_mkt)
        lines.append(f"\n🔥 [현재 바이비트 선물 수급 주도주]: {mkt_str}")

    report_text = "\n".join(lines)
    print(report_text)
    send_telegram_weekly(report_text)
    git_push_saved_files()


if __name__ == "__main__":
    main()
# === END OF FILE ===
