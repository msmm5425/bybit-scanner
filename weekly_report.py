import os, json, time, requests
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    import config as cfg
    STATE_FILE = getattr(cfg, "STATE_FILE", "scanner_state.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception:
    cfg = None
    STATE_FILE = "scanner_state.json"
    KST = timezone(timedelta(hours=9))
    FIXED_SL_PCT = 6

from api import get_scanner_candidates, get_valid_bybit_linear_symbols, clean_url

SCHEME = "ht" + "tps://"


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


def load_json_safe(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d if isinstance(d, type(default_val)) else default_val
        except Exception:
            pass
    return default_val


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

    state = load_json_safe(STATE_FILE, {})
    episodes = state.get("signal_episodes") or {}
    sl_history = state.get("sl_history") or {}

    valid_syms, _ = get_valid_bybit_linear_symbols()
    candidates, _, _ = get_scanner_candidates()
    if valid_syms and candidates:
        candidates = [c for c in candidates if c.get("symbol") in valid_syms]

    weekly_eps = []
    for ep in episodes.values():
        if not isinstance(ep, dict):
            continue
        c_at = int(ep.get("created_at", 0))
        sym = str(ep.get("symbol", "")).upper()
        if c_at >= week_ago_ts and (not valid_syms or sym in valid_syms):
            weekly_eps.append(ep)

    total_signals = len(weekly_eps)
    tp1_hits = sum(1 for e in weekly_eps if e.get("tp1_hit"))
    tp2_hits = sum(1 for e in weekly_eps if e.get("tp2_hit"))
    tp3_hits = sum(1 for e in weekly_eps if e.get("tp3_alerted") or e.get("tp3_completed"))
    be_saves = sum(1 for e in weekly_eps if e.get("be_armed") and not e.get("tp1_hit") and not e.get("sl_alerted"))
    sl_hits = sum(1 for e in weekly_eps if e.get("sl_alerted"))
    active_cnt = max(0, total_signals - (tp1_hits + be_saves + sl_hits))

    win_or_save_cnt = tp1_hits + be_saves
    closed_cnt = win_or_save_cnt + sl_hits
    win_rate = round((win_or_save_cnt / closed_cnt) * 100.0, 1) if closed_cnt > 0 else 0.0

    fav_list = [to_float(e.get("max_favorable_pct")) for e in weekly_eps if to_float(e.get("max_favorable_pct")) > 0]
    avg_mfe = round(float(np.mean(fav_list)), 1) if fav_list else 0.0
    med_mfe = round(float(np.median(fav_list)), 1) if fav_list else 0.0

    w1_rate = round(tp1_hits / max(total_signals, 1) * 100.0, 1)
    w2_retention = round(tp2_hits / max(tp1_hits, 1) * 100.0, 1) if tp1_hits > 0 else 0.0
    w3_retention = round(tp3_hits / max(tp2_hits, 1) * 100.0, 1) if tp2_hits > 0 else 0.0

    track_stats = {"TRACK_A": [0, 0, []], "TRACK_B": [0, 0, []], "TRACK_C": [0, 0, []]}
    for e in weekly_eps:
        tk = str(e.get("track_type", "TRACK_A"))
        if tk not in track_stats:
            track_stats[tk] = [0, 0, []]
        track_stats[tk][0] += 1
        if e.get("tp1_hit") or (e.get("be_armed") and not e.get("sl_alerted")):
            track_stats[tk][1] += 1
        track_stats[tk][2].append(to_float(e.get("max_favorable_pct")))

    best_5 = sorted(weekly_eps, key=lambda x: to_float(x.get("max_favorable_pct")), reverse=True)[:5]
    sl_eps = [e for e in weekly_eps if e.get("sl_alerted")]

    lines = [
        f"📅 [Bybit 멀티프레임 스캐너 주간 종합 & 3파동 심층 리포트] ({now_str})",
        "",
        "📊 [1. 최근 7일 실전 시그널 종합 성적]",
        f"• 주간 총 포착 종목: {total_signals}건 (마감 {closed_cnt}건 / 실시간 추적 중 {active_cnt}건)",
        f"• 실전 방어 포함 승률: {win_rate:.1f}% (익절/본전방어 {win_or_save_cnt}건 vs 손절 {sl_hits}건)",
        f"• 단계별 달성: 1차익절(TP1) {tp1_hits}건 | 2차익절(TP2) {tp2_hits}건 | 3차졸업/수익마감 {tp3_hits}건 | 본전방어(1R) {be_saves}건",
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
        lines.append(f"• {label}: {tot_t}건 중 {win_t}건 수익/방어 (승률 {rate_t:.1f}% | 평균 최고 +{avg_t:.1f}%)")

    lines.extend([
        "",
        "🌊 [3. 3파동(TP1→TP2→TP3) 도달률 & 꺾임 구간 심층 해부]",
        f"• 1파동(TP1, +8%권) 돌파율: 전체 포착 중 {w1_rate:.1f}% ({tp1_hits}/{total_signals}건)",
        f"• 2파동(TP2, +15%권) 연장 유지력: TP1 돌파 종목 중 {w2_retention:.1f}%가 2차 파동 안착 ({tp2_hits}/{max(tp1_hits, 1)}건)",
        f"• 3파동(TP3, +22%권) 클라이맥스 도달률: TP2 돌파 종목 중 {w3_retention:.1f}% 최종 졸업 ({tp3_hits}/{max(tp2_hits, 1)}건)",
    ])

    stall_1_3 = sum(1 for e in sl_eps if 1.0 <= to_float(e.get("max_favorable_pct")) < 4.0)
    stall_4_7 = sum(1 for e in sl_eps if 4.0 <= to_float(e.get("max_favorable_pct")) < 7.5)
    lines.append(
        f"• 주요 꺾임(저항) 구간 진단: 손절 종목 중 +1~4% 초기 저항 후 반락 {stall_1_3}건 / +4~7% 본전방어 직전 반락 {stall_4_7}건"
    )

    lines.append("\n🏆 [4. 주간 베스트 수익률 Top 5 타점]")
    if best_5:
        for idx, b in enumerate(best_5, 1):
            sym = b.get("symbol", "")
            base = sym.replace("USDT", "")
            entry_p = to_float(b.get("entry_price"))
            max_h = to_float(b.get("max_high_price"))
            fav_pct = to_float(b.get("max_favorable_pct"))
            stage = "3차졸업" if b.get("tp3_alerted") else ("2차돌파" if b.get("tp2_hit") else ("1차돌파" if b.get("tp1_hit") else "추적중"))
            lines.append(f"{idx}. ${base} ({sym}): 진입 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (+{fav_pct:.1f}% | {stage})")
    else:
        lines.append("• 집계된 주간 타점 기록이 아직 없습니다.")

    lines.append(f"\n📓 [5. 주간 손절(SL) 오답노트 원인 해부 (총 {len(sl_eps)}건)]")
    if sl_eps:
        sl_mfes = [to_float(x.get("max_favorable_pct")) for x in sl_eps]
        avg_sl_mfe = round(float(np.mean(sl_mfes)), 1) if sl_mfes else 0.0
        lines.append(f"• 손절 종목 진입 후 평균 최고 반등폭: +{avg_sl_mfe:.1f}% (고정 손절폭 -{FIXED_SL_PCT}%)")
        for s in sorted(sl_eps, key=lambda x: int(x.get("created_at", 0)), reverse=True)[:5]:
            base = str(s.get("symbol", "")).replace("USDT", "")
            fav = to_float(s.get("max_favorable_pct"))
            sc = int(s.get("score", 85))
            tk = str(s.get("track_type", "TRACK_A"))
            cause = "돌파 직후 매수세 실종(가짜 돌파)" if fav < 1.5 else f"+{fav:.1f}% 반등 후 차익 매물 출회"
            lines.append(f"  ↳ ⚠️ ${base} ({tk}·{sc}점): 최고 +{fav:.1f}% 도달 후 -{FIXED_SL_PCT}% 이탈 ({cause})")
    else:
        lines.append("• 최근 7일 내 손절 마감된 종목이 없습니다.")

    rec_be = "4.5~5.0%" if stall_4_7 >= 2 else f"{FIXED_SL_PCT}.0%"
    lines.extend([
        "",
        "⚙️ [6. 차주 매매 전략 & 자율 학습 권고안]",
        f"• 주력 우위 트랙: 현재 장세에서는 [{best_track_label}] 타점의 방어력이 가장 우수합니다.",
        f"• 리스크 관리 팁: 진입 후 +{rec_be} 이상 상승 시 본전 방어선(+0.4%)을 조기 가동하면 손절률을 크게 낮출 수 있습니다.",
    ])

    if candidates:
        top_mkt = candidates[:3]
        mkt_str = ", ".join(f"${c['base_coin']}({c['change_24h']:+.1f}%)" for c in top_mkt)
        lines.append(f"\n🔥 [현재 바이비트 선물 수급 주도주]: {mkt_str}")

    report_text = "\n".join(lines)
    print(report_text)
    send_telegram_weekly(report_text)


if __name__ == "__main__":
    main()
# === END OF FILE ===
