```python
import os
import json
import time
import requests
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

from api import get_scanner_candidates, get_valid_bybit_linear_symbols

HISTORY_FILE = "analyzer_history.json"


def to_float(val, default=0.0):
    try:
        if val is None or val == "":
            return default
        v = float(val)
        return default if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return default


def fmt_price(price):
    p = to_float(price, 0.0)
    if p <= 0:
        return "0.0000"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 10:
        return f"{p:.3f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    return f"{p:.6f}"


def load_json_safe(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, type(default_val)) else default_val
        except Exception:
            pass
    return default_val


def send_telegram_weekly(text):
    if not text:
        return False
    token = (
        os.getenv("TELEGRAM_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or getattr(cfg, "TELEGRAM_BOT_TOKEN", "")
        or ""
    ).strip()
    chat_id = (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("CHAT_ID")
        or getattr(cfg, "TELEGRAM_CHAT_ID", "")
        or ""
    ).strip()

    if not token or not chat_id:
        print("[경고] 텔레그램 환경변수가 비어 있습니다.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = [text[i : i + 3800] for i in range(0, len(text), 3800)]
    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": True,
        }
        for _ in range(3):
            try:
                r = requests.post(url, json=payload, timeout=15)
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
    episodes = state.get("signal_episodes", {})

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
    be_saves = sum(
        1 for e in weekly_eps
        if e.get("be_armed") and not e.get("tp1_hit") and not e.get("sl_alerted")
    )
    sl_hits = sum(1 for e in weekly_eps if e.get("sl_alerted"))

    win_or_save_cnt = tp1_hits + be_saves
    closed_cnt = win_or_save_cnt + sl_hits
    win_rate = round((win_or_save_cnt / closed_cnt) * 100.0, 1) if closed_cnt > 0 else 0.0

    track_stats = {"TRACK_A": [0, 0], "TRACK_B": [0, 0], "TRACK_C": [0, 0]}
    for e in weekly_eps:
        t_type = str(e.get("track_type", "TRACK_A"))
        if t_type not in track_stats:
            track_stats[t_type] = [0, 0]
        track_stats[t_type][0] += 1
        if e.get("tp1_hit") or (e.get("be_armed") and not e.get("sl_alerted")):
            track_stats[t_type][1] += 1

    best_5 = sorted(
        weekly_eps,
        key=lambda x: to_float(x.get("max_favorable_pct")),
        reverse=True,
    )[:5]

    lines = [
        f"📅 [Bybit 멀티프레임 스캐너 주간 종합 리포트] ({now_str})",
        "",
        "📊 [최근 7일 실전 시그널 종합 성적]",
        f"• 주간 총 포착 종목: {total_signals}건 (마감 {closed_cnt}건)",
        f"• 실전 방어 포함 승률: {win_rate:.1f}% (익절/본전방어 {win_or_save_cnt}건 vs 손절 {sl_hits}건)",
        f"• 단계별 달성: 1차익절(TP1) {tp1_hits}건 | 2차익절(TP2) {tp2_hits}건 | 3차졸업/수익마감 {tp3_hits}건 | 본전방어(1R) {be_saves}건",
        "",
        "🧭 [트랙별(Track A·B·C) 성과 분석]",
    ]

    track_names = {
        "TRACK_A": "Track A (눌림목/스퀴즈)",
        "TRACK_B": "Track B (수급 돌파)",
        "TRACK_C": "Track C (대장주 되돌림)",
    }
    for tk, label in track_names.items():
        tot_t, win_t = track_stats.get(tk, [0, 0])
        rate_t = round((win_t / tot_t) * 100.0, 1) if tot_t > 0 else 0.0
        lines.append(f"• {label}: {tot_t}건 중 {win_t}건 수익/방어 ({rate_t:.1f}%)")

    lines.append("\n🏆 [주간 베스트 수익률 Top 5 타점]")
    if best_5:
        for idx, b in enumerate(best_5, 1):
            sym = b.get("symbol", "")
            base = sym.replace("USDT", "")
            entry_p = to_float(b.get("entry_price"))
            max_h = to_float(b.get("max_high_price"))
            fav_pct = to_float(b.get("max_favorable_pct"))
            lines.append(
                f"{idx}. ${base} ({sym}): 진입 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (+{fav_pct:.1f}%)"
            )
    else:
        lines.append("• 집계된 주간 타점 기록이 아직 없습니다.")

    if candidates:
        top_mkt = candidates[:3]
        mkt_str = ", ".join(
            f"${c['base_coin']}({c['change_24h']:+.1f}%)" for c in top_mkt
        )
        lines.append(f"\n🔥 [현재 바이비트 선물 수급 주도주]: {mkt_str}")

    report_text = "\n".join(lines)
    print(report_text)
    send_telegram_weekly(report_text)


if __name__ == "__main__":
    main()
```