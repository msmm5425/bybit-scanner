import os, json, time, requests
import numpy as np
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

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

try:
    import postmortem
except Exception:
    postmortem = None

from api import get_scanner_candidates, fetch_klines

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
        if not base or base in _FALLBACK_STOCKS:
            return True
        return any(k in base for k in ("SAMSUNG", "HYNIX", "HYUNDAI", "KOSPI", "TSLA", "NVDA", "AAPL"))

    def clean_url(raw_url):
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", str(raw_url or "").strip())
        return s.replace("[", "").replace("]", "")

SCHEME = "ht" + "tps://"
DNA_PROFILE_FILE = "top5_dna_profile.json"
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
    saved_w = to_float(e.get("realized_weighted_pct"), 0.0)
    if saved_w > 0:
        return round(saved_w, 2)

    is_long = bool(e.get("is_long_term") or e.get("track_type") == "TRACK_D")
    trail_drop = 10.0 if is_long else 4.0
    entry_ref = max(to_float(e.get("entry_price")), 1e-8)
    tps = e.get("tps") or []
    tp1_p = to_float(tps[0]) if (len(tps) > 0 and to_float(tps[0]) > entry_ref) else entry_ref * 1.08
    tp2_p = to_float(tps[1]) if (len(tps) > 1 and to_float(tps[1]) > entry_ref) else entry_ref * 1.15
    tp3_p = to_float(tps[2]) if (len(tps) > 2 and to_float(tps[2]) > entry_ref) else entry_ref * 1.22

    g1 = round((tp1_p - entry_ref) / entry_ref * 100.0, 1)
    g2 = round((tp2_p - entry_ref) / entry_ref * 100.0, 1)
    g3 = round((tp3_p - entry_ref) / entry_ref * 100.0, 1)

    if c["tp3"]:
        trail_exit_g = max(g2, round(c["fav"] - trail_drop, 1))
        return round(g3 * 0.50 + trail_exit_g * 0.50, 2)
    if c["tp2"]:
        return g1 if c["trailing_closed"] else round(c["fav"], 2)
    if c["sl"]:
        return -float(int(round(to_float(e.get("sl_pct"), 20 if is_long else FIXED_SL_PCT))))
    return round(c["fav"], 2) if c["tp1"] else 0.0


def load_merged_state_readonly():
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
                            merged[k][sk] = dict(sv) if isinstance(sv, dict) else sv
        except Exception:
            pass
    return merged


def _load_dna_summary_line():
    if not os.path.exists(DNA_PROFILE_FILE):
        return ""
    try:
        with open(DNA_PROFILE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        af = (raw or {}).get("active_dna_formula") or {}
        if af.get("enabled"):
            return (
                f"• 🧬 Top 5 자동학습 DNA (Track H): 14일 표본 {af.get('sample_count', 0)}건(사각지대 {af.get('missed_count', 0)}건) | "
                f"주력 [{af.get('dominant_tag', '-')}] · 응축<={af.get('bb_width_max', 0)}% · "
                f"발사거래량>={af.get('launch_vol_min', 0)}x · OI>={af.get('oi_ratio_min', 0)}%"
            )
    except Exception:
        pass
    return ""


def _backfill_single_ep(ep_k, e):
    sym = str(e.get("symbol", "")).upper()
    entry_p = to_float(e.get("entry_price"))
    c_at = int(e.get("created_at", 0))
    df_15m = fetch_klines(sym, "15", limit=96)
    if not df_15m.empty and "timestamp" in df_15m.columns:
        post_df = df_15m[df_15m["timestamp"] >= (c_at * 1000 - 900000)]
        if not post_df.empty:
            h_max = to_float(post_df["high"].max(), entry_p)
            if 0.40 <= (h_max / entry_p) <= 15.0 and h_max > to_float(e.get("max_high_price"), entry_p):
                e["max_high_price"] = h_max
                e["max_favorable_pct"] = round((h_max - entry_p) / entry_p * 100.0, 2)
    return ep_k, e


def backfill_in_memory_only(episodes, now_ts):
    to_bf = []
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
        if sym and entry_p > 0 and c_at > 0 and (now_ts - c_at) <= 172800 and fav < 19.5 and not e.get("sl_alerted"):
            to_bf.append((ep_k, e))

    if to_bf:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(_backfill_single_ep, k, v) for k, v in to_bf[:10]]
            for fut in as_completed(futs):
                try:
                    k, v = fut.result()
                    episodes[k] = v
                except Exception:
                    pass


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
    tp2 = fav >= max(12.0, tp2_req)
    tp3 = fav >= max(19.0, tp3_req)
    is_sl = bool(e.get("sl_alerted")) and not tp2
    trailing_closed = bool(e.get("tp3_completed") or e.get("closed_alerted")) and tp2 and not is_sl
    return {"fav": fav, "tp1": tp1, "tp2": tp2, "tp3": tp3, "be_save": False, "sl": is_sl, "trailing_closed": trailing_closed}


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
                r = requests.post(url, json={"chat_id": chat_id, "text": chunk, "disable_web_page_preview": True}, timeout=12)
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

    state = load_merged_state_readonly()
    episodes = state.get("signal_episodes") or {}
    tracked = state.get("tracked") or {}
    sl_history = state.get("sl_history") or {}

    for sym, t_item in list(tracked.items()):
        if is_stock_or_tradfi_symbol(sym):
            continue
        if isinstance(t_item, dict):
            c_at = int(t_item.get("created_at", 0))
            ep_k = f"{sym}_{c_at}"
            if c_at > 0 and ep_k not in episodes:
                episodes[ep_k] = dict(t_item, symbol=sym)

    backfill_in_memory_only(episodes, now_ts)
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
    active_cnt = max(0, total_signals - (tp2_hits + sl_hits))

    closed_cnt = tp2_hits + sl_hits
    win_rate = round((tp2_hits / closed_cnt) * 100.0, 1) if closed_cnt > 0 else 0.0

    fav_list = [c["fav"] for _, c in weekly_eps if c["fav"] > 0]
    avg_mfe = round(float(np.mean(fav_list)), 1) if fav_list else 0.0
    med_mfe = round(float(np.median(fav_list)), 1) if fav_list else 0.0

    w1_rate = round(tp1_hits / max(total_signals, 1) * 100.0, 1)
    w2_retention = round(tp2_hits / max(tp1_hits, 1) * 100.0, 1) if tp1_hits > 0 else 0.0
    w3_retention = round(tp3_hits / max(tp2_hits, 1) * 100.0, 1) if tp2_hits > 0 else 0.0

    track_names = {
        "TRACK_A": "Track A (TTM 스퀴즈·델타)",
        "TRACK_B": "Track B (24H 돌파·S-R Flip)",
        "TRACK_C": "Track C (대장주 심층 눌림)",
        "TRACK_D": "Track D (장기 포지션 · D1~D5)",
        "TRACK_E": "Track E (SFP 유동성 스윕)",
        "TRACK_F": "Track F (BTC 역행 RS 대장주)",
        "TRACK_G": "Track G (숏스퀴즈 강제청산)",
        "TRACK_H": "Track H (Top 5 급등 DNA 자동학습)",
    }
    track_stats = {tk: [0, 0, 0, []] for tk in track_names}
    for e, c in weekly_eps:
        tk = str(e.get("track_type", "TRACK_A"))
        if tk not in track_stats:
            track_stats[tk] = [0, 0, 0, []]
        track_stats[tk][0] += 1
        if c["tp2"]:
            track_stats[tk][1] += 1
        if c["sl"]:
            track_stats[tk][2] += 1
        track_stats[tk][3].append(c["fav"])

    best_5 = sorted(weekly_eps, key=lambda x: x[1]["fav"], reverse=True)[:5]
    sl_eps = [(e, c) for e, c in weekly_eps if c["sl"]]
    sls_recent = [s for s in sl_history.values() if isinstance(s, dict) and not is_stock_or_tradfi_symbol(s.get("symbol", "")) and (now_ts - int(s.get("sl_at", 0))) <= (7 * 86400)]

    lines = [
        f"📅 [Bybit 멀티프레임 스캐너 주간 종합 & 3파동 심층 리포트] ({now_str})",
        "",
        "📊 [1. 최근 7일 실전 시그널 종합 성적]",
        f"• 주간 총 포착 종목: {total_signals}건 (확정마감 {closed_cnt}건 / 실시간 홀딩·추적 중 {active_cnt}건)",
        f"• 수익 확정 승률(TP2 방어선 이상): {win_rate:.1f}% (수익확보 {tp2_hits}건 vs 손절 {sl_hits}건)",
        f"• 단계별 실제 달성: 1차돌파(전량홀딩) {tp1_hits}건 | 2차돌파(TP1방어) {tp2_hits}건 | 3차돌파(50%익절+트레일링) {tp3_hits}건 | 수익마감 {trail_closes}건",
        f"• 포착 종목 평균 최고 상승폭(MFE): +{avg_mfe:.1f}% (중앙값 +{med_mfe:.1f}%)",
        "",
        "🧭 [2. 트랙별(Track A~H) 성과 및 평균 상승폭]",
    ]

    best_track_label = "Track A (TTM 스퀴즈·델타)"
    best_track_rate = -1.0
    for tk, label in track_names.items():
        tot_t, win_t, sl_t, mfes = track_stats.get(tk, [0, 0, 0, []])
        if tot_t == 0 and tk in ("TRACK_E", "TRACK_F", "TRACK_G", "TRACK_H"):
            continue
        closed_t = win_t + sl_t
        rate_t = round((win_t / closed_t) * 100.0, 1) if closed_t > 0 else 0.0
        avg_t = round(float(np.mean(mfes)), 1) if mfes else 0.0
        if closed_t > 0 and rate_t > best_track_rate:
            best_track_rate = rate_t
            best_track_label = label
        lines.append(f"• {label}: 총 {tot_t}건 (수익확보 {win_t}건 vs 손절 {sl_t}건 | 확정승률 {rate_t:.1f}% | 평균 최고 +{avg_t:.1f}%)")

    lines.extend([
        "",
        "🌊 [3. 3파동(TP1→TP2→TP3) 실제 도달률 심층 해부]",
        f"• 1파동(TP1) 돌파율: 전체 포착 중 {w1_rate:.1f}% ({tp1_hits}/{total_signals}건)",
        f"• 2파동(TP2) 연장 유지력: TP1 돌파 종목 중 {w2_retention:.1f}%가 2차 구간 안착 ({tp2_hits}/{max(tp1_hits, 1)}건)",
        f"• 3파동(TP3) 클라이맥스 도달률: TP2 돌파 종목 중 {w3_retention:.1f}% 3차 돌파 ({tp3_hits}/{max(tp2_hits, 1)}건)",
        "\n🏆 [4. 주간 베스트 수익률 Top 5 타점 (전량 홀딩 · 3차 50%+50% 합산)]",
    ])

    if best_5:
        for idx, (b, c) in enumerate(best_5, 1):
            sym = b.get("symbol", "")
            base = sym.replace("USDT", "")
            entry_p = to_float(b.get("entry_price"))
            max_h = to_float(b.get("max_high_price"))
            fav_pct = c["fav"]
            w_pct = calc_weighted_realized_pct(b, c)
            h_tag = "[장기] " if (b.get("is_long_term") or b.get("track_type") == "TRACK_D") else ""
            stage = "3차50%익절·트레일링" if c["tp3"] else ("2차돌파·TP1방어" if c["tp2"] else ("1차돌파·전량홀딩" if c["tp1"] else "추적중"))
            lines.append(
                f"{idx}. {h_tag}${base} ({sym}): 진입 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (최고 +{fav_pct:.1f}% | 합산 +{w_pct:.2f}% | {stage})"
            )
    else:
        lines.append("• 집계된 주간 타점 기록이 아직 없습니다.")

    lines.append(f"\n📓 [5. 주간 손절(SL) 오답노트 v2.0 & 사후 부검 (총 {len(sl_eps)}건)]")
    if postmortem is not None and hasattr(postmortem, "build_daily_postmortem_summary"):
        try:
            lines.extend(postmortem.build_daily_postmortem_summary(sls_recent))
        except Exception:
            pass
    elif sl_eps:
        for s, c in sorted(sl_eps, key=lambda x: int(x[0].get("created_at", 0)), reverse=True)[:5]:
            base = str(s.get("symbol", "")).replace("USDT", "")
            fav = c["fav"]
            sc = int(s.get("score", 90))
            tk = str(s.get("track_type", "TRACK_A"))
            sl_p_val = int(round(to_float(s.get("sl_pct"), 20 if tk == "TRACK_D" else FIXED_SL_PCT)))
            cause = "돌파 직후 매수세 실종(가짜 돌파)" if fav < 1.5 else f"+{fav:.1f}% 반등 후 차익 매물 출회"
            lines.append(f"  ↳ ⚠️ ${base} ({tk}·{sc}점): 최고 +{fav:.1f}% 도달 후 -{sl_p_val}% 이탈 ({cause})")
    else:
        lines.append("• 최근 7일 내 손절 마감된 종목이 없습니다.")

    lines.extend([
        "",
        "⚙️ [6. 차주 매매 전략 & 자율 학습 현황]",
        f"• 주력 우위 트랙: 현재 장세에서는 [{best_track_label}] 타점의 확정 승률이 가장 우수합니다.",
    ])
    dna_line = _load_dna_summary_line()
    if dna_line:
        lines.append(dna_line)
    lines.append(
        "• 운용 원칙: 단기(-6% SL) · 장기(-20% SL) | 1차 돌파(전량 홀딩) → 2차 돌파(전량 홀딩 · 방어선 TP1) → 3차 돌파(50% 익절 · 방어선 TP2 & 잔여 50% 단기 -4% / 장기 -10% 자동 트레일링)."
    )

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
