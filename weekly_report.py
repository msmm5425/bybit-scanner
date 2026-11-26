import os, json, time, requests
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from config import (
    STATE_FILE,
    KST,
    FIXED_SL_PCT,
    LONGTERM_SL_PCT,
    ROUNDTRIP_FEE_SLIPPAGE_PCT,
    DNA_PROFILE_FILE,
    to_float,
)
from api import (
    get_scanner_candidates,
    fetch_klines,
    is_stock_or_tradfi_symbol,
    clean_url,
)

try:
    import postmortem
except Exception:
    postmortem = None

SCHEME = "ht" + "tps://"
FEE_SLIPPAGE_PCT = float(ROUNDTRIP_FEE_SLIPPAGE_PCT)
STATE_CANDIDATE_FILES = list(dict.fromkeys([STATE_FILE, "bot_state.json"]))


def fmt_price(price):
    p = to_float(price, 0.0)
    if p <= 0:
        return "0.0000"
    return f"{p:,.2f}" if p >= 1000 else (f"{p:.3f}" if p >= 10 else (f"{p:.4f}" if p >= 1 else (f"{p:.5f}" if p >= 0.01 else f"{p:.6f}")))


def calc_weighted_realized_pct(e, c):
    """
    [P1-1 수정] 실체결가(exit_price) 및 왕복 수수료·슬리피지(0.15%)가 반영된 실현 수익률 산출
    - 손절: 원장 realized_weighted_pct / exit_price 최우선 사용 (고정 -6.0% 왜곡 제거)
    - 익절/만료: 원장 확정 수익률 최우선 사용
    """
    is_long = bool(e.get("is_long_term") or e.get("track_type") == "TRACK_D")
    entry_ref = max(to_float(e.get("entry_price")), 1e-8)
    saved_w = to_float(e.get("realized_weighted_pct"), 0.0)
    exit_p = to_float(e.get("exit_price"), 0.0)

    if c["sl"]:
        if saved_w < 0:
            return round(saved_w, 2)
        if exit_p > 0 and entry_ref > 0:
            return round(((exit_p - entry_ref) / entry_ref * 100.0) - FEE_SLIPPAGE_PCT, 2)
        sl_base = float(int(round(to_float(e.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT))))
        return round(-sl_base - FEE_SLIPPAGE_PCT, 2)

    if c["expired"]:
        if abs(saved_w) > 0:
            return round(saved_w, 2)
        if exit_p > 0 and entry_ref > 0:
            return round(((exit_p - entry_ref) / entry_ref * 100.0) - FEE_SLIPPAGE_PCT, 2)
        return 0.0

    if abs(saved_w) > 0:
        return round(saved_w, 2)

    trail_drop = 10.0 if is_long else 4.0
    tps = e.get("tps") or []
    tp1_p = to_float(tps[0]) if (len(tps) > 0 and to_float(tps[0]) > entry_ref) else entry_ref * 1.08
    tp2_p = to_float(tps[1]) if (len(tps) > 1 and to_float(tps[1]) > entry_ref) else entry_ref * 1.15
    tp3_p = to_float(tps[2]) if (len(tps) > 2 and to_float(tps[2]) > entry_ref) else entry_ref * 1.22

    g1 = round((tp1_p - entry_ref) / entry_ref * 100.0, 1)
    g2 = round((tp2_p - entry_ref) / entry_ref * 100.0, 1)
    g3 = round((tp3_p - entry_ref) / entry_ref * 100.0, 1)

    if c["tp3"]:
        if c["trailing_closed"] and exit_p > 0:
            trail_exit_g = round((exit_p - entry_ref) / entry_ref * 100.0, 2)
            return round((g3 * 0.50 + trail_exit_g * 0.50) - FEE_SLIPPAGE_PCT, 2)
        trail_est_g = max(g2, round(c["fav"] - trail_drop, 1))
        return round((g3 * 0.50 + trail_est_g * 0.50) - FEE_SLIPPAGE_PCT, 2)

    if c["tp2"]:
        if c["trailing_closed"]:
            if exit_p > 0:
                return round(((exit_p - entry_ref) / entry_ref * 100.0) - FEE_SLIPPAGE_PCT, 2)
            return round(g1 - FEE_SLIPPAGE_PCT, 2)
        return round(c["fav"], 2)

    if c["is_closed"] and exit_p > 0:
        return round(((exit_p - entry_ref) / entry_ref * 100.0) - FEE_SLIPPAGE_PCT, 2)

    return round(c["fav"], 2) if c["tp1"] else 0.0


def load_merged_state_readonly():
    """완전 읽기 전용 상태 로드 (디스크 및 GitHub에 절대 쓰지 않음)"""
    merged = {"tracked": {}, "alert_history": {}, "sl_history": {}, "signal_episodes": {}}
    lock_flags = (
        "tp1_hit", "tp2_hit", "tp3_hit",
        "tp1_alerted", "tp2_alerted", "tp3_alerted",
        "tp3_completed", "trailing_closed", "sl_alerted", "expired_closed", "closed_alerted",
    )
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
                        prev_sv = merged[k].get(sk)
                        if prev_sv is None:
                            merged[k][sk] = dict(sv) if isinstance(sv, dict) else sv
                        elif isinstance(sv, dict) and isinstance(prev_sv, dict):
                            cand = dict(prev_sv)
                            if to_float(sv.get("max_favorable_pct")) >= to_float(prev_sv.get("max_favorable_pct")):
                                cand.update(sv)
                            for fg in lock_flags:
                                if prev_sv.get(fg) or sv.get(fg):
                                    cand[fg] = True
                            cand["max_high_price"] = max(to_float(prev_sv.get("max_high_price")), to_float(sv.get("max_high_price")))
                            cand["max_favorable_pct"] = max(to_float(prev_sv.get("max_favorable_pct")), to_float(sv.get("max_favorable_pct")))
                            merged[k][sk] = cand
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
        ver = str(af.get("dna_version") or "legacy")
        n_samp = int(to_float(af.get("sample_count"), 0))
        if af.get("enabled"):
            return (
                f"• 🧬 Top 5 정제 차트 DNA (Track H · {ver}): 14일 표본 {n_samp}건(사각지대 {af.get('missed_count', 0)}건) | "
                f"주력 [{af.get('dominant_tag', '-')}] · 응축<={af.get('bb_width_max', 0)}% · "
                f"건조<={af.get('vol_dry_max', 0)} · 발사양봉>={af.get('launch_vol_min', 0)}x · "
                f"직전RSI {af.get('pre_rsi_min', 0)}~{af.get('pre_rsi_max', 0)}"
            )
        return f"• 🧬 Top 5 정제 차트 DNA (Track H · {ver}): 정제 표본 축적 중 ({n_samp}/5건 · 과열 표본 배제)"
    except Exception:
        pass
    return ""


def _backfill_single_ep(ep_k, e):
    sym = str(e.get("symbol", "")).upper()
    entry_p = to_float(e.get("entry_price"))
    c_at = int(e.get("created_at", 0))
    df_15m = fetch_klines(sym, "15", limit=96)
    if not df_15m.empty and "timestamp" in df_15m.columns:
        post_df = df_15m[df_15m["timestamp"] >= (c_at * 1000)]
        if not post_df.empty:
            h_max = to_float(post_df["high"].max(), entry_p)
            if 0.40 <= (h_max / entry_p) <= 15.0 and h_max > to_float(e.get("max_high_price"), entry_p):
                e["max_high_price"] = h_max
                e["max_favorable_pct"] = round((h_max - entry_p) / entry_p * 100.0, 2)
    return ep_k, e


def backfill_in_memory_only(episodes, now_ts):
    """리포트 MFE 표시 정확도용 메모리 내 최고가 보정 (파일 쓰기·승패 플래그 변조 없음)"""
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
        if (
            sym
            and entry_p > 0
            and c_at > 0
            and (now_ts - c_at) <= 172800
            and fav < 19.5
            and not e.get("sl_alerted")
            and not e.get("closed_alerted")
        ):
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


def classify_episode_by_actual_flags(e):
    """
    [P1-1 수정] 실제 체결 플래그 최우선 분류 + 만료/레거시 마감·실시간 활성 포지션 정확 분리
    """
    fav = to_float(e.get("max_favorable_pct"))
    entry_p = to_float(e.get("entry_price"))
    max_h = to_float(e.get("max_high_price"))
    if entry_p > 0 and max_h > entry_p:
        fav = max(fav, round((max_h - entry_p) / entry_p * 100.0, 2))
    tps = e.get("tps") or []
    tp1_req = ((to_float(tps[0]) - entry_p) / entry_p * 100.0 - 0.3) if (len(tps) > 0 and entry_p > 0 and to_float(tps[0]) > entry_p) else 6.8

    is_sl = bool(e.get("sl_alerted"))
    is_expired = (not is_sl) and bool(e.get("expired_closed"))
    tp3 = (not is_sl) and bool(e.get("tp3_hit") or e.get("tp3_alerted"))
    tp2 = (not is_sl) and (tp3 or bool(e.get("tp2_hit") or e.get("tp2_alerted")))
    tp1 = tp2 or bool(e.get("tp1_hit") or e.get("tp1_alerted")) or (fav >= max(6.5, tp1_req))
    trailing_closed = (not is_sl) and (not is_expired) and tp2 and bool(e.get("tp3_completed") or e.get("trailing_closed") or e.get("closed_alerted"))
    is_closed = bool(is_sl or is_expired or trailing_closed or e.get("closed_alerted") or e.get("tp3_completed"))
    legacy_closed = (not is_sl) and (not is_expired) and (not tp2) and is_closed

    return {
        "fav": fav,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl": is_sl,
        "expired": is_expired,
        "legacy_closed": legacy_closed,
        "trailing_closed": trailing_closed,
        "is_closed": is_closed,
        "is_active": not is_closed,
    }


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
            weekly_eps.append((ep, classify_episode_by_actual_flags(ep)))

    total_signals = len(weekly_eps)
    tp1_hits = sum(1 for _, c in weekly_eps if c["tp1"])
    tp2_hits = sum(1 for _, c in weekly_eps if c["tp2"])
    tp3_hits = sum(1 for _, c in weekly_eps if c["tp3"])
    trail_closes = sum(1 for _, c in weekly_eps if c["trailing_closed"])
    sl_hits = sum(1 for _, c in weekly_eps if c["sl"])
    expired_cnt = sum(1 for _, c in weekly_eps if c["expired"])
    legacy_cnt = sum(1 for _, c in weekly_eps if c["legacy_closed"])
    # [P1-1 수정] 실제 미마감 포지션만 직접 집계 (만료/레거시 마감 착시 제거)
    active_cnt = sum(1 for _, c in weekly_eps if c["is_active"])

    closed_win_loss_cnt = tp2_hits + sl_hits
    win_rate = round((tp2_hits / closed_win_loss_cnt) * 100.0, 1) if closed_win_loss_cnt > 0 else 0.0

    # [P1-1] 확정 마감 종목 기준 실현 손익 및 거래당 기대값(Expectancy · 비용반영) 산출
    closed_pnls = [calc_weighted_realized_pct(e, c) for e, c in weekly_eps if c["is_closed"]]
    cum_realized_pct = round(sum(closed_pnls), 2) if closed_pnls else 0.0
    expectancy_pct = round(float(np.mean(closed_pnls)), 2) if closed_pnls else 0.0
    win_pnls = [p for p in closed_pnls if p > 0]
    loss_pnls = [p for p in closed_pnls if p <= 0]
    avg_win_pct = round(float(np.mean(win_pnls)), 2) if win_pnls else 0.0
    avg_loss_pct = round(float(np.mean(loss_pnls)), 2) if loss_pnls else 0.0

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
        "TRACK_D": "Track D (장기 포지션 · D1~D6)",
        "TRACK_E": "Track E (SFP 유동성 스윕)",
        "TRACK_F": "Track F (BTC 역행 RS 대장주)",
        "TRACK_G": "Track G (숏스퀴즈 강제청산)",
        "TRACK_H": "Track H (Top 5 정제 DNA)",
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

    best_5 = sorted(
        [(e, c) for e, c in weekly_eps if not c["sl"] and not c["expired"]],
        key=lambda x: (calc_weighted_realized_pct(x[0], x[1]), x[1]["fav"]),
        reverse=True,
    )[:5]
    sl_eps = [(e, c) for e, c in weekly_eps if c["sl"]]

    extra_closed_note = ""
    if expired_cnt > 0 or legacy_cnt > 0:
        extra_closed_note = f" / 기한만료 {expired_cnt}건" + (f" · 기타마감 {legacy_cnt}건" if legacy_cnt > 0 else "")

    lines = [
        f"📅 [Bybit 멀티프레임 스캐너 주간 종합 & 3파동 심층 리포트] ({now_str})",
        "",
        "📊 [1. 최근 7일 실전 시그널 종합 성적 (실체결가 · 왕복비용 0.15% 반영)]",
        f"• 주간 총 포착 종목: {total_signals}건 (확정승패 {closed_win_loss_cnt}건{extra_closed_note} / 실시간 홀딩 {active_cnt}건)",
        f"• 수익 확정 승률(TP2 종가안착 이상): {win_rate:.1f}% (수익확보 {tp2_hits}건 vs 손절 {sl_hits}건)",
        f"• 마감 실현 손익({len(closed_pnls)}건): 누적 {cum_realized_pct:+.2f}% | 거래당 기대값 {expectancy_pct:+.2f}% (평균익절 {avg_win_pct:+.2f}% vs 평균손실 {avg_loss_pct:+.2f}%)",
        f"• 단계별 실제 달성: 1차돌파(전량홀딩) {tp1_hits}건 | 2차안착(TP1방어) {tp2_hits}건 | 3차돌파(50%익절+트레일링) {tp3_hits}건 | 최종수익마감 {trail_closes}건",
        f"• 포착 종목 평균 최고 상승폭(MFE): +{avg_mfe:.1f}% (중앙값 +{med_mfe:.1f}%)",
        "",
        "🧭 [2. 트랙별(Track A~H) 확정 성과 및 평균 상승폭]",
    ]

    best_track_label = "Track B (24H 돌파·S-R Flip)"
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
        f"• 2파동(TP2) 종가안착 유지력: TP1 돌파 종목 중 {w2_retention:.1f}%가 2차 구간 안착 ({tp2_hits}/{max(tp1_hits, 1)}건)",
        f"• 3파동(TP3) 클라이맥스 도달률: TP2 안착 종목 중 {w3_retention:.1f}% 3차 돌파 ({tp3_hits}/{max(tp2_hits, 1)}건)",
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
            stage = (
                "3차50%·트레일링마감"
                if (c["tp3"] and c["trailing_closed"])
                else (
                    "3차50%·트레일링중"
                    if c["tp3"]
                    else (
                        "TP1방어선마감"
                        if (c["tp2"] and c["trailing_closed"])
                        else ("2차안착·TP1방어중" if c["tp2"] else ("1차돌파·전량홀딩" if c["tp1"] else "추적중"))
                    )
                )
            )
            lines.append(
                f"{idx}. {h_tag}${base} ({sym}): 진입 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (최고 +{fav_pct:.1f}% | 합산 {w_pct:+.2f}% | {stage})"
            )
    else:
        lines.append("• 집계된 주간 수익 타점 기록이 아직 없습니다.")

    lines.append(f"\n📓 [5. 주간 손절(SL) 오답노트 v3 & 25h 사후 부검 (주간 전체 SL {len(sl_eps)}건)]")
    if postmortem is not None and hasattr(postmortem, "build_daily_postmortem_summary"):
        try:
            # [P1-2 수정] 구버전 sl_history 혼용 없이 순수 v3 DB 기준 출력
            lines.extend(postmortem.build_daily_postmortem_summary())
        except Exception:
            pass
    elif sl_eps:
        for s, c in sorted(sl_eps, key=lambda x: int(x[0].get("created_at", 0)), reverse=True)[:5]:
            base = str(s.get("symbol", "")).replace("USDT", "")
            fav = c["fav"]
            sc = int(s.get("score", 88))
            tk = str(s.get("track_type", "TRACK_A"))
            act_loss = calc_weighted_realized_pct(s, c)
            cause = "돌파 직후 매수세 실종(가짜 돌파)" if fav < 1.5 else f"+{fav:.1f}% 반등 후 차익 매물 출회"
            lines.append(f"  ↳ ⚠️ ${base} ({tk}·{sc}점): 최고 +{fav:.1f}% 도달 후 {act_loss:+.2f}% 이탈 ({cause})")
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
        "• 운용 원칙: 단기(-6% SL) · 장기(-20% SL) | 1차 돌파(전량 홀딩) → 2차 종가안착(전량 홀딩 · 방어선 TP1) → 3차 돌파(50% 익절 · 방어선 TP2 & 잔여 50% 단기 -4% / 장기 -10% 종가 트레일링)."
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
