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
except Exception:
    cfg = None
    STATE_FILE = "bot_state.json"
    KST = timezone(timedelta(hours=9))
    FIXED_SL_PCT = 6

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


def send_telegram_report(text):
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
    return {"fav": fav, "tp1": tp1, "tp2": tp2, "tp3": tp3, "sl": is_sl, "trailing_closed": trailing_closed}


def calc_rsi(series, period=6):
    delta = series.diff()
    up = delta.clip(lower=0.0).ewm(com=period - 1, adjust=True, min_periods=period).mean()
    down = (-1.0 * delta.clip(upper=0.0)).ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = up / down.replace(0, np.nan)
    return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)


def analyze_top_gainer_structure(coin):
    sym = coin["symbol"]
    base = coin.get("base_coin") or sym.replace("USDT", "")
    price = to_float(coin.get("price"))
    chg_24h = to_float(coin.get("change_24h"))
    turn_m = to_float(coin.get("turnover_m"))
    oi_m = to_float(coin.get("oi_m"))
    oi_ratio = round((oi_m / max(turn_m, 0.1)) * 100.0, 1)
    fund_pct = to_float(coin.get("funding_pct"))

    df_15m = fetch_klines(sym, interval="15", limit=96)
    df_1h = fetch_klines(sym, interval="60", limit=48)

    vol_spike, rsi15, squeeze_pct = 1.5, 55.0, 3.8
    pattern_note = "수급 집중 돌파"

    if not df_15m.empty and len(df_15m) >= 25:
        closes, vols = df_15m["close"], df_15m["volume"]
        live_p = to_float(closes.iloc[-1], 0.0)
        if live_p > 0 and (price <= 0 or 0.50 <= (live_p / price) <= 2.00):
            price = live_p

        rsi15 = round(to_float(calc_rsi(closes, 6).iloc[-1], 55.0), 1)
        avg_v = to_float(vols.iloc[-25:-5].mean(), 1.0)
        max_v = to_float(vols.iloc[-12:].max(), avg_v)
        vol_spike = round(max_v / max(avg_v, 1e-8), 1)

        hi_w = to_float(df_15m["high"].iloc[-40:-15].max(), price)
        lo_w = to_float(df_15m["low"].iloc[-40:-15].min(), price * 0.96)
        if lo_w > 0:
            squeeze_pct = round((hi_w - lo_w) / lo_w * 100.0, 1)

        if squeeze_pct <= 5.0 and vol_spike >= 2.5:
            pattern_note = f"박스권 초응축({squeeze_pct:.1f}%) 후 거래량 {vol_spike:.1f}배 폭발"
        elif fund_pct < 0:
            pattern_note = f"음수 펀딩({fund_pct:+.4f}%) 숏스퀴즈 + 거래량 {vol_spike:.1f}배"
        else:
            pattern_note = f"15분봉 거래량 {vol_spike:.1f}배 유입 + 추세 가속"

    htf_trend = "1H 정배열 확산"
    if not df_1h.empty and len(df_1h) >= 20:
        ma20_1h = to_float(df_1h["close"].rolling(20).mean().iloc[-1], price)
        htf_trend = "1H 정배열 돌파·확산" if price >= ma20_1h else "1H 급등 후 중심선 눌림"

    return {
        "symbol": sym, "base": base, "price": price, "change_24h": chg_24h,
        "turnover_m": turn_m, "oi_m": oi_m, "oi_ratio": oi_ratio, "funding_pct": fund_pct,
        "vol_spike": vol_spike, "rsi15": rsi15, "squeeze_pct": squeeze_pct,
        "pattern_note": pattern_note, "htf_trend": htf_trend,
    }


def verify_48h_history(history_data, current_map, now_ts):
    records = history_data.get("records", [])
    verified_lines, updated_records, seen_syms = [], [], set()

    for rec in sorted(records, key=lambda x: int(x.get("timestamp", 0)), reverse=True):
        if not isinstance(rec, dict):
            continue
        rec_ts = int(rec.get("timestamp", 0))
        age_h = (now_ts - rec_ts) / 3600.0
        sym = str(rec.get("symbol", "")).upper()
        entry_p = to_float(rec.get("price"), 0.0)
        if age_h > 96 or age_h < 0 or not sym or entry_p <= 0 or is_stock_or_tradfi_symbol(sym):
            continue

        curr_p = to_float(current_map.get(sym, {}).get("price"), entry_p)
        max_h = max(to_float(rec.get("max_high_48h"), entry_p), curr_p)

        if 0.2 <= age_h <= 54 and sym not in seen_syms and len(seen_syms) < 6:
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
        updated_records.append(rec)

        if age_h >= 0.2 and sym not in seen_syms and len(verified_lines) < 6:
            seen_syms.add(sym)
            base = sym.replace("USDT", "")
            status = "🚀추가급등" if max_gain >= 12.0 else ("✅상승유지" if curr_gain >= 0 else "📉차익조정")
            verified_lines.append(
                f"• ${base} ({age_h:.1f}h 경과 | {status}): 기준 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (+{max_gain:.1f}%) | 현재 {curr_gain:+.1f}%"
            )

    history_data["records"] = updated_records[:50]
    return verified_lines


def summarize_scanner_episodes(now_ts):
    state = load_merged_state()
    episodes = state.get("signal_episodes") or {}
    tracked = state.get("tracked") or {}
    sl_history = state.get("sl_history") or {}

    eps_list = [e for e in episodes.values() if isinstance(e, dict) and not is_stock_or_tradfi_symbol(e.get("symbol", "")) and (now_ts - int(e.get("created_at", 0))) <= (7 * 86400)]
    sls_recent = [s for s in sl_history.values() if isinstance(s, dict) and not is_stock_or_tradfi_symbol(s.get("symbol", "")) and (now_ts - int(s.get("sl_at", 0))) <= (7 * 86400)]
    active_list = [
        (sym, t) for sym, t in tracked.items()
        if isinstance(t, dict) and not is_stock_or_tradfi_symbol(sym) and not t.get("sl_alerted") and not t.get("tp3_completed") and not t.get("closed_alerted")
    ]

    if not eps_list and not sls_recent and not active_list:
        return ["• 현재 기록된 스캐너 타점 에피소드가 없습니다."]

    parsed = [(e, classify_episode_by_mfe(e)) for e in eps_list]
    tot = len(parsed)
    tp1_c = sum(1 for _, c in parsed if c["tp1"])
    tp2_c = sum(1 for _, c in parsed if c["tp2"])
    tp3_c = sum(1 for _, c in parsed if c["tp3"])
    trail_c = sum(1 for _, c in parsed if c["trailing_closed"])
    sl_c = sum(1 for _, c in parsed if c["sl"])
    closed_c = tp2_c + sl_c
    win_r = round((tp2_c / closed_c) * 100.0, 1) if closed_c > 0 else 0.0

    lines = [
        f"• [최근 7일 누적 성적]: 총 {tot}건 포착 | 수익 확정 승률(TP2 방어선 이상) {win_r:.1f}%",
        f"• 단계별 실제 달성: 1차돌파(전량홀딩) {tp1_c}건 | 2차돌파(TP1방어) {tp2_c}건 | 3차돌파(50%익절+트레일링) {tp3_c}건 | 수익마감 {trail_c}건 | 손절 {sl_c}건",
    ]

    best_eps = sorted(parsed, key=lambda x: x[1]["fav"], reverse=True)[:4]
    for b, c in best_eps:
        sym = b.get("symbol", "").replace("USDT", "")
        h_tag = "[장기] " if (b.get("is_long_term") or b.get("track_type") == "TRACK_D") else ""
        fav = c["fav"]
        ep_p = to_float(b.get("entry_price"))
        mh_p = to_float(b.get("max_high_price"))
        tag = "3차50%익절·트레일링" if c["tp3"] else ("2차돌파·TP1방어" if c["tp2"] else ("1차돌파·전량홀딩" if c["tp1"] else "추적중"))
        if fav > 0:
            lines.append(f"  ↳ 🏆 베스트 타점: {h_tag}${sym} 진입 {fmt_price(ep_p)} → 최고 {fmt_price(mh_p)} (+{fav:.1f}% | {tag})")

    if active_list:
        lines.append(f"• 🔄 [현재 실시간 감시/보유 중 ({len(active_list)}종목)]:")
        for sym, t in sorted(active_list, key=lambda x: to_float(x[1].get("max_favorable_pct")), reverse=True)[:4]:
            base = sym.replace("USDT", "")
            h_tag = "[장기] " if (t.get("is_long_term") or t.get("track_type") == "TRACK_D") else ""
            c = classify_episode_by_mfe(t)
            st = "3차50%익절·트레일링" if c["tp3"] else ("2차돌파·TP1방어" if c["tp2"] else ("1차돌파·전량홀딩" if c["tp1"] else "초입추적중"))
            lines.append(f"  ↳ 🟢 {h_tag}${base}: 진입 {fmt_price(t.get('entry_price'))} | 최고 +{c['fav']:.1f}% ({st})")

    # postmortem.py 오답노트 자동 피드백 요약 연동
    if postmortem is not None and hasattr(postmortem, "build_daily_postmortem_summary"):
        try:
            lines.extend(postmortem.build_daily_postmortem_summary(sls_recent))
        except Exception:
            pass
    elif sls_recent:
        lines.append(f"• 📓 [손절(SL) 오답 노트 & 원인 해부 ({len(sls_recent)}건)]:")
        for s in sorted(sls_recent, key=lambda x: int(x.get("sl_at", 0)), reverse=True)[:3]:
            sym = s.get("symbol", "").replace("USDT", "")
            hold_m = int(s.get("hold_minutes", 0))
            max_f = to_float(s.get("max_favorable_pct", 0.0))
            sl_p_val = int(round(to_float(s.get("sl_pct"), 20 if s.get("track_type") == "TRACK_D" else FIXED_SL_PCT)))
            lines.append(f"  ↳ ⚠️ ${sym}: 보유 {hold_m}분 후 -{sl_p_val}% 손절 (최고 +{max_f:.1f}% 반등)")

    return lines


def main():
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
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
        f_map = {ex.submit(analyze_top_gainer_structure, c): idx for idx, c in enumerate(top5)}
        temp_list = [None] * len(top5)
        for fut in as_completed(f_map):
            idx = f_map[fut]
            try:
                temp_list[idx] = fut.result()
            except Exception:
                pass
        analyzed_top5 = [x for x in temp_list if x is not None]

    for info in analyzed_top5:
        history_data["records"].insert(0, {
            "symbol": info["symbol"], "price": info["price"], "change_24h": info["change_24h"],
            "max_high_48h": info["price"], "max_gain_pct": 0.0, "timestamp": now_ts,
        })

    scanner_lines = summarize_scanner_episodes(now_ts)
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

    report = [
        f"📊 [Bybit 선물 Top 5 일간 분석 & 48h 사후검증] ({now_str})",
        "",
        "🔥 [금일 바이비트 선물 급등 Top 5 초입 패턴 정밀 해부]",
    ]
    for idx, it in enumerate(analyzed_top5, 1):
        report.append(
            f"{idx}. ${it['base']} ({it['symbol']}) | 24h {it['change_24h']:+.1f}%\n"
            f"   • 현재가: {fmt_price(it['price'])} USDT | 거래대금 {it['turnover_m']:.1f}M | OI {it['oi_m']:.1f}M (비중 {it['oi_ratio']:.0f}%)\n"
            f"   • 수급/지표: 펀딩 {it['funding_pct']:+.4f}% | 15m RSI6 {it['rsi15']:.1f} | 응축폭 {it['squeeze_pct']:.1f}%\n"
            f"   • 초입 특징: {it['pattern_note']} ({it['htf_trend']})"
        )

    if analyzed_top5:
        avg_sq = np.mean([x["squeeze_pct"] for x in analyzed_top5])
        avg_vol = np.mean([x["vol_spike"] for x in analyzed_top5])
        neg_f = sum(1 for x in analyzed_top5 if x["funding_pct"] < 0)
        report.extend([
            "",
            "💡 [금일 급등 대장주 공통 초입 공식 요약]",
            f"• 평균 박스권 응축폭 {avg_sq:.1f}% 수렴 후 15분봉 거래량 평균 {avg_vol:.1f}배 폭발",
            f"• Top 5 중 음수 펀딩비(숏스퀴즈 연료) 종목 비중: {neg_f}/5종목",
        ])

    report.append("\n⏱️ [48시간 사후검증 추적 결과 (이전 급등주 사후 흐름)]")
    if verified_48h_lines:
        report.extend(verified_48h_lines)
    else:
        for it in analyzed_top5[:3]:
            report.append(f"• ${it['base']} (신규 추적 등록): 기준가 {fmt_price(it['price'])} USDT (24h {it['change_24h']:+.1f}%)")

    report.append("\n🎯 [스캐너 실전 타점 성적표 & 오답노트 해부]")
    report.extend(scanner_lines)
    report.append(f"\n⚙️ [자율 학습 기준 파라미터]: TP1 +{p1_l:.1f}% / TP2 +{p2_l:.1f}% / TP3 +{p3_l:.1f}% (실전 타점은 개별 차트 저항선 적용)")

    full_msg = "\n".join(report)
    print(full_msg)
    send_telegram_report(full_msg)


if __name__ == "__main__":
    main()
# === END OF FILE ===
