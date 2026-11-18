import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
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

from api import (
    get_scanner_candidates,
    fetch_klines,
    get_valid_bybit_linear_symbols,
    clean_url,
)

SCHEME = "ht" + "tps://"
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


def send_telegram_report(text):
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
        print("[경고] 텔레그램 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        return False

    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendMessage")
    chunks = [text[i : i + 3800] for i in range(0, len(text), 3800)]

    for chunk in chunks:
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": True,
        }
        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=15)
                if resp.status_code == 200:
                    time.sleep(0.6)
                    break
                if resp.status_code == 429:
                    wait_s = int(resp.json().get("parameters", {}).get("retry_after", 3))
                    time.sleep(wait_s + 0.5)
            except Exception as e:
                print(f"[텔레그램 전송 재시도 {attempt + 1}/3] {e}")
            time.sleep(1.0)
    return True


def load_json_safe(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, type(default_val)) else default_val
        except Exception as e:
            print(f"[파일 로드 경고] {path}: {e}")
    return default_val


def save_json_safe(path, data):
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except Exception as e:
        print(f"[파일 저장 오류] {path}: {e}")


def calc_rsi(series, period=6):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -1.0 * delta.clip(upper=0.0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def analyze_top_gainer_structure(coin):
    symbol = coin["symbol"]
    base = coin.get("base_coin") or symbol.replace("USDT", "")
    price = to_float(coin.get("price"))
    chg_24h = to_float(coin.get("change_24h"))
    turn_m = to_float(coin.get("turnover_m"))
    oi_m = to_float(coin.get("oi_m"))
    fund_pct = to_float(coin.get("funding_pct"))

    df_15m = fetch_klines(symbol, interval="15", limit=100)
    df_1h = fetch_klines(symbol, interval="60", limit=60)

    vol_spike = 1.0
    rsi15 = 50.0
    pre_squeeze_pct = 4.5
    pattern_note = "수급 집중 돌파"

    if not df_15m.empty and len(df_15m) >= 25:
        closes = df_15m["close"]
        vols = df_15m["volume"]
        rsi_s = calc_rsi(closes, 6)
        rsi15 = to_float(rsi_s.iloc[-1], 50.0)

        if abs(chg_24h) < 0.05 and len(df_15m) >= 90:
            open_24h = to_float(df_15m["open"].iloc[-min(len(df_15m), 96)], price)
            if open_24h > 0:
                chg_24h = round(((price - open_24h) / open_24h) * 100.0, 2)

        avg_vol = to_float(vols.iloc[-25:-5].mean(), 1.0)
        max_recent_vol = to_float(vols.iloc[-12:].max(), avg_vol)
        if avg_vol > 0:
            vol_spike = round(max_recent_vol / avg_vol, 1)

        hi_window = to_float(df_15m["high"].iloc[-40:-15].max(), price)
        lo_window = to_float(df_15m["low"].iloc[-40:-15].min(), price * 0.95)
        if lo_window > 0:
            pre_squeeze_pct = round((hi_window - lo_window) / lo_window * 100.0, 1)

        if pre_squeeze_pct <= 5.0 and vol_spike >= 2.5:
            pattern_note = f"박스권 초응축({pre_squeeze_pct:.1f}%) 후 거래량 {vol_spike:.1f}배 폭발"
        elif fund_pct < 0:
            pattern_note = f"음수 펀딩({fund_pct:+.3f}%) 숏스퀴즈 + 거래량 {vol_spike:.1f}배"
        else:
            pattern_note = f"15분봉 거래량 {vol_spike:.1f}배 유입 + 추세 가속"

    htf_trend = "1H 정배열 확산"
    if not df_1h.empty and len(df_1h) >= 20:
        ma20_1h = to_float(df_1h["close"].rolling(20).mean().iloc[-1], price)
        if price < ma20_1h:
            htf_trend = "1H 급등 후 눌림"

    return {
        "symbol": symbol,
        "base": base,
        "price": price,
        "change_24h": chg_24h,
        "turnover_m": turn_m,
        "oi_m": oi_m,
        "funding_pct": fund_pct,
        "vol_spike": vol_spike,
        "rsi15": rsi15,
        "pattern_note": pattern_note,
        "htf_trend": htf_trend,
    }


def verify_48h_history(history_data, current_map, now_ts):
    records = history_data.get("records", [])
    if not isinstance(records, list):
        records = []

    verified_lines = []
    updated_records = []

    for rec in records:
        if not isinstance(rec, dict):
            continue
        rec_ts = int(rec.get("timestamp", 0))
        age_hours = (now_ts - rec_ts) / 3600.0
        if age_hours > 168:
            continue

        sym = str(rec.get("symbol", "")).upper()
        entry_p = to_float(rec.get("price"), 0.0)
        if not sym or entry_p <= 0:
            continue

        curr_p = to_float(current_map.get(sym, {}).get("price"), 0.0)
        max_h = to_float(rec.get("max_high_48h"), entry_p)

        if age_hours <= 52:
            df_1h = fetch_klines(sym, interval="60", limit=50)
            if not df_1h.empty and "timestamp" in df_1h.columns:
                post_df = df_1h[df_1h["timestamp"] >= rec_ts * 1000]
                if not post_df.empty:
                    max_h = max(max_h, to_float(post_df["high"].max(), entry_p))
            if curr_p > max_h:
                max_h = curr_p
            rec["max_high_48h"] = max_h

        max_gain_pct = round((max_h - entry_p) / entry_p * 100.0, 1)
        curr_gain_pct = round((curr_p - entry_p) / entry_p * 100.0, 1) if curr_p > 0 else 0.0
        rec["max_gain_pct"] = max_gain_pct

        if 18 <= age_hours <= 54 and not rec.get("verified_reported"):
            rec["verified_reported"] = True
            base = sym.replace("USDT", "")
            verified_lines.append(
                f"• ${base} (포착 후 {int(age_hours)}h 경과): 기준가 {fmt_price(entry_p)} → 최고 {fmt_price(max_h)} (+{max_gain_pct:.1f}%) | 현재 {curr_gain_pct:+.1f}%"
            )

        updated_records.append(rec)

    history_data["records"] = updated_records
    return verified_lines


def summarize_24h_scanner_episodes(now_ts):
    state = load_json_safe(STATE_FILE, {})
    episodes = state.get("signal_episodes", {})
    sl_history = state.get("sl_history", {})

    recent_eps = []
    for ep in episodes.values():
        if isinstance(ep, dict) and (now_ts - int(ep.get("created_at", 0))) <= 86400:
            recent_eps.append(ep)

    recent_sls = []
    for sl in sl_history.values():
        if isinstance(sl, dict) and (now_ts - int(sl.get("sl_at", 0))) <= 86400:
            recent_sls.append(sl)

    if not recent_eps and not recent_sls:
        return ["• 최근 24시간 내 신규 마감/진행 중인 스캐너 에피소드가 없습니다."]

    total_cnt = len(recent_eps)
    tp3_cnt = sum(1 for e in recent_eps if e.get("tp3_alerted"))
    tp2_cnt = sum(1 for e in recent_eps if e.get("tp2_hit"))
    tp1_cnt = sum(1 for e in recent_eps if e.get("tp1_hit"))
    be_cnt = sum(1 for e in recent_eps if e.get("be_armed") and not e.get("tp1_hit") and not e.get("sl_alerted"))
    sl_cnt = sum(1 for e in recent_eps if e.get("sl_alerted"))

    lines = [
        f"• 24h 포착 종목: 총 {total_cnt}건 (1차익절 이상 {tp1_cnt}건 | 2차익절 {tp2_cnt}건 | 3차익절 {tp3_cnt}건 | 본전방어 {be_cnt}건 | 손절 {sl_cnt}건)"
    ]

    best_eps = sorted(recent_eps, key=lambda x: to_float(x.get("max_favorable_pct")), reverse=True)[:3]
    for b in best_eps:
        sym = b.get("symbol", "")
        base = sym.replace("USDT", "")
        fav = to_float(b.get("max_favorable_pct"))
        if fav >= 3.0:
            lines.append(f"  ↳ 🏆 우수 타점: ${base} 최고 +{fav:.1f}% 도달")

    if recent_sls:
        lines.append(f"• 📓 [손절 오답 노트 ({len(recent_sls)}건)]:")
        for s in recent_sls[:3]:
            sym = s.get("symbol", "").replace("USDT", "")
            hold_m = int(s.get("hold_minutes", 0))
            max_f = to_float(s.get("max_favorable_pct", 0.0))
            lines.append(f"  ↳ ⚠️ ${sym}: 보유 {hold_m}분 후 -{FIXED_SL_PCT}% 이탈 (최고 반등 +{max_f:.1f}%)")

    return lines


def main():
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    candidates, _, _ = get_scanner_candidates()
    valid_syms, _ = get_valid_bybit_linear_symbols()

    if valid_syms:
        candidates = [c for c in candidates if c.get("symbol") in valid_syms]

    history_data = load_json_safe(HISTORY_FILE, {"records": [], "learned_tp_params": {}})

    if not candidates:
        print("[경고] 현재 시점 후보군 조회 결과가 비어 있어 기존 상태 기록 기반으로 리포트를 생성합니다.")
        scanner_lines = summarize_24h_scanner_episodes(now_ts)
        fallback_msg = "\n".join([
            f"📊 [Bybit 선물 일일 리포트 & 스캐너 점검] ({now_str})",
            "",
            "📈 [최근 24시간 스캐너 타점 성적표]",
            *scanner_lines,
        ])
        send_telegram_report(fallback_msg)
        save_json_safe(HISTORY_FILE, history_data)
        return

    current_map = {c["symbol"]: c for c in candidates}
    top5 = sorted(candidates, key=lambda x: to_float(x.get("change_24h")), reverse=True)[:5]

    analyzed_top5 = []
    for c in top5:
        info = analyze_top_gainer_structure(c)
        analyzed_top5.append(info)
        history_data.setdefault("records", []).append({
            "symbol": info["symbol"],
            "price": info["price"],
            "change_24h": info["change_24h"],
            "max_high_48h": info["price"],
            "max_gain_pct": 0.0,
            "timestamp": now_ts,
            "verified_reported": False,
        })

    verified_48h_lines = verify_48h_history(history_data, current_map, now_ts)
    scanner_24h_lines = summarize_24h_scanner_episodes(now_ts)

    gains = [to_float(x.get("max_gain_pct")) for x in history_data.get("records", []) if to_float(x.get("max_gain_pct")) >= 6.0]
    if len(gains) >= 5:
        p1_learn = float(np.clip(np.percentile(gains, 35), 7.5, 11.0))
        p2_learn = float(np.clip(np.percentile(gains, 65), 14.0, 20.0))
        p3_learn = float(np.clip(np.percentile(gains, 85), 22.0, 32.0))
    else:
        p1_learn, p2_learn, p3_learn = 8.0, 16.0, 24.0

    history_data["learned_tp_params"] = {
        "tp1_min_pct": round(p1_learn, 1),
        "tp2_min_pct": round(p2_learn, 1),
        "tp3_min_pct": round(p3_learn, 1),
        "updated_at": now_ts,
    }
    save_json_safe(HISTORY_FILE, history_data)

    report_lines = [
        f"📊 [Bybit 선물 Top 5 일간 분석 & 48h 사후검증] ({now_str})",
        "",
        "🔥 [금일 바이비트 선물 급등 Top 5 초입 패턴 해부]",
    ]

    for idx, item in enumerate(analyzed_top5, 1):
        report_lines.append(
            f"{idx}. ${item['base']} ({item['symbol']}) | 24h {item['change_24h']:+.1f}%\n"
            f"   • 현재가: {fmt_price(item['price'])} USDT | 거래대금 {item['turnover_m']:.1f}M | OI {item['oi_m']:.1f}M\n"
            f"   • 초입 특징: {item['pattern_note']} ({item['htf_trend']})"
        )

    report_lines.append("\n⏱️ [48시간 사후검증 추적 결과]")
    if verified_48h_lines:
        report_lines.extend(verified_48h_lines[:6])
    else:
        report_lines.append("• 현재 48시간 사후검증 대상 종목들을 추적 중입니다.")

    report_lines.append("\n🎯 [최근 24시간 스캐너 실전 타점 성적 & 오답노트]")
    report_lines.extend(scanner_24h_lines)

    report_lines.append(
        f"\n⚙️ [자율 학습 익절 파라미터]: TP1 +{p1_learn:.1f}% / TP2 +{p2_learn:.1f}% / TP3 +{p3_learn:.1f}%"
    )

    full_msg = "\n".join(report_lines)
    print(full_msg)
    send_telegram_report(full_msg)


if __name__ == "__main__":
    main()
