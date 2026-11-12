import os
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import config as cfg
from config import STATE_FILE, KST, to_float, is_tradfi_ticker
from api import get_scanner_candidates, fetch_verified_1m_klines
from analyzer import analyze_symbol
from notifier import build_briefing_message, build_x_post_message, send_telegram_single, fmt_price

# 일반 종목 재알림 쿨타임 (12시간 = 43,200초)
RE_ALERT_COOLDOWN_SEC = 43200
# 스탑헌팅 손절 후 V자 탈환(88점 이상) 재진입 허용 쿨타임 (4시간 = 14,400초)
SL_RECOVERY_COOLDOWN_SEC = getattr(cfg, "SL_RECOVERY_COOLDOWN_SEC", 14400)
# 오답 노트(손절 기록) 및 성과 보관 주기 (72시간 = 259,200초)
HISTORY_RETENTION_SEC = 259200


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    if not isinstance(data.get("tracked"), dict):
                        data["tracked"] = {}
                    if not isinstance(data.get("alert_history"), dict):
                        data["alert_history"] = {}
                    if not isinstance(data.get("sl_history"), dict):
                        data["sl_history"] = {}
                    return data
        except Exception:
            pass
    return {"tracked": {}, "alert_history": {}, "sl_history": {}}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[상태 저장 오류] {e}")


def check_tracked_sl_and_tp_1m(symbol, t_item, sl_history, now_ts):
    stop_p = to_float(t_item.get("stop_price"))
    entry_low = to_float(t_item.get("entry_low"))
    entry_high = to_float(t_item.get("entry_high"))
    entry_ref = to_float(t_item.get("entry_price"))
    if entry_ref <= 0:
        entry_ref = (entry_low + entry_high) / 2.0 if (entry_low > 0 and entry_high > 0) else max(entry_low, entry_high)
    created_at = int(t_item.get("created_at", now_ts))
    created_ms = created_at * 1000

    if stop_p <= 0 or entry_ref <= 0:
        return False

    df_1m = fetch_verified_1m_klines(symbol, entry_ref=entry_ref, limit=25)
    if df_1m.empty:
        return False

    post_1m = df_1m[df_1m["timestamp"] >= created_ms].copy()
    if post_1m.empty:
        return False

    base_coin = symbol.replace("USDT", "")
    window_high = to_float(post_1m["high"].max(), default=entry_ref)
    prev_max_high = to_float(t_item.get("max_high_price"), default=entry_ref)
    max_high = max(prev_max_high, window_high, entry_ref)
    max_fav_pct = round((max_high - entry_ref) / max(entry_ref, 1e-8) * 100.0, 2)
    t_item["max_high_price"] = max_high
    t_item["max_favorable_pct"] = max_fav_pct

    sent_event = False
    tps_list = t_item.get("tps") or []

    # 1. 3차 목표가(TP3) 전량 익절 졸업 확인
    if len(tps_list) >= 3 and to_float(tps_list[2]) > 0 and max_high >= to_float(tps_list[2]):
        t_item["tp1_hit"] = True
        t_item["tp2_hit"] = True
        t_item["tp3_completed"] = True
        t_item["tp3_at"] = now_ts
        if not t_item.get("tp3_alerted"):
            t_item["tp3_alerted"] = True
            tp3_msg = (
                f"🎯 [3차 최종 목표가(TP3) 전량 익절 졸업] ${base_coin} ({symbol})\n"
                f"• 진입 기준가: {fmt_price(entry_ref)} → 도달 최고가: {fmt_price(max_high)} (+{max_fav_pct:.1f}%)\n"
                f"• 3단계 피보나치 목표가를 모두 달성하여 추적을 성공적으로 종료합니다."
            )
            send_telegram_single(tp3_msg)
        return True

    # 2. 2차 목표가(TP2) 달성 확인 및 익절선 상향(+8% 보장)
    if len(tps_list) >= 2 and to_float(tps_list[1]) > 0 and max_high >= to_float(tps_list[1]):
        t_item["tp1_hit"] = True
        t_item["tp2_hit"] = True
        new_stop = max(stop_p, entry_ref * 1.080)
        t_item["stop_price"] = new_stop
        stop_p = new_stop
        if not t_item.get("tp2_alerted"):
            t_item["tp2_alerted"] = True
            tp2_msg = (
                f"✅ [2차 목표가(TP2) 돌파] ${base_coin} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav_pct:.1f}%)\n"
                f"• 분할 익절 권장 및 수익 보호선을 {fmt_price(stop_p)} (+8.0%) 위로 상향 고정합니다."
            )
            send_telegram_single(tp2_msg)
            sent_event = True

    # 3. 1차 목표가(TP1) 달성 확인 및 익절선 상향(+4% 보장)
    elif len(tps_list) >= 1 and to_float(tps_list[0]) > 0 and max_high >= to_float(tps_list[0]):
        t_item["tp1_hit"] = True
        new_stop = max(stop_p, entry_ref * 1.040)
        t_item["stop_price"] = new_stop
        stop_p = new_stop
        if not t_item.get("tp1_alerted"):
            t_item["tp1_alerted"] = True
            tp1_msg = (
                f"✅ [1차 목표가(TP1) 달성] ${base_coin} ({symbol})\n"
                f"• 진입가 {fmt_price(entry_ref)} → 현재 최고 {fmt_price(max_high)} (+{max_fav_pct:.1f}%)\n"
                f"• 1차 분할 익절 권장 및 손절선을 수익 보존선({fmt_price(stop_p)}, +4.0%)으로 상향합니다."
            )
            send_telegram_single(tp1_msg)
            sent_event = True

    # 4. [오답 노트 $XCN(+7.9% 후 반락) 처방] +4.5% 도달 시 본전 방어(Breakeven) 스탑 자동 활성화
    be_trigger = getattr(cfg, "BREAKEVEN_TRIGGER_PCT", 4.5)
    if max_fav_pct >= be_trigger and not t_item.get("be_armed") and not t_item.get("tp1_hit"):
        t_item["be_armed"] = True
        be_stop = max(stop_p, entry_ref * 1.003)
        t_item["stop_price"] = be_stop
        stop_p = be_stop
        be_msg = (
            f"🛡️ [본전 방어(Breakeven) 자동 발동] ${base_coin} ({symbol})\n"
            f"• 진입 후 최고 +{max_fav_pct:.1f}% 수익권 도달 확인 ({fmt_price(max_high)} USDT)\n"
            f"• 차익 매물 반락에 대비해 손절선을 진입가 위({fmt_price(stop_p)} USDT)로 상향 고정합니다."
        )
        send_telegram_single(be_msg)
        sent_event = True

    # 5. 1분봉 기준 손절선(또는 본전/익절 보호선) 이탈 확인
    min_1m_low = to_float(post_1m["low"].min(), default=entry_ref)
    min_1m_close = to_float(post_1m["close"].min(), default=entry_ref)
    last_1m_close = to_float(post_1m["close"].iloc[-1], default=entry_ref)

    if min_1m_low <= stop_p and (min_1m_close <= stop_p or last_1m_close <= stop_p):
        trigger_price = min(min_1m_close, last_1m_close)
        hold_min = max(1, int((now_ts - created_at) // 60))

        # 이미 TP1을 찍었거나 본전 방어(be_armed)가 켜진 상태에서 내려온 경우 -> 손실이 아닌 수익/본전 보호 마감
        if t_item.get("tp1_hit") or t_item.get("be_armed"):
            t_item["tp3_completed"] = True
            t_item["tp3_at"] = now_ts
            protect_msg = (
                f"🔒 [수익/본전 보호선 마감 완료] ${base_coin} ({symbol})\n"
                f"• 진입 후 최고 +{max_fav_pct:.1f}% 도달 후 보호선({fmt_price(stop_p)}) 터치로 안전 마감합니다."
            )
            send_telegram_single(protect_msg)
            return True

        t_item["sl_alerted"] = True
        t_item["sl_at"] = now_ts
        t_item["sl_trigger_price"] = trigger_price
        t_item["hold_minutes"] = hold_min

        sl_key = f"{symbol}_{created_at}"
        sl_history[sl_key] = {
            "symbol": symbol,
            "track_type": t_item.get("track_type", "TRACK_A"),
            "score": int(t_item.get("score", 85)),
            "entry_price": entry_ref,
            "stop_price": stop_p,
            "sl_trigger_price": trigger_price,
            "sl_pct": to_float(t_item.get("sl_pct", 3)),
            "created_at": created_at,
            "sl_at": now_ts,
            "hold_minutes": hold_min,
            "max_high_price": max_high,
            "max_favorable_pct": max_fav_pct,
            "tp1_hit_before_sl": False,
            "vol_mult": to_float(t_item.get("vol_mult", 1.0)),
            "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
            "kdj_j_15m": to_float(t_item.get("kdj_j_15m", 50.0)),
            "oi_m": to_float(t_item.get("oi_m", 0.0)),
            "turnover_m": to_float(t_item.get("turnover_m", 0.0)),
            "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
            "reasons": t_item.get("reasons", []),
        }

        sl_msg = (
            f"⚠️ [손절 기준가 이탈] ${base_coin} ({symbol})\n"
            f"• 이탈 확인가: {fmt_price(trigger_price)} USDT (전략적 손절선 {fmt_price(stop_p)} 1분봉 이탈 확인)\n"
            f"• 진입 후 최고 상승폭: +{max_fav_pct:.1f}% (보유 {hold_min}분)\n"
            f"• 리스크 관리 원칙 준수를 권장합니다. (일일 리포트 오답 노트에 자동 기록됨)"
        )
        send_telegram_single(sl_msg)
        return True

    return sent_event


def main():
    start_t = time.time()
    state = load_state()
    tracked = state.get("tracked", {})
    alert_history = state.get("alert_history", {})
    sl_history = state.get("sl_history", {})
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    candidates, pure_crypto_bases, rwa_blocked_bases = get_scanner_candidates()
    sent_count = 0

    for h_sym in list(alert_history.keys()):
        if now_ts - int(alert_history.get(h_sym, 0)) > 86400:
            del alert_history[h_sym]

    for sl_k in list(sl_history.keys()):
        sl_item = sl_history.get(sl_k, {})
        if now_ts - int(sl_item.get("sl_at", 0)) > HISTORY_RETENTION_SEC:
            del sl_history[sl_k]

    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        t_item = tracked[sym]
        created_at = int(t_item.get("created_at", 0))
        age_sec = now_ts - created_at
        is_sl_done = bool(t_item.get("sl_alerted", False))
        is_tp3_done = bool(t_item.get("tp3_completed", False))

        if sym not in alert_history and created_at > 0:
            alert_history[sym] = created_at

        if is_tradfi_ticker(base_c) or base_c in rwa_blocked_bases:
            del tracked[sym]
            continue

        if is_sl_done and (now_ts - int(t_item.get("sl_at", created_at)) > 172800):
            del tracked[sym]
            continue
        elif is_tp3_done and (now_ts - int(t_item.get("tp3_at", created_at)) > 172800):
            del tracked[sym]
            continue
        elif age_sec > 604800:
            del tracked[sym]
            continue

        if not is_sl_done and not is_tp3_done:
            if check_tracked_sl_and_tp_1m(sym, t_item, sl_history, now_ts):
                sent_count += 1

    if not candidates:
        print("[알림] 조건을 만족하는 순수 암호화폐 후보군 데이터를 가져오지 못했습니다.")
        state["tracked"] = tracked
        state["alert_history"] = alert_history
        state["sl_history"] = sl_history
        save_state(state)
        return

    # 이미 활성 추적 중인 종목은 제외하되, 스탑헌팅 손절 후 4시간 경과한 종목은 V자 탈환 검사를 위해 후보에 포함
    to_analyze = []
    for coin in candidates:
        sym = coin["symbol"]
        last_alert_ts = int(alert_history.get(sym, 0))
        prev_t = tracked.get(sym)

        if prev_t and not prev_t.get("sl_alerted") and not prev_t.get("tp3_completed"):
            continue

        was_stopped = bool(prev_t and prev_t.get("sl_alerted"))
        cooldown = SL_RECOVERY_COOLDOWN_SEC if was_stopped else RE_ALERT_COOLDOWN_SEC
        if (now_ts - last_alert_ts) < cooldown:
            continue

        to_analyze.append(coin)

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_map = {executor.submit(analyze_symbol, c): c["symbol"] for c in to_analyze}
        for future in as_completed(future_map):
            sym = future_map[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[스캔 예외] {sym}: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)

    for res in results:
        symbol = res["symbol"]
        last_alert_ts = int(alert_history.get(symbol, 0))
        prev_t = tracked.get(symbol)
        was_stopped = bool(prev_t and prev_t.get("sl_alerted"))

        # 스탑헌팅 손절 후 4~12시간 사이 재진입일 경우: 88점 이상 & 이전 진입가 탈환 시에만 특별 허용 ($Q 패턴)
        if was_stopped and (now_ts - last_alert_ts) < RE_ALERT_COOLDOWN_SEC:
            prev_entry_p = to_float(prev_t.get("entry_price"), 0.0)
            if res["score"] < 88 or (prev_entry_p > 0 and res["price"] < prev_entry_p * 0.998):
                continue
            res["track_title"] = "🔄 [스탑헌팅 후 V자 탈환 재진입 타점]"
            res["reasons"].insert(0, "스탑헌팅 바닥 확인 후 진입가 V자 재탈환")
            res["reason_str"] = " / ".join(res["reasons"])

        tracked[symbol] = {
            "created_at": now_ts,
            "track_type": res["track_type"],
            "score": res["score"],
            "entry_price": res["price"],
            "swing_high_line": res["swing_high_line"],
            "entry_low": res["entry_low"],
            "entry_high": res["entry_high"],
            "stop_price": res["stop_price"],
            "sl_pct": res["sl_pct"],
            "tps": res["tps"],
            "tp_pcts": res["tp_pcts"],
            "vol_mult": res.get("vol_mult", 1.0),
            "rsi6_15m": res.get("rsi6_15m", 50.0),
            "kdj_j_15m": res.get("kdj_j_15m", 50.0),
            "oi_m": res.get("oi_m", 0.0),
            "turnover_m": res.get("turnover_m", 0.0),
            "funding_pct": res.get("funding_pct", 0.0),
            "reasons": res.get("reasons", []),
            "max_high_price": res["price"],
            "max_favorable_pct": 0.0,
            "be_armed": False,
            "tp1_hit": False,
            "tp2_hit": False,
            "tp1_alerted": False,
            "tp2_alerted": False,
            "tp3_alerted": False,
            "breakout_alerted": (res["track_type"] == "TRACK_B"),
            "sl_alerted": False,
            "tp3_completed": False,
        }
        alert_history[symbol] = now_ts

        briefing_msg = build_briefing_message(res, now_str)
        send_telegram_single(briefing_msg)

        x_msg = build_x_post_message(res)
        send_telegram_single(x_msg)

        sent_count += 1

    state["tracked"] = tracked
    state["alert_history"] = alert_history
    state["sl_history"] = sl_history
    save_state(state)

    elapsed = time.time() - start_t
    if sent_count == 0:
        print(f"[{now_str}] 멀티프레임 스캔 완료 ({elapsed:.1f}초 소요) - 85점 이상 신규 초입 종목 없음.")
    else:
        print(f"[{now_str}] 멀티프레임 스캔 완료 ({elapsed:.1f}초 소요) - 총 {sent_count}건 알림 전송.")


if __name__ == "__main__":
    main()
