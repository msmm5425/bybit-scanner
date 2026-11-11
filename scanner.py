import os
import json
import time
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import STATE_FILE, KST, to_float, is_tradfi_ticker, strip_multiplier_prefix
from api import get_scanner_candidates, fetch_verified_1m_klines, get_fast_pure_crypto_whitelist
from analyzer import analyze_symbol
from notifier import build_briefing_message, build_x_post_message, send_telegram_single, fmt_price


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
    return {"tracked": {}}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[상태 저장 오류] {e}")


def check_tracked_sl_and_tp_1m(symbol, t_item, now_ts):
    stop_p = to_float(t_item.get("stop_price"))
    entry_low = to_float(t_item.get("entry_low"))
    entry_high = to_float(t_item.get("entry_high"))
    entry_ref = (entry_low + entry_high) / 2.0 if (entry_low > 0 and entry_high > 0) else max(entry_low, entry_high)
    created_ms = int(t_item.get("created_at", 0)) * 1000

    if stop_p <= 0 or entry_ref <= 0:
        return False

    df_1m = fetch_verified_1m_klines(symbol, entry_ref=entry_ref, limit=25)
    if df_1m.empty:
        return False

    post_1m = df_1m[df_1m["timestamp"] >= created_ms].copy()
    if post_1m.empty:
        return False

    tps_list = t_item.get("tps") or []
    if len(tps_list) >= 3 and to_float(tps_list[2]) > 0:
        if to_float(post_1m["high"].max()) >= to_float(tps_list[2]):
            t_item["tp3_completed"] = True
            t_item["tp3_at"] = now_ts
            return False

    min_1m_low = to_float(post_1m["low"].min(), default=entry_ref)
    min_1m_close = to_float(post_1m["close"].min(), default=entry_ref)
    last_1m_close = to_float(post_1m["close"].iloc[-1], default=entry_ref)

    if min_1m_low <= stop_p and (min_1m_close <= stop_p or last_1m_close <= stop_p):
        trigger_price = min(min_1m_close, last_1m_close)
        t_item["sl_alerted"] = True
        t_item["sl_at"] = now_ts
        sl_msg = (
            f"⚠️ [손절 기준가 이탈] ${symbol.replace('USDT', '')} ({symbol})\n"
            f"• 이탈 확인가: {fmt_price(trigger_price)} USDT (전략적 손절선 {fmt_price(stop_p)} 1분봉 이탈 확인)\n"
            f"• 리스크 관리 원칙 준수를 권장합니다."
        )
        send_telegram_single(sl_msg)
        return True

    return False


def main():
    start_t = time.time()
    state = load_state()
    tracked = state.get("tracked", {})
    now_ts = int(time.time())
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    candidates, pure_crypto_bases, rwa_blocked_bases = get_fast_pure_crypto_whitelist()
    sent_count = 0

    for sym in list(tracked.keys()):
        base_c = sym[:-4].upper() if sym.endswith("USDT") else sym
        t_item = tracked[sym]
        age_sec = now_ts - int(t_item.get("created_at", 0))
        is_sl_done = bool(t_item.get("sl_alerted", False))
        is_tp3_done = bool(t_item.get("tp3_completed", False))

        if (
            is_tradfi_ticker(base_c)
            or base_c in rwa_blocked_bases
            or (pure_crypto_bases and base_c not in pure_crypto_bases and strip_multiplier_prefix(base_c) not in pure_crypto_bases)
        ):
            del tracked[sym]
            continue

        if is_sl_done and (now_ts - int(t_item.get("sl_at", t_item.get("created_at", 0))) > 21600):
            del tracked[sym]
            continue
        elif is_tp3_done and (now_ts - int(t_item.get("tp3_at", t_item.get("created_at", 0))) > 86400):
            del tracked[sym]
            continue
        elif age_sec > 604800:
            del tracked[sym]
            continue

        if not is_sl_done and not is_tp3_done:
            if check_tracked_sl_and_tp_1m(sym, t_item, now_ts):
                sent_count += 1

    if not candidates:
        print("[알림] 조건을 만족하는 순수 암호화폐 후보군 데이터를 가져오지 못했습니다.")
        state["tracked"] = tracked
        save_state(state)
        return

    to_analyze = [coin for coin in candidates if coin["symbol"] not in tracked]

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
        if symbol in tracked and not tracked[symbol].get("sl_alerted", False):
            continue

        tracked[symbol] = {
            "created_at": now_ts,
            "track_type": res["track_type"],
            "swing_high_line": res["swing_high_line"],
            "entry_low": res["entry_low"],
            "entry_high": res["entry_high"],
            "stop_price": res["stop_price"],
            "sl_pct": res["sl_pct"],
            "tps": res["tps"],
            "tp_pcts": res["tp_pcts"],
            "breakout_alerted": (res["track_type"] == "TRACK_B"),
            "sl_alerted": False,
            "tp3_completed": False
        }

        briefing_msg = build_briefing_message(res, now_str)
        send_telegram_single(briefing_msg)

        x_msg = build_x_post_message(res)
        send_telegram_single(x_msg)

        sent_count += 1

    state["tracked"] = tracked
    save_state(state)

    elapsed = time.time() - start_t
    if sent_count == 0:
        print(f"[{now_str}] 멀티프레임 스캔 완료 ({elapsed:.1f}초 소요) - 85점 이상 신규 초입 종목 없음.")
    else:
        print(f"[{now_str}] 멀티프레임 스캔 완료 ({elapsed:.1f}초 소요) - 총 {sent_count}건 알림 전송.")


if __name__ == "__main__":
    main()
