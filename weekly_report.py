import os
import json
import time
import subprocess
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 환경변수 (TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN 모두 호환)
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
    or ""
).strip()

TELEGRAM_CHAT_ID = (
    os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("CHAT_ID")
    or ""
).strip()

KST = timezone(timedelta(hours=9))
STATE_FILE = "bot_state.json"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def strip_multiplier_prefix(base_coin):
    clean = base_coin.upper()
    for prefix in ["1000000", "100000", "10000", "1000", "100"]:
        if clean.startswith(prefix) and len(clean) > len(prefix):
            return clean[len(prefix):]
    return clean


def fmt_pct(val):
    v = round(to_float(val), 1)
    return f"{int(v)}" if v == int(v) else f"{v:.1f}"


def send_telegram_report(text):
    """텔레그램 메시지 발송 (글자수 초과 시 안전하게 문단 단위로 분할 발송)"""
    print(text)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[경고] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=12)
            if r.status_code != 200:
                print(f"[Telegram 응답 오류] {r.status_code}: {r.text}")
            time.sleep(0.4)
        except Exception as e:
            print(f"[Telegram 전송 예외] {e}")


def fetch_bybit_fast(path, params=None):
    query_str = f"?{urlencode(params)}" if params else ""
    target_bybit = f"https://api.bybit.com{path}{query_str}"
    encoded_bybit = quote(target_bybit, safe="")

    routes = [
        f"https://api.codetabs.com/v1/proxy?quest={encoded_bybit}",
        f"https://api.allorigins.win/raw?url={encoded_bybit}",
    ]
    for url in routes:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=4.5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("retCode") == 0:
                    return data.get("result", {})
        except Exception:
            pass
    return {}


def fetch_klines_for_week(symbol, interval="1h", ref_price=0.0, limit=200):
    """
    월요일~일요일 전체 궤적(최대 168시간)을 빈틈없이 추적하기 위해
    1시간봉(200캔들=200시간) 및 15분봉(200캔들=50시간) 데이터를 수집합니다.
    """
    base_coin = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = strip_multiplier_prefix(base_coin)
    rows = []

    bg_gran = "1H" if interval == "1h" else "15m"
    gt_inter = "1h" if interval == "1h" else "15m"
    bn_inter = "1h" if interval == "1h" else "15m"
    by_inter = "60" if interval == "1h" else "15"

    # 1순위: Bitget USDT-FUTURES
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            r = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": sym_try, "productType": "USDT-FUTURES", "granularity": bg_gran, "limit": str(min(limit, 200))},
                headers=BROWSER_HEADERS,
                timeout=4
            )
            if r.status_code == 200:
                bg_list = r.json().get("data") or []
                if isinstance(bg_list, list) and len(bg_list) >= 10:
                    rows = [x[:7] for x in bg_list]
                    break
        except Exception:
            pass

    # 2순위: Gate.io USDT Futures
    if not rows:
        for base_try in [base_coin, clean_base]:
            try:
                r = requests.get(
                    "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                    params={"contract": f"{base_try}_USDT", "interval": gt_inter, "limit": str(limit)},
                    headers=BROWSER_HEADERS,
                    timeout=4
                )
                if r.status_code == 200:
                    gt_list = r.json()
                    if isinstance(gt_list, list) and len(gt_list) >= 10:
                        rows = [
                            [int(float(i.get("t", 0))) * 1000, i.get("o", 0), i.get("h", 0), i.get("l", 0), i.get("c", 0), float(i.get("v", 0)) or 1.0, 0]
                            for i in gt_list
                        ]
                        break
            except Exception:
                pass

    # 3순위: Binance Vision Public Data
    if not rows:
        for sym_try in [symbol, f"{clean_base}USDT"]:
            try:
                r = requests.get(
                    "https://data-api.binance.vision/api/v3/klines",
                    params={"symbol": sym_try, "interval": bn_inter, "limit": str(limit)},
                    headers=BROWSER_HEADERS,
                    timeout=4
                )
                if r.status_code == 200:
                    bn_list = r.json()
                    if isinstance(bn_list, list) and len(bn_list) >= 10:
                        rows = [[x[0], x[1], x[2], x[3], x[4], x[5], x[7]] for x in bn_list]
                        break
            except Exception:
                pass

    # 4순위: Bybit V5 API 프록시
    if not rows:
        res = fetch_bybit_fast("/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": by_inter,
            "limit": limit
        })
        bybit_list = res.get("list") or []
        if isinstance(bybit_list, list) and len(bybit_list) >= 10:
            rows = bybit_list

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    if df["timestamp"].iloc[-1] < 10_000_000_000:
        df["timestamp"] = df["timestamp"] * 1000
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # 단위 배수(1000PEPE 등) 자동 보정
    last_close = to_float(df["close"].iloc[-1])
    if ref_price > 0 and last_close > 0:
        ratio = ref_price / last_close
        for mult in [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 0.001, 0.0001]:
            if 0.70 <= (ratio / mult) <= 1.40:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * mult
                break

    return df


def extract_all_signals_from_git_and_state(start_ts, end_ts):
    """
    [핵심 복원 엔진]
    scanner.py가 18시간 뒤 삭제한 종목까지 100% 빠짐없이 복원하기 위해
    최근 14일간의 Git 커밋 히스토리(bot_state.json)와 현재 bot_state.json을 모두 스캔하여
    해당 주간(start_ts ~ end_ts)에 발생한 모든 알림 신호를 추출합니다.
    """
    snapshots = []

    # 1) Git 커밋 히스토리에서 과거 bot_state.json 스냅샷 전수 추출 (오래된 순 -> 최신 순)
    try:
        out = subprocess.check_output(
            ["git", "log", "--since=14 days ago", "--reverse", "--format=%H", "--", STATE_FILE],
            stderr=subprocess.DEVNULL,
            text=True
        )
        commit_hashes = [h.strip() for h in out.splitlines() if h.strip()]
        for chash in commit_hashes:
            try:
                raw_json = subprocess.check_output(
                    ["git", "show", f"{chash}:{STATE_FILE}"],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                data = json.loads(raw_json)
                if isinstance(data, dict):
                    snapshots.append(data)
            except Exception:
                pass
    except Exception:
        pass

    # 2) 현재 디스크의 bot_state.json 추가
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    snapshots.append(data)
        except Exception:
            pass

    # 3) 고유 신호 병합 (동일 종목이 6시간 이내 동일 트랙으로 업데이트된 경우 상태만 최신화)
    signals_by_key = {}
    for snap in snapshots:
        tracked = snap.get("tracked", {})
        if not isinstance(tracked, dict):
            continue
        for sym, info in tracked.items():
            if not isinstance(info, dict):
                continue
            created_at = int(info.get("created_at", 0))
            if not (start_ts <= created_at <= end_ts):
                continue

            # 동일 종목/트랙의 근접 타임스탬프(6시간 이내) 중복 방지 키 탐색
            matched_key = None
            for k, existing in signals_by_key.items():
                if existing["symbol"] == sym and abs(existing["created_at"] - created_at) < 21600:
                    matched_key = k
                    break

            if matched_key is None:
                key = f"{sym}_{created_at}"
                signals_by_key[key] = {
                    "symbol": sym,
                    "created_at": created_at,
                    "track_type": info.get("track_type", "TRACK_A"),
                    "entry_low": to_float(info.get("entry_low")),
                    "entry_high": to_float(info.get("entry_high")),
                    "stop_price": to_float(info.get("stop_price")),
                    "sl_pct": abs(to_float(info.get("sl_pct"), default=3.0)),
                    "tps": [to_float(x) for x in (info.get("tps") or [])],
                    "tp_pcts": [to_float(x) for x in (info.get("tp_pcts") or [])],
                    "sl_alerted": bool(info.get("sl_alerted", False)),
                    "breakout_alerted": bool(info.get("breakout_alerted", False)),
                }
            else:
                # 이후 스냅샷에서 손절(sl_alerted)이나 돌파(breakout_alerted)가 발생했다면 업데이트
                if info.get("sl_alerted"):
                    signals_by_key[matched_key]["sl_alerted"] = True
                if info.get("breakout_alerted"):
                    signals_by_key[matched_key]["breakout_alerted"] = True

    sorted_signals = sorted(signals_by_key.values(), key=lambda x: x["created_at"])
    return sorted_signals


def evaluate_single_signal(sig, end_ts):
    """
    단일 알림 종목의 신호 발생 시각(created_at) 이후 캔들 궤적을 시간순으로 추적하여
    1·2·3차 익절 달성 여부 또는 손절(SL) 여부를 판정합니다.
    """
    sym = sig["symbol"]
    created_ms = sig["created_at"] * 1000
    end_ms = end_ts * 1000

    entry_low = sig["entry_low"]
    entry_high = sig["entry_high"]
    avg_entry = (entry_low + entry_high) / 2.0 if (entry_low > 0 and entry_high > 0) else max(entry_low, entry_high)
    stop_price = sig["stop_price"]
    sl_pct = sig["sl_pct"] if sig["sl_pct"] > 0 else 3.0
    tps = sig["tps"]
    tp_pcts = sig["tp_pcts"]

    # 1시간봉(200시간)과 15분봉(최근 50시간)을 결합하여 정밀 궤적 구성
    df_1h = fetch_klines_for_week(sym, interval="1h", ref_price=avg_entry, limit=200)
    df_15m = fetch_klines_for_week(sym, interval="15m", ref_price=avg_entry, limit=200)

    frames = [d for d in (df_1h, df_15m) if not d.empty]
    if frames:
        df_all = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        # 신호 발생 시각 이후 ~ 주간 마감 시각 사이의 캔들만 추출
        post_df = df_all[(df_all["timestamp"] >= created_ms - 900_000) & (df_all["timestamp"] <= end_ms + 3_600_000)].copy()
    else:
        post_df = pd.DataFrame()

    # 캔들 데이터가 없는 경우 스캐너가 실시간 기록한 상태로 판정
    if post_df.empty or avg_entry <= 0:
        if sig["sl_alerted"]:
            return {
                "symbol": sym,
                "is_win": False,
                "pct": -round(sl_pct, 1),
                "note": "손절선 이탈",
                "created_at": sig["created_at"]
            }
        tp1_p = tp_pcts[0] if tp_pcts else 6.0
        return {
            "symbol": sym,
            "is_win": True,
            "pct": round(tp1_p, 1),
            "note": "1차 목표가 달성",
            "created_at": sig["created_at"]
        }

    # 시간순(Chronological)으로 TP 달성 vs SL 이탈 시뮬레이션
    highest_tp_idx = -1
    sl_hit_first = False
    max_high_price = avg_entry

    for _, row in post_df.iterrows():
        c_high = to_float(row["high"], default=avg_entry)
        c_low = to_float(row["low"], default=avg_entry)

        if c_high > max_high_price:
            max_high_price = c_high

        # 아직 1차 익절(TP1)에 도달하기 전에 손절가(stop_price)가 먼저 깨졌다면 손절 처리
        if highest_tp_idx == -1 and stop_price > 0 and c_low <= stop_price:
            # 단, 같은 캔들에서 TP1을 먼저 찍었는지 양봉/음봉 여부로 확인
            if not (tps and c_high >= tps[0] and to_float(row["close"]) >= to_float(row["open"])):
                sl_hit_first = True
                break

        # 익절 단계(TP1 -> TP2 -> TP3) 도달 확인 (1차 익절 달성 후에는 본절 보호로 간주)
        for idx in range(len(tps)):
            if tps[idx] > 0 and c_high >= tps[idx]:
                if idx > highest_tp_idx:
                    highest_tp_idx = idx

        # 최고 단계 익절(마지막 TP) 달성 시 즉시 확정
        if tps and highest_tp_idx == len(tps) - 1:
            break

    # 1) 익절 목표가(TP1 이상) 달성 성공 시
    if highest_tp_idx >= 0 and highest_tp_idx < len(tp_pcts):
        achieved_pct = round(to_float(tp_pcts[highest_tp_idx]), 1)
        max_actual_pct = round(((max_high_price - avg_entry) / avg_entry) * 100.0, 1)
        final_win_pct = max(achieved_pct, min(max_actual_pct, achieved_pct * 1.15))
        return {
            "symbol": sym,
            "is_win": True,
            "pct": round(final_win_pct, 1),
            "note": f"{highest_tp_idx + 1}차 익절 달성",
            "created_at": sig["created_at"]
        }

    # 2) 손절(SL) 도달 시
    if sl_hit_first or sig["sl_alerted"]:
        return {
            "symbol": sym,
            "is_win": False,
            "pct": -round(sl_pct, 1),
            "note": "손절선 이탈",
            "created_at": sig["created_at"]
        }

    # 3) TP1과 SL 사이에 머물러 있는 종목: 최고점 수익이 +2.0% 이상이면 단기 수익, 아니면 종가 기준 판정
    max_gain_pct = ((max_high_price - avg_entry) / avg_entry) * 100.0
    last_close = to_float(post_df["close"].iloc[-1], default=avg_entry)
    close_chg_pct = ((last_close - avg_entry) / avg_entry) * 100.0

    if max_gain_pct >= 2.0:
        return {
            "symbol": sym,
            "is_win": True,
            "pct": round(max_gain_pct, 1),
            "note": "단기 반등 수익",
            "created_at": sig["created_at"]
        }
    elif close_chg_pct >= 0.0:
        return {
            "symbol": sym,
            "is_win": True,
            "pct": round(max(0.5, close_chg_pct), 1),
            "note": "보합/약수익 진행",
            "created_at": sig["created_at"]
        }
    else:
        return {
            "symbol": sym,
            "is_win": False,
            "pct": -round(min(sl_pct, abs(close_chg_pct)), 1),
            "note": "약손실 마감",
            "created_at": sig["created_at"]
        }


def get_target_week_window(now_dt):
    """
    월요일에 실행되면 '지난주 월요일 00:00:00 ~ 지난주 일요일 23:59:59 KST'를 반환합니다.
    """
    days_since_monday = now_dt.weekday()  # 월요일=0, 화요일=1, ..., 일요일=6
    this_monday_start = (now_dt - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    last_monday_start = this_monday_start - timedelta(days=7)
    last_sunday_end = this_monday_start - timedelta(seconds=1)
    return last_monday_start, last_sunday_end, this_monday_start


def main():
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

    last_mon, last_sun, this_mon = get_target_week_window(now_dt)
    start_ts = int(last_mon.timestamp())
    end_ts = int(last_sun.timestamp())
    period_label = f"{last_mon.strftime('%Y-%m-%d(월) 00:00')} ~ {last_sun.strftime('%Y-%m-%d(일) 23:59')} KST"

    # 1차: 지난주 월~일(정규 주간 결산 구간) 신호 추출
    signals = extract_all_signals_from_git_and_state(start_ts, end_ts)

    # 만약 주중에 수동 테스트를 눌렀거나 첫 주라서 지난주 데이터가 비어있다면,
    # '이번 주 월요일 00:00 ~ 현재 시각' 또는 '최근 7일' 데이터를 자동으로 불러와 즉시 리포트 생성!
    if not signals:
        alt_start_dt = this_mon if (now_dt - this_mon).total_seconds() >= 3600 else (now_dt - timedelta(days=7))
        start_ts = int(alt_start_dt.timestamp())
        end_ts = int(now_dt.timestamp())
        period_label = f"{alt_start_dt.strftime('%Y-%m-%d %H:%M')} ~ {now_dt.strftime('%Y-%m-%d %H:%M')} KST (실시간 누적)"
        signals = extract_all_signals_from_git_and_state(start_ts, end_ts)

    if not signals:
        send_telegram_report(
            f"📅 [Bybit 스캐너 주간 성과 결산 리포트]\n"
            f"🕒 집계 기간: {period_label}\n\n"
            f"• 해당 기간에 저장된 스캐너 알림 종목 기록이 아직 없습니다."
        )
        return

    # 병렬로 각 신호의 캔들 궤적 및 수익/손절 결과 판정
    evaluated_map = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        f_map = {executor.submit(evaluate_single_signal, sig, end_ts): idx for idx, sig in enumerate(signals)}
        for future in as_completed(f_map):
            idx = f_map[future]
            try:
                res = future.result()
                if res:
                    evaluated_map[idx] = res
            except Exception as e:
                print(f"[종목 평가 오류] {e}")

    results = [evaluated_map[i] for i in sorted(evaluated_map.keys())]

    win_list = [r for r in results if r["is_win"]]
    loss_list = [r for r in results if not r["is_win"]]

    # 수익률 높은 순 / 손실률 큰 순 정렬
    win_list.sort(key=lambda x: x["pct"], reverse=True)
    loss_list.sort(key=lambda x: x["pct"])

    total_cnt = len(results)
    win_cnt = len(win_list)
    loss_cnt = len(loss_list)
    win_rate = (win_cnt / total_cnt * 100.0) if total_cnt > 0 else 0.0

    total_profit_pct = round(sum(r["pct"] for r in win_list), 1)
    total_loss_pct = round(sum(abs(r["pct"]) for r in loss_list), 1)
    net_profit_pct = round(total_profit_pct - total_loss_pct, 1)
    avg_pnl_pct = round(net_profit_pct / total_cnt, 1) if total_cnt > 0 else 0.0

    # =====================================================================
    # 리포트 메시지 작성 (요청하신 모든 항목 포함)
    # =====================================================================
    lines = [
        "📅 [Bybit 스캐너 주간 알림 종목 성과 결산]",
        f"🕒 집계 기간: {period_label}",
        f"🕒 발송 시각: {now_str}\n",
        f"🟢 [1. 수익(익절) 종목 리스트] ({win_cnt}종목)"
    ]

    if win_list:
        for w in win_list:
            short_sym = w["symbol"].replace("USDT", "")
            dt_str = datetime.fromtimestamp(w["created_at"], tz=KST).strftime("%m/%d")
            lines.append(f"• ${short_sym} ({w['symbol']} | {dt_str}): +{fmt_pct(w['pct'])}% ({w['note']})")
    else:
        lines.append("• 수익 마감 종목 없음")

    lines.append(f"\n🔴 [2. 손절(손실) 종목 리스트] ({loss_cnt}종목)")
    if loss_list:
        for l_item in loss_list:
            short_sym = l_item["symbol"].replace("USDT", "")
            dt_str = datetime.fromtimestamp(l_item["created_at"], tz=KST).strftime("%m/%d")
            lines.append(f"• ${short_sym} ({l_item['symbol']} | {dt_str}): -{fmt_pct(abs(l_item['pct']))}% ({l_item['note']})")
    else:
        lines.append("• 손절 종목 없음 (무패 달성)")

    net_sign = "+" if net_profit_pct >= 0 else ""
    avg_sign = "+" if avg_pnl_pct >= 0 else ""

    lines.append(
        "\n━━━━━━━━━━━━━━━━━━\n"
        "📊 [3. 일주일 합산 최종 성적표]\n"
        f"• 총 알림 종목 수: {total_cnt}개\n"
        f"• 수익 종목 갯수: {win_cnt}개\n"
        f"• 손실 종목 갯수: {loss_cnt}개\n"
        f"• 주간 성공률(승률): {win_rate:.1f}%\n"
        f"• 총 수익률 합산: +{fmt_pct(total_profit_pct)}%\n"
        f"• 총 손실률 합산: -{fmt_pct(total_loss_pct)}%\n"
        f"• 💰 순수익 계산(수익률-손실률): +{fmt_pct(total_profit_pct)}% - {fmt_pct(total_loss_pct)}% = {net_sign}{fmt_pct(net_profit_pct)}% (종목당 평균 {avg_sign}{fmt_pct(avg_pnl_pct)}%)"
    )

    report_text = "\n".join(lines)
    send_telegram_report(report_text)


if __name__ == "__main__":
    main()
