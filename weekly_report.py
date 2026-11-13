import os
import sys
import json
import time
import subprocess
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import config as cfg
    TELEGRAM_BOT_TOKEN = getattr(cfg, "TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = getattr(cfg, "TELEGRAM_CHAT_ID", "")
    STATE_FILE = getattr(cfg, "STATE_FILE", "bot_state.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
    is_tradfi_ticker = getattr(cfg, "is_tradfi_ticker", lambda x: False)
except Exception:
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    STATE_FILE = "bot_state.json"
    KST = timezone(timedelta(hours=9))
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}
    is_tradfi_ticker = lambda x: False

try:
    import api as api_mod
except Exception:
    api_mod = None

try:
    import notifier as notifier_mod
except Exception:
    notifier_mod = None

if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = (
        os.getenv("TELEGRAM_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or ""
    ).strip()

if not TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("CHAT_ID")
        or ""
    ).strip()

# 동일 심볼 캔들 중복 호출 방지용 메모리 캐시
_KLINE_CACHE = {}


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def fmt_price(price):
    if notifier_mod and hasattr(notifier_mod, "fmt_price"):
        try:
            return notifier_mod.fmt_price(price)
        except Exception:
            pass
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


def send_telegram_chunked(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[경고] 텔레그램 환경변수 미설정 - 콘솔 출력:\n" + text)
        return False

    max_len = 3800
    lines = text.split("\n")
    chunks, current, curr_len = [], [], 0
    for line in lines:
        if curr_len + len(line) + 1 > max_len and current:
            chunks.append("\n".join(current))
            current = [line]
            curr_len = len(line) + 1
        else:
            current.append(line)
            curr_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))

    sent_ok = False
    for chunk in chunks:
        if notifier_mod and hasattr(notifier_mod, "send_telegram_single"):
            try:
                notifier_mod.send_telegram_single(chunk)
                sent_ok = True
                time.sleep(0.5)
                continue
            except Exception:
                pass
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "disable_web_page_preview": True},
                timeout=15,
            )
            if r.status_code == 200:
                sent_ok = True
            time.sleep(0.5)
        except Exception as e:
            print(f"[텔레그램 전송 오류] {e}")
    return sent_ok


def fetch_klines_cached(symbol, interval="15", limit=500):
    """동일 종목에 대해 여러 번 신호가 발생하더라도 거래소 API는 단 1회만 호출하도록 캐싱"""
    cache_key = (symbol, str(interval), int(limit))
    if cache_key in _KLINE_CACHE:
        return _KLINE_CACHE[cache_key]

    df = pd.DataFrame()
    if api_mod and hasattr(api_mod, "fetch_klines"):
        try:
            df = api_mod.fetch_klines(symbol, str(interval), limit=int(limit))
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        iv_min = int(interval)
        base = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
        gate_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1h"}.get(iv_min, "15m")
        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": f"{base}_USDT", "interval": gate_iv, "limit": int(limit)},
                headers=BROWSER_HEADERS,
                timeout=5,
            )
            if r.status_code == 200 and isinstance(r.json(), list) and r.json():
                raw_df = pd.DataFrame(r.json()).rename(
                    columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                )
                raw_df["timestamp"] = pd.to_numeric(raw_df["timestamp"], errors="coerce") * 1000.0
                for col in ("open", "high", "low", "close", "volume"):
                    raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
                df = raw_df.dropna(subset=["close"]).sort_values("timestamp", ascending=True).reset_index(drop=True)
        except Exception:
            pass

    if df is None:
        df = pd.DataFrame()
    _KLINE_CACHE[cache_key] = df
    return df


def _merge_episode_dict(episodes_map, sym, raw_item):
    if not isinstance(raw_item, dict):
        return
    base = sym[:-4].upper() if sym.endswith("USDT") else sym.upper()
    if is_tradfi_ticker(base):
        return

    c_at = int(to_float(raw_item.get("created_at"), 0))
    if c_at <= 0:
        return

    # 30분 이내 동일 종목 스냅샷은 동일 에피소드로 병합하되, 다른 시간대 재진입은 독립 에피소드로 보존
    bucket = c_at // 1800
    ep_key = f"{sym}_{bucket}"

    entry_low = to_float(raw_item.get("entry_low"))
    entry_high = to_float(raw_item.get("entry_high"))
    entry_p = to_float(raw_item.get("entry_price"))
    stop_p = to_float(raw_item.get("stop_price"))
    if entry_p <= 0:
        entry_p = (entry_low + entry_high) / 2.0 if entry_low > 0 else stop_p

    if entry_p <= 0:
        return

    existing = episodes_map.get(ep_key)
    if not existing:
        episodes_map[ep_key] = {
            "symbol": sym,
            "base": base,
            "created_at": c_at,
            "track_type": raw_item.get("track_type", "TRACK_A"),
            "score": int(to_float(raw_item.get("score"), 85)),
            "entry_price": entry_p,
            "stop_price": to_float(raw_item.get("initial_stop_price") or stop_p, entry_p * 0.96),
            "sl_pct": to_float(raw_item.get("sl_pct"), 4.0),
            "tps": raw_item.get("tps") or [entry_p * 1.08, entry_p * 1.15, entry_p * 1.22],
            "tp1_hit": bool(raw_item.get("tp1_hit", False)),
            "tp2_hit": bool(raw_item.get("tp2_hit", False)),
            "tp3_completed": bool(raw_item.get("tp3_completed", False)),
            "be_armed": bool(raw_item.get("be_armed", False)),
            "sl_alerted": bool(raw_item.get("sl_alerted", False)),
            "sl_at": int(to_float(raw_item.get("sl_at"), 0)),
            "max_favorable_pct": to_float(raw_item.get("max_favorable_pct"), 0.0),
        }
    else:
        existing["tp1_hit"] = existing["tp1_hit"] or bool(raw_item.get("tp1_hit", False))
        existing["tp2_hit"] = existing["tp2_hit"] or bool(raw_item.get("tp2_hit", False))
        existing["tp3_completed"] = existing["tp3_completed"] or bool(raw_item.get("tp3_completed", False))
        existing["be_armed"] = existing["be_armed"] or bool(raw_item.get("be_armed", False))
        existing["sl_alerted"] = existing["sl_alerted"] or bool(raw_item.get("sl_alerted", False))
        if int(to_float(raw_item.get("sl_at"), 0)) > existing["sl_at"]:
            existing["sl_at"] = int(to_float(raw_item.get("sl_at"), 0))
        existing["max_favorable_pct"] = max(
            existing["max_favorable_pct"], to_float(raw_item.get("max_favorable_pct"), 0.0)
        )


def collect_weekly_episodes(now_ts, days=7):
    """
    IDE 권고 반영: 수천 개의 커밋을 전수 조사하지 않고
    1) 현재 bot_state.json의 signal_episodes / tracked / sl_history를 먼저 읽은 뒤,
    2) Git 히스토리가 존재할 경우 최소 2시간 간격으로만 샘플링(최대 60개 이하)하여 2초 내 수집 완료
    """
    cutoff_ts = now_ts - (days * 86400)
    episodes_map = {}

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict):
                for ep in state.get("signal_episodes", {}).values():
                    if isinstance(ep, dict) and ep.get("symbol"):
                        _merge_episode_dict(episodes_map, ep["symbol"], ep)
                for sym, item in state.get("tracked", {}).items():
                    _merge_episode_dict(episodes_map, sym, item)
                for item in state.get("sl_history", {}).values():
                    if isinstance(item, dict) and item.get("symbol"):
                        item_copy = dict(item)
                        item_copy["sl_alerted"] = True
                        _merge_episode_dict(episodes_map, item["symbol"], item_copy)
        except Exception as e:
            print(f"[현재 bot_state.json 로드 경고] {e}")

    # Git 히스토리에서 2시간(7200초) 간격으로만 샘플링해 이전 누락분 보완 (최대 50회 미만)
    try:
        cmd = ["git", "log", f"--since={days + 1} days ago", "--format=%H %ct", "--", STATE_FILE]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            sampled_hashes = []
            last_sampled_ts = 0
            for line in proc.stdout.strip().split("\n"):
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                c_hash, c_ts = parts[0], int(parts[1])
                if c_ts < cutoff_ts:
                    continue
                if last_sampled_ts == 0 or abs(last_sampled_ts - c_ts) >= 7200:
                    sampled_hashes.append(c_hash)
                    last_sampled_ts = c_ts
                if len(sampled_hashes) >= 50:
                    break

            for h in sampled_hashes:
                try:
                    sh = subprocess.run(
                        ["git", "show", f"{h}:{STATE_FILE}"],
                        capture_output=True,
                        text=True,
                        timeout=3,
                    )
                    if sh.returncode == 0 and sh.stdout:
                        st = json.loads(sh.stdout)
                        if isinstance(st, dict):
                            for sym, item in st.get("tracked", {}).items():
                                _merge_episode_dict(episodes_map, sym, item)
                except Exception:
                    continue
    except Exception as e:
        print(f"[Git 샘플링 안내] {e}")

    valid_episodes = [
        ep for ep in episodes_map.values()
        if cutoff_ts <= int(ep.get("created_at", 0)) <= now_ts
    ]
    valid_episodes.sort(key=lambda x: int(x["created_at"]))
    return valid_episodes


def resolve_same_candle_tie_5m(symbol, candle_ts_ms, tp_price, sl_price):
    """
    IDE 권고 반영: 15분봉 단일 캔들 내에서 고가>=TP와 저가<=SL이 동시에 발생한 경우
    해당 15분 구간의 5분봉 3개를 조회해 어느 가격이 먼저 도달했는지 정밀 판정
    """
    df_5m = fetch_klines_cached(symbol, "5", limit=180)
    if df_5m is not None and not df_5m.empty and "timestamp" in df_5m.columns:
        sub = df_5m[(df_5m["timestamp"] >= candle_ts_ms) & (df_5m["timestamp"] < candle_ts_ms + 900000)]
        for _, row in sub.iterrows():
            h = to_float(row["high"])
            l = to_float(row["low"])
            if l <= sl_price and h < tp_price:
                return "SL_FIRST"
            if h >= tp_price and l > sl_price:
                return "TP_FIRST"
    # 5분봉에서도 동시 터치이거나 구할 수 없으면 보수적으로 손절(SL) 우선 판정
    return "SL_FIRST"


def evaluate_episode_performance(ep, now_ts):
    """
    개별 신호(Episode) 단위로 손절 시점 이전까지만 고점을 추적하며,
    단순 수익률과 실전 분할익절(40:30:30 + 본전방어) 수익률을 함께 계산합니다.
    """
    sym = ep["symbol"]
    entry_p = to_float(ep["entry_price"])
    init_stop_p = to_float(ep["stop_price"], entry_p * 0.96)
    sl_pct = max(2.0, min(5.0, abs((entry_p - init_stop_p) / max(entry_p, 1e-8) * 100.0)))
    created_ms = int(ep["created_at"]) * 1000

    tps = ep.get("tps") or []
    tp1_p = to_float(tps[0], entry_p * 1.08) if len(tps) >= 1 else entry_p * 1.08
    tp2_p = to_float(tps[1], entry_p * 1.15) if len(tps) >= 2 else entry_p * 1.15
    tp3_p = to_float(tps[2], entry_p * 1.22) if len(tps) >= 3 else entry_p * 1.22

    tp1_pct = (tp1_p - entry_p) / entry_p * 100.0
    tp2_pct = (tp2_p - entry_p) / entry_p * 100.0
    tp3_pct = (tp3_p - entry_p) / entry_p * 100.0
    be_trigger_p = entry_p * (1.0 + getattr(cfg, "BREAKEVEN_TRIGGER_PCT", 4.5) / 100.0)

    df_15m = fetch_klines_cached(sym, "15", limit=500)
    post_df = pd.DataFrame()
    if df_15m is not None and not df_15m.empty and "timestamp" in df_15m.columns:
        post_df = df_15m[df_15m["timestamp"] >= created_ms].copy()

    # 캔들을 시간순으로 순회하며 실제 트레이딩과 동일하게 상태 전이 시뮬레이션 (손절 후 고점 착시 완전 배제)
    highest_before_exit = entry_p
    highest_after_sl = entry_p
    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False
    hit_be = False
    stopped_out = False
    be_closed = False

    if not post_df.empty:
        for _, row in post_df.iterrows():
            c_ts = int(to_float(row["timestamp"]))
            h = to_float(row["high"], entry_p)
            l = to_float(row["low"], entry_p)

            if stopped_out or be_closed or hit_tp3:
                if stopped_out and h > highest_after_sl:
                    highest_after_sl = h
                continue

            active_stop = init_stop_p
            if hit_tp2:
                active_stop = max(active_stop, entry_p * 1.08)
            elif hit_tp1:
                active_stop = max(active_stop, entry_p * 1.04)
            elif hit_be:
                active_stop = max(active_stop, entry_p * 1.003)

            # 동일 15분봉에서 목표가와 손절선을 동시에 건드린 경우 5분봉으로 선후관계 판별
            next_tp_target = tp3_p if hit_tp2 else (tp2_p if hit_tp1 else tp1_p)
            if l <= active_stop and h >= next_tp_target:
                order = resolve_same_candle_tie_5m(sym, c_ts, next_tp_target, active_stop)
                if order == "SL_FIRST":
                    if hit_tp1 or hit_be:
                        be_closed = True
                    else:
                        stopped_out = True
                        highest_after_sl = h
                    continue

            if l <= active_stop:
                if hit_tp1 or hit_be:
                    be_closed = True
                else:
                    stopped_out = True
                    highest_after_sl = h
                continue

            if h > highest_before_exit:
                highest_before_exit = h

            if h >= tp3_p:
                hit_tp1 = hit_tp2 = hit_tp3 = True
            elif h >= tp2_p:
                hit_tp1 = hit_tp2 = True
            elif h >= tp1_p:
                hit_tp1 = True
            elif h >= be_trigger_p:
                hit_be = True
    else:
        # 캔들 미수집 시 스캐너가 실시간 기록한 상태값 사용
        hit_tp1 = ep.get("tp1_hit", False)
        hit_tp2 = ep.get("tp2_hit", False)
        hit_tp3 = ep.get("tp3_completed", False)
        hit_be = ep.get("be_armed", False)
        stopped_out = ep.get("sl_alerted", False) and not hit_tp1
        highest_before_exit = entry_p * (1.0 + to_float(ep.get("max_favorable_pct", 0.0)) / 100.0)

    valid_max_gain_pct = max(0.0, (highest_before_exit - entry_p) / entry_p * 100.0)
    post_sl_rebound_pct = max(0.0, (highest_after_sl - init_stop_p) / max(init_stop_p, 1e-8) * 100.0)

    # 최종 상태 및 실전 분할익절(40% / 30% / 30%) 수익률 계산
    if hit_tp3:
        status = "TP3_WIN"
        badge = "🎯 3차졸업"
        split_pnl_pct = (tp1_pct * 0.40) + (tp2_pct * 0.30) + (tp3_pct * 0.30)
        simple_pnl_pct = tp3_pct
    elif hit_tp2:
        status = "TP2_WIN"
        badge = "✅ 2차달성"
        split_pnl_pct = (tp1_pct * 0.40) + (tp2_pct * 0.30) + (8.0 * 0.30)
        simple_pnl_pct = tp2_pct
    elif hit_tp1:
        status = "TP1_WIN"
        badge = "✅ 1차달성"
        split_pnl_pct = (tp1_pct * 0.40) + (4.0 * 0.60)
        simple_pnl_pct = tp1_pct
    elif be_closed:
        status = "BE_PROTECT"
        badge = "🛡️ 본전보호(+0.3%)"
        split_pnl_pct = 0.3
        simple_pnl_pct = 0.3
    elif stopped_out:
        status = "SL_LOSS"
        badge = "❌ 손절이탈"
        split_pnl_pct = -sl_pct
        simple_pnl_pct = -sl_pct
    else:
        status = "ACTIVE"
        badge = "⏳ 순항중"
        curr_c = to_float(post_df["close"].iloc[-1], entry_p) if not post_df.empty else entry_p
        curr_pnl = (curr_c - entry_p) / entry_p * 100.0
        split_pnl_pct = curr_pnl
        simple_pnl_pct = curr_pnl

    return {
        "symbol": sym,
        "base": ep["base"],
        "created_at": ep["created_at"],
        "track_type": ep["track_type"],
        "score": ep["score"],
        "entry_price": entry_p,
        "status": status,
        "badge": badge,
        "valid_max_gain_pct": round(valid_max_gain_pct, 1),
        "post_sl_rebound_pct": round(post_sl_rebound_pct, 1),
        "split_pnl_pct": round(split_pnl_pct, 2),
        "simple_pnl_pct": round(simple_pnl_pct, 2),
    }


def main():
    start_t = time.time()
    now_ts = int(time.time())
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")
    start_date_str = (now_dt - timedelta(days=7)).strftime("%m/%d")
    end_date_str = now_dt.strftime("%m/%d")

    episodes = collect_weekly_episodes(now_ts, days=7)
    if not episodes:
        msg = f"📊 [주간 성과 리포트 ({start_date_str}~{end_date_str})]\n• 최근 7일간 기록된 알림 에피소드가 없습니다."
        send_telegram_chunked(msg)
        return

    # 심볼별 15분봉 병렬 프리패칭 (중복 호출 제거 및 실행 시간 단축)
    unique_syms = list({ep["symbol"] for ep in episodes})
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_klines_cached, s, "15", 500) for s in unique_syms]
        for _ in as_completed(futures):
            pass

    evaluated = [evaluate_episode_performance(ep, now_ts) for ep in episodes]

    total_cnt = len(evaluated)
    tp3_list = [x for x in evaluated if x["status"] == "TP3_WIN"]
    tp2_list = [x for x in evaluated if x["status"] == "TP2_WIN"]
    tp1_list = [x for x in evaluated if x["status"] == "TP1_WIN"]
    be_list = [x for x in evaluated if x["status"] == "BE_PROTECT"]
    sl_list = [x for x in evaluated if x["status"] == "SL_LOSS"]
    act_list = [x for x in evaluated if x["status"] == "ACTIVE"]

    win_cnt = len(tp3_list) + len(tp2_list) + len(tp1_list)
    be_cnt = len(be_list)
    sl_cnt = len(sl_list)
    decided_cnt = win_cnt + be_cnt + sl_cnt

    strict_win_rate = (win_cnt / decided_cnt * 100.0) if decided_cnt > 0 else 0.0
    defended_rate = ((win_cnt + be_cnt) / decided_cnt * 100.0) if decided_cnt > 0 else 0.0

    # 트랙별(Track A 선취매 vs Track B 돌파) 성과 비교
    track_a = [x for x in evaluated if x["track_type"] == "TRACK_A" and x["status"] != "ACTIVE"]
    track_b = [x for x in evaluated if x["track_type"] == "TRACK_B" and x["status"] != "ACTIVE"]
    win_a = sum(1 for x in track_a if x["status"].endswith("WIN"))
    win_b = sum(1 for x in track_b if x["status"].endswith("WIN"))
    rate_a = (win_a / len(track_a) * 100.0) if track_a else 0.0
    rate_b = (win_b / len(track_b) * 100.0) if track_b else 0.0

    # 수익률 집계 (단순 합산 vs 실전 분할익절 기대값 vs 타점당 계좌 10% 투입 시 실제 계좌 증감률)
    decided_items = [x for x in evaluated if x["status"] != "ACTIVE"]
    simple_sum_pct = sum(x["simple_pnl_pct"] for x in decided_items)
    split_sum_pct = sum(x["split_pnl_pct"] for x in decided_items)
    avg_expectancy_pct = (split_sum_pct / len(decided_items)) if decided_items else 0.0
    account_growth_10pct_pos = split_sum_pct * 0.10

    top_winners = sorted(evaluated, key=lambda x: x["valid_max_gain_pct"], reverse=True)[:5]
    stop_hunted = [x for x in sl_list if x["post_sl_rebound_pct"] >= 4.0]

    lines = [
        f"📅 [Bybit 스캐너 주간 실전 성과 리포트 ({start_date_str} ~ {end_date_str})]",
        f"🕒 집계 시각: {now_str} (생존 편향 제거 · 전수 신호 독립 집계)\n",
        "1️⃣ [주간 전체 신호(Episode) 성적표]",
        f"• 총 발생 신호: {total_cnt}건 (종결 {decided_cnt}건 | 순항중 {len(act_list)}건)",
        f"• 익절 달성: {win_cnt}건 (🎯3차 {len(tp3_list)}건 | ✅2차 {len(tp2_list)}건 | ✅1차 {len(tp1_list)}건)",
        f"• 본전 방어(🛡️+4.5% 도달 후 보호마감): {be_cnt}건 | 손절 이탈(❌): {sl_cnt}건",
        f"• 순수 익절 승률: {strict_win_rate:.1f}% | 본전방어 포함 무패율: {defended_rate:.1f}%\n",
        "2️⃣ [실전 계좌 수익률 분석 (분할익절 40:30:30 반영)]",
        f"• 신호 1건당 평균 기대수익률(Expectancy): {avg_expectancy_pct:+.2f}%",
        f"• 실전 계좌 수익률 (타점당 시드 10% 투입 기준): {account_growth_10pct_pos:+.2f}%",
        f"• 단순 산술 누적 수익률 (분할익절 기준): {split_sum_pct:+.1f}% (단순 고점 합산 {simple_sum_pct:+.1f}%)\n",
        "3️⃣ [트랙별 적중률 비교 (Track A vs Track B)]",
        f"• 🟢 Track A (1차 선취매/눌림): 종결 {len(track_a)}건 중 {win_a}건 익절 ({rate_a:.1f}%)",
        f"• 🚀 Track B (2차 돌파/S&R/포켓피벗): 종결 {len(track_b)}건 중 {win_b}건 익절 ({rate_b:.1f}%)\n",
        "4️⃣ [주간 베스트 수익 타점 Top 5 (손절 전 유효 최고가 기준)]",
    ]

    for idx, item in enumerate(top_winners, 1):
        dt_s = datetime.fromtimestamp(item["created_at"], tz=KST).strftime("%m/%d %H:%M")
        lines.append(
            f"  {idx}) ${item['base']} ({dt_s} 포착 | {item['score']}점) → 최고 +{item['valid_max_gain_pct']:.1f}% {item['badge']} (실현 {item['split_pnl_pct']:+.1f}%)"
        )

    lines.append("\n5️⃣ [주간 손절(SL) 복기 및 스탑헌팅 진단]")
    if not sl_list:
        lines.append("• 최근 7일간 손절 마감된 종목이 없습니다.")
    else:
        lines.append(
            f"• 총 손절 {sl_cnt}건 중 아랫꼬리 스탑헌팅 후 재상승(+4%↑) 사례: {len(stop_hunted)}건"
        )
        for idx, item in enumerate(sl_list[-5:], 1):
            dt_s = datetime.fromtimestamp(item["created_at"], tz=KST).strftime("%m/%d %H:%M")
            sh_note = f" (⚠️손절후 +{item['post_sl_rebound_pct']:.1f}% 반등)" if item["post_sl_rebound_pct"] >= 4.0 else ""
            lines.append(
                f"  - ❌ ${item['base']} ({dt_s} | {item['track_type']}): 진입 후 유효최고 +{item['valid_max_gain_pct']:.1f}% → {item['split_pnl_pct']:+.1f}%{sh_note}"
            )

    elapsed = time.time() - start_t
    lines.append(f"\n⏱️ 리포트 생성 소요 시간: {elapsed:.1f}초")
    send_telegram_chunked("\n".join(lines))
    print(f"[{now_str}] 주간 성과 리포트 전송 완료 ({elapsed:.1f}초 소요)")


if __name__ == "__main__":
    main()
