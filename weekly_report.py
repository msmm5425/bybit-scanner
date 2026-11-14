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
    HISTORY_FILE = getattr(cfg, "HISTORY_FILE", "analyzer_history.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
    is_tradfi_ticker = getattr(cfg, "is_tradfi_ticker", lambda x: False)
    is_heavy_major_coin = getattr(cfg, "is_heavy_major_coin", lambda x: False)
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception:
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    STATE_FILE = "bot_state.json"
    HISTORY_FILE = "analyzer_history.json"
    KST = timezone(timedelta(hours=9))
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}
    is_tradfi_ticker = lambda x: False
    is_heavy_major_coin = lambda x: False
    FIXED_SL_PCT = 6

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


def calc_rsi(series, period=6):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def calc_kdj_j(df, period=9):
    low_min = df["low"].rolling(window=period, min_periods=3).min()
    high_max = df["high"].rolling(window=period, min_periods=3).max()
    denom = (high_max - low_min).replace(0, np.nan)
    rsv = ((df["close"] - low_min) / denom * 100.0).fillna(50.0)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    return 3.0 * k - 2.0 * d


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
    if not df.empty and "ma20" not in df.columns:
        for p in (5, 10, 20, 50, 112):
            df[f"ma{p}"] = df["close"].rolling(p, min_periods=2).mean()
        df["rsi6"] = calc_rsi(df["close"], 6)
        df["kdj_j"] = calc_kdj_j(df, 9)
        df["vol_ma20"] = df["volume"].rolling(20, min_periods=3).mean().shift(1).bfill()
        df["vol_mult"] = (df["volume"] / df["vol_ma20"].replace(0, np.nan)).fillna(1.0)

    _KLINE_CACHE[cache_key] = df
    return df


def extract_three_wave_structure(df, ign_pos, peak_pos, entry_price, support_line, wave_h):
    """주간 개별 에피소드의 1차(첫 조정 고점) · 2차(중간 조정 고점) · 3차(최종 고점) 로데이터 정밀 추출"""
    sub = df.iloc[ign_pos: peak_pos + 1].copy()
    if sub.empty:
        sub = df.iloc[ign_pos: ign_pos + 1].copy()

    positions = list(sub.index)
    n_bars = len(positions)
    final_idx = positions[-1]
    final_high = max(to_float(df.loc[final_idx, "high"], entry_price), entry_price * 1.005)
    total_pump_pct = max(0.5, (final_high - entry_price) / max(entry_price, 1e-8) * 100.0)

    min_w1_gain_thr = max(5.5, min(14.0, total_pump_pct * 0.22)) if total_pump_pct >= 15.0 else max(3.0, total_pump_pct * 0.30)

    swing_peaks = []
    running_max_h = entry_price
    running_max_idx = positions[0]

    for k in range(len(positions) - 1):
        curr_idx = positions[k]
        h_val = to_float(df.loc[curr_idx, "high"], entry_price)
        if h_val > running_max_h:
            running_max_h = h_val
            running_max_idx = curr_idx

        gain_so_far = (running_max_h - entry_price) / max(entry_price, 1e-8) * 100.0
        if gain_so_far < min_w1_gain_thr:
            continue

        lookahead_indices = positions[k + 1: min(len(positions), k + 5)]
        if not lookahead_indices:
            continue
        min_low_ahead = min(to_float(df.loc[idx, "low"], running_max_h) for idx in lookahead_indices)
        pullback_pct = (running_max_h - min_low_ahead) / max(running_max_h, 1e-8) * 100.0

        next_idx = positions[k + 1]
        is_red_turn = to_float(df.loc[next_idx, "close"]) < to_float(df.loc[next_idx, "open"])

        if pullback_pct >= 1.2 or (is_red_turn and pullback_pct >= 0.8):
            if not swing_peaks or (
                swing_peaks[-1]["idx"] != running_max_idx
                and running_max_h > swing_peaks[-1]["high"] * 1.02
            ):
                trough_idx = min(lookahead_indices, key=lambda x: to_float(df.loc[x, "low"], running_max_h))
                trough_low = to_float(df.loc[trough_idx, "low"], running_max_h)
                ma5_t = to_float(df.loc[trough_idx, "ma5"], trough_low)
                ma10_t = to_float(df.loc[trough_idx, "ma10"], trough_low)
                ma20_t = to_float(df.loc[trough_idx, "ma20"], trough_low)

                if trough_low >= ma5_t * 0.996:
                    hold_tag = "MA5"
                elif trough_low >= ma10_t * 0.995:
                    hold_tag = "MA10"
                elif trough_low >= ma20_t * 0.993:
                    hold_tag = "MA20"
                else:
                    hold_tag = "박스상단S/R"

                swing_peaks.append({
                    "idx": running_max_idx,
                    "high": running_max_h,
                    "pullback_pct": round(pullback_pct, 1),
                    "hold_tag": hold_tag,
                })

    if len(swing_peaks) >= 2:
        w1_info = swing_peaks[0]
        mid_candidates = [
            sp for sp in swing_peaks[1:]
            if sp["idx"] != final_idx and sp["high"] < final_high * 0.985
        ]
        if mid_candidates:
            target_w2_price = entry_price + (final_high - entry_price) * 0.62
            w2_info = min(mid_candidates, key=lambda x: abs(x["high"] - target_w2_price))
        else:
            w2_info = swing_peaks[1]
    elif len(swing_peaks) == 1:
        w1_info = swing_peaks[0]
        w1_pos_in_sub = positions.index(w1_info["idx"]) if w1_info["idx"] in positions else 0
        mid_pos = positions[min(n_bars - 1, max(w1_pos_in_sub + 1, (w1_pos_in_sub + n_bars - 1) // 2))]
        w2_info = {
            "idx": mid_pos,
            "high": max(to_float(df.loc[mid_pos, "high"]), w1_info["high"] + (final_high - w1_info["high"]) * 0.55),
            "pullback_pct": 1.5,
            "hold_tag": "MA10",
        }
    else:
        idx1 = positions[max(0, n_bars // 3)]
        idx2 = positions[max(0, (n_bars * 2) // 3)]
        h1_est = max(to_float(df.loc[idx1, "high"]), entry_price + (final_high - entry_price) * 0.35)
        h2_est = max(to_float(df.loc[idx2, "high"]), entry_price + (final_high - entry_price) * 0.68)
        w1_info = {"idx": idx1, "high": h1_est, "pullback_pct": 1.2, "hold_tag": "MA5"}
        w2_info = {"idx": idx2, "high": h2_est, "pullback_pct": 1.8, "hold_tag": "MA10"}

    def _build_metrics(idx_val, high_override, pb_pct=0.0, hold_str=""):
        row = df.loc[idx_val]
        h_p = max(to_float(row["high"]), to_float(high_override), entry_price * 1.005)
        ma20_v = to_float(row.get("ma20"), entry_price)
        ma50_v = to_float(row.get("ma50"), ma20_v)
        ma112_v = to_float(row.get("ma112"), ma50_v)
        rsi_v = to_float(row.get("rsi6"), 68.0)
        kdj_v = to_float(row.get("kdj_j"), 75.0)
        vol_m = to_float(row.get("vol_mult"), 2.0)

        gain_pct = max(0.2, (h_p - entry_price) / max(entry_price, 1e-8) * 100.0)
        fib_mult = max(1.05, min(8.50, (h_p - support_line) / max(wave_h, 1e-8)))
        ma20_ext_pct = max(0.1, (h_p - ma20_v) / max(ma20_v, 1e-8) * 100.0)

        if abs(h_p - ma112_v) / max(ma112_v, 1e-8) <= 0.015:
            resist_tag = "MA112저항"
        elif abs(h_p - ma50_v) / max(ma50_v, 1e-8) <= 0.015:
            resist_tag = "MA50저항"
        else:
            resist_tag = f"MA20+{ma20_ext_pct:.1f}%이격"

        return {
            "price": h_p,
            "gain_pct": round(gain_pct, 1),
            "fib_mult": round(fib_mult, 2),
            "rsi6": round(rsi_v, 1),
            "kdj_j": round(kdj_v, 1),
            "ma20_ext_pct": round(ma20_ext_pct, 1),
            "vol_mult": round(vol_m, 1),
            "pullback_pct": round(pb_pct, 1),
            "hold_tag": hold_str,
            "resist_tag": resist_tag,
        }

    w1 = _build_metrics(w1_info["idx"], w1_info["high"], w1_info["pullback_pct"], w1_info["hold_tag"])
    w2 = _build_metrics(w2_info["idx"], max(w2_info["high"], w1["price"] * 1.02), w2_info["pullback_pct"], w2_info["hold_tag"])
    w3 = _build_metrics(final_idx, max(final_high, w2["price"] * 1.02), 0.0, "클라이맥스")

    return {"w1": w1, "w2": w2, "w3": w3}


def _merge_episode_dict(episodes_map, sym, raw_item):
    if not isinstance(raw_item, dict):
        return
    base = sym[:-4].upper() if sym.endswith("USDT") else sym.upper()
    if not base.isascii() or is_tradfi_ticker(base) or is_heavy_major_coin(base):
        return

    c_at = int(to_float(raw_item.get("created_at"), 0))
    if c_at <= 0:
        return

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

    # 주간 시뮬레이션도 신규 원칙(-6% 고정 손절 및 +9% / +17% / +29% 목표가)으로 통일 검증
    fixed_stop_p = entry_p * (1.0 - FIXED_SL_PCT / 100.0)
    existing = episodes_map.get(ep_key)
    if not existing:
        episodes_map[ep_key] = {
            "symbol": sym,
            "base": base,
            "created_at": c_at,
            "track_type": raw_item.get("track_type", "TRACK_A"),
            "score": int(to_float(raw_item.get("score"), 85)),
            "entry_price": entry_p,
            "swing_high_line": to_float(raw_item.get("swing_high_line"), entry_p * 1.02),
            "stop_price": fixed_stop_p,
            "sl_pct": float(FIXED_SL_PCT),
            "tps": [entry_p * 1.09, entry_p * 1.17, entry_p * 1.29],
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
    return "SL_FIRST"


def evaluate_episode_performance(ep, now_ts):
    sym = ep["symbol"]
    entry_p = to_float(ep["entry_price"])
    sl_pct = float(FIXED_SL_PCT)
    init_stop_p = entry_p * (1.0 - sl_pct / 100.0)
    created_ms = int(ep["created_at"]) * 1000

    tp1_pct = to_float(getattr(cfg, "TP1_MIN_PCT", 9.0), 9.0)
    tp2_pct = to_float(getattr(cfg, "TP2_MIN_PCT", 17.0), 17.0)
    tp3_pct = to_float(getattr(cfg, "TP3_MIN_PCT", 29.0), 29.0)

    tp1_p = entry_p * (1.0 + tp1_pct / 100.0)
    tp2_p = entry_p * (1.0 + tp2_pct / 100.0)
    tp3_p = entry_p * (1.0 + tp3_pct / 100.0)
    be_trigger_p = entry_p * (1.0 + sl_pct / 100.0)

    df_15m = fetch_klines_cached(sym, "15", limit=500)
    post_df = pd.DataFrame()
    if df_15m is not None and not df_15m.empty and "timestamp" in df_15m.columns:
        post_df = df_15m[df_15m["timestamp"] >= created_ms].copy()

    highest_before_exit = entry_p
    highest_after_sl = entry_p
    hit_tp1 = False
    hit_tp2 = False
    hit_tp3 = False
    hit_be = False
    stopped_out = False
    be_closed = False

    ign_pos = 0
    peak_pos_before_exit = 0
    exit_ts_ms = now_ts * 1000

    if not post_df.empty:
        ign_pos = int(df_15m.index.get_loc(post_df.index[0]))
        peak_pos_before_exit = ign_pos

        for idx_val, row in post_df.iterrows():
            c_ts = int(to_float(row["timestamp"]))
            h = to_float(row["high"], entry_p)
            l = to_float(row["low"], entry_p)

            if stopped_out or be_closed or hit_tp3:
                if stopped_out and h > highest_after_sl:
                    highest_after_sl = h
                continue

            active_stop = init_stop_p
            if hit_tp2:
                # 2차 TP 달성 시에는 1차 TP(+9%)로 보호선 고정
                active_stop = max(active_stop, entry_p * (1.0 + tp1_pct / 100.0))
            elif hit_tp1:
                # 1차 TP 달성 시에는 40% 익절 후 남은 60%가 1차 조정 눌림을 견디도록 +0.6% 본전 보호선 유지 ($ARIA +30.9% 홀딩 공식)
                active_stop = max(active_stop, entry_p * 1.006)
            elif hit_be:
                active_stop = max(active_stop, entry_p * 1.004)

            next_tp_target = tp3_p if hit_tp2 else (tp2_p if hit_tp1 else tp1_p)
            if l <= active_stop and h >= next_tp_target:
                order = resolve_same_candle_tie_5m(sym, c_ts, next_tp_target, active_stop)
                if order == "SL_FIRST":
                    exit_ts_ms = c_ts
                    if hit_tp1 or hit_be:
                        be_closed = True
                    else:
                        stopped_out = True
                        highest_after_sl = h
                    continue

            if l <= active_stop:
                exit_ts_ms = c_ts
                if hit_tp1 or hit_be:
                    be_closed = True
                else:
                    stopped_out = True
                    highest_after_sl = h
                continue

            if h > highest_before_exit:
                highest_before_exit = h
                peak_pos_before_exit = int(df_15m.index.get_loc(idx_val))

            if h >= tp3_p:
                hit_tp1 = hit_tp2 = hit_tp3 = True
                exit_ts_ms = c_ts
            elif h >= tp2_p:
                hit_tp1 = hit_tp2 = True
            elif h >= tp1_p:
                hit_tp1 = True
            elif h >= be_trigger_p:
                hit_be = True

        sup_p = entry_p * 0.960
        w_h = max(to_float(ep.get("swing_high_line"), entry_p * 1.02) - sup_p, entry_p * 0.050)
        waves = extract_three_wave_structure(df_15m, ign_pos, max(ign_pos, peak_pos_before_exit), entry_p, sup_p, w_h)
    else:
        fav_g = max(0.5, to_float(ep.get("max_favorable_pct", 0.0)))
        hit_tp3 = fav_g >= tp3_pct
        hit_tp2 = fav_g >= tp2_pct
        hit_tp1 = fav_g >= tp1_pct
        hit_be = fav_g >= sl_pct
        stopped_out = ep.get("sl_alerted", False) and not hit_tp1 and not hit_be
        highest_before_exit = entry_p * (1.0 + fav_g / 100.0)
        waves = {
            "w1": {"price": highest_before_exit, "gain_pct": fav_g, "fib_mult": 1.85, "rsi6": 70.0, "kdj_j": 78.0, "ma20_ext_pct": 4.5, "vol_mult": 2.5, "pullback_pct": 1.5, "hold_tag": "MA10", "resist_tag": "MA20+4.5%이격"},
            "w2": {"price": highest_before_exit, "gain_pct": fav_g, "fib_mult": 2.75, "rsi6": 76.0, "kdj_j": 84.0, "ma20_ext_pct": 7.7, "vol_mult": 3.0, "pullback_pct": 1.8, "hold_tag": "MA10", "resist_tag": "MA50저항"},
            "w3": {"price": highest_before_exit, "gain_pct": fav_g, "fib_mult": 4.04, "rsi6": 84.0, "kdj_j": 90.0, "ma20_ext_pct": 12.5, "vol_mult": 4.0, "pullback_pct": 0.0, "hold_tag": "클라이맥스", "resist_tag": "클라이맥스"},
        }

    valid_max_gain_pct = max(0.0, (highest_before_exit - entry_p) / entry_p * 100.0)
    post_sl_rebound_pct = max(0.0, (highest_after_sl - init_stop_p) / max(init_stop_p, 1e-8) * 100.0)

    if hit_tp3:
        status = "TP3_WIN"
        badge = "🎯 3차졸업"
        split_pnl_pct = (tp1_pct * 0.40) + (tp2_pct * 0.30) + (tp3_pct * 0.30)
        simple_pnl_pct = tp3_pct
    elif hit_tp2:
        status = "TP2_WIN"
        badge = "✅ 2차달성"
        split_pnl_pct = (tp1_pct * 0.40) + (tp2_pct * 0.30) + (tp1_pct * 0.30)
        simple_pnl_pct = tp2_pct
    elif hit_tp1:
        status = "TP1_WIN"
        badge = "✅ 1차달성"
        split_pnl_pct = (tp1_pct * 0.40) + (0.6 * 0.60)
        simple_pnl_pct = tp1_pct
    elif be_closed:
        status = "BE_PROTECT"
        badge = "🛡️ 본전보호(+0.4%)"
        split_pnl_pct = 0.4
        simple_pnl_pct = 0.4
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
        "exit_ts_ms": exit_ts_ms,
        "track_type": ep["track_type"],
        "score": ep["score"],
        "entry_price": entry_p,
        "sl_pct": sl_pct,
        "status": status,
        "badge": badge,
        "valid_max_gain_pct": round(valid_max_gain_pct, 1),
        "post_sl_rebound_pct": round(post_sl_rebound_pct, 1),
        "split_pnl_pct": round(split_pnl_pct, 2),
        "simple_pnl_pct": round(simple_pnl_pct, 2),
        "waves": waves,
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

    unique_syms = list({ep["symbol"] for ep in episodes} | {"BTCUSDT"})
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

    track_a = [x for x in evaluated if x["track_type"] == "TRACK_A" and x["status"] != "ACTIVE"]
    track_b = [x for x in evaluated if x["track_type"] == "TRACK_B" and x["status"] != "ACTIVE"]
    win_a = sum(1 for x in track_a if x["status"].endswith("WIN"))
    win_b = sum(1 for x in track_b if x["status"].endswith("WIN"))
    rate_a = (win_a / len(track_a) * 100.0) if track_a else 0.0
    rate_b = (win_b / len(track_b) * 100.0) if track_b else 0.0

    decided_items = [x for x in evaluated if x["status"] != "ACTIVE"]
    simple_sum_pct = sum(x["simple_pnl_pct"] for x in decided_items)
    split_sum_pct = sum(x["split_pnl_pct"] for x in decided_items)
    avg_expectancy_pct = (split_sum_pct / len(decided_items)) if decided_items else 0.0
    account_growth_10pct_pos = split_sum_pct * 0.10

    top_winners = sorted(evaluated, key=lambda x: x["valid_max_gain_pct"], reverse=True)[:5]
    stop_hunted = [x for x in sl_list if x["post_sl_rebound_pct"] >= 4.0]

    w1_Pool = [x["waves"]["w1"] for x in evaluated if x["valid_max_gain_pct"] >= 6.0]
    w2_Pool = [x["waves"]["w2"] for x in evaluated if x["valid_max_gain_pct"] >= 12.0]
    w3_Pool = [x["waves"]["w3"] for x in evaluated if x["valid_max_gain_pct"] >= 18.0]

    wk_w1_pct = round(float(np.median([w["gain_pct"] for w in w1_Pool])) if w1_Pool else 9.0, 1)
    wk_w1_fib = round(float(np.median([w["fib_mult"] for w in w1_Pool])) if w1_Pool else 1.85, 2)
    wk_w1_ext = round(float(np.median([w["ma20_ext_pct"] for w in w1_Pool])) if w1_Pool else 4.5, 1)
    wk_w1_rsi = round(float(np.median([w["rsi6"] for w in w1_Pool])) if w1_Pool else 72.0, 0)

    wk_w2_pct = round(float(np.median([w["gain_pct"] for w in w2_Pool])) if w2_Pool else 17.0, 1)
    wk_w2_fib = round(float(np.median([w["fib_mult"] for w in w2_Pool])) if w2_Pool else 2.75, 2)
    wk_w2_ext = round(float(np.median([w["ma20_ext_pct"] for w in w2_Pool])) if w2_Pool else 7.7, 1)
    wk_w2_rsi = round(float(np.median([w["rsi6"] for w in w2_Pool])) if w2_Pool else 78.0, 0)

    wk_w3_pct = round(float(np.median([w["gain_pct"] for w in w3_Pool])) if w3_Pool else 29.0, 1)
    wk_w3_fib = round(float(np.median([w["fib_mult"] for w in w3_Pool])) if w3_Pool else 4.04, 2)
    wk_w3_ext = round(float(np.median([w["ma20_ext_pct"] for w in w3_Pool])) if w3_Pool else 12.5, 1)
    wk_w3_rsi = round(float(np.median([w["rsi6"] for w in w3_Pool])) if w3_Pool else 84.0, 0)

    lines = [
        f"📅 [Bybit 스캐너 주간 실전 성과 & 1·2·3차 파동 심층 리포트 ({start_date_str} ~ {end_date_str})]",
        f"🕒 집계 시각: {now_str} (-{FIXED_SL_PCT}% 고정SL · 1R 본전방어 · 무거운 메이저 배제 기준)\n",
        "1️⃣ [주간 전체 신호(Episode) 성적표]",
        f"• 총 발생 신호: {total_cnt}건 (종결 {decided_cnt}건 | 순항중 {len(act_list)}건)",
        f"• 익절 달성: {win_cnt}건 (🎯3차 {len(tp3_list)}건 | ✅2차 {len(tp2_list)}건 | ✅1차 {len(tp1_list)}건)",
        f"• 본전 방어(🛡️1R 도달 후 보호마감): {be_cnt}건 | 손절 이탈(❌): {sl_cnt}건",
        f"• 순수 익절 승률: {strict_win_rate:.1f}% | 본전방어 포함 무패율: {defended_rate:.1f}%\n",
        "2️⃣ [실전 계좌 수익률 분석 (분할익절 40:30:30 반영)]",
        f"• 신호 1건당 평균 기대수익률(Expectancy): {avg_expectancy_pct:+.2f}%",
        f"• 실전 계좌 수익률 (타점당 시드 10% 투입 기준): {account_growth_10pct_pos:+.2f}%",
        f"• 단순 산술 누적 수익률 (분할익절 기준): {split_sum_pct:+.1f}% (단순 고점 합산 {simple_sum_pct:+.1f}%)\n",
        "3️⃣ [트랙별 적중률 비교 (Track A vs Track B)]",
        f"• 🟢 Track A (1차 선취매/눌림): 종결 {len(track_a)}건 중 {win_a}건 익절 ({rate_a:.1f}%)",
        f"• 🚀 Track B (2차 돌파/S&R/포켓피벗): 종결 {len(track_b)}건 중 {win_b}건 익절 ({rate_b:.1f}%)\n",
        "4️⃣ [주간 베스트 수익 타점 Top 5 & 1·2·3차 파동 로데이터]",
    ]

    for idx, item in enumerate(top_winners, 1):
        dt_s = datetime.fromtimestamp(item["created_at"], tz=KST).strftime("%m/%d %H:%M")
        w1, w2, w3 = item["waves"]["w1"], item["waves"]["w2"], item["waves"]["w3"]
        lines.append(
            f"  {idx}) ${item['base']} ({dt_s} 포착 | {item['score']}점) → 유효최고 +{item['valid_max_gain_pct']:.1f}% {item['badge']} (실현 {item['split_pnl_pct']:+.1f}%)"
        )
        lines.append(
            f"     - [파동궤적]: 1차첫조정 +{w1['gain_pct']:.1f}%(Fib {w1['fib_mult']:.2f}·MA20+{w1['ma20_ext_pct']:.1f}%·{w1['hold_tag']} 지지) "
            f"→ 2차중간고점 +{w2['gain_pct']:.1f}%(Fib {w2['fib_mult']:.2f}·MA20+{w2['ma20_ext_pct']:.1f}%) "
            f"→ 3차최종 +{w3['gain_pct']:.1f}%(Fib {w3['fib_mult']:.2f}·RSI {w3['rsi6']:.0f})"
        )

    lines.append(f"\n5️⃣ [주간 오답 노트: -{FIXED_SL_PCT}% 고정SL 기준 손절 복기 및 저항 로데이터 진단]")
    if not sl_list:
        lines.append("• 최근 7일간 손절 마감된 종목이 없습니다.")
    else:
        lines.append(
            f"• 총 손절 {sl_cnt}건 중 아랫꼬리 스탑헌팅 후 재상승(+4%↑) 사례: {len(stop_hunted)}건"
        )
        for idx, item in enumerate(sl_list[-5:], 1):
            dt_s = datetime.fromtimestamp(item["created_at"], tz=KST).strftime("%m/%d %H:%M")
            w1 = item["waves"]["w1"]
            sh_note = f" | ⚠️손절후 +{item['post_sl_rebound_pct']:.1f}% V자반등" if item["post_sl_rebound_pct"] >= 4.0 else ""
            lines.append(
                f"  - ❌ ${item['base']} ({dt_s} | SL -{item['sl_pct']:g}%): 첫고점 +{item['valid_max_gain_pct']:.1f}% "
                f"(Fib {w1['fib_mult']:.2f}배·{w1['resist_tag']}·RSI {w1['rsi6']:.0f} 저항) → {item['split_pnl_pct']:+.1f}%{sh_note}"
            )

    rec_tp1 = int(np.clip(round(wk_w1_pct), 9, 14))
    rec_tp2 = int(np.clip(round(max(rec_tp1 + 7, wk_w2_pct)), 16, 26))
    rec_tp3 = int(np.clip(round(max(rec_tp2 + 10, wk_w3_pct)), 28, 50))

    lines.extend(
        [
            "\n6️⃣ [주간 전체 코인 1·2·3차 고점 로데이터 종합 결산 & 차주 TP 튜닝안]",
            f"• 1차(첫 조정 형성 지점 중앙값): +{wk_w1_pct:.1f}% | Fib {wk_w1_fib:.2f}배 | 15m MA20이격 +{wk_w1_ext:.1f}% | RSI6 {wk_w1_rsi:.0f}",
            f"• 2차(최종고점 전 중간조정 고점): +{wk_w2_pct:.1f}% | Fib {wk_w2_fib:.2f}배 | 15m MA20이격 +{wk_w2_ext:.1f}% | RSI6 {wk_w2_rsi:.0f}",
            f"• 3차(최종 클라이맥스 고점): +{wk_w3_pct:.1f}% | Fib {wk_w3_fib:.2f}배 | 15m MA20이격 +{wk_w3_ext:.1f}% | RSI6 {wk_w3_rsi:.0f}",
            f"• 💡 [차주 권장 3단계 익절 세팅]: TP1 = +{rec_tp1}% (Fib {wk_w1_fib:.2f}) | TP2 = +{rec_tp2}% (Fib {wk_w2_fib:.2f}) | TP3 = +{rec_tp3}% (Fib {wk_w3_fib:.2f})",
        ]
    )

    elapsed = time.time() - start_t
    lines.append(f"\n⏱️ 리포트 생성 소요 시간: {elapsed:.1f}초")
    send_telegram_chunked("\n".join(lines))
    print(f"[{now_str}] 주간 성과 및 3파동 로데이터 리포트 전송 완료 ({elapsed:.1f}초 소요)")


if __name__ == "__main__":
    main()
