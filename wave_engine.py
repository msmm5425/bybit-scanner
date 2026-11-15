import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import timedelta, timezone

try:
    import config as cfg
    TELEGRAM_BOT_TOKEN = getattr(cfg, "TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = getattr(cfg, "TELEGRAM_CHAT_ID", "")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
except Exception:
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    KST = timezone(timedelta(hours=9))
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}

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


def calc_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=3).mean().bfill()


def calc_obv_and_cvd(df):
    close_diff = df["close"].diff().fillna(0.0)
    direction = np.where(close_diff > 0, 1.0, np.where(close_diff < 0, -1.0, 0.0))
    obv = (df["volume"] * direction).cumsum()

    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    buy_ratio = ((df["close"] - df["low"]) / candle_range).fillna(0.5).clip(0.0, 1.0)
    net_delta = df["volume"] * (2.0 * buy_ratio - 1.0)
    cvd = net_delta.cumsum()
    return obv, cvd, net_delta


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


def extract_three_wave_structure(df, ign_pos, peak_pos, entry_price, support_line, wave_h):
    """
    발화점(ign_pos)부터 최종 고점(peak_pos)까지의 캔들 궤적을 스캔하여:
    - 1차(Wave 1): 발화 직후 미세 노이즈를 배제하고 최소 +5.5%(또는 전체 파동의 22%) 이상 상승 후 첫 조정이 형성된 1차 저항 고점
    - 2차(Wave 2): 최종 고점 직전 중간 조정이 형성된 2차 저항 고점
    - 3차(Wave 3): 최종 클라이맥스 고점
    """
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
                ma10_t = to_float(df.loc[trough_idx, "ma10"], trough_low) if "ma10" in df.columns else ma5_t
                ma20_t = to_float(df.loc[trough_idx, "ma20"], trough_low)
                ma50_t = to_float(df.loc[trough_idx, "ma50"], trough_low) if "ma50" in df.columns else ma20_t

                if trough_low >= ma5_t * 0.996:
                    hold_tag = "MA5"
                elif trough_low >= ma10_t * 0.995:
                    hold_tag = "MA10"
                elif trough_low >= ma20_t * 0.993:
                    hold_tag = "MA20"
                elif trough_low >= ma50_t * 0.993:
                    hold_tag = "MA50"
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
        ma50_ext_pct = max(0.1, (h_p - ma50_v) / max(ma50_v, 1e-8) * 100.0)

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
            "ma50_ext_pct": round(ma50_ext_pct, 1),
            "vol_mult": round(vol_m, 1),
            "pullback_pct": round(pb_pct, 1),
            "hold_tag": hold_str,
            "resist_tag": resist_tag,
        }

    w1 = _build_metrics(w1_info["idx"], w1_info["high"], w1_info["pullback_pct"], w1_info["hold_tag"])
    w2 = _build_metrics(w2_info["idx"], max(w2_info["high"], w1["price"] * 1.02), w2_info["pullback_pct"], w2_info["hold_tag"])
    w3 = _build_metrics(final_idx, max(final_high, w2["price"] * 1.02), 0.0, "클라이맥스")

    return {"w1": w1, "w2": w2, "w3": w3}
