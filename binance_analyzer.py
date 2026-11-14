import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

try:
    import config as cfg
    TELEGRAM_BOT_TOKEN = getattr(cfg, "TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = getattr(cfg, "TELEGRAM_CHAT_ID", "")
    STATE_FILE = getattr(cfg, "STATE_FILE", "bot_state.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
    is_tradfi_ticker = getattr(cfg, "is_tradfi_ticker", lambda x: False)
    NON_CRYPTO_META_KEYWORDS = getattr(cfg, "NON_CRYPTO_META_KEYWORDS", ())
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception as e:
    print(f"[설정 로드 경고] config.py 기본값 대체: {e}")
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    STATE_FILE = "bot_state.json"
    KST = timezone(timedelta(hours=9))
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}
    is_tradfi_ticker = lambda x: False
    NON_CRYPTO_META_KEYWORDS = ()
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

HISTORY_FILE = "analyzer_history.json"


def check_required_secrets():
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[경고] TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID 환경변수가 설정되지 않았습니다.")
        return False
    return True


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def normalize_funding_pct(raw_val):
    v = to_float(raw_val, 0.0)
    if abs(v) > 0.15:
        return v / 100.0 if abs(v) <= 2.5 else v
    return v


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


def _normalize_kline_df(raw, expected_interval_min=None):
    df = pd.DataFrame()
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        df = raw.copy()
        col_map = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ("start", "starttime", "open_time", "time", "date", "datetime", "t"):
                col_map[c] = "timestamp"
            elif cl in ("o", "openprice"):
                col_map[c] = "open"
            elif cl in ("h", "highprice"):
                col_map[c] = "high"
            elif cl in ("l", "lowprice"):
                col_map[c] = "low"
            elif cl in ("c", "closeprice"):
                col_map[c] = "close"
            elif cl in ("v", "vol", "basevol"):
                col_map[c] = "volume"
        if col_map:
            df = df.rename(columns=col_map)
    elif isinstance(raw, list) and len(raw) > 0:
        try:
            if isinstance(raw[0], dict):
                return _normalize_kline_df(pd.DataFrame(raw), expected_interval_min)
            cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(raw[0])]
            df = pd.DataFrame(raw, columns=cols)
        except Exception:
            return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            return pd.DataFrame()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" in df.columns:
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.notna().any():
            if ts.max() < 1e11:
                ts = ts * 1000.0
            df["timestamp"] = ts
            df = df.sort_values("timestamp", ascending=True)
            if expected_interval_min is not None and len(df) >= 5:
                median_diff_sec = df["timestamp"].diff().median() / 1000.0
                target_sec = expected_interval_min * 60.0
                if not (target_sec * 0.6 <= median_diff_sec <= target_sec * 1.4):
                    return pd.DataFrame()

    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_klines(symbol, interval="5", limit=300):
    iv_min = int(interval)
    base = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = base
    for prefix in ("1000000", "100000", "10000", "1000", "100"):
        if clean_base.startswith(prefix) and len(clean_base) > len(prefix):
            clean_base = clean_base[len(prefix):]
            break

    gate_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h"}.get(iv_min, "5m")
    for contract in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            resp = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": contract, "interval": gate_iv, "limit": int(limit)},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                df = _normalize_kline_df(resp.json(), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 20:
                    return df
        except requests.RequestException:
            pass

    bg_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1H", 240: "4H"}.get(iv_min, "5m")
    for bg_sym in (symbol, f"{clean_base}USDT"):
        try:
            resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": bg_sym, "productType": "USDT-FUTURES", "granularity": bg_iv, "limit": str(min(int(limit), 200))},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                df = _normalize_kline_df(resp.json().get("data", []), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 20:
                    return df
        except requests.RequestException:
            pass

    bybit_params = {"category": "linear", "symbol": symbol, "interval": str(interval), "limit": int(limit)}
    for host in ("https://api.bybit-tr.com", "https://api.bybit.kz", "https://api.bytick.com"):
        try:
            resp = requests.get(f"{host}/v5/market/kline", params=bybit_params, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                df = _normalize_kline_df(resp.json().get("result", {}).get("list", []), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 20:
                    return df
        except requests.RequestException:
            pass

    mexc_iv = {1: "Min1", 5: "Min5", 15: "Min15", 60: "Min60"}.get(iv_min, "Min5")
    for m_sym in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            resp = requests.get(
                f"https://contract.mexc.com/api/v1/contract/kline/{m_sym}",
                params={"interval": mexc_iv},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                d = resp.json().get("data", {})
                if isinstance(d, dict) and d.get("close"):
                    df = pd.DataFrame({
                        "timestamp": pd.to_numeric(d.get("time", []), errors="coerce") * 1000.0,
                        "open": pd.to_numeric(d.get("open", []), errors="coerce"),
                        "high": pd.to_numeric(d.get("high", []), errors="coerce"),
                        "low": pd.to_numeric(d.get("low", []), errors="coerce"),
                        "close": pd.to_numeric(d.get("close", []), errors="coerce"),
                        "volume": pd.to_numeric(d.get("vol", []), errors="coerce"),
                    })
                    df = _normalize_kline_df(df.iloc[-int(limit):], expected_interval_min=iv_min)
                    if not df.empty and len(df) >= 20:
                        return df
        except requests.RequestException:
            pass

    return pd.DataFrame()


def enrich_ticker_metrics(candidates):
    ticker_map = {}
    for host in ("https://api.bybit-tr.com", "https://api.bybit.kz", "https://api.bytick.com"):
        try:
            r = requests.get(f"{host}/v5/market/tickers", params={"category": "linear"}, headers=BROWSER_HEADERS, timeout=5)
            if r.status_code == 200:
                items = r.json().get("result", {}).get("list", [])
                if items:
                    for it in items:
                        s = str(it.get("symbol", ""))
                        ticker_map[s] = {
                            "turnover_m": to_float(it.get("turnover24h")) / 1e6,
                            "oi_m": to_float(it.get("openInterestValue")) / 1e6,
                            "funding_pct": to_float(it.get("fundingRate")) * 100.0,
                        }
                    break
        except requests.RequestException:
            pass

    gate_map = {}
    try:
        r = requests.get("https://api.gateio.ws/api/v4/futures/usdt/tickers", headers=BROWSER_HEADERS, timeout=5)
        if r.status_code == 200:
            for it in r.json():
                sym = str(it.get("contract", "")).replace("_", "")
                vol_usd_m = to_float(it.get("volume_24h_settle") or it.get("volume_24h_quote")) / 1e6
                mark_p = to_float(it.get("mark_price") or it.get("last"))
                quanto = to_float(it.get("quanto_multiplier"), 1.0)
                total_size = to_float(it.get("total_size"))
                oi_usd_m = (total_size * quanto * mark_p) / 1e6 if quanto > 0 else 0.0
                gate_map[sym] = {
                    "turnover_m": vol_usd_m,
                    "oi_m": oi_usd_m,
                    "funding_pct": to_float(it.get("funding_rate")) * 100.0,
                }
    except requests.RequestException:
        pass

    filtered_candidates = []
    for c in candidates:
        sym = c.get("symbol", "")
        base = sym[:-4].upper() if sym.endswith("USDT") else sym.upper()
        if is_tradfi_ticker(base):
            continue

        t_m = to_float(c.get("turnover_m") or c.get("turnover24h_m") or c.get("vol_m"))
        if t_m <= 0 and to_float(c.get("turnover24h")) > 0:
            t_m = to_float(c.get("turnover24h")) / 1e6

        o_m = to_float(c.get("oi_m") or c.get("open_interest_m") or c.get("oi_usd_m"))
        if o_m <= 0 and to_float(c.get("openInterestValue")) > 0:
            o_m = to_float(c.get("openInterestValue")) / 1e6

        ext = ticker_map.get(sym) or gate_map.get(sym) or {}
        if t_m <= 0:
            t_m = to_float(ext.get("turnover_m"), 0.0)
        if o_m <= 0:
            o_m = to_float(ext.get("oi_m"), 0.0)

        if "funding_pct" in c and c.get("funding_pct") is not None:
            f_p = normalize_funding_pct(c.get("funding_pct"))
        elif c.get("fundingRate") is not None:
            f_p = normalize_funding_pct(to_float(c.get("fundingRate")) * 100.0)
        elif c.get("funding_rate") is not None:
            f_p = normalize_funding_pct(c.get("funding_rate"))
        else:
            f_p = normalize_funding_pct(ext.get("funding_pct", 0.0))

        if f_p == 0.0 and ext.get("funding_pct") is not None:
            f_p = normalize_funding_pct(ext.get("funding_pct"))

        c["turnover_m"] = t_m
        c["oi_m"] = o_m
        c["funding_pct"] = f_p
        filtered_candidates.append(c)

    return filtered_candidates


def fetch_top_gainer_candidates():
    if api_mod is not None and hasattr(api_mod, "get_scanner_candidates"):
        try:
            res = api_mod.get_scanner_candidates()
            cands = res[0] if isinstance(res, tuple) else res
            if isinstance(cands, list) and len(cands) > 0:
                return enrich_ticker_metrics(cands)
        except Exception as e:
            print(f"[api.get_scanner_candidates 실패 -> 직통 폴백 전환] {e}")

    try:
        bg_resp = requests.get(
            "https://api.bitget.com/api/v2/mix/market/tickers",
            params={"productType": "USDT-FUTURES"},
            headers=BROWSER_HEADERS,
            timeout=6,
        )
        if bg_resp.status_code == 200:
            items = bg_resp.json().get("data", [])
            fallback_cands = []
            for it in items:
                sym = str(it.get("symbol", ""))
                if not sym.endswith("USDT"):
                    continue
                base = sym[:-4].upper()
                if is_tradfi_ticker(base):
                    continue
                turnover_m = to_float(it.get("usdtVolume") or it.get("quoteVolume")) / 1e6
                if turnover_m < 2.0:
                    continue
                fallback_cands.append({
                    "symbol": sym,
                    "base_coin": base,
                    "price": to_float(it.get("lastPr")),
                    "change_24h": to_float(it.get("change24h")) * 100.0,
                    "turnover_m": turnover_m,
                    "oi_m": to_float(it.get("holdingAmount")) * to_float(it.get("lastPr")) / 1e6,
                    "funding_pct": to_float(it.get("fundingRate")) * 100.0,
                })
            if fallback_cands:
                return enrich_ticker_metrics(fallback_cands)
    except Exception as e:
        print(f"[자체 폴백 후보군 조회 오류] {e}")

    return []


def send_telegram_chunked(text):
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

    sent_any = False
    for chunk in chunks:
        if notifier_mod is not None and hasattr(notifier_mod, "send_telegram_single"):
            try:
                notifier_mod.send_telegram_single(chunk)
                sent_any = True
                time.sleep(0.5)
                continue
            except Exception:
                pass

        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            try:
                resp = requests.post(
                    url,
                    json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "disable_web_page_preview": True},
                    timeout=15,
                )
                if resp.status_code == 200:
                    sent_any = True
                    time.sleep(0.5)
            except Exception as e:
                print(f"[텔레그램 전송 예외] {e}")

    return sent_any


def extract_ts_ms(df, pos, now_ts, interval_min=5):
    fallback_ms = int((now_ts - max(0, (len(df) - 1 - pos)) * interval_min * 60) * 1000)
    if "timestamp" in df.columns:
        val = df["timestamp"].iloc[pos]
        if isinstance(val, pd.Timestamp):
            return int(val.timestamp() * 1000)
        try:
            num = float(val)
            if not np.isnan(num) and num > 0:
                return int(num * 1000) if num < 1e11 else int(num)
        except Exception:
            pass
    return fallback_ms


def slice_df_window_ms(df, start_ms, end_ms, now_ts, interval_min=5):
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        try:
            ts_s = pd.to_numeric(df["timestamp"], errors="coerce")
            if ts_s.notna().any():
                if ts_s.max() < 1e11:
                    ts_s = ts_s * 1000.0
                sub = df[(ts_s >= start_ms) & (ts_s <= end_ms)]
                if not sub.empty:
                    return sub
        except Exception:
            pass
    n = len(df)
    end_bars_ago = max(0, int((now_ts * 1000 - end_ms) / (interval_min * 60000)))
    start_bars_ago = max(end_bars_ago + 1, int((now_ts * 1000 - start_ms) / (interval_min * 60000)) + 1)
    idx_start = max(0, n - start_bars_ago)
    idx_end = max(idx_start + 1, n - end_bars_ago)
    return df.iloc[idx_start:idx_end]


def load_json_file(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[JSON 로드 경고] {path}: {e}")
    return default_val


def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[파일 저장 오류] {path}: {e}")


# =====================================================================
# [핵심 엔진: 1차(첫 조정 고점) · 2차(중간 조정 고점) · 3차(최종 고점) 로데이터 추출]
# =====================================================================
def extract_three_wave_structure(df, ign_pos, peak_pos, entry_price, support_line, wave_h):
    """
    발화점(ign_pos)부터 최종 고점(peak_pos)까지의 캔들 궤적을 정밀 스캔하여:
    - 1차(Wave 1): 첫 조정이 형성된 첫 저항 고점
    - 2차(Wave 2): 최종 고점 직전 중간 조정이 형성된 2차 저항 고점
    - 3차(Wave 3): 최종 클라이맥스 고점
    각 지점의 [상승률%, Fib배수, MA20이격%, MA50이격%, RSI6, KDJ_J, 거래량배수, 조정폭%, 지지이평선] 추출
    """
    sub = df.iloc[ign_pos: peak_pos + 1].copy()
    if sub.empty:
        sub = df.iloc[ign_pos: ign_pos + 1].copy()

    positions = list(sub.index)
    n_bars = len(positions)

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
        if gain_so_far < 2.0:
            continue

        lookahead_indices = positions[k + 1: min(len(positions), k + 6)]
        if not lookahead_indices:
            continue
        min_low_ahead = min(to_float(df.loc[idx, "low"], running_max_h) for idx in lookahead_indices)
        pullback_pct = (running_max_h - min_low_ahead) / max(running_max_h, 1e-8) * 100.0

        next_idx = positions[k + 1]
        is_red_turn = to_float(df.loc[next_idx, "close"]) < to_float(df.loc[next_idx, "open"])

        if pullback_pct >= 1.2 or (is_red_turn and pullback_pct >= 0.7):
            if not swing_peaks or swing_peaks[-1]["idx"] != running_max_idx:
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
                    hold_tag = "S/R박스상단"

                swing_peaks.append({
                    "idx": running_max_idx,
                    "high": running_max_h,
                    "pullback_pct": round(pullback_pct, 1),
                    "hold_tag": hold_tag,
                })

    final_idx = positions[-1]
    if len(swing_peaks) >= 2:
        w1_info = swing_peaks[0]
        mid_candidates = [sp for sp in swing_peaks[1:] if sp["idx"] != final_idx]
        w2_info = mid_candidates[-1] if mid_candidates else swing_peaks[-1]
    elif len(swing_peaks) == 1:
        w1_info = swing_peaks[0]
        mid_pos = positions[min(n_bars - 1, max(0, (positions.index(w1_info["idx"]) + n_bars - 1) // 2))]
        w2_info = {
            "idx": mid_pos,
            "high": max(to_float(df.loc[mid_pos, "high"]), (w1_info["high"] + to_float(df.loc[final_idx, "high"])) * 0.5),
            "pullback_pct": 1.5,
            "hold_tag": "MA10",
        }
    else:
        idx1 = positions[max(0, n_bars // 3)]
        idx2 = positions[max(0, (n_bars * 2) // 3)]
        w1_info = {"idx": idx1, "high": to_float(df.loc[idx1, "high"], entry_price * 1.08), "pullback_pct": 1.2, "hold_tag": "MA5"}
        w2_info = {"idx": idx2, "high": to_float(df.loc[idx2, "high"], entry_price * 1.15), "pullback_pct": 1.8, "hold_tag": "MA10"}

    def _build_wave_metrics(idx_val, high_override, pb_pct=0.0, hold_str=""):
        row = df.loc[idx_val]
        h_p = max(to_float(row["high"]), to_float(high_override), entry_price * 1.01)
        ma20_v = to_float(row.get("ma20"), entry_price)
        ma50_v = to_float(row.get("ma50"), ma20_v)
        rsi_v = to_float(row.get("rsi6"), 68.0)
        kdj_v = to_float(row.get("kdj_j"), 75.0)
        vol_m = to_float(row.get("vol_mult"), 2.0)

        gain_pct = max(0.5, (h_p - entry_price) / max(entry_price, 1e-8) * 100.0)
        fib_mult = max(1.0, (h_p - support_line) / max(wave_h, 1e-8))
        ma20_ext_pct = max(0.2, (h_p - ma20_v) / max(ma20_v, 1e-8) * 100.0)
        ma50_ext_pct = max(0.2, (h_p - ma50_v) / max(ma50_v, 1e-8) * 100.0)

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
        }

    w1 = _build_wave_metrics(w1_info["idx"], w1_info["high"], w1_info["pullback_pct"], w1_info["hold_tag"])
    w2 = _build_wave_metrics(w2_info["idx"], max(w2_info["high"], w1["price"] * 1.02), w2_info["pullback_pct"], w2_info["hold_tag"])
    w3 = _build_wave_metrics(final_idx, max(to_float(df.loc[final_idx, "high"]), w2["price"] * 1.02), 0.0, "클라이맥스")

    return {"w1": w1, "w2": w2, "w3": w3}


# =====================================================================
# [1. 전문가급 급등 Top 5 초동 발화점 및 1·2·3차 파동 로데이터 역추적]
# =====================================================================
def reverse_engineer_top_coin(coin, now_ts):
    symbol = coin.get("symbol", "UNKNOWN")
    try:
        df_5m = fetch_klines(symbol, "5", limit=300)
        if df_5m is None or df_5m.empty or len(df_5m) < 25:
            return None

        df_1h = fetch_klines(symbol, "60", limit=75)
        df_5m = df_5m.reset_index(drop=True)

        for p in (5, 10, 20, 50, 112):
            df_5m[f"ma{p}"] = df_5m["close"].rolling(p, min_periods=3).mean()
            df_5m[f"ema{p}"] = df_5m["close"].ewm(span=p, adjust=False).mean()

        df_5m["rsi6"] = calc_rsi(df_5m["close"], 6)
        df_5m["kdj_j"] = calc_kdj_j(df_5m, 9)
        df_5m["vol_ma20"] = df_5m["volume"].rolling(20, min_periods=3).mean().shift(1).bfill()
        df_5m["vol_mult"] = (df_5m["volume"] / df_5m["vol_ma20"].replace(0, np.nan)).fillna(1.0)
        atr_5m = calc_atr(df_5m, 14)

        ribbon_max = df_5m[["ema5", "ema20", "ema50"]].max(axis=1)
        ribbon_min = df_5m[["ema5", "ema20", "ema50"]].min(axis=1)
        df_5m["ribbon_spread"] = (ribbon_max - ribbon_min) / df_5m["close"].replace(0, np.nan) * 100.0

        obv, cvd, _ = calc_obv_and_cvd(df_5m)
        df_5m["obv"] = obv
        df_5m["cvd"] = cvd

        start_idx = min(20, max(3, len(df_5m) // 8))
        search_df = df_5m.iloc[start_idx:-2].copy()
        if search_df.empty:
            search_df = df_5m.iloc[:-1].copy()

        overall_low = to_float(search_df["low"].min(), to_float(df_5m["close"].iloc[-1]))
        overall_high = to_float(search_df["high"].max(), to_float(df_5m["close"].iloc[-1]))
        price_ceiling_for_ignition = overall_low + (overall_high - overall_low) * 0.55

        ign_candidates = search_df[
            (search_df["close"] > search_df["open"])
            & (search_df["vol_mult"] >= 2.2)
            & (search_df["close"] <= price_ceiling_for_ignition * 1.05)
        ]
        if ign_candidates.empty:
            ign_candidates = search_df[(search_df["close"] > search_df["open"]) & (search_df["vol_mult"] >= 1.6)]
        if ign_candidates.empty:
            ign_idx = search_df["vol_mult"].idxmax()
        else:
            ign_idx = ign_candidates.index[0]

        ign_pos = int(df_5m.index.get_loc(ign_idx))
        ign_row = df_5m.iloc[ign_pos]
        ign_price = to_float(ign_row["close"])
        ign_open = to_float(ign_row["open"], ign_price)
        ign_high = to_float(ign_row["high"], ign_price)
        ign_low = to_float(ign_row["low"], ign_price)
        ign_vol = to_float(ign_row["volume"], 1.0)
        ign_vol_mult = to_float(ign_row["vol_mult"], 2.5)

        ign_ts_ms = extract_ts_ms(df_5m, ign_pos, now_ts, interval_min=5)
        ign_dt = datetime.fromtimestamp(ign_ts_ms / 1000.0, tz=KST)
        ign_time_str = ign_dt.strftime("%m/%d %H:%M KST")
        ign_minute = ign_dt.minute

        post_df = df_5m.iloc[ign_pos:]
        peak_pos_rel = post_df["high"].idxmax()
        peak_pos = int(df_5m.index.get_loc(peak_pos_rel))
        peak_high = to_float(post_df["high"].max(), ign_price)
        pump_from_ign_pct = max(1.0, (peak_high - ign_price) / max(ign_price, 1e-8) * 100.0)
        hours_to_peak = max(0.2, round((peak_pos - ign_pos) * 5.0 / 60.0, 1))

        pre_start = max(0, ign_pos - 24)
        pre_df = df_5m.iloc[pre_start:ign_pos]
        if pre_df.empty:
            pre_df = df_5m.iloc[:max(1, ign_pos)]
        pre_high = to_float(pre_df["high"].max(), ign_price)
        pre_low = to_float(pre_df["low"].min(), ign_price)
        pre_box_pct = max(1.0, (pre_high - pre_low) / max(pre_low, 1e-8) * 100.0)

        wave_h = max(ign_high - pre_low, ign_price * 0.045)
        waves = extract_three_wave_structure(df_5m, ign_pos, peak_pos, ign_price, pre_low, wave_h)

        pre5_df = df_5m.iloc[max(0, ign_pos - 6):ign_pos]
        if pre5_df.empty:
            pre5_df = pre_df
        vol_ma20_at_ign = to_float(ign_row["vol_ma20"], 1.0)
        pre5_min_vol_ratio = to_float(pre5_df["volume"].min(), vol_ma20_at_ign) / max(vol_ma20_at_ign, 1e-8)
        pre5_min_ribbon = to_float(pre5_df["ribbon_spread"].min(), 2.5)

        ma5_i = to_float(ign_row["ma5"], ign_price)
        ma20_i = to_float(ign_row["ma20"], ign_price)
        ma50_i = to_float(ign_row["ma50"], ign_price)
        ma112_i = to_float(ign_row["ma112"], ign_price)

        full_aligned = ma5_i > ma20_i > ma50_i > ma112_i
        short_aligned = ma5_i > ma20_i > ma50_i

        min_body_floor = pre5_df["close"] * 0.0015
        body_s = np.maximum((pre5_df["close"] - pre5_df["open"]).abs(), min_body_floor)
        l_wick = (pre5_df[["open", "close"]].min(axis=1) - pre5_df["low"]).clip(lower=0.0)
        spring_wick_ratio = min(15.0, to_float((l_wick / body_s).max(), 0.0))
        cond_spring = (not short_aligned) and (spring_wick_ratio >= 1.4) and (ign_price > ma20_i)

        obv_up = to_float(pre_df["obv"].iloc[-1], 0) > to_float(pre_df["obv"].iloc[:10].mean(), 0) if len(pre_df) >= 4 else True
        cvd_up = to_float(pre_df["cvd"].iloc[-1], 0) > to_float(pre_df["cvd"].iloc[:10].mean(), 0) if len(pre_df) >= 4 else True

        bb_std = df_5m["close"].rolling(20, min_periods=5).std()
        bb_u = df_5m["ma20"] + 2.0 * bb_std
        bb_l = df_5m["ma20"] - 2.0 * bb_std
        kc_u = df_5m["ma20"] + 1.5 * atr_5m
        kc_l = df_5m["ma20"] - 1.5 * atr_5m
        ttm_sqz = bool(((bb_u < kc_u) & (bb_l > kc_l)).iloc[max(0, ign_pos - 8):max(1, ign_pos)].any())

        after_4 = df_5m.iloc[ign_pos + 1: min(len(df_5m), ign_pos + 5)]
        if not after_4.empty:
            pullback_low = to_float(after_4["low"].min(), ign_price)
            pullback_min_vol = to_float(after_4["volume"].min(), ign_vol)
            dryup_ratio = pullback_min_vol / max(ign_vol, 1e-8)
        else:
            pullback_low = ign_low
            dryup_ratio = 0.25

        sr_err_pct = abs(pullback_low - pre_high) / max(pre_high, 1e-8) * 100.0
        impulse_len = max(ign_high - pre_high, ign_price * 0.004)
        bull_flag_line = ign_high - impulse_len * 0.382
        candle_mid_fvg = (ign_open + ign_high) / 2.0

        is_sr_flip = sr_err_pct <= 1.50 and dryup_ratio <= 0.40
        is_bull_flag = (pullback_low >= bull_flag_line) and (dryup_ratio <= 0.45)
        is_fvg_50_hold = (pullback_low >= candle_mid_fvg * 0.998) and (dryup_ratio <= 0.45)
        is_pocket_pivot = (pre5_min_vol_ratio <= 0.42) and (ign_vol_mult >= 2.8)

        pre_peak_df = df_5m.iloc[ign_pos: peak_pos + 1]
        min_after_ign = to_float(pre_peak_df["low"].min(), ign_low)
        mae_pct = max(1.5, (ign_price - min_after_ign) / max(ign_price, 1e-8) * 100.0)

        if df_1h is not None and not df_1h.empty and len(df_1h) >= 3:
            ma20_1h = to_float(df_1h["close"].rolling(20, min_periods=2).mean().iloc[-1], ign_price)
            htf_aligned = to_float(df_5m["close"].iloc[-1]) >= ma20_1h * 0.99
        else:
            htf_aligned = to_float(df_5m["close"].iloc[-1]) >= ma112_i * 0.99

        turnover_m = to_float(coin.get("turnover_m"))
        oi_m = to_float(coin.get("oi_m"))
        funding_pct = normalize_funding_pct(coin.get("funding_pct", 0.0))
        change_24h = to_float(coin.get("change_24h"), to_float(coin.get("price24hPcnt")) * 100.0)
        oi_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0 if turnover_m > 0 else 0.0

        alphas = []
        if is_bull_flag:
            alphas.append(f"High-Tight Bull Flag 공중눌림(돌파봉 상단 38.2% 사수·거래량 {int(round(dryup_ratio*100))}% 건조)")
        if is_pocket_pivot:
            alphas.append(f"포켓피벗(Pocket Pivot) 초동 직격(발화전 거래량 {int(round(pre5_min_vol_ratio*100))}% 극건조 후 {ign_vol_mult:.1f}배 폭발)")
        if is_fvg_50_hold and not is_bull_flag:
            alphas.append("장대양봉 중심값(50% Equilibrium/FVG) 칼지지 반등")
        if cond_spring:
            alphas.append(f"유동성 스윕(Spring 아랫꼬리 {spring_wick_ratio:.1f}배 스탑헌팅 후 MA20 탈환)")
        if cvd_up:
            alphas.append("CVD 강세/흡수 다이버전스(선행 시장가 매수 누적)")
        if ttm_sqz:
            alphas.append("TTM 켈트너 스퀴즈 폭발(BB⊂KC 변동성 압축 후 상단 분출)")
        if pre5_min_ribbon <= 2.2:
            alphas.append(f"이평선 리본 초밀집 확산(5·20·50 이격 {pre5_min_ribbon:.1f}% 수렴 후 분출)")
        if ign_minute in (0, 5, 10, 15):
            alphas.append(f"정각 오프닝 드라이브({ign_dt.strftime('%H:%M')} 새 1시간봉 시동 수급)")
        elif ign_minute in (45, 50, 55):
            alphas.append(f"정각 마감 전 종가관리 피벗({ign_dt.strftime('%H:%M')} 선취매 수급)")
        if htf_aligned:
            alphas.append("상위프레임(1H 중심선) 구조적 지지 반등")
        if is_sr_flip:
            alphas.append(f"돌파후 박스상단 S/R Flip 리테스트(오차 {sr_err_pct:.2f}%·거래량 {int(round(dryup_ratio*100))}% 건조)")
        if funding_pct <= 0.015:
            alphas.append(f"숏스퀴즈/청정 수급(펀딩 {funding_pct:+.4f}%·OI비중 {int(round(oi_ratio))}%)")

        if is_sr_flip:
            post_pattern_str = f"전고점 S/R Flip 칼지지(오차 {sr_err_pct:.2f}%) (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"
        elif is_bull_flag:
            post_pattern_str = f"High-Tight Bull Flag 공중눌림(상단 38.2% 지지) (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"
        elif is_fvg_50_hold:
            post_pattern_str = f"장대양봉 50%(FVG) 중심선 지지 반등 (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"
        else:
            post_pattern_str = f"단기 자율 눌림 후 반등 (거래량 {int(round(dryup_ratio*100))}% 수준 건조)"

        if full_aligned:
            ma_str = "[✅ 완전 정배열(5>20>50>112)]"
        elif short_aligned:
            ma_str = "[✅ 단기 정배열(5>20>50)]"
        else:
            ma_str = "[⚠️ 비정배열(역배열/혼조 돌파)]"

        obv_str = "[✅ 우상향 선행매집]" if (obv_up or cvd_up) else "[➖ 중립/횡보]"

        return {
            "symbol": symbol,
            "base": symbol.replace("USDT", ""),
            "price": to_float(df_5m["close"].iloc[-1]),
            "change_24h": change_24h,
            "turnover_m": turnover_m,
            "oi_m": oi_m,
            "oi_ratio": oi_ratio,
            "funding_pct": funding_pct,
            "ign_time_str": ign_time_str,
            "ign_price": ign_price,
            "peak_high": peak_high,
            "pump_from_ign_pct": pump_from_ign_pct,
            "waves": waves,
            "hours_to_peak": hours_to_peak,
            "ma_str": ma_str,
            "full_aligned": full_aligned,
            "short_aligned": short_aligned,
            "cond_spring": cond_spring,
            "pre_box_pct": pre_box_pct,
            "obv_str": obv_str,
            "obv_up": obv_up,
            "cvd_up": cvd_up,
            "ign_vol_mult": ign_vol_mult,
            "pre5_min_vol_ratio": pre5_min_vol_ratio,
            "pre5_min_ribbon": pre5_min_ribbon,
            "dryup_ratio": dryup_ratio,
            "is_sr_flip": is_sr_flip,
            "is_bull_flag": is_bull_flag,
            "is_fvg_50_hold": is_fvg_50_hold,
            "is_pocket_pivot": is_pocket_pivot,
            "ttm_sqz": ttm_sqz,
            "htf_aligned": htf_aligned,
            "post_pattern_str": post_pattern_str,
            "alphas": alphas,
            "mae_pct": mae_pct,
            "support_line": pre_low,
            "swing_high_line": ign_high,
        }
    except Exception as e:
        print(f"[종목 역추적 오류] {symbol}: {type(e).__name__} - {e}")
        return None


# =====================================================================
# [2. 내 스캐너 포착 종목(다른 코인들)의 1·2·3차 고점 형성 지점 & 오답 노트 심층 역추적]
# =====================================================================
def analyze_scanner_coins_and_postmortem(now_ts):
    state = load_json_file(STATE_FILE, {"tracked": {}, "sl_history": {}, "signal_episodes": {}})
    if not isinstance(state, dict):
        state = {"tracked": {}, "sl_history": {}, "signal_episodes": {}}

    tracked = state.get("tracked", {})
    if not isinstance(tracked, dict):
        tracked = {}

    sl_history = state.get("sl_history", {})
    if not isinstance(sl_history, dict):
        sl_history = {}

    tp_win_count = 0
    be_protect_count = 0
    active_count = 0
    sl_count_48h = 0

    # 스캐너 포착 코인(다른 코인들)의 1차(첫 조정)·2차(중간 고점)·3차(최종 고점) 로데이터 저장소
    scanner_wave_stats = {
        "w1_gains": [], "w1_fibs": [], "w1_ma20_exts": [], "w1_ma50_exts": [], "w1_rsis": [], "w1_kdjs": [], "w1_holds": [],
        "w2_gains": [], "w2_fibs": [], "w2_ma20_exts": [], "w2_ma50_exts": [], "w2_rsis": [], "w2_kdjs": [],
        "w3_gains": [], "w3_fibs": [], "w3_ma20_exts": [], "w3_ma50_exts": [], "w3_rsis": [], "w3_kdjs": [],
        "coin_wave_samples": [],
    }

    for sym, t_item in tracked.items():
        if not isinstance(t_item, dict):
            continue
        c_at = int(t_item.get("created_at", 0))
        if (now_ts - c_at) > 172800:
            continue

        stop_p_val = to_float(t_item.get("initial_stop_price") or t_item.get("stop_price"))
        entry_low = to_float(t_item.get("entry_low"))
        entry_high = to_float(t_item.get("entry_high"))
        entry_p = to_float(t_item.get("entry_price"))
        if entry_p <= 0:
            entry_p = (entry_low + entry_high) / 2.0 if entry_low > 0 else stop_p_val

        raw_sl_pct = to_float(t_item.get("sl_pct"), 0.0)
        if raw_sl_pct <= 0 and entry_p > 0 and stop_p_val > 0:
            raw_sl_pct = round(abs(entry_p - stop_p_val) / entry_p * 100.0, 1)
        item_sl_pct = float(np.clip(raw_sl_pct if raw_sl_pct > 0 else float(FIXED_SL_PCT), 2.0, 8.0))

        tps_list = t_item.get("tps") or []
        tp1_val = to_float(tps_list[0]) if len(tps_list) >= 1 else entry_p * 1.100

        if entry_p > 0:
            df_15m_chk = fetch_klines(sym, "15", limit=180)
            if not df_15m_chk.empty:
                for p in (5, 10, 20, 50, 112):
                    df_15m_chk[f"ma{p}"] = df_15m_chk["close"].rolling(p, min_periods=2).mean()
                df_15m_chk["rsi6"] = calc_rsi(df_15m_chk["close"], 6)
                df_15m_chk["kdj_j"] = calc_kdj_j(df_15m_chk, 9)
                df_15m_chk["vol_ma20"] = df_15m_chk["volume"].rolling(20, min_periods=3).mean().shift(1).bfill()
                df_15m_chk["vol_mult"] = (df_15m_chk["volume"] / df_15m_chk["vol_ma20"].replace(0, np.nan)).fillna(1.0)

                # 손절된 종목은 손절 시점 이전까지만 고점 측정
                end_eval_ms = int(t_item.get("sl_at") or now_ts) * 1000 if t_item.get("sl_alerted") else now_ts * 1000
                win_15m = slice_df_window_ms(df_15m_chk, c_at * 1000, end_eval_ms, now_ts, interval_min=15)
                if not win_15m.empty:
                    ign_pos_15 = int(df_15m_chk.index.get_loc(win_15m.index[0]))
                    peak_idx_15 = win_15m["high"].idxmax()
                    peak_pos_15 = int(df_15m_chk.index.get_loc(peak_idx_15))
                    max_h_15m = to_float(win_15m["high"].max(), entry_p)
                    fav_pct = round((max_h_15m - entry_p) / entry_p * 100.0, 2)
                    t_item["max_favorable_pct"] = max(to_float(t_item.get("max_favorable_pct")), max(0.0, fav_pct))

                    if max_h_15m >= tp1_val:
                        t_item["tp1_hit"] = True
                    elif fav_pct >= item_sl_pct:
                        t_item["be_armed"] = True

                    # +3.5% 이상 유의미한 상승 파동을 그린 모든 스캐너 포착 코인의 1·2·3차 고점 로데이터 추출
                    if fav_pct >= 3.5 and peak_pos_15 >= ign_pos_15:
                        sup_p = entry_p * 0.965
                        w_h = max(to_float(t_item.get("swing_high_line"), entry_p * 1.02) - sup_p, entry_p * 0.045)
                        w_res = extract_three_wave_structure(df_15m_chk, ign_pos_15, peak_pos_15, entry_p, sup_p, w_h)
                        t_item["wave_raw"] = w_res

                        scanner_wave_stats["w1_gains"].append(w_res["w1"]["gain_pct"])
                        scanner_wave_stats["w1_fibs"].append(w_res["w1"]["fib_mult"])
                        scanner_wave_stats["w1_ma20_exts"].append(w_res["w1"]["ma20_ext_pct"])
                        scanner_wave_stats["w1_ma50_exts"].append(w_res["w1"]["ma50_ext_pct"])
                        scanner_wave_stats["w1_rsis"].append(w_res["w1"]["rsi6"])
                        scanner_wave_stats["w1_kdjs"].append(w_res["w1"]["kdj_j"])
                        scanner_wave_stats["w1_holds"].append(w_res["w1"]["hold_tag"])

                        if fav_pct >= 10.0:
                            scanner_wave_stats["w2_gains"].append(w_res["w2"]["gain_pct"])
                            scanner_wave_stats["w2_fibs"].append(w_res["w2"]["fib_mult"])
                            scanner_wave_stats["w2_ma20_exts"].append(w_res["w2"]["ma20_ext_pct"])
                            scanner_wave_stats["w2_ma50_exts"].append(w_res["w2"]["ma50_ext_pct"])
                            scanner_wave_stats["w2_rsis"].append(w_res["w2"]["rsi6"])
                            scanner_wave_stats["w2_kdjs"].append(w_res["w2"]["kdj_j"])

                        if fav_pct >= 16.0:
                            scanner_wave_stats["w3_gains"].append(w_res["w3"]["gain_pct"])
                            scanner_wave_stats["w3_fibs"].append(w_res["w3"]["fib_mult"])
                            scanner_wave_stats["w3_ma20_exts"].append(w_res["w3"]["ma20_ext_pct"])
                            scanner_wave_stats["w3_ma50_exts"].append(w_res["w3"]["ma50_ext_pct"])
                            scanner_wave_stats["w3_rsis"].append(w_res["w3"]["rsi6"])
                            scanner_wave_stats["w3_kdjs"].append(w_res["w3"]["kdj_j"])

                        scanner_wave_stats["coin_wave_samples"].append({
                            "base": sym.replace("USDT", ""),
                            "max_fav": fav_pct,
                            "w1": w_res["w1"],
                            "w2": w_res["w2"],
                            "w3": w_res["w3"],
                        })

        fav_now = to_float(t_item.get("max_favorable_pct"), 0.0)
        if t_item.get("tp1_hit") or t_item.get("tp2_hit") or (t_item.get("tp3_completed") and fav_now >= 9.5):
            tp_win_count += 1
        elif t_item.get("be_armed") and t_item.get("tp3_completed"):
            be_protect_count += 1
        elif t_item.get("sl_alerted"):
            sl_count_48h += 1
        else:
            active_count += 1

        if t_item.get("sl_alerted"):
            sl_key = f"{sym}_{c_at}"
            if sl_key not in sl_history:
                sl_history[sl_key] = {
                    "symbol": sym,
                    "track_type": t_item.get("track_type", "TRACK_A"),
                    "score": int(t_item.get("score", 85)),
                    "entry_price": entry_p,
                    "stop_price": stop_p_val,
                    "sl_trigger_price": to_float(t_item.get("sl_trigger_price"), stop_p_val),
                    "sl_pct": item_sl_pct,
                    "created_at": c_at,
                    "sl_at": int(t_item.get("sl_at", 0)),
                    "hold_minutes": int(t_item.get("hold_minutes", 0)),
                    "max_favorable_pct": fav_now,
                    "vol_mult": to_float(t_item.get("vol_mult", 1.0)),
                    "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
                    "oi_m": to_float(t_item.get("oi_m", 0.0)),
                    "turnover_m": to_float(t_item.get("turnover_m", 0.0)),
                    "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
                    "reasons": t_item.get("reasons", []),
                }

    # [스캐너 포착 '다른 코인들'의 1·2·3차 고점 형성 지점 로데이터 요약 섹션 생성]
    other_lines = [
        "━━━━━━━━━━━━━━━━━━",
        "🔬 [내 스캐너 포착 '다른 코인들'의 1차·2차·3차 고점 형성 로데이터 심층 분석]",
    ]
    samples = sorted(scanner_wave_stats["coin_wave_samples"], key=lambda x: x["max_fav"], reverse=True)
    if not samples:
        other_lines.append("• 최근 48h 포착 종목의 파동 로데이터를 누적 중입니다.")
    else:
        w1_g = float(np.median(scanner_wave_stats["w1_gains"]))
        w1_f = float(np.median(scanner_wave_stats["w1_fibs"]))
        w1_e20 = float(np.median(scanner_wave_stats["w1_ma20_exts"]))
        w1_r = float(np.median(scanner_wave_stats["w1_rsis"]))
        w1_k = float(np.median(scanner_wave_stats["w1_kdjs"]))

        w2_g = float(np.median(scanner_wave_stats["w2_gains"])) if scanner_wave_stats["w2_gains"] else w1_g * 1.65
        w2_f = float(np.median(scanner_wave_stats["w2_fibs"])) if scanner_wave_stats["w2_fibs"] else w1_f * 1.45
        w2_e20 = float(np.median(scanner_wave_stats["w2_ma20_exts"])) if scanner_wave_stats["w2_ma20_exts"] else w1_e20 * 1.6
        w2_r = float(np.median(scanner_wave_stats["w2_rsis"])) if scanner_wave_stats["w2_rsis"] else 76.0

        w3_g = float(np.median(scanner_wave_stats["w3_gains"])) if scanner_wave_stats["w3_gains"] else w2_g * 1.45
        w3_f = float(np.median(scanner_wave_stats["w3_fibs"])) if scanner_wave_stats["w3_fibs"] else w2_f * 1.30
        w3_e20 = float(np.median(scanner_wave_stats["w3_ma20_exts"])) if scanner_wave_stats["w3_ma20_exts"] else w2_e20 * 1.4
        w3_r = float(np.median(scanner_wave_stats["w3_rsis"])) if scanner_wave_stats["w3_rsis"] else 82.0

        other_lines.extend([
            f"• 분석 표본: 최근 48h 스캐너 포착 상승 파동 {len(samples)}종목 전수 역추적",
            f"  - [1차: 첫 조정 형성 지점]: 평균 +{w1_g:.1f}% | Fib {w1_f:.2f}배 | 15m MA20이격 +{w1_e20:.1f}% | RSI6 {w1_r:.0f}·KDJ_J {w1_k:.0f}",
            f"  - [2차: 최종고점 전 중간조정 고점]: 평균 +{w2_g:.1f}% | Fib {w2_f:.2f}배 | 15m MA20이격 +{w2_e20:.1f}% | RSI6 {w2_r:.0f}",
            f"  - [3차: 최종 클라이맥스 고점]: 평균 +{w3_g:.1f}% | Fib {w3_f:.2f}배 | 15m MA20이격 +{w3_e20:.1f}% | RSI6 {w3_r:.0f}",
            "• [주요 포착 코인별 1·2·3차 파동 실측 로데이터 상위 3선]:",
        ])
        for s_item in samples[:3]:
            sw1, sw2, sw3 = s_item["w1"], s_item["w2"], s_item["w3"]
            other_lines.append(
                f"  - ${s_item['base']}: 1차첫조정 +{sw1['gain_pct']:.1f}%(Fib {sw1['fib_mult']:.2f}·MA20+{sw1['ma20_ext_pct']:.1f}%·{sw1['hold_tag']}지지) "
                f"→ 2차고점 +{sw2['gain_pct']:.1f}%(Fib {sw2['fib_mult']:.2f}) "
                f"→ 3차최종 +{sw3['gain_pct']:.1f}%(Fib {sw3['fib_mult']:.2f}·RSI {sw3['rsi6']:.0f})"
            )

    recent_sl_list = [
        v for v in sl_history.values()
        if isinstance(v, dict) and (now_ts - int(v.get("created_at", now_ts))) <= 172800
    ]
    recent_sl_list.sort(key=lambda x: int(x.get("sl_at") or x.get("created_at") or 0), reverse=True)

    total_eval = tp_win_count + be_protect_count + sl_count_48h + active_count

    lines = [
        "\n".join(other_lines),
        "━━━━━━━━━━━━━━━━━━",
        "🚨 [내 스캐너 알림 종목 오답 노트 (최근 48h 손절 심층 역추적 & 첫 저항 로데이터)]",
    ]

    if total_eval == 0 and not recent_sl_list:
        lines.append("• 최근 48시간 내 손절 이탈 종목이 없습니다. (방어 필터 클린 유지 중)")
        return "\n".join(lines), [], scanner_wave_stats

    decided = tp_win_count + be_protect_count + sl_count_48h
    win_rate_str = f"{((tp_win_count + be_protect_count) / decided * 100.0):.1f}%" if decided > 0 else "집계중"
    lines.append(
        f"• 최근 48h 스캐너 성적표: 총 {total_eval}건 포착 | 익절·본전방어 {tp_win_count + be_protect_count}건(익절 {tp_win_count}·1R방어 {be_protect_count}) | 손절 {sl_count_48h}건 | 순항중 {active_count}건 (확정방어승률 {win_rate_str})"
    )

    if not recent_sl_list:
        lines.append(f"• ✅ 손절(SL) 발생 0건: 현재 -{FIXED_SL_PCT}% 고정 손절 및 1R(+{FIXED_SL_PCT}%) 본전 방어가 완벽히 작동하고 있습니다.")
        return "\n".join(lines), [], scanner_wave_stats

    df_btc_15m = fetch_klines("BTCUSDT", "15", limit=200)
    cause_counter = {
        "BTC_DUMP": 0,
        "STOP_HUNT_REBOUND": 0,
        "PROFIT_GIVEBACK": 0,
        "FAKEOUT_WHIPSAW": 0,
        "LOW_OI_DRAIN": 0,
    }
    prescription_notes = []

    for idx, sl_item in enumerate(recent_sl_list[:5], 1):
        sym = sl_item.get("symbol", "UNKNOWN")
        base = sym.replace("USDT", "")
        t_type = "Track B(돌파)" if sl_item.get("track_type") == "TRACK_B" else "Track A(선취매)"
        entry_p = to_float(sl_item.get("entry_price"))
        stop_p = to_float(sl_item.get("stop_price"))
        raw_sl = to_float(sl_item.get("sl_pct"), 0.0)
        if raw_sl <= 0 and entry_p > 0 and stop_p > 0:
            raw_sl = round(abs(entry_p - stop_p) / entry_p * 100.0, 1)
        sl_pct = float(np.clip(raw_sl if raw_sl > 0 else float(FIXED_SL_PCT), 2.0, 8.0))

        created_at = int(sl_item.get("created_at", now_ts - 3600))
        sl_at = int(sl_item.get("sl_at", 0))
        max_fav = to_float(sl_item.get("max_favorable_pct", 0.0))
        rsi_entry = to_float(sl_item.get("rsi6_15m", 50.0))

        age_hours = (now_ts - created_at) / 3600.0
        iv_str = "5" if age_hours <= 14.0 else "15"
        iv_min = 5 if iv_str == "5" else 15
        df_coin = fetch_klines(sym, iv_str, limit=200)

        rebound_after_sl_pct = 0.0
        peak_fib_at_sl = 1.4
        peak_ma20_ext_at_sl = 1.8
        peak_rsi_at_sl = 60.0
        resist_ma_tag = "단기매물대"

        if df_coin is not None and not df_coin.empty and entry_p > 0:
            df_coin["ma20"] = df_coin["close"].rolling(20, min_periods=5).mean()
            df_coin["ma50"] = df_coin["close"].rolling(50, min_periods=10).mean()
            df_coin["ma112"] = df_coin["close"].rolling(112, min_periods=20).mean()
            df_coin["rsi6"] = calc_rsi(df_coin["close"], 6)
            after_entry_df = slice_df_window_ms(df_coin, created_at * 1000, now_ts * 1000, now_ts, interval_min=iv_min)
            if not after_entry_df.empty:
                if (sl_at <= 0 or sl_at >= now_ts - 120) and stop_p > 0:
                    breach_rows = after_entry_df[after_entry_df["low"] <= stop_p]
                    if not breach_rows.empty:
                        first_b_pos = df_coin.index.get_loc(breach_rows.index[0])
                        sl_at = int(extract_ts_ms(df_coin, first_b_pos, now_ts, interval_min=iv_min) // 1000)

                if sl_at <= created_at:
                    sl_at = min(now_ts, created_at + 1800)

                hold_win = slice_df_window_ms(df_coin, created_at * 1000, (sl_at + iv_min * 60) * 1000, now_ts, interval_min=iv_min)
                if not hold_win.empty:
                    pk_idx = hold_win["high"].idxmax()
                    actual_high = to_float(hold_win.loc[pk_idx, "high"], entry_p)
                    calc_fav = round((actual_high - entry_p) / entry_p * 100.0, 2)
                    max_fav = max(max_fav, max(0.0, calc_fav))

                    ma20_pk = to_float(df_coin.loc[pk_idx, "ma20"], entry_p)
                    ma50_pk = to_float(df_coin.loc[pk_idx, "ma50"], entry_p)
                    ma112_pk = to_float(df_coin.loc[pk_idx, "ma112"], entry_p)
                    peak_ma20_ext_at_sl = round(max(0.1, (actual_high - ma20_pk) / max(ma20_pk, 1e-8) * 100.0), 1)
                    peak_rsi_at_sl = round(to_float(df_coin.loc[pk_idx, "rsi6"], 60.0), 1)
                    peak_fib_at_sl = round(max(1.0, (actual_high - entry_p * 0.965) / (entry_p * 0.045)), 2)

                    if abs(actual_high - ma112_pk) / max(ma112_pk, 1e-8) <= 0.015:
                        resist_ma_tag = "MA112 장기이평 저항"
                    elif abs(actual_high - ma50_pk) / max(ma50_pk, 1e-8) <= 0.015:
                        resist_ma_tag = "MA50 중기이평 저항"
                    else:
                        resist_ma_tag = f"MA20 +{peak_ma20_ext_at_sl}% 이격저항"

                after_sl_win = slice_df_window_ms(df_coin, sl_at * 1000, (sl_at + 7200) * 1000, now_ts, interval_min=iv_min)
                if not after_sl_win.empty and stop_p > 0:
                    post_sl_high = to_float(after_sl_win["high"].max(), stop_p)
                    rebound_after_sl_pct = (post_sl_high - stop_p) / max(stop_p, 1e-8) * 100.0

        if sl_at <= created_at:
            sl_at = now_ts
        hold_min = max(5, int((sl_at - created_at) // 60))

        btc_drop_pct = 0.0
        if df_btc_15m is not None and not df_btc_15m.empty:
            btc_win = slice_df_window_ms(df_btc_15m, (sl_at - 3600) * 1000, (sl_at + 900) * 1000, now_ts, interval_min=15)
            if not btc_win.empty:
                b_high = to_float(btc_win["high"].max(), 1.0)
                b_low = to_float(btc_win["low"].min(), 1.0)
                btc_drop_pct = (b_low - b_high) / max(b_high, 1e-8) * 100.0

        if max_fav >= sl_pct:
            cause_tag = f"유형3 [1R(+{sl_pct:g}%) 도달(최고 +{max_fav:.1f}%) 후 반락]"
            cause_detail = f"첫 조정 고점(Fib {peak_fib_at_sl}배·{resist_ma_tag}·RSI {peak_rsi_at_sl})에서 꺾임 (1R 본전방어 및 TP1 학습 반영)"
            cause_counter["PROFIT_GIVEBACK"] += 1
        elif btc_drop_pct <= -0.80:
            cause_tag = f"유형1 [BTC 도미노 급락({btc_drop_pct:.2f}%)]"
            cause_detail = f"반락 전 첫 고점 +{max_fav:.1f}%(Fib {peak_fib_at_sl}배·{resist_ma_tag}) 형성 후 BTC 투매 연동 하락"
            cause_counter["BTC_DUMP"] += 1
        elif rebound_after_sl_pct >= 4.0:
            cause_tag = f"유형2 [스탑헌팅 후 +{rebound_after_sl_pct:.1f}% V자 재상승]"
            cause_detail = f"아랫꼬리 털기 후 급반등 (신규 -{FIXED_SL_PCT}% 고정 손절 적용 시 휩소 방어 가능)"
            cause_counter["STOP_HUNT_REBOUND"] += 1
        elif hold_min <= 45 and max_fav < 1.2:
            cause_tag = f"유형4 [돌파 직후 윗꼬리 휩소(Fakeout·{hold_min}분 만에 이탈)]"
            cause_detail = f"15m RSI({rsi_entry:.1f}) 단기 고점권에서 후속 매수세 실종으로 즉시 반락"
            cause_counter["FAKEOUT_WHIPSAW"] += 1
        else:
            cause_tag = "유형5 [박스 하단 수급 이탈 및 추세 둔화]"
            cause_detail = f"초동 반등(+{max_fav:.1f}%·{resist_ma_tag}) 후 후속 거래량 부족으로 지지선 이탈"
            cause_counter["LOW_OI_DRAIN"] += 1

        sl_dt_str = datetime.fromtimestamp(sl_at, tz=KST).strftime("%m/%d %H:%M")
        lines.append(
            f"  {idx}) ❌ ${base} ({t_type} | SL -{sl_pct:g}% | {sl_dt_str} 이탈 | 보유 {hold_min}분 | 첫고점 +{max_fav:.1f}%)"
        )
        lines.append(f"     - [패인 & 저항로데이터]: {cause_tag} — {cause_detail}")

    if cause_counter["BTC_DUMP"] > 0:
        prescription_notes.append(
            "• [BTC 급락 방어]: BTC 15m(-0.7%↓) 및 1H(-1.2%↓) 투매 시 컷오프 85점→90점 상향 & 상대강도(24h +2%↑) 디커플링 필터 유지"
        )
    if cause_counter["STOP_HUNT_REBOUND"] > 0:
        prescription_notes.append(
            f"• [스탑헌팅 방어]: 급등 전 아랫꼬리 휩소 견디기 위해 FIXED_SL_PCT = {FIXED_SL_PCT}% 고정 손절 및 4h V자 재진입 유지"
        )
    if cause_counter["PROFIT_GIVEBACK"] > 0:
        prescription_notes.append(
            f"• [1R 본전 방어 & 1차 조정지점 학습]: +{FIXED_SL_PCT}% 도달 시 본전 방어 고정 & 첫 조정 저항 지점(Fib·MA20 이격)에 맞춰 TP1 조율"
        )
    if cause_counter["FAKEOUT_WHIPSAW"] > 0:
        prescription_notes.append(
            "• [윗꼬리 휩소 방어]: 돌파 직후 꺾인 Fakeout 방지 — PULLBACK_DRYUP_MAX를 0.28 이하로 유지 권장"
        )
    if cause_counter["LOW_OI_DRAIN"] > 0:
        prescription_notes.append(
            "• [수급 둔화 방어]: 1시간봉 중심선(MA20) 구조적 지지 필수화 및 무거운 저탄력 메이저 코인 스캔 배제"
        )

    if prescription_notes:
        lines.append("• 🛠️ [오답 노트 기반 스캐너 업데이트 처방전]:")
        lines.extend([f"  {note}" for note in prescription_notes])

    return "\n".join(lines), prescription_notes, scanner_wave_stats


# =====================================================================
# [3. Top 5 + 내 스캐너 코인(오답노트 포함) 1·2·3차 파동 로데이터 결합 튜닝 엔진]
# =====================================================================
def calibrate_3wave_tp_system(analyzed_top5, scanner_wave_stats):
    t5_w1_gain = [x["waves"]["w1"]["gain_pct"] for x in analyzed_top5]
    t5_w1_fib = [x["waves"]["w1"]["fib_mult"] for x in analyzed_top5]
    t5_w1_ext = [x["waves"]["w1"]["ma20_ext_pct"] for x in analyzed_top5]
    t5_w1_rsi = [x["waves"]["w1"]["rsi6"] for x in analyzed_top5]

    t5_w2_gain = [x["waves"]["w2"]["gain_pct"] for x in analyzed_top5]
    t5_w2_fib = [x["waves"]["w2"]["fib_mult"] for x in analyzed_top5]
    t5_w2_ext = [x["waves"]["w2"]["ma20_ext_pct"] for x in analyzed_top5]
    t5_w2_rsi = [x["waves"]["w2"]["rsi6"] for x in analyzed_top5]

    t5_w3_gain = [x["waves"]["w3"]["gain_pct"] for x in analyzed_top5]
    t5_w3_fib = [x["waves"]["w3"]["fib_mult"] for x in analyzed_top5]
    t5_w3_ext = [x["waves"]["w3"]["ma20_ext_pct"] for x in analyzed_top5]
    t5_w3_rsi = [x["waves"]["w3"]["rsi6"] for x in analyzed_top5]

    pm_w1_gain = scanner_wave_stats.get("w1_gains", [])
    pm_w1_fib = scanner_wave_stats.get("w1_fibs", [])
    pm_w1_ext = scanner_wave_stats.get("w1_ma20_exts", [])

    pm_w2_gain = scanner_wave_stats.get("w2_gains", [])
    pm_w2_fib = scanner_wave_stats.get("w2_fibs", [])
    pm_w2_ext = scanner_wave_stats.get("w2_ma20_exts", [])

    pm_w3_gain = scanner_wave_stats.get("w3_gains", [])
    pm_w3_fib = scanner_wave_stats.get("w3_fibs", [])
    pm_w3_ext = scanner_wave_stats.get("w3_ma20_exts", [])

    def _blend(t5_arr, pm_arr, w_t5=0.50, default_v=10.0):
        m_t5 = float(np.median(t5_arr)) if t5_arr else default_v
        if pm_arr:
            m_pm = float(np.median(pm_arr))
            return (m_t5 * w_t5) + (m_pm * (1.0 - w_t5))
        return m_t5

    # 1차 익절(첫 조정 고점): 스캐너 포착 코인들의 첫 조정 고점(55%) + Top 5 첫 조정 고점(45%)
    tp1_pct = int(np.clip(round(_blend(t5_w1_gain, pm_w1_gain, w_t5=0.45, default_v=11.0)), 10, 16))
    tp1_fib = round(float(np.clip(_blend(t5_w1_fib, pm_w1_fib, w_t5=0.45, default_v=2.13), 1.85, 2.55)), 2)
    tp1_ext = round(float(np.clip(_blend(t5_w1_ext, pm_w1_ext, w_t5=0.45, default_v=4.5), 3.0, 9.0)), 1)
    tp1_rsi = round(float(np.median(t5_w1_rsi + scanner_wave_stats.get("w1_rsis", []))) if t5_w1_rsi else 74.0, 1)

    # 2차 익절(최종고점 전 중간 조정 고점): Top 5 2차 고점(55%) + 스캐너 코인 2차 고점(45%)
    raw_tp2 = _blend(t5_w2_gain, pm_w2_gain, w_t5=0.55, default_v=20.0)
    tp2_pct = int(np.clip(round(max(tp1_pct + 7, raw_tp2)), 18, 28))
    tp2_fib = round(float(np.clip(max(tp1_fib + 0.70, _blend(t5_w2_fib, pm_w2_fib, w_t5=0.55, default_v=3.15)), 2.75, 3.70)), 2)
    tp2_ext = round(float(np.clip(max(tp1_ext + 3.0, _blend(t5_w2_ext, pm_w2_ext, w_t5=0.55, default_v=8.5)), 6.0, 16.0)), 1)
    tp2_rsi = round(float(np.median(t5_w2_rsi + scanner_wave_stats.get("w2_rsis", []))) if t5_w2_rsi else 79.0, 1)

    # 3차 익절(최종 클라이맥스 고점): Top 5 최종 고점(65%) + 스캐너 졸업 코인 최종 고점(35%)
    raw_tp3 = _blend(t5_w3_gain, pm_w3_gain, w_t5=0.65, default_v=35.0)
    tp3_pct = int(np.clip(round(max(tp2_pct + 10, raw_tp3)), 28, 50))
    tp3_fib = round(float(np.clip(max(tp2_fib + 0.80, _blend(t5_w3_fib, pm_w3_fib, w_t5=0.65, default_v=4.20)), 3.60, 4.80)), 2)
    tp3_ext = round(float(np.clip(max(tp2_ext + 5.0, _blend(t5_w3_ext, pm_w3_ext, w_t5=0.65, default_v=14.5)), 11.0, 26.0)), 1)
    tp3_rsi = round(float(np.median(t5_w3_rsi + scanner_wave_stats.get("w3_rsis", []))) if t5_w3_rsi else 84.0, 1)

    return {
        "tp1_min_pct": float(tp1_pct),
        "tp2_min_pct": float(tp2_pct),
        "tp3_min_pct": float(tp3_pct),
        "tp1_fib_mult": tp1_fib,
        "tp2_fib_mult": tp2_fib,
        "tp3_fib_mult": tp3_fib,
        "tp1_ma20_ext_pct": tp1_ext,
        "tp2_ma20_ext_pct": tp2_ext,
        "tp3_ma20_ext_pct": tp3_ext,
        "tp1_rsi6": tp1_rsi,
        "tp2_rsi6": tp2_rsi,
        "tp3_rsi6": tp3_rsi,
        "top5_w1_med": round(float(np.median(t5_w1_gain)) if t5_w1_gain else 10.0, 1),
        "top5_w2_med": round(float(np.median(t5_w2_gain)) if t5_w2_gain else 20.0, 1),
        "top5_w3_med": round(float(np.median(t5_w3_gain)) if t5_w3_gain else 35.0, 1),
        "pm_w1_med": round(float(np.median(pm_w1_gain)) if pm_w1_gain else tp1_pct, 1),
        "pm_w2_med": round(float(np.median(pm_w2_gain)) if pm_w2_gain else tp2_pct, 1),
        "pm_w3_med": round(float(np.median(pm_w3_gain)) if pm_w3_gain else tp3_pct, 1),
        "pm_sample_cnt": len(pm_w1_gain),
    }


def _extract_flat_cohorts(obj):
    results = []
    if isinstance(obj, list):
        for el in obj:
            results.extend(_extract_flat_cohorts(el))
    elif isinstance(obj, dict):
        c_list = obj.get("coins") or obj.get("items") or obj.get("records") or obj.get("top5")
        if (
            str(obj.get("date_str", "")) in ("cohorts", "snapshots", "history")
            and isinstance(c_list, list)
            and c_list
            and isinstance(c_list[0], dict)
            and ("symbol" not in c_list[0])
        ):
            results.extend(_extract_flat_cohorts(c_list))
            return results

        for key in ("cohorts", "snapshots", "history", "data"):
            if isinstance(obj.get(key), list):
                results.extend(_extract_flat_cohorts(obj[key]))

        if isinstance(c_list, list) and c_list and isinstance(c_list[0], dict) and ("symbol" in c_list[0]):
            ts = int(to_float(obj.get("timestamp") or obj.get("captured_ts") or obj.get("created_at") or obj.get("ts"), 0))
            date_s = str(obj.get("date_str") or obj.get("captured_at") or obj.get("date") or obj.get("cohort_id") or "")
            if (not date_s or date_s in ("cohorts", "snapshots")) and ts > 0:
                date_s = datetime.fromtimestamp(ts, tz=KST).strftime("%Y-%m-%d %H:%M")
            results.append({
                "date_str": date_s or "이전 포착분",
                "timestamp": ts if ts > 0 else int(time.time()) - 172800,
                "coins": c_list,
            })
    return results


def normalize_history_data(raw_data):
    flat = _extract_flat_cohorts(raw_data)
    dedup = {}
    for item in flat:
        k = item.get("date_str", "")
        dedup[k] = item
    res = list(dedup.values())
    res.sort(key=lambda x: int(x.get("timestamp", 0)))
    return res


def evaluate_48h_history(history_list, now_ts, tp_tuning):
    today_prefix = datetime.now(KST).strftime("%Y-%m-%d")
    target_snap = None
    for snap in reversed(history_list):
        if not isinstance(snap, dict):
            continue
        age_h = (now_ts - int(snap.get("timestamp", 0))) / 3600.0
        if 20.0 <= age_h <= 96.0 and snap.get("coins"):
            target_snap = snap
            break
    if not target_snap:
        for snap in reversed(history_list):
            if not str(snap.get("date_str", "")).startswith(today_prefix) and snap.get("coins"):
                target_snap = snap
                break

    lines = [
        "━━━━━━━━━━━━━━━━━━",
        "⏱️ [48시간 경과 동일 상위종목 실전 익절(TP1~TP3) 사후 검증]",
    ]
    if not target_snap or not target_snap.get("coins"):
        lines.append("• 오늘 첫 정상 스냅샷이 저장되었습니다. (다음 리포트부터 실전 익절 도달률이 자동 검증됩니다)")
        return "\n".join(lines)

    snap_date = target_snap.get("date_str", "이전 포착분")
    lines[1] = f"⏱️ [48시간 경과 동일 상위종목({snap_date} 포착분) 실전 익절(TP1~TP3) 사후 검증]"

    t1_thr = tp_tuning.get("tp1_min_pct", 11.0)
    t2_thr = tp_tuning.get("tp2_min_pct", 20.0)
    t3_thr = tp_tuning.get("tp3_min_pct", 35.0)

    added_rows = 0
    for item in target_snap.get("coins", []):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", ""))
        if not sym:
            continue
        base = sym.replace("USDT", "")
        ign_p = to_float(item.get("ign_price") or item.get("ignition_price") or item.get("entry_price") or item.get("price"))
        ign_ts = int(item.get("timestamp_ms", int(target_snap.get("timestamp", now_ts - 172800)) * 1000))

        df_1h = fetch_klines(sym, "60", limit=200)
        if df_1h is None or df_1h.empty or ign_p <= 0:
            continue
        post_1h = slice_df_window_ms(df_1h, ign_ts, now_ts * 1000, now_ts, interval_min=60)
        max_h = to_float(post_1h["high"].max() if not post_1h.empty else df_1h["high"].max(), ign_p)
        max_gain_pct = max(0.0, (max_h - ign_p) / ign_p * 100.0)

        if max_gain_pct >= t3_thr:
            badge = "🎯 [3차 TP 전량익절 졸업]"
        elif max_gain_pct >= t2_thr:
            badge = "✅ [2차 TP 돌파]"
        elif max_gain_pct >= t1_thr:
            badge = "✅ [1차 TP 달성]"
        else:
            badge = "⏳ [보합/눌림 진행]"

        lines.append(
            f"• ${base}: 발화가 {fmt_price(ign_p)} → 48h 최고 {fmt_price(max_h)} (+{max_gain_pct:.1f}%) {badge}"
        )
        added_rows += 1

    if added_rows == 0:
        lines.append("• 오늘 첫 정상 스냅샷이 저장되었습니다. (다음 리포트부터 실전 익절 도달률이 자동 검증됩니다)")

    return "\n".join(lines)


def main():
    check_required_secrets()
    now_ts = int(time.time())
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

    candidates = fetch_top_gainer_candidates()
    if not candidates:
        print("[치명적 오류] Top 5 후보군 조회 실패")
        sys.exit(1)

    candidates.sort(
        key=lambda x: to_float(x.get("change_24h"), to_float(x.get("price24hPcnt")) * 100.0),
        reverse=True,
    )

    analyzed_top5 = []
    for c in candidates[:25]:
        res = reverse_engineer_top_coin(c, now_ts)
        if res:
            analyzed_top5.append(res)
        if len(analyzed_top5) >= 5:
            break

    if not analyzed_top5:
        print("[치명적 오류] 상위 종목 캔들 분석 결과 없음")
        sys.exit(1)

    num_icons = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    msg_lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 1·2·3차 파동 로데이터 학습]",
        f"🕒 기준 시각: {now_str} (순수 코인 상위 5개)\n",
    ]

    for i, item in enumerate(analyzed_top5):
        icon = num_icons[i] if i < len(num_icons) else f"{i+1}."
        alpha_str = " / ".join(item["alphas"]) if item["alphas"] else "표준 수급 돌파"
        w1 = item["waves"]["w1"]
        w2 = item["waves"]["w2"]
        w3 = item["waves"]["w3"]

        msg_lines.extend(
            [
                f"{icon} ${item['base']} ({item['symbol']} | +{item['change_24h']:.2f}% | 현재가 {fmt_price(item['price'])} USDT)",
                f"• 거래대금: {item['turnover_m']:.1f}M | 미결제약정(OI): {item['oi_m']:.1f}M (비중 {item['oi_ratio']:.0f}%) | 펀딩비: {item['funding_pct']:+.4f}%",
                f"• [초동 발화점]: {item['ign_time_str']} (발화가 {fmt_price(item['ign_price'])} → 최종고점까지 +{item['pump_from_ign_pct']:.1f}% 폭등 | {item['hours_to_peak']}h 소요)",
                "• [🌊 1·2·3차 파동별 조정·고점 형성 로데이터(Raw Data) 분석]:",
                f"  - 1차(첫 조정 고점): +{w1['gain_pct']:.1f}% ({fmt_price(w1['price'])}) | Fib {w1['fib_mult']:.2f}배 | MA20이격 +{w1['ma20_ext_pct']:.1f}% | RSI6 {w1['rsi6']:.0f}·J {w1['kdj_j']:.0f} → -{w1['pullback_pct']:.1f}% 눌림({w1['hold_tag']} 지지)",
                f"  - 2차(중간 조정 고점): +{w2['gain_pct']:.1f}% ({fmt_price(w2['price'])}) | Fib {w2['fib_mult']:.2f}배 | MA20이격 +{w2['ma20_ext_pct']:.1f}% | RSI6 {w2['rsi6']:.0f}·J {w2['kdj_j']:.0f} → -{w2['pullback_pct']:.1f}% 숨고르기({w2['hold_tag']} 지지)",
                f"  - 3차(최종 클라이맥스): +{w3['gain_pct']:.1f}% ({fmt_price(w3['price'])}) | Fib {w3['fib_mult']:.2f}배 | MA20이격 +{w3['ma20_ext_pct']:.1f}% | RSI6 {w3['rsi6']:.0f}·J {w3['kdj_j']:.0f} | 거래량 {w3['vol_mult']:.1f}배",
                "• [발화 직전 컨펌된 초동 전략]:",
                f"  - 5분봉 이평선: {item['ma_str']} | 직전 2h 박스 {item['pre_box_pct']:.1f}% 응축 | OBV/CVD {item['obv_str']}",
                f"  - 초동 수급: 평소 대비 [{item['ign_vol_mult']:.1f}배 폭발] (발화전 포켓 {int(round(item['pre5_min_vol_ratio']*100))}%) | 돌파 후: {item['post_pattern_str']}",
                "• [🔍 포착된 전문가급 신규 알파 전략]:",
                f"  - {alpha_str}\n",
            ]
        )

    sec_postmortem, prescriptions, scanner_wave_stats = analyze_scanner_coins_and_postmortem(now_ts)
    tp_tuning = calibrate_3wave_tp_system(analyzed_top5, scanner_wave_stats)

    raw_history = load_json_file(HISTORY_FILE, {"cohorts": []})
    history_list = normalize_history_data(raw_history)
    sec_48h = evaluate_48h_history(history_list, now_ts, tp_tuning)
    msg_lines.append(sec_48h)
    msg_lines.append(sec_postmortem)

    today_key = now_dt.strftime("%Y-%m-%d")
    history_list = [h for h in history_list if not str(h.get("date_str", "")).startswith(today_key)]
    history_list.append(
        {
            "date_str": today_key,
            "timestamp": now_ts,
            "coins": [
                {
                    "symbol": x["symbol"],
                    "ign_price": x["ign_price"],
                    "support_line": x["support_line"],
                    "swing_high_line": x["swing_high_line"],
                }
                for x in analyzed_top5
            ],
        }
    )
    history_list = history_list[-10:]
    save_json_file(HISTORY_FILE, {
        "cohorts": history_list,
        "dynamic_tp_tuning": tp_tuning,
        "updated_at": now_str,
    })

    n = len(analyzed_top5)
    c_short = sum(1 for x in analyzed_top5 if x["short_aligned"])
    c_full = sum(1 for x in analyzed_top5 if x["full_aligned"])
    c_obv = sum(1 for x in analyzed_top5 if x["obv_up"])
    c_cvd = sum(1 for x in analyzed_top5 if x["cvd_up"])
    c_spring = sum(1 for x in analyzed_top5 if x["cond_spring"])
    c_htf = sum(1 for x in analyzed_top5 if x["htf_aligned"])
    c_pocket = sum(1 for x in analyzed_top5 if x["is_pocket_pivot"])
    c_ribbon = sum(1 for x in analyzed_top5 if x["pre5_min_ribbon"] <= 2.2)
    c_bull_flag = sum(1 for x in analyzed_top5 if x["is_bull_flag"])
    c_fvg50 = sum(1 for x in analyzed_top5 if x["is_fvg_50_hold"])
    c_sr = sum(1 for x in analyzed_top5 if x["is_sr_flip"])
    c_sqz = sum(1 for x in analyzed_top5 if x["ttm_sqz"])
    c_fund = sum(1 for x in analyzed_top5 if x["funding_pct"] <= 0.015)

    rec_vol_min = round(max(2.6, min(4.0, float(np.percentile([x["ign_vol_mult"] for x in analyzed_top5], 15)) * 0.85)), 1)
    rec_box_max = round(max(5.5, min(9.5, float(np.median([x["pre_box_pct"] for x in analyzed_top5])) * 1.6)), 1)
    rec_dryup = round(max(0.18, min(0.35, float(np.median([x["dryup_ratio"] for x in analyzed_top5])) + 0.05)), 2)
    avg_hours = round(float(np.mean([x["hours_to_peak"] for x in analyzed_top5])), 1)

    tp1_rec = int(tp_tuning["tp1_min_pct"])
    tp2_rec = int(tp_tuning["tp2_min_pct"])
    tp3_rec = int(tp_tuning["tp3_min_pct"])
    fib1_rec = tp_tuning["tp1_fib_mult"]
    fib2_rec = tp_tuning["tp2_fib_mult"]
    fib3_rec = tp_tuning["tp3_fib_mult"]
    ext1_rec = tp_tuning["tp1_ma20_ext_pct"]
    ext2_rec = tp_tuning["tp2_ma20_ext_pct"]
    ext3_rec = tp_tuning["tp3_ma20_ext_pct"]

    msg_lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━",
            "💡 [내 전략(scanner.py / analyzer.py) 실시간 업데이트 피드백 (제미나이 복붙용)]\n",
            "[1. 기존 vs 전문가 신규 알파 전략 적중률 비교]",
            f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {c_short}종목 (완전정배열 {c_full}종목)",
            f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {c_obv}종목 | CVD 흡수 다이버전스: {c_cvd}종목",
            f"• [전문가] High-Tight Bull Flag 공중눌림(상단 38.2% 사수): {n}종목 중 {c_bull_flag}종목",
            f"• [전문가] 포켓 피벗(Pocket Pivot 초동 양봉 직격): {n}종목 중 {c_pocket}종목",
            f"• [전문가] 장대양봉 50% 중심선(FVG/Equilibrium) 칼지지: {n}종목 중 {c_fvg50}종목",
            f"• [전문가] 전고점 S/R Flip 칼지지: {n}종목 중 {c_sr}종목 | 유동성 스윕(Spring): {c_spring}종목",
            f"• [전문가] 이평선 리본 초밀집(<=2.2%): {n}종목 중 {c_ribbon}종목 | TTM 켈트너 스퀴즈: {c_sqz}종목",
            f"• [전문가] 상위프레임(1H 중심선) 구조적 지지: {n}종목 중 {c_htf}종목 | 청정/음수 펀딩: {c_fund}종목\n",
            "[2. analyzer.py & config.py 파라미터 튜닝 권장값 (자동 동기화 완료)]",
            f"• EARLY_VOL_MULT_MIN = {rec_vol_min} (초동 최소 거래량 배수)",
            f"• PRE_BOX_RANGE_MAX_PCT = {rec_box_max}% (직전 2시간 변동폭 상한)",
            f"• PULLBACK_DRYUP_MAX = {rec_dryup} (발화봉 대비 눌림 거래량 상한)",
            f"• TP1_MIN_PCT = {tp1_rec}% (Fib {fib1_rec}배 / MA20이격 +{ext1_rec}%)",
            f"• TP2_MIN_PCT = {tp2_rec}% (Fib {fib2_rec}배 / MA20이격 +{ext2_rec}%)",
            f"• TP3_MIN_PCT = {tp3_rec}% (Fib {fib3_rec}배 / MA20이격 +{ext3_rec}%)\n",
            "[3. analyzer.py 추가/보완할 전문가 파이썬 조건식]",
            "• cond_bull_flag = (close > prev_box_high) and (pullback_low >= ignition_high - (ignition_high - prev_box_high) * 0.382) and (close >= ma5 * 0.998)",
            "• cond_pocket_pivot = is_current_ignition and (close > open_p) and (pre_vol_pocket_ratio <= 0.40) and (curr_vol_mult >= EARLY_VOL_MULT_MIN)",
            "• cond_fvg_mid_hold = (pullback_low >= (ignition_open + ignition_high) * 0.5 * 0.998) and (pullback_vol_ratio <= 0.35)\n",
            "[4. 오답 노트(손절 역추적) 기반 방어 필터 업데이트 요약]",
        ]
    )

    if prescriptions:
        msg_lines.extend(prescriptions)
    else:
        msg_lines.append("• 최근 48h 손절 특이사항 없음 (현행 방어 필터 유지 권장)")

    msg_lines.extend(
        [
            "\n[5. Top 5 대장주 + 내 스캐너 코인(오답노트) 로데이터 결합 1·2·3차 익절(TP) 학습 결과]",
            f"• 실전 고정 손절폭(SL): -{FIXED_SL_PCT}% (급등 전 휩소 방어 & 1R +{FIXED_SL_PCT}% 본전방어 연동)",
            f"• 파동별 고점 형성 위치 심층 분석 (Top5 5건 + 스캐너 포착 {tp_tuning['pm_sample_cnt']}건 결합 | 최종고점 평균 {avg_hours}h 소요):",
            f"  - 1차 TP (첫 조정 형성 고점): +{tp1_rec}% (Top5 첫조정 +{tp_tuning['top5_w1_med']}%·스캐너코인 첫조정 +{tp_tuning['pm_w1_med']}% 결합 | Fib {fib1_rec}배 | MA20이격 +{ext1_rec}% | RSI6 {tp_tuning['tp1_rsi6']:.0f})",
            f"  - 2차 TP (최종고점 전 중간조정 고점): +{tp2_rec}% (Top5 2차고점 +{tp_tuning['top5_w2_med']}%·스캐너코인 2차고점 +{tp_tuning['pm_w2_med']}% 결합 | Fib {fib2_rec}배 | MA20이격 +{ext2_rec}% | RSI6 {tp_tuning['tp2_rsi6']:.0f})",
            f"  - 3차 TP (최종 클라이맥스 고점): +{tp3_rec}% (Top5 최종고점 +{tp_tuning['top5_w3_med']}%·스캐너코인 최종고점 +{tp_tuning['pm_w3_med']}% 결합 | Fib {fib3_rec}배 | MA20이격 +{ext3_rec}% | RSI6 {tp_tuning['tp3_rsi6']:.0f})",
        ]
    )

    full_report = "\n".join(msg_lines)
    send_telegram_chunked(full_report)
    print(f"[{now_str}] 전문가급 Top 5 + 스캐너 코인 3파동 로데이터 심층 분석 브리핑 전송 완료")


if __name__ == "__main__":
    main()
