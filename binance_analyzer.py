import os
import sys
import json
import time
import inspect
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# 1. 외부 모듈 안전 로드 (특정 함수명 불일치로 인한 ImportError 원천 차단)
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

HISTORY_FILE = "analyzer_history.json"


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


def _normalize_kline_df(raw):
    """어떤 형태의 캔들 데이터(DataFrame 또는 list)가 들어와도 표준 5분봉/1시간봉 DataFrame으로 변환"""
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return pd.DataFrame()
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
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df

    if isinstance(raw, list) and len(raw) > 0:
        try:
            if isinstance(raw[0], dict):
                df = pd.DataFrame(raw)
                return _normalize_kline_df(df)
            cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(raw[0])]
            df = pd.DataFrame(raw, columns=cols)
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            for col in ("open", "high", "low", "close", "volume"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
            return df
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def fetch_klines(symbol, interval="5", limit=150):
    """
    1순위: api.py 내부에 정의된 캔들 함수(이름 무관 자동 탐색) 호출
    2순위: 미국 서버(GitHub Actions) 방화벽을 우회하는 멀티 거래소/미러 폴백 호출
    """
    if api_mod is not None:
        candidate_names = (
            "fetch_klines", "fetch_kline", "get_klines", "get_kline",
            "fetch_bybit_klines", "get_bybit_klines", "fetch_candles",
            "get_candles", "fetch_kline_df", "get_kline_df", "fetch_ohlcv", "get_ohlcv",
        )
        excluded = {
            "get_scanner_candidates", "fetch_verified_1m_klines",
            "to_float", "is_tradfi_ticker", "strip_multiplier_prefix", "strip_tokenized_suffix",
        }
        funcs_to_try = []
        for name in candidate_names:
            fn = getattr(api_mod, name, None)
            if callable(fn):
                funcs_to_try.append(fn)

        for name, fn in inspect.getmembers(api_mod, inspect.isfunction):
            if name not in excluded and name not in candidate_names and not name.startswith("_"):
                if fn.__module__ == api_mod.__name__:
                    funcs_to_try.append(fn)

        for fn in funcs_to_try:
            for args in [(symbol, str(interval), int(limit)), (symbol, int(interval), int(limit)), (symbol, str(interval))]:
                try:
                    res = fn(*args)
                    df = _normalize_kline_df(res)
                    if not df.empty and len(df) >= 15:
                        return df
                except Exception:
                    continue

    # 2순위A: Bybit 글로벌/지역 미러 엔드포인트 호출
    bybit_Params = {"category": "linear", "symbol": symbol, "interval": str(interval), "limit": int(limit)}
    for host in ("https://api.bybit-tr.com", "https://api.bybit.kz", "https://api.bytick.com", "https://api.bybit.com"):
        try:
            resp = requests.get(f"{host}/v5/market/kline", params=bybit_Params, headers=BROWSER_HEADERS, timeout=6)
            if resp.status_code == 200:
                rows = resp.json().get("result", {}).get("list", [])
                df = _normalize_kline_df(rows)
                if not df.empty and len(df) >= 15:
                    return df
        except Exception:
            pass

    base = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = base
    for prefix in ("1000000", "100000", "10000", "1000", "100"):
        if clean_base.startswith(prefix) and len(clean_base) > len(prefix):
            clean_base = clean_base[len(prefix):]
            break

    # 2순위B: Gate.io USDT 선물 캔들 (미국 GitHub Actions 서버 100% 무차단)
    gate_iv = {"1": "1m", "5": "5m", "15": "15m", "60": "1h", "240": "4h"}.get(str(interval), "5m")
    for contract in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            resp = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": contract, "interval": gate_iv, "limit": int(limit)},
                headers=BROWSER_HEADERS,
                timeout=6,
            )
            if resp.status_code == 200:
                rows = resp.json()
                if isinstance(rows, list) and len(rows) >= 15:
                    df = pd.DataFrame(rows)
                    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce") * 1000.0
                    for col in ("open", "high", "low", "close", "volume"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    df = df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
                    if not df.empty:
                        return df
        except Exception:
            pass

    # 2순위C: Bitget USDT 선물 캔들 (미국 GitHub Actions 서버 무차단)
    bg_iv = {"1": "1m", "5": "5m", "15": "15m", "60": "1H", "240": "4H"}.get(str(interval), "5m")
    for bg_sym in (symbol, f"{clean_base}USDT"):
        try:
            resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": bg_sym, "productType": "USDT-FUTURES", "granularity": bg_iv, "limit": str(limit)},
                headers=BROWSER_HEADERS,
                timeout=6,
            )
            if resp.status_code == 200:
                rows = resp.json().get("data", [])
                df = _normalize_kline_df(rows)
                if not df.empty and len(df) >= 15:
                    return df
        except Exception:
            pass

    # 2순위D: MEXC 선물 캔들 (중소형 급등 알트코인 커버리지 최강)
    mexc_iv = {"1": "Min1", "5": "Min5", "15": "Min15", "60": "Min60"}.get(str(interval), "Min5")
    for m_sym in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            resp = requests.get(
                f"https://contract.mexc.com/api/v1/contract/kline/{m_sym}",
                params={"interval": mexc_iv},
                headers=BROWSER_HEADERS,
                timeout=6,
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
                    df = df.dropna(subset=["close"]).sort_values("timestamp").iloc[-int(limit):].reset_index(drop=True)
                    if len(df) >= 15:
                        return df
        except Exception:
            pass

    # 2순위E: Binance Data Vision (미국 공식 허용 엔드포인트)
    bn_iv = {"1": "1m", "5": "5m", "15": "15m", "60": "1h"}.get(str(interval), "5m")
    for bn_sym in (symbol, f"{clean_base}USDT"):
        try:
            resp = requests.get(
                "https://data-api.binance.vision/api/v3/klines",
                params={"symbol": bn_sym, "interval": bn_iv, "limit": int(limit)},
                headers=BROWSER_HEADERS,
                timeout=6,
            )
            if resp.status_code == 200:
                rows = resp.json()
                if isinstance(rows, list) and len(rows) >= 15:
                    df = pd.DataFrame([r[:6] for r in rows], columns=["timestamp", "open", "high", "low", "close", "volume"])
                    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
                    for col in ("open", "high", "low", "close", "volume"):
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                    return df.dropna(subset=["close"]).reset_index(drop=True)
        except Exception:
            pass

    return pd.DataFrame()


def fetch_top_gainer_candidates():
    if api_mod is not None and hasattr(api_mod, "get_scanner_candidates"):
        try:
            res = api_mod.get_scanner_candidates()
            cands = res[0] if isinstance(res, tuple) else res
            if isinstance(cands, list) and len(cands) > 0:
                return cands
        except Exception as e:
            print(f"[get_scanner_candidates 예외] {e}")
    return []


def send_telegram_chunked(text):
    max_len = 3800
    lines = text.split("\n")
    chunks = []
    current = []
    curr_len = 0

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
                time.sleep(0.4)
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
                    time.sleep(0.4)
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


def filter_df_since_ts(df, since_ms, now_ts, interval_min=5):
    if df is None or df.empty:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        try:
            ts_series = pd.to_numeric(df["timestamp"], errors="coerce")
            if ts_series.notna().any():
                if ts_series.max() < 1e11:
                    ts_series = ts_series * 1000.0
                sub = df[ts_series >= since_ms]
                if not sub.empty:
                    return sub
        except Exception:
            pass
    elapsed_min = max(1, int((now_ts * 1000 - since_ms) / 60000))
    bars_count = max(1, int(elapsed_min // interval_min) + 1)
    return df.iloc[-min(len(df), bars_count):]


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


def load_json_file(path, default_val):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default_val


def save_json_file(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[파일 저장 오류] {path}: {e}")


# =====================================================================
# [1. 전문가급 급등 Top 5 초동 발화점 심층 역추적 엔진]
# =====================================================================
def reverse_engineer_top_coin(coin, now_ts):
    try:
        symbol = coin["symbol"]
        df_5m = fetch_klines(symbol, "5", limit=150)
        if df_5m is None or df_5m.empty or len(df_5m) < 20:
            return None

        df_1h = fetch_klines(symbol, "60", limit=75)
        df_5m = df_5m.reset_index(drop=True)

        for p in (5, 20, 50, 112):
            df_5m[f"ma{p}"] = df_5m["close"].rolling(p, min_periods=3).mean()
            df_5m[f"ema{p}"] = df_5m["close"].ewm(span=p, adjust=False).mean()

        df_5m["vol_ma20"] = df_5m["volume"].rolling(20, min_periods=3).mean().shift(1).bfill()
        df_5m["vol_mult"] = (df_5m["volume"] / df_5m["vol_ma20"].replace(0, np.nan)).fillna(1.0)
        atr_5m = calc_atr(df_5m, 14)

        ribbon_max = df_5m[["ema5", "ema20", "ema50"]].max(axis=1)
        ribbon_min = df_5m[["ema5", "ema20", "ema50"]].min(axis=1)
        df_5m["ribbon_spread"] = (ribbon_max - ribbon_min) / df_5m["close"].replace(0, np.nan) * 100.0

        obv, cvd, _ = calc_obv_and_cvd(df_5m)
        df_5m["obv"] = obv
        df_5m["cvd"] = cvd

        start_idx = min(15, max(3, len(df_5m) // 6))
        search_df = df_5m.iloc[start_idx:-2].copy()
        if search_df.empty:
            search_df = df_5m.iloc[:-1].copy()

        overall_low = to_float(search_df["low"].min(), to_float(df_5m["close"].iloc[-1]))
        overall_high = to_float(search_df["high"].max(), to_float(df_5m["close"].iloc[-1]))
        price_ceiling_for_ignition = overall_low + (overall_high - overall_low) * 0.55

        ign_candidates = search_df[
            (search_df["close"] > search_df["open"])
            & (search_df["vol_mult"] >= 1.8)
            & (search_df["close"] <= price_ceiling_for_ignition * 1.08)
        ]
        if ign_candidates.empty:
            ign_candidates = search_df[(search_df["close"] > search_df["open"]) & (search_df["vol_mult"] >= 1.5)]
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

        body_s = (pre5_df["close"] - pre5_df["open"]).abs().replace(0, 1e-8)
        l_wick = pre5_df[["open", "close"]].min(axis=1) - pre5_df["low"]
        spring_wick_ratio = to_float((l_wick / body_s).max(), 0.0)
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
        if turnover_m <= 0:
            turnover_m = to_float(coin.get("turnover24h")) / 1e6

        oi_m = to_float(coin.get("oi_m"))
        if oi_m <= 0:
            oi_m = to_float(coin.get("openInterestValue")) / 1e6

        if "funding_pct" in coin:
            funding_pct = to_float(coin.get("funding_pct"))
        else:
            funding_pct = to_float(coin.get("fundingRate")) * 100.0

        if "change_24h" in coin:
            change_24h = to_float(coin.get("change_24h"))
        else:
            change_24h = to_float(coin.get("price24hPcnt")) * 100.0

        oi_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0

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
        if ign_minute in (0, 5, 10):
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
        print(f"[종목 분석 예외] {coin.get('symbol')}: {e}")
        return None


# =====================================================================
# [2. 내 스캐너 알림 종목 오답 노트 (손절 5대 패인 심층 역추적 엔진)]
# =====================================================================
def analyze_scanner_sl_postmortem(now_ts):
    state = load_json_file(STATE_FILE, {"tracked": {}, "sl_history": {}})
    if not isinstance(state, dict):
        state = {"tracked": {}, "sl_history": {}}

    tracked = state.get("tracked", {})
    if not isinstance(tracked, dict):
        tracked = {}

    sl_history = state.get("sl_history", {})
    if not isinstance(sl_history, dict):
        sl_history = {}

    for sym, t_item in tracked.items():
        if not isinstance(t_item, dict):
            continue
        if t_item.get("sl_alerted"):
            c_at = int(t_item.get("created_at", 0))
            sl_key = f"{sym}_{c_at}"
            if sl_key not in sl_history:
                stop_p_val = to_float(t_item.get("stop_price"))
                entry_low = to_float(t_item.get("entry_low"))
                entry_high = to_float(t_item.get("entry_high"))
                entry_p = to_float(t_item.get("entry_price"))
                if entry_p <= 0:
                    entry_p = (entry_low + entry_high) / 2.0 if entry_low > 0 else stop_p_val
                sl_history[sl_key] = {
                    "symbol": sym,
                    "track_type": t_item.get("track_type", "TRACK_A"),
                    "score": int(t_item.get("score", 85)),
                    "entry_price": entry_p,
                    "stop_price": stop_p_val,
                    "sl_trigger_price": to_float(t_item.get("sl_trigger_price"), stop_p_val),
                    "sl_pct": to_float(t_item.get("sl_pct", 3)),
                    "created_at": c_at,
                    "sl_at": int(t_item.get("sl_at", now_ts)),
                    "hold_minutes": max(1, int((int(t_item.get("sl_at", now_ts)) - c_at) // 60)),
                    "max_favorable_pct": to_float(t_item.get("max_favorable_pct", 0.0)),
                    "vol_mult": to_float(t_item.get("vol_mult", 1.0)),
                    "rsi6_15m": to_float(t_item.get("rsi6_15m", 50.0)),
                    "oi_m": to_float(t_item.get("oi_m", 0.0)),
                    "turnover_m": to_float(t_item.get("turnover_m", 0.0)),
                    "funding_pct": to_float(t_item.get("funding_pct", 0.0)),
                    "reasons": t_item.get("reasons", []),
                }

    recent_tracked = [
        v for v in tracked.values()
        if isinstance(v, dict) and (now_ts - int(v.get("created_at", 0))) <= 172800
    ]
    recent_sl_list = [
        v for v in sl_history.values()
        if isinstance(v, dict) and (now_ts - int(v.get("sl_at", 0))) <= 172800
    ]
    recent_sl_list.sort(key=lambda x: int(x.get("sl_at", 0)), reverse=True)

    tp_win_count = sum(
        1 for v in recent_tracked if v.get("tp1_hit") or v.get("tp2_hit") or v.get("tp3_completed")
    )
    active_count = sum(
        1 for v in recent_tracked if not v.get("sl_alerted") and not v.get("tp3_completed")
    )
    sl_count = len(recent_sl_list)
    total_eval = max(len(recent_tracked), tp_win_count + sl_count + active_count)

    lines = [
        "━━━━━━━━━━━━━━━━━━",
        "🚨 [내 스캐너 알림 종목 오답 노트 (최근 48h 손절 심층 역추적 & 처방)]",
    ]

    if total_eval == 0 and sl_count == 0:
        lines.append("• 최근 48시간 내 손절 이탈 종목이 없습니다. (방어 필터 클린 유지 중)")
        return "\n".join(lines), []

    decided = tp_win_count + sl_count
    win_rate_str = f"{(tp_win_count / decided * 100.0):.1f}%" if decided > 0 else "집계중"
    lines.append(
        f"• 최근 48h 스캐너 성적표: 총 {total_eval}건 포착 | 익절권 도달 {tp_win_count}건 | 손절 {sl_count}건 | 순항중 {active_count}건 (확정승률 {win_rate_str})"
    )

    if sl_count == 0:
        lines.append("• ✅ 손절(SL) 발생 0건: 현재 유동 손절선(-2%~-5%) 및 컷오프 필터가 완벽히 작동하고 있습니다.")
        return "\n".join(lines), []

    df_btc = fetch_klines("BTCUSDT", "5", limit=150)
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
        created_at = int(sl_item.get("created_at", now_ts - 3600))
        sl_at = int(sl_item.get("sl_at", now_ts))
        hold_min = int(sl_item.get("hold_minutes", 15))
        max_fav = to_float(sl_item.get("max_favorable_pct", 0.0))
        oi_m = to_float(sl_item.get("oi_m", 0.0))
        turnover_m = to_float(sl_item.get("turnover_m", 1.0))
        oi_ratio = (oi_m / max(turnover_m, 1e-8)) * 100.0
        rsi_entry = to_float(sl_item.get("rsi6_15m", 50.0))

        df_coin = fetch_klines(sym, "5", limit=150)
        btc_drop_pct = 0.0
        if df_btc is not None and not df_btc.empty:
            btc_win = filter_df_since_ts(df_btc, (created_at - 300) * 1000, now_ts, interval_min=5)
            if not btc_win.empty:
                b_high = to_float(btc_win["high"].max(), 1.0)
                b_low = to_float(btc_win["low"].min(), 1.0)
                btc_drop_pct = (b_low - b_high) / max(b_high, 1e-8) * 100.0

        rebound_after_sl_pct = 0.0
        if df_coin is not None and not df_coin.empty and entry_p > 0:
            after_sl_win = filter_df_since_ts(df_coin, sl_at * 1000, now_ts, interval_min=5)
            if not after_sl_win.empty and stop_p > 0:
                post_sl_high = to_float(after_sl_win["high"].max(), stop_p)
                rebound_after_sl_pct = (post_sl_high - stop_p) / max(stop_p, 1e-8) * 100.0

        if btc_drop_pct <= -0.75:
            cause_tag = f"유형1 [BTC 도미노 급락({btc_drop_pct:.2f}%)]"
            cause_detail = "개별 타점 결함이 아닌 비트코인 순간 투매에 의한 시장 연동 하락"
            cause_counter["BTC_DUMP"] += 1
        elif rebound_after_sl_pct >= 3.5:
            cause_tag = f"유형2 [스탑헌팅 후 +{rebound_after_sl_pct:.1f}% V자 재상승]"
            cause_detail = "방향성은 적중했으나 손절선 ATR 버퍼가 좁아 아랫꼬리에 털린 후 반등"
            cause_counter["STOP_HUNT_REBOUND"] += 1
        elif max_fav >= 3.5:
            cause_tag = f"유형3 [진입 후 +{max_fav:.1f}% 상승 뒤 반락]"
            cause_detail = "초동 수익 구간 진입에는 성공했으나 단기 차익 매물에 밀려 본전/손절 이탈"
            cause_counter["PROFIT_GIVEBACK"] += 1
        elif hold_min <= 35 and max_fav < 1.2:
            cause_tag = f"유형4 [돌파 직후 윗꼬리 휩소(Fakeout·{hold_min}분 만에 이탈)]"
            cause_detail = f"15m RSI({rsi_entry:.1f}) 단기 고점권에서 후속 매수세 실종으로 즉시 반락"
            cause_counter["FAKEOUT_WHIPSAW"] += 1
        else:
            cause_tag = f"유형5 [선물 OI 실수급 부족(OI비중 {oi_ratio:.0f}%)]"
            cause_detail = "거래대금 대비 미결제약정(OI) 잔류 비중이 낮아 단타 매도세에 지지선 붕괴"
            cause_counter["LOW_OI_DRAIN"] += 1

        sl_dt_str = datetime.fromtimestamp(sl_at, tz=KST).strftime("%m/%d %H:%M")
        lines.append(
            f"  {idx}) ❌ ${base} ({t_type} | {sl_dt_str} 이탈 | 보유 {hold_min}분 | 최고 +{max_fav:.1f}%)"
        )
        lines.append(f"     - [패인 진단]: {cause_tag} — {cause_detail}")

    if cause_counter["BTC_DUMP"] > 0:
        prescription_notes.append(
            "• [BTC 급락 방어]: BTCUSDT 15분봉이 직전 고점 대비 -0.7% 이상 급락 중일 때는 알트코인 진입 컷오프를 85점→90점으로 자동 상향 권장"
        )
    if cause_counter["STOP_HUNT_REBOUND"] > 0:
        prescription_notes.append(
            "• [스탑헌팅 방어]: 손절 후 V자 반등 종목 포착 — analyzer.py의 ATR 손절 버퍼를 1.2배→1.45배로 확대하고 최소 손절폭을 -3% 이상으로 완화 권장"
        )
    if cause_counter["PROFIT_GIVEBACK"] > 0:
        prescription_notes.append(
            "• [수익 반납 방어]: 진입 후 +4.0% 이상 도달 시 scanner.py에서 손절선을 진입가(본전)로 올리는 브레이크이븐(Breakeven) 스탑 활성화 권장"
        )
    if cause_counter["FAKEOUT_WHIPSAW"] > 0:
        prescription_notes.append(
            "• [윗꼬리 휩소 방어]: 돌파 즉시 꺾인 Fakeout 방지 — PULLBACK_DRYUP_MAX를 0.35→0.28로 조이고 돌파봉 윗꼬리 허용치를 0.42→0.32로 축소 권장"
        )
    if cause_counter["LOW_OI_DRAIN"] > 0:
        prescription_notes.append(
            "• [빈집 붕괴 방어]: OI/거래대금 비중이 8% 미만인 종목은 만점 배점에서 5점 감점 처리하여 허수 거래량 걸러내기 권장"
        )

    if prescription_notes:
        lines.append("• 🛠️ [오답 노트 기반 스캐너 업데이트 처방전]:")
        lines.extend([f"  {note}" for note in prescription_notes])

    return "\n".join(lines), prescription_notes


# =====================================================================
# [3. 48시간 사후검증 (기존 dict/list 구조 모두 호환)]
# =====================================================================
def normalize_history_data(raw_data):
    if isinstance(raw_data, list):
        return [x for x in raw_data if isinstance(x, dict)]
    if isinstance(raw_data, dict):
        if isinstance(raw_data.get("snapshots"), list):
            return [x for x in raw_data["snapshots"] if isinstance(x, dict)]
        if isinstance(raw_data.get("history"), list):
            return [x for x in raw_data["history"] if isinstance(x, dict)]
        converted = []
        for k, v in raw_data.items():
            if isinstance(v, dict) and "coins" in v:
                converted.append(v)
            elif isinstance(v, list):
                converted.append({"date_str": str(k), "timestamp": int(time.time()) - 172800, "coins": v})
        return converted
    return []


def evaluate_48h_history(history_list, now_ts):
    target_snap = None
    for snap in history_list:
        if not isinstance(snap, dict):
            continue
        age_h = (now_ts - int(snap.get("timestamp", 0))) / 3600.0
        if 36.0 <= age_h <= 72.0:
            target_snap = snap
            break

    lines = [
        "━━━━━━━━━━━━━━━━━━",
        "⏱️ [48시간 경과 동일 상위종목 실전 익절(TP1~TP3) 사후 검증]",
    ]
    if not target_snap:
        lines.append("• 아직 48시간이 경과된 이전 스냅샷 데이터가 없습니다. (자동 누적 중)")
        return "\n".join(lines)

    snap_date = target_snap.get("date_str", "이전 포착분")
    lines[1] = f"⏱️ [48시간 경과 동일 상위종목({snap_date} 포착분) 실전 익절(TP1~TP3) 사후 검증]"

    for item in target_snap.get("coins", []):
        if not isinstance(item, dict):
            continue
        sym = str(item.get("symbol", ""))
        if not sym:
            continue
        base = sym.replace("USDT", "")
        ign_p = to_float(item.get("ign_price") or item.get("ignition_price") or item.get("entry_price"))
        ign_ts = int(item.get("timestamp_ms", int(target_snap.get("timestamp", 0)) * 1000))
        sup_p = to_float(item.get("support_line"), ign_p * 0.96)
        sw_high = to_float(item.get("swing_high_line"), ign_p * 1.02)
        wave_h = max(sw_high - sup_p, ign_p * 0.032)

        tp1 = max(ign_p * 1.08, sup_p + wave_h * 2.13)
        tp2 = max(ign_p * 1.15, sup_p + wave_h * 3.06)
        tp3 = max(ign_p * 1.22, sup_p + wave_h * 3.86)

        df_1h = fetch_klines(sym, "60", limit=75)
        if df_1h is None or df_1h.empty or ign_p <= 0:
            continue
        post_1h = filter_df_since_ts(df_1h, ign_ts, now_ts, interval_min=60)
        max_h = to_float(post_1h["high"].max() if not post_1h.empty else df_1h["high"].max(), ign_p)
        max_gain_pct = (max_h - ign_p) / ign_p * 100.0

        if max_h >= tp3:
            badge = "🎯 [3차 TP 전량익절 졸업]"
        elif max_h >= tp2:
            badge = "✅ [2차 TP 돌파]"
        elif max_h >= tp1:
            badge = "✅ [1차 TP 달성]"
        else:
            badge = "⏳ [보합/눌림 진행]"

        lines.append(
            f"• ${base}: 발화가 {fmt_price(ign_p)} → 48h 최고 {fmt_price(max_h)} (+{max_gain_pct:.1f}%) {badge}"
        )

    return "\n".join(lines)


def main():
    now_ts = int(time.time())
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

    candidates = fetch_top_gainer_candidates()
    if not candidates:
        print("[알림] Top 5 후보군 조회 실패")
        sys.exit(1)

    candidates.sort(
        key=lambda x: to_float(x.get("change_24h"), to_float(x.get("price24hPcnt")) * 100.0),
        reverse=True,
    )

    analyzed_top5 = []
    for c in candidates[:20]:
        res = reverse_engineer_top_coin(c, now_ts)
        if res:
            analyzed_top5.append(res)
        if len(analyzed_top5) >= 5:
            break

    if not analyzed_top5:
        print("[알림] 상위 종목 캔들 분석 결과 없음")
        sys.exit(1)

    num_icons = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    msg_lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 48h 실전 익절 검증]",
        f"🕒 기준 시각: {now_str} (순수 코인 상위 5개)\n",
    ]

    for i, item in enumerate(analyzed_top5):
        icon = num_icons[i] if i < len(num_icons) else f"{i+1}."
        alpha_str = " / ".join(item["alphas"]) if item["alphas"] else "표준 수급 돌파"
        msg_lines.extend(
            [
                f"{icon} ${item['base']} ({item['symbol']} | +{item['change_24h']:.2f}% | 현재가 {fmt_price(item['price'])} USDT)",
                f"• 거래대금: {item['turnover_m']:.1f}M | 미결제약정(OI): {item['oi_m']:.1f}M (비중 {item['oi_ratio']:.0f}%) | 펀딩비: {item['funding_pct']:+.4f}%",
                f"• [초동 발화점]: {item['ign_time_str']} (발화가 {fmt_price(item['ign_price'])} → 고점까지 +{item['pump_from_ign_pct']:.1f}% 폭등 | {item['hours_to_peak']}h 소요)",
                "• [발화 직전 컨펌된 초동 전략]:",
                f"  - 5분봉 이평선: {item['ma_str']}",
                f"  - 사전 매집: 직전 2시간 변동폭 {item['pre_box_pct']:.1f}% 응축 | OBV/CVD {item['obv_str']}",
                f"  - 초동 수급: 평소 대비 거래량 [{item['ign_vol_mult']:.1f}배 폭발] (발화전 포켓 {int(round(item['pre5_min_vol_ratio']*100))}%)",
                f"  - 돌파 후 패턴: {item['post_pattern_str']}",
                "• [🔍 포착된 전문가급 신규 알파 전략]:",
                f"  - {alpha_str}\n",
            ]
        )

    raw_history = load_json_file(HISTORY_FILE, [])
    history_list = normalize_history_data(raw_history)
    sec_48h = evaluate_48h_history(history_list, now_ts)
    msg_lines.append(sec_48h)

    today_key = now_dt.strftime("%Y-%m-%d")
    if not any(isinstance(h, dict) and h.get("date_str") == today_key for h in history_list):
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
        save_json_file(HISTORY_FILE, history_list)

    sec_postmortem, prescriptions = analyze_scanner_sl_postmortem(now_ts)
    msg_lines.append(sec_postmortem)

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

    rec_vol_min = round(max(2.5, min(4.0, float(np.percentile([x["ign_vol_mult"] for x in analyzed_top5], 15)) * 0.85)), 1)
    rec_box_max = round(max(5.5, min(9.5, float(np.median([x["pre_box_pct"] for x in analyzed_top5])) * 1.6)), 1)
    rec_dryup = round(max(0.20, min(0.40, float(np.median([x["dryup_ratio"] for x in analyzed_top5])) + 0.05)), 2)
    rec_sl_pct = int(np.clip(round(float(np.median([x["mae_pct"] for x in analyzed_top5])) + 1.0), 3, 5))
    avg_hours = round(float(np.mean([x["hours_to_peak"] for x in analyzed_top5])), 1)
    med_pump = float(np.median([x["pump_from_ign_pct"] for x in analyzed_top5]))
    tp1_rec = int(max(8, round(med_pump * 0.45)))
    tp2_rec = int(max(14, round(med_pump * 0.75)))
    tp3_rec = int(max(20, round(med_pump * 1.10)))

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
            "[2. analyzer.py & config.py 파라미터 튜닝 권장값]",
            f"• EARLY_VOL_MULT_MIN = {rec_vol_min} (초동 최소 거래량 배수)",
            f"• PRE_BOX_RANGE_MAX_PCT = {rec_box_max}% (직전 2시간 변동폭 상한)",
            f"• PULLBACK_DRYUP_MAX = {rec_dryup} (발화봉 대비 눌림 거래량 상한)",
            "• BULL_FLAG_MAX_RETRACE = 0.382 (돌파 양봉 상단 38.2% 공중 지지 기준)\n",
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
            "\n[5. 48시간 사후검증 기반 유동적 손절(SL) & 3단계 익절(TP) 공식 튜닝안]",
            f"• 실전 검증 유동 손절폭(SL): -{rec_sl_pct}% (발화 후 고점 전 최대 흔들기 기준 -2%~-5% 유동 적용)",
            f"• 48h 실전 3단계 익절(TP) 중앙값 (고점 도달 평균 {avg_hours}시간 소요):",
            f"  - 1차 TP(첫 저항눌림): +{tp1_rec}% (실전 파동 Fib 2.13배 구간)",
            f"  - 2차 TP(중기 저항대): +{tp2_rec}% (실전 파동 Fib 3.06배 구간)",
            f"  - 3차 TP(48h 최종고점): +{tp3_rec}% (실전 파동 Fib 3.86배 구간)",
        ]
    )

    full_report = "\n".join(msg_lines)
    send_telegram_chunked(full_report)
    print(f"[{now_str}] 전문가급 Top 5 역추적 + 오답 노트 브리핑 전송 완료")


if __name__ == "__main__":
    main()
