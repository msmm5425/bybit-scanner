import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote

from config import (
    BROWSER_HEADERS,
    NON_CRYPTO_META_KEYWORDS,
    to_float,
    strip_multiplier_prefix,
    is_tradfi_ticker,
)


def get_fast_pure_crypto_whitelist():
    pure_crypto_bases = set()
    rwa_blocked_bases = set()

    try:
        r = requests.get(
            "https://api.bitget.com/api/v2/mix/market/contracts",
            params={"productType": "USDT-FUTURES"},
            headers=BROWSER_HEADERS,
            timeout=5
        )
        if r.status_code == 200:
            for item in (r.json().get("data") or []):
                base = str(item.get("baseCoin", "")).upper().strip()
                is_rwa = str(item.get("isRwa", "NO")).upper().strip()
                if not base:
                    continue
                if is_rwa == "YES" or is_tradfi_ticker(base):
                    rwa_blocked_bases.add(base)
                    rwa_blocked_bases.add(strip_multiplier_prefix(base))
                else:
                    pure_crypto_bases.add(base)
                    pure_crypto_bases.add(strip_multiplier_prefix(base))
    except Exception:
        pass

    for url, mode in [
        ("https://www.okx.com/api/v5/public/instruments?instType=SWAP", "okx"),
        ("https://api-futures.kucoin.com/api/v1/contracts/active", "kucoin"),
        ("https://api.gateio.ws/api/v4/futures/usdt/tickers", "gate"),
        ("https://data-api.binance.vision/api/v3/ticker/price", "bn"),
    ]:
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=5)
            if r.status_code == 200:
                data = r.json()
                items = (data.get("data") or []) if mode in ("okx", "kucoin") else (data if isinstance(data, list) else [])
                for item in items:
                    base = ""
                    if mode == "okx":
                        inst = str(item.get("instId", "")).upper()
                        base = inst.split("-")[0] if inst.endswith("-USDT-SWAP") else ""
                    elif mode == "kucoin":
                        sym = str(item.get("symbol", "")).upper()
                        base = sym[:-5] if sym.endswith("USDTM") else ""
                        if base == "XBT":
                            base = "BTC"
                    elif mode == "gate":
                        c = str(item.get("contract", "")).upper()
                        base = c[:-5] if c.endswith("_USDT") else ""
                    else:
                        s = str(item.get("symbol", "")).upper()
                        base = s[:-4] if s.endswith("USDT") else ""
                    if base and not is_tradfi_ticker(base) and base not in rwa_blocked_bases:
                        pure_crypto_bases.add(base)
                        pure_crypto_bases.add(strip_multiplier_prefix(base))
        except Exception:
            pass

    return pure_crypto_bases, rwa_blocked_bases


def get_scanner_candidates():
    pure_crypto_bases, rwa_blocked_bases = get_fast_pure_crypto_whitelist()
    candidates = []

    url = "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=unexpired"
    try:
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
        if resp.status_code == 200:
            for t in (resp.json().get("tickers") or []):
                if str(t.get("contract_type", "")).lower() != "perpetual":
                    continue
                symbol = str(t.get("symbol", "")).upper().strip()
                if not symbol.endswith("USDT") or "-" in symbol or "_" in symbol:
                    continue

                base_coin = symbol[:-4]
                clean_base = strip_multiplier_prefix(base_coin)
                coin_id = str(t.get("coin_id", "")).lower().strip()
                target_str = str(t.get("target", "")).lower().strip()

                if is_tradfi_ticker(base_coin) or base_coin in rwa_blocked_bases or clean_base in rwa_blocked_bases:
                    continue

                if any(k in coin_id for k in NON_CRYPTO_META_KEYWORDS) or any(k in target_str for k in NON_CRYPTO_META_KEYWORDS):
                    continue

                if pure_crypto_bases and (base_coin not in pure_crypto_bases and clean_base not in pure_crypto_bases):
                    continue

                last_price = to_float(t.get("last"))
                change_24h = to_float(
                    t.get("h24_percentage_change")
                    if t.get("h24_percentage_change") is not None
                    else t.get("price_percentage_change_24h")
                )
                turnover_24h = to_float((t.get("converted_volume") or {}).get("usd"))

                if last_price > 0 and (-8.0 <= change_24h <= 34.0) and turnover_24h >= 1_200_000:
                    candidates.append({
                        "symbol": symbol,
                        "change_24h": change_24h,
                        "turnover_24h": turnover_24h,
                        "oi_value": to_float(t.get("open_interest_usd")),
                        "funding_rate": to_float(t.get("funding_rate")),
                        "last_price": last_price
                    })

            if candidates:
                candidates.sort(
                    key=lambda x: np.log10(max(10.0, x["turnover_24h"])) * (1.0 + max(0.0, x["change_24h"]) * 0.08),
                    reverse=True
                )
                return candidates[:28], pure_crypto_bases, rwa_blocked_bases
    except Exception as e:
        print(f"[후보군 수집 예외] {e}")

    return [], pure_crypto_bases, rwa_blocked_bases


def fetch_bybit_fallback(path, params=None):
    query_str = f"?{urlencode(params)}" if params else ""
    target_bybit = f"https://api.bybit.com{path}{query_str}"
    encoded_bybit = quote(target_bybit, safe="")

    for url in [
        f"https://api.codetabs.com/v1/proxy?quest={encoded_bybit}",
        f"https://api.allorigins.win/raw?url={encoded_bybit}",
    ]:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("retCode") == 0:
                    return data.get("result", {})
        except Exception:
            pass
    return {}


def fetch_raw_klines_multi_source(symbol, interval="5m", limit=200):
    base_coin = symbol[:-4].upper()
    clean_base = strip_multiplier_prefix(base_coin)

    if interval == "1m":
        bg_gran, gt_inter, bn_inter, by_inter, min_len = "1m", "1m", "1m", "1", 3
    elif interval == "5m":
        bg_gran, gt_inter, bn_inter, by_inter, min_len = "5m", "5m", "5m", "5", 60
    else:
        bg_gran, gt_inter, bn_inter, by_inter, min_len = "1H", "1h", "1h", "60", 30

    bg_limit = min(int(limit), 200)
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bg_url = "https://api.bitget.com/api/v2/mix/market/candles"
            resp = requests.get(bg_url, params={
                "symbol": sym_try,
                "productType": "USDT-FUTURES",
                "granularity": bg_gran,
                "limit": str(bg_limit)
            }, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                bg_data = resp.json().get("data") or []
                if isinstance(bg_data, list) and len(bg_data) >= min_len:
                    return [r[:7] for r in bg_data]
        except Exception:
            pass

    for base_try in [base_coin, clean_base]:
        try:
            gt_url = "https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
            resp = requests.get(gt_url, params={
                "contract": f"{base_try}_USDT",
                "interval": gt_inter,
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                gt_data = resp.json()
                if isinstance(gt_data, list) and len(gt_data) >= min_len:
                    converted = []
                    for item in gt_data:
                        ts_ms = int(float(item.get("t", 0))) * 1000
                        v = float(item.get("v", 0)) or float(item.get("sum", 1))
                        converted.append([
                            ts_ms, item.get("o", 0), item.get("h", 0),
                            item.get("l", 0), item.get("c", 0), v, v
                        ])
                    return converted
        except Exception:
            pass

    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            bn_url = "https://data-api.binance.vision/api/v3/klines"
            resp = requests.get(bn_url, params={
                "symbol": sym_try,
                "interval": bn_inter,
                "limit": str(limit)
            }, headers=BROWSER_HEADERS, timeout=4)
            if resp.status_code == 200:
                bn_data = resp.json()
                if isinstance(bn_data, list) and len(bn_data) >= min_len:
                    return [[r[0], r[1], r[2], r[3], r[4], r[5], r[7]] for r in bn_data]
        except Exception:
            pass

    res = fetch_bybit_fallback("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": by_inter,
        "limit": limit
    })
    rows = res.get("list") or []
    if isinstance(rows, list) and len(rows) >= min_len:
        return rows

    return []


def clean_and_scale_df(rows, ref_price=0.0, min_len=30, allow_fine_scale=True):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    if len(df) < min_len:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    if df["timestamp"].iloc[-1] < 10_000_000_000:
        df["timestamp"] = df["timestamp"] * 1000

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    last_close = to_float(df["close"].iloc[-1])
    if ref_price > 0 and last_close > 0:
        ratio = ref_price / last_close
        for mult in [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 0.001, 0.0001]:
            if 0.75 <= (ratio / mult) <= 1.35:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * mult
                last_close = to_float(df["close"].iloc[-1])
                break
        if allow_fine_scale:
            fine_ratio = ref_price / last_close if last_close > 0 else 1.0
            if 0.96 <= fine_ratio <= 1.04:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * fine_ratio

    return df


def fetch_verified_1m_klines(symbol, entry_ref, limit=25):
    """
    [1분봉 정밀 손절/익절 검증 전용]
    - CoinGecko의 반올림된 last_price나 타 거래소 동명이인 티커를 차단합니다.
    - 인위적인 fine_ratio 보정(allow_fine_scale=False)을 끄고 실제 1분봉 가격 그대로 반환합니다.
    """
    base_coin = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = strip_multiplier_prefix(base_coin)

    # 1순위: Bybit V5 API 프록시 (바이비트 원본 1분봉)
    res = fetch_bybit_fallback("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": "1",
        "limit": limit
    })
    bybit_rows = res.get("list") or []
    if isinstance(bybit_rows, list) and len(bybit_rows) >= 2:
        df = clean_and_scale_df(bybit_rows, ref_price=entry_ref, min_len=2, allow_fine_scale=False)
        if not df.empty:
            last_c = to_float(df["close"].iloc[-1])
            if entry_ref <= 0 or (0.65 <= (last_c / entry_ref) <= 1.55):
                return df

    # 2순위: Bitget USDT-FUTURES 1분봉
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            r = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": sym_try, "productType": "USDT-FUTURES", "granularity": "1m", "limit": str(limit)},
                headers=BROWSER_HEADERS,
                timeout=4
            )
            if r.status_code == 200:
                bg_data = r.json().get("data") or []
                if isinstance(bg_data, list) and len(bg_data) >= 2:
                    df = clean_and_scale_df([x[:7] for x in bg_data], ref_price=entry_ref, min_len=2, allow_fine_scale=False)
                    if not df.empty:
                        last_c = to_float(df["close"].iloc[-1])
                        if entry_ref <= 0 or (0.70 <= (last_c / entry_ref) <= 1.45):
                            return df
        except Exception:
            pass

    # 3순위: Gate.io USDT Futures 1분봉
    for base_try in [base_coin, clean_base]:
        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": f"{base_try}_USDT", "interval": "1m", "limit": str(limit)},
                headers=BROWSER_HEADERS,
                timeout=4
            )
            if r.status_code == 200:
                gt_data = r.json()
                if isinstance(gt_data, list) and len(gt_data) >= 2:
                    rows = [
                        [int(float(i.get("t", 0))) * 1000, i.get("o", 0), i.get("h", 0), i.get("l", 0), i.get("c", 0), float(i.get("v", 0)) or 1.0, 0]
                        for i in gt_data
                    ]
                    df = clean_and_scale_df(rows, ref_price=entry_ref, min_len=2, allow_fine_scale=False)
                    if not df.empty:
                        last_c = to_float(df["close"].iloc[-1])
                        if entry_ref <= 0 or (0.70 <= (last_c / entry_ref) <= 1.45):
                            return df
        except Exception:
            pass

    # 4순위: Binance Vision 1분봉
    for sym_try in [symbol, f"{clean_base}USDT"]:
        try:
            r = requests.get(
                "https://data-api.binance.vision/api/v3/klines",
                params={"symbol": sym_try, "interval": "1m", "limit": str(limit)},
                headers=BROWSER_HEADERS,
                timeout=4
            )
            if r.status_code == 200:
                bn_data = r.json()
                if isinstance(bn_data, list) and len(bn_data) >= 2:
                    rows = [[x[0], x[1], x[2], x[3], x[4], x[5], x[7]] for x in bn_data]
                    df = clean_and_scale_df(rows, ref_price=entry_ref, min_len=2, allow_fine_scale=False)
                    if not df.empty:
                        last_c = to_float(df["close"].iloc[-1])
                        if entry_ref <= 0 or (0.70 <= (last_c / entry_ref) <= 1.45):
                            return df
        except Exception:
            pass

    return pd.DataFrame()
