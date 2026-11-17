import os
import json
import time
import urllib.parse
import requests
import numpy as np
import pandas as pd

import config as cfg
from config import (
    BROWSER_HEADERS,
    STATE_FILE,
    to_float,
    normalize_funding_pct,
    is_tradfi_ticker,
    is_heavy_major_coin,
)

BYBIT_HOSTS = (
    "https://api.bybit-tr.com",
    "https://api.bybit.kz",
    "https://api.bybit.nl",
    "https://api.bytick.com",
    "https://api.bybit.com",
)

_VALID_BYBIT_SYMBOLS_CACHE = {"ts": 0, "symbols": set(), "bases": set()}


def _fetch_bybit_json_via_proxy(path_and_query, timeout=6):
    """미국 IP에서 바이비트 직접 접속이 403으로 차단될 때 글로벌 프록시를 통해 공식 V5 JSON 조회"""
    target_url = f"https://api.bybit.com{path_and_query}"
    encoded = urllib.parse.quote(target_url, safe="")
    proxy_urls = (
        f"https://api.codetabs.com/v1/proxy?quest={encoded}",
        f"https://api.allorigins.win/raw?url={encoded}",
    )
    for purl in proxy_urls:
        try:
            resp = requests.get(purl, headers=BROWSER_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("retCode") == 0:
                    return data
        except Exception:
            continue
    return None


def _normalize_kline_df(rows, expected_interval_min=None):
    if not isinstance(rows, list) or len(rows) == 0:
        return pd.DataFrame()
    try:
        cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][: len(rows[0])]
        df = pd.DataFrame(rows, columns=cols)
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df["timestamp"].max() < 1e11:
            df["timestamp"] = df["timestamp"] * 1000.0

        df = (
            df.dropna(subset=["timestamp", "open", "high", "low", "close"])
            .sort_values("timestamp", ascending=True)
            .reset_index(drop=True)
        )

        if expected_interval_min is not None and len(df) >= 5:
            median_diff_sec = df["timestamp"].diff().median() / 1000.0
            target_sec = expected_interval_min * 60.0
            if not (target_sec * 0.5 <= median_diff_sec <= target_sec * 1.5):
                return pd.DataFrame()

        return df
    except Exception:
        return pd.DataFrame()


def get_valid_bybit_linear_symbols():
    """
    [바이비트 선물(Linear USDT Perpetual) 실제 상장 종목 화이트리스트 확보]
    1순위: 바이비트 V5 공식 미러 서버 직접 조회
    2순위: 글로벌 프록시 경유 바이비트 V5 공식 상장 목록 조회
    3순위: 코인게코 바이비트 선물(Derivatives) 전용 API 조회 (미국 IP 무차단)
    """
    now = time.time()
    if now - _VALID_BYBIT_SYMBOLS_CACHE["ts"] < 900 and len(_VALID_BYBIT_SYMBOLS_CACHE["symbols"]) >= 50:
        return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]

    valid_syms = set()
    valid_bases = set()

    def _parse_bybit_instruments(items):
        for it in items:
            sym = str(it.get("symbol", "")).upper().strip()
            quote = str(it.get("quoteCoin", "")).upper().strip()
            status = str(it.get("status", "")).strip()
            contract_type = str(it.get("contractType", "LinearPerpetual")).strip()
            if (
                sym.endswith("USDT")
                and quote == "USDT"
                and status == "Trading"
                and contract_type == "LinearPerpetual"
            ):
                base = sym[:-4]
                if base.isascii() and not is_tradfi_ticker(base):
                    valid_syms.add(sym)
                    valid_bases.add(base)

    # 1순위: 바이비트 공식 도메인 직접 조회
    for host in BYBIT_HOSTS:
        try:
            resp = requests.get(
                f"{host}/v5/market/instruments-info",
                params={"category": "linear", "status": "Trading", "limit": 1000},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                items = resp.json().get("result", {}).get("list", [])
                _parse_bybit_instruments(items)
                if len(valid_syms) >= 50:
                    _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                    _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                    _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                    return valid_syms, valid_bases
        except Exception:
            continue

    # 2순위: 글로벌 프록시 경유 바이비트 V5 공식 상장 목록 조회
    proxy_data = _fetch_bybit_json_via_proxy("/v5/market/instruments-info?category=linear&status=Trading&limit=1000")
    if proxy_data:
        items = proxy_data.get("result", {}).get("list", [])
        _parse_bybit_instruments(items)
        if len(valid_syms) >= 50:
            _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
            _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
            _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
            return valid_syms, valid_bases

    # 3순위: 코인게코 바이비트 선물(Derivatives) 전용 엔드포인트 조회
    try:
        cg_deriv_resp = requests.get(
            "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all",
            headers=BROWSER_HEADERS,
            timeout=7,
        )
        if cg_deriv_resp.status_code == 200:
            tickers = cg_deriv_resp.json().get("tickers", [])
            for t in tickers:
                sym = str(t.get("symbol", "")).upper().strip()
                c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                if sym.endswith("USDT") and c_type == "perpetual":
                    base = sym[:-4]
                    if base.isascii() and not is_tradfi_ticker(base):
                        valid_syms.add(sym)
                        valid_bases.add(base)
            if len(valid_syms) >= 30:
                _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                return valid_syms, valid_bases
    except Exception:
        pass

    # 4순위: 코인게코 전체 파생상품 목록에서 Bybit (Futures) 필터링
    try:
        cg_all_resp = requests.get(
            "https://api.coingecko.com/api/v3/derivatives",
            headers=BROWSER_HEADERS,
            timeout=7,
        )
        if cg_all_resp.status_code == 200:
            for t in cg_all_resp.json():
                market_name = str(t.get("market", "")).lower()
                sym = str(t.get("symbol", "")).upper().strip()
                c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                if "bybit" in market_name and sym.endswith("USDT") and c_type == "perpetual":
                    base = sym[:-4]
                    if base.isascii() and not is_tradfi_ticker(base):
                        valid_syms.add(sym)
                        valid_bases.add(base)
            if len(valid_syms) >= 30:
                _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                return valid_syms, valid_bases
    except Exception:
        pass

    return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]


def get_scanner_candidates():
    """
    [바이비트 실제 상장 선물 전용 후보군 수집 + 미국 IP 차단 우회]
    - 반드시 get_valid_bybit_linear_symbols() 화이트리스트에 포함된 심볼만 통과시킵니다.
    - 바이비트 직접 접속이 미국 IP로 차단될 경우 프록시 -> 코인게코 바이비트 선물 -> 화이트리스트 검증된 글로벌 선물 티커 순으로 수집합니다.
    """
    valid_syms, valid_bases = get_valid_bybit_linear_symbols()
    candidates = []
    pure_crypto_bases = set(valid_bases)
    rwa_blocked_bases = set()

    def _parse_bybit_ticker_items(items):
        out = []
        for it in items:
            sym = str(it.get("symbol", "")).upper().strip()
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            if valid_syms and sym not in valid_syms:
                rwa_blocked_bases.add(base)
                continue
            if not base.isascii() or is_tradfi_ticker(base):
                rwa_blocked_bases.add(base)
                continue
            pure_crypto_bases.add(base)
            if is_heavy_major_coin(base):
                continue

            price = to_float(it.get("lastPrice"))
            turnover_m = to_float(it.get("turnover24h")) / 1e6
            oi_m = to_float(it.get("openInterestValue")) / 1e6
            change_24h = to_float(it.get("price24hPcnt")) * 100.0
            funding_pct = normalize_funding_pct(to_float(it.get("fundingRate")) * 100.0)

            if price <= 0 or turnover_m < 2.0:
                continue

            out.append({
                "symbol": sym,
                "base_coin": base,
                "price": price,
                "turnover_m": turnover_m,
                "oi_m": oi_m,
                "change_24h": change_24h,
                "funding_pct": funding_pct,
            })
        return out

    # 1순위: 바이비트 공식 도메인 직접 조회
    for host in BYBIT_HOSTS:
        try:
            resp = requests.get(
                f"{host}/v5/market/tickers",
                params={"category": "linear"},
                headers=BROWSER_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                items = resp.json().get("result", {}).get("list", [])
                candidates = _parse_bybit_ticker_items(items)
                if candidates:
                    break
        except Exception:
            continue

    # 2순위: 글로벌 프록시 경유 바이비트 V5 공식 티커 조회
    if not candidates:
        proxy_data = _fetch_bybit_json_via_proxy("/v5/market/tickers?category=linear", timeout=7)
        if proxy_data:
            items = proxy_data.get("result", {}).get("list", [])
            candidates = _parse_bybit_ticker_items(items)

    # 3순위: 코인게코 바이비트 선물 전용 티커 조회 (100% 바이비트 선물 데이터, 미국 IP 무차단)
    if not candidates:
        try:
            cg_resp = requests.get(
                "https://api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all",
                headers=BROWSER_HEADERS,
                timeout=8,
            )
            if cg_resp.status_code == 200:
                tickers = cg_resp.json().get("tickers", [])
                for t in tickers:
                    sym = str(t.get("symbol", "")).upper().strip()
                    c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                    if not sym.endswith("USDT") or c_type != "perpetual":
                        continue
                    base = sym[:-4]
                    if not base.isascii() or is_tradfi_ticker(base):
                        rwa_blocked_bases.add(base)
                        continue
                    valid_syms.add(sym)
                    pure_crypto_bases.add(base)
                    if is_heavy_major_coin(base):
                        continue

                    price = to_float(t.get("last"))
                    conv_vol = t.get("converted_volume") or {}
                    turnover_m = to_float(conv_vol.get("usd")) / 1e6
                    oi_m = to_float(t.get("open_interest_usd")) / 1e6
                    change_24h = to_float(t.get("price_percentage_change_24h"))
                    funding_pct = normalize_funding_pct(to_float(t.get("funding_rate")))

                    if price <= 0 or turnover_m < 2.0:
                        continue

                    candidates.append({
                        "symbol": sym,
                        "base_coin": base,
                        "price": price,
                        "turnover_m": turnover_m,
                        "oi_m": oi_m,
                        "change_24h": change_24h,
                        "funding_pct": funding_pct,
                    })
        except Exception:
            pass

    # 4순위: 화이트리스트(valid_syms)가 확보된 상태에서 비트겟 선물 티커로 시세 보강 (미상장 코인 100% 차단)
    if not candidates and valid_syms:
        try:
            bg_resp = requests.get(
                "https://api.bitget.com/api/v2/mix/market/tickers",
                params={"productType": "USDT-FUTURES"},
                headers=BROWSER_HEADERS,
                timeout=6,
            )
            if bg_resp.status_code == 200:
                for it in bg_resp.json().get("data", []) or []:
                    sym = str(it.get("symbol", "")).upper().strip()
                    if sym not in valid_syms:
                        continue
                    base = sym[:-4]
                    if is_heavy_major_coin(base) or is_tradfi_ticker(base):
                        continue
                    price = to_float(it.get("lastPr"))
                    turnover_m = to_float(it.get("usdtVolume")) / 1e6
                    oi_m = to_float(it.get("holdingAmount")) * price / 1e6
                    change_24h = to_float(it.get("change24h")) * 100.0
                    funding_pct = normalize_funding_pct(to_float(it.get("fundingRate")) * 100.0)
                    if price <= 0 or turnover_m < 2.0:
                        continue
                    candidates.append({
                        "symbol": sym,
                        "base_coin": base,
                        "price": price,
                        "turnover_m": turnover_m,
                        "oi_m": oi_m,
                        "change_24h": change_24h,
                        "funding_pct": funding_pct,
                    })
        except Exception:
            pass

    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates, pure_crypto_bases, rwa_blocked_bases


def fetch_klines(symbol, interval="5", limit=150):
    """
    [바이비트 상장 화이트리스트 검증 캔들 조회 + 미국 IP 차단 우회]
    - 바이비트 화이트리스트에 없는 종목은 즉시 빈 데이터프레임을 반환합니다.
    - 바이비트 직접 호출이 미국 IP에서 차단될 경우 프록시 -> Bitget -> Gate.io -> OKX 순으로 우회 조회합니다.
    """
    sym = str(symbol).upper().strip()
    if not sym.endswith("USDT"):
        return pd.DataFrame()

    valid_syms, _ = get_valid_bybit_linear_symbols()
    if valid_syms and sym not in valid_syms:
        return pd.DataFrame()

    iv_str = str(interval)
    iv_min_map = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240, "D": 1440}
    iv_min = iv_min_map.get(iv_str, 5)
    lim = int(min(limit, 500))

    # 1순위: 바이비트 공식 도메인 직접 조회
    params = {"category": "linear", "symbol": sym, "interval": iv_str, "limit": lim}
    for host in BYBIT_HOSTS:
        try:
            resp = requests.get(
                f"{host}/v5/market/kline",
                params=params,
                headers=BROWSER_HEADERS,
                timeout=3.5,
            )
            if resp.status_code == 200:
                rows = resp.json().get("result", {}).get("list", [])
                df = _normalize_kline_df(rows, expected_interval_min=iv_min)
                if not df.empty and len(df) >= 15:
                    return df
        except Exception:
            continue

    # 2순위: 비트겟 USDT-FUTURES 캔들 우회 조회 (오직 바이비트 화이트리스트 통과 종목만)
    bg_gran_map = {
        "1": "1m", "3": "3m", "5": "5m", "15": "15m",
        "30": "30m", "60": "1H", "120": "2H", "240": "4H", "D": "1D",
    }
    bg_gran = bg_gran_map.get(iv_str, "5m")
    try:
        bg_resp = requests.get(
            "https://api.bitget.com/api/v2/mix/market/candles",
            params={"symbol": sym, "productType": "USDT-FUTURES", "granularity": bg_gran, "limit": str(lim)},
            headers=BROWSER_HEADERS,
            timeout=4,
        )
        if bg_resp.status_code == 200:
            rows = bg_resp.json().get("data", [])
            df = _normalize_kline_df(rows, expected_interval_min=iv_min)
            if not df.empty and len(df) >= 15:
                return df
    except Exception:
        pass

    # 3순위: 게이트아이오 USDT 선물 캔들 우회 조회 (오직 바이비트 화이트리스트 통과 종목만)
    gate_iv_map = {
        "1": "1m", "5": "5m", "15": "15m", "30": "30m",
        "60": "1h", "240": "4h", "D": "1d",
    }
    gate_iv = gate_iv_map.get(iv_str, "5m")
    contract = f"{sym[:-4]}_USDT"
    try:
        g_resp = requests.get(
            "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
            params={"contract": contract, "interval": gate_iv, "limit": lim},
            headers=BROWSER_HEADERS,
            timeout=4,
        )
        if g_resp.status_code == 200:
            g_list = g_resp.json()
            if isinstance(g_list, list) and len(g_list) >= 15:
                rows = [
                    [
                        float(x.get("t", 0)) * 1000.0,
                        x.get("o"),
                        x.get("h"),
                        x.get("l"),
                        x.get("c"),
                        x.get("v", 0),
                        x.get("sum", 0),
                    ]
                    for x in g_list
                ]
                df = _normalize_kline_df(rows, expected_interval_min=iv_min)
                if not df.empty and len(df) >= 15:
                    return df
    except Exception:
        pass

    # 4순위: 글로벌 프록시 경유 바이비트 공식 캔들 조회
    proxy_q = f"/v5/market/kline?category=linear&symbol={sym}&interval={iv_str}&limit={lim}"
    proxy_data = _fetch_bybit_json_via_proxy(proxy_q, timeout=5)
    if proxy_data:
        rows = proxy_data.get("result", {}).get("list", [])
        df = _normalize_kline_df(rows, expected_interval_min=iv_min)
        if not df.empty and len(df) >= 15:
            return df

    return pd.DataFrame()


def fetch_verified_1m_klines(symbol, entry_ref=0.0, limit=25):
    df = fetch_klines(symbol, interval="1", limit=limit)
    if df.empty:
        return pd.DataFrame()

    if entry_ref and entry_ref > 0:
        last_close = to_float(df["close"].iloc[-1], 0.0)
        if last_close <= 0 or not (0.40 <= (last_close / entry_ref) <= 2.60):
            return pd.DataFrame()

    return df
