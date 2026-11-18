import os
import re
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

SCHEME = "ht" + "tps://"

BYBIT_HOST_NAMES = (
    "api.bybit-tr.com",
    "api.bybit.kz",
    "api.bybit.nl",
    "api.bytick.com",
    "api.bybit.com",
)

_VALID_BYBIT_SYMBOLS_CACHE = {"ts": 0, "symbols": set(), "bases": set()}


def clean_url(raw_url):
    """모바일 복사 시 발생할 수 있는 [url](url) 마크다운 변환을 완벽히 제거"""
    s = str(raw_url or "").strip()
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", s)
    s = s.replace("[", "").replace("]", "")
    if not s.startswith("http"):
        s = SCHEME + s.lstrip("/")
    return s


def _fetch_bybit_json_via_proxy(path_and_query, timeout=7):
    target_url = clean_url(f"{SCHEME}api.bybit.com{path_and_query}")
    encoded = urllib.parse.quote(target_url, safe="")
    proxy_urls = (
        clean_url(f"{SCHEME}api.codetabs.com/v1/proxy?quest={encoded}"),
        clean_url(f"{SCHEME}api.allorigins.win/raw?url={encoded}"),
        clean_url(f"{SCHEME}corsproxy.io/?{encoded}"),
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


def _fetch_24h_change_backup_map():
    """
    미국 IP에서 바이비트 직접 접속이 막혀 코인게코 백업을 사용할 때,
    누락될 수 있는 24h 등락률(%)을 실시간 선물 티커에서 보충합니다.
    """
    chg_map = {}
    # 1. Bitget USDT-FUTURES 실시간 24h 변동률
    try:
        u1 = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES")
        r1 = requests.get(u1, headers=BROWSER_HEADERS, timeout=5)
        if r1.status_code == 200:
            for it in r1.json().get("data", []):
                sym = str(it.get("symbol", "")).upper().strip()
                c24 = to_float(it.get("change24h")) * 100.0
                if sym.endswith("USDT") and abs(c24) > 0.001:
                    chg_map[sym] = round(c24, 2)
    except Exception:
        pass

    # 2. Gate.io USDT Futures 실시간 24h 변동률 보충
    try:
        u2 = clean_url(f"{SCHEME}api.gateio.ws/api/v4/futures/usdt/tickers")
        r2 = requests.get(u2, headers=BROWSER_HEADERS, timeout=5)
        if r2.status_code == 200:
            for it in r2.json():
                contract = str(it.get("contract", "")).upper().replace("_", "")
                c24 = to_float(it.get("change_percentage"))
                if contract.endswith("USDT") and contract not in chg_map and abs(c24) > 0.001:
                    chg_map[contract] = round(c24, 2)
    except Exception:
        pass

    return chg_map


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

        # 최근 캔들이 6시간 이상 멈춘 상폐/거래정지 코인 차단
        latest_ts_ms = to_float(df["timestamp"].iloc[-1], 0.0)
        if latest_ts_ms > 0 and (time.time() * 1000.0 - latest_ts_ms) > (6 * 3600 * 1000.0):
            return pd.DataFrame()

        return df
    except Exception:
        return pd.DataFrame()


def get_valid_bybit_linear_symbols():
    now = time.time()
    if now - _VALID_BYBIT_SYMBOLS_CACHE["ts"] < 900 and len(_VALID_BYBIT_SYMBOLS_CACHE["symbols"]) >= 80:
        return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]

    valid_syms = set()
    valid_bases = set()

    def _parse_bybit_instruments(items):
        for it in items:
            sym = str(it.get("symbol", "")).upper().strip()
            quote = str(it.get("quoteCoin", "")).upper().strip()
            status = str(it.get("status", "")).strip()
            contract_type = str(it.get("contractType", "")).strip()
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

    for host in BYBIT_HOST_NAMES:
        try:
            url = clean_url(f"{SCHEME}{host}/v5/market/instruments-info")
            resp = requests.get(
                url,
                params={"category": "linear", "status": "Trading", "limit": 1000},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                items = resp.json().get("result", {}).get("list", [])
                _parse_bybit_instruments(items)
                if len(valid_syms) >= 80:
                    _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                    _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                    _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                    return valid_syms, valid_bases
        except Exception:
            continue

    proxy_data = _fetch_bybit_json_via_proxy("/v5/market/instruments-info?category=linear&status=Trading&limit=1000", timeout=7)
    if proxy_data:
        items = proxy_data.get("result", {}).get("list", [])
        _parse_bybit_instruments(items)
        if len(valid_syms) >= 80:
            _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
            _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
            _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
            return valid_syms, valid_bases

    # 3순위: 코인게코 바이비트 파생상품 상장 심볼 + 활성 거래 심볼 교차 검증
    try:
        cg_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all")
        cg_resp = requests.get(cg_url, headers=BROWSER_HEADERS, timeout=7)
        if cg_resp.status_code == 200:
            tickers = cg_resp.json().get("tickers", [])
            for t in tickers:
                sym = str(t.get("symbol", "")).upper().strip()
                c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                expired = t.get("expired_at")
                conv_vol = to_float((t.get("converted_volume") or {}).get("usd"))
                if sym.endswith("USDT") and c_type == "perpetual" and not expired and conv_vol >= 500000:
                    base = sym[:-4]
                    if base.isascii() and not is_tradfi_ticker(base):
                        valid_syms.add(sym)
                        valid_bases.add(base)
            if len(valid_syms) >= 50:
                _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                return valid_syms, valid_bases
    except Exception:
        pass

    return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]


def get_scanner_candidates():
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
            prev_24h = to_float(it.get("prevPrice24h"))
            turnover_m = to_float(it.get("turnover24h")) / 1e6
            oi_m = to_float(it.get("openInterestValue")) / 1e6
            change_24h = to_float(it.get("price24hPcnt")) * 100.0

            if abs(change_24h) < 1e-6 and prev_24h > 0 and price > 0:
                change_24h = ((price - prev_24h) / prev_24h) * 100.0

            funding_pct = normalize_funding_pct(to_float(it.get("fundingRate")) * 100.0)

            if price <= 0 or turnover_m < 2.0 or oi_m <= 0.1:
                continue

            out.append({
                "symbol": sym,
                "base_coin": base,
                "price": price,
                "turnover_m": turnover_m,
                "oi_m": oi_m,
                "change_24h": round(change_24h, 2),
                "funding_pct": funding_pct,
            })
        return out

    # 1순위: 바이비트 공식 도메인 직접 조회
    for host in BYBIT_HOST_NAMES:
        try:
            url = clean_url(f"{SCHEME}{host}/v5/market/tickers")
            resp = requests.get(
                url,
                params={"category": "linear"},
                headers=BROWSER_HEADERS,
                timeout=4,
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

    # 3순위: 코인게코 바이비트 선물 전용 API + 실시간 24h 등락률 교차 보충
    if not candidates:
        chg_backup_map = _fetch_24h_change_backup_map()
        try:
            cg_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all")
            cg_resp = requests.get(cg_url, headers=BROWSER_HEADERS, timeout=8)
            if cg_resp.status_code == 200:
                tickers = cg_resp.json().get("tickers", [])
                for t in tickers:
                    sym = str(t.get("symbol", "")).upper().strip()
                    c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                    if not sym.endswith("USDT") or c_type != "perpetual" or t.get("expired_at"):
                        continue
                    if valid_syms and sym not in valid_syms:
                        continue
                    # 거래소 교차 검증: 비정상 유령 티커($OPG 등) 방어를 위해 실제 활성 선물 마켓에 존재하는 심볼만 허용
                    if chg_backup_map and sym not in chg_backup_map:
                        continue

                    base = sym[:-4]
                    if not base.isascii() or is_tradfi_ticker(base) or is_heavy_major_coin(base):
                        continue

                    price = to_float(t.get("last"))
                    conv_vol = t.get("converted_volume") or {}
                    turnover_m = to_float(conv_vol.get("usd")) / 1e6
                    oi_m = to_float(t.get("open_interest_usd")) / 1e6

                    raw_chg = t.get("h24_percentage_change")
                    if raw_chg is None or abs(to_float(raw_chg)) < 1e-5:
                        raw_chg = t.get("price_percentage_change_24h")
                    change_24h = to_float(raw_chg)
                    if abs(change_24h) < 1e-5 and sym in chg_backup_map:
                        change_24h = chg_backup_map[sym]

                    funding_pct = normalize_funding_pct(to_float(t.get("funding_rate")))

                    if price <= 0 or turnover_m < 2.0 or oi_m <= 0.1:
                        continue

                    pure_crypto_bases.add(base)
                    candidates.append({
                        "symbol": sym,
                        "base_coin": base,
                        "price": price,
                        "turnover_m": turnover_m,
                        "oi_m": oi_m,
                        "change_24h": round(change_24h, 2),
                        "funding_pct": funding_pct,
                    })
        except Exception as e:
            print(f"[후보군 수집 백업 예외] {e}")

    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates, pure_crypto_bases, rwa_blocked_bases


def fetch_klines(symbol, interval="5", limit=150):
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

    params = {"category": "linear", "symbol": sym, "interval": iv_str, "limit": lim}
    for host in BYBIT_HOST_NAMES:
        try:
            url = clean_url(f"{SCHEME}{host}/v5/market/kline")
            resp = requests.get(
                url,
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

    proxy_q = f"/v5/market/kline?category=linear&symbol={sym}&interval={iv_str}&limit={lim}"
    proxy_data = _fetch_bybit_json_via_proxy(proxy_q, timeout=5)
    if proxy_data:
        rows = proxy_data.get("result", {}).get("list", [])
        df = _normalize_kline_df(rows, expected_interval_min=iv_min)
        if not df.empty and len(df) >= 15:
            return df

    # 3순위: 바이비트 화이트리스트 통과 종목에 한해 Bitget 선물 캔들 백업
    bg_gran_map = {
        "1": "1m", "3": "3m", "5": "5m", "15": "15m",
        "30": "30m", "60": "1H", "120": "2H", "240": "4H", "D": "1D",
    }
    bg_gran = bg_gran_map.get(iv_str, "5m")
    try:
        bg_url = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/candles")
        bg_resp = requests.get(
            bg_url,
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
