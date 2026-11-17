import os
import json
import time
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

# 깃허브 액션(미국 IP) 방화벽을 우회하기 위한 바이비트 글로벌 V5 공식 미러 도메인
BYBIT_HOSTS = (
    "https://api.bybit-tr.com",
    "https://api.bybit.kz",
    "https://api.bybit.nl",
    "https://api.bytick.com",
    "https://api.bybit.com",
)

_VALID_BYBIT_SYMBOLS_CACHE = {"ts": 0, "symbols": set(), "bases": set()}


def _normalize_bybit_kline_df(rows, expected_interval_min=None):
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
            if not (target_sec * 0.6 <= median_diff_sec <= target_sec * 1.4):
                return pd.DataFrame()

        return df
    except Exception:
        return pd.DataFrame()


def get_valid_bybit_linear_symbols():
    """
    [바이비트 V5 선물 화이트리스트 검증]
    바이비트 선물(linear)에 실제로 상장되어 거래 중인 USDT 무기한 심볼 목록만 확보합니다.
    """
    now = time.time()
    if now - _VALID_BYBIT_SYMBOLS_CACHE["ts"] < 600 and _VALID_BYBIT_SYMBOLS_CACHE["symbols"]:
        return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]

    valid_syms = set()
    valid_bases = set()

    for host in BYBIT_HOSTS:
        try:
            resp = requests.get(
                f"{host}/v5/market/instruments-info",
                params={"category": "linear", "status": "Trading", "limit": 1000},
                headers=BROWSER_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                items = resp.json().get("result", {}).get("list", [])
                for it in items:
                    sym = str(it.get("symbol", "")).upper()
                    quote = str(it.get("quoteCoin", "")).upper()
                    status = str(it.get("status", ""))
                    contract_type = str(it.get("contractType", "LinearPerpetual"))
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
                if valid_syms:
                    _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                    _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                    _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                    return valid_syms, valid_bases
        except Exception:
            continue

    # 미러 서버 직접 접속이 막힐 경우 CoinGecko API를 통해 바이비트 상장 심볼 목록 백업 동기화
    try:
        cg_resp = requests.get(
            "https://api.coingecko.com/api/v3/exchanges/bybit/tickers",
            headers=BROWSER_HEADERS,
            timeout=6,
        )
        if cg_resp.status_code == 200:
            cg_items = cg_resp.json().get("tickers", [])
            for t in cg_items:
                if str(t.get("target", "")).upper() == "USDT" and str(t.get("trade_url", "")).find("futures") != -1:
                    base = str(t.get("base", "")).upper()
                    sym = f"{base}USDT"
                    if base.isascii() and not is_tradfi_ticker(base):
                        valid_syms.add(sym)
                        valid_bases.add(base)
            if valid_syms:
                _VALID_BYBIT_SYMBOLS_CACHE["ts"] = now
                _VALID_BYBIT_SYMBOLS_CACHE["symbols"] = valid_syms
                _VALID_BYBIT_SYMBOLS_CACHE["bases"] = valid_bases
                return valid_syms, valid_bases
    except Exception:
        pass

    return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]


def get_scanner_candidates():
    """
    [바이비트 실제 상장 선물 전용 후보군 수집]
    - 타 거래소(비트겟·게이트아이오) 유입을 원천 차단하고 바이비트 화이트리스트만 사용합니다.
    """
    valid_syms, valid_bases = get_valid_bybit_linear_symbols()
    candidates = []
    pure_crypto_bases = set(valid_bases)
    rwa_blocked_bases = set()

    for host in BYBIT_HOSTS:
        try:
            resp = requests.get(
                f"{host}/v5/market/tickers",
                params={"category": "linear"},
                headers=BROWSER_HEADERS,
                timeout=6,
            )
            if resp.status_code != 200:
                continue

            items = resp.json().get("result", {}).get("list", [])
            if not isinstance(items, list) or not items:
                continue

            for it in items:
                sym = str(it.get("symbol", "")).upper()
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

                candidates.append({
                    "symbol": sym,
                    "base_coin": base,
                    "price": price,
                    "turnover_m": turnover_m,
                    "oi_m": oi_m,
                    "change_24h": change_24h,
                    "funding_pct": funding_pct,
                })

            if candidates:
                break
        except Exception:
            continue

    if os.path.exists(STATE_FILE) and pure_crypto_bases:
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                st = json.load(f)
            for sym_k in list((st.get("tracked") or {}).keys()):
                b_c = sym_k[:-4].upper() if sym_k.endswith("USDT") else sym_k.upper()
                if valid_syms and sym_k not in valid_syms:
                    rwa_blocked_bases.add(b_c)
        except Exception:
            pass

    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates, pure_crypto_bases, rwa_blocked_bases


def fetch_klines(symbol, interval="5", limit=150):
    """오직 Bybit V5 공식 캔들만 조회 (미상장 종목 및 타 거래소 차단)"""
    sym = str(symbol).upper()
    if not sym.endswith("USDT"):
        return pd.DataFrame()

    valid_syms, _ = get_valid_bybit_linear_symbols()
    if valid_syms and sym not in valid_syms:
        return pd.DataFrame()

    iv_str = str(interval)
    iv_min_map = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240, "D": 1440}
    iv_min = iv_min_map.get(iv_str, 5)

    params = {
        "category": "linear",
        "symbol": sym,
        "interval": iv_str,
        "limit": int(min(limit, 1000)),
    }

    for host in BYBIT_HOSTS:
        try:
            resp = requests.get(
                f"{host}/v5/market/kline",
                params=params,
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                rows = resp.json().get("result", {}).get("list", [])
                df = _normalize_bybit_kline_df(rows, expected_interval_min=iv_min)
                if not df.empty and len(df) >= 15:
                    return df
        except Exception:
            continue

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
