import time
import requests
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import config as cfg

BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
to_float = cfg.to_float
is_tradfi_ticker = cfg.is_tradfi_ticker
strip_multiplier_prefix = cfg.strip_multiplier_prefix

# 프록시 없이 직통 호출 가능한 글로벌/지역 엔드포인트
BYBIT_DIRECT_HOSTS = (
    "https://api.bybit-tr.com",
    "https://api.bybit.kz",
    "https://api.bytick.com",
    "https://api.bybit.com",
)


def _normalize_funding(val):
    v = to_float(val, 0.0)
    if abs(v) > 0.15:
        return v / 100.0 if abs(v) <= 2.5 else v
    return v


def _fetch_bybit_tickers():
    for host in BYBIT_DIRECT_HOSTS:
        try:
            r = requests.get(
                f"{host}/v5/market/tickers",
                params={"category": "linear"},
                headers=BROWSER_HEADERS,
                timeout=5,
            )
            if r.status_code == 200:
                rows = r.json().get("result", {}).get("list", [])
                if rows:
                    out = {}
                    for it in rows:
                        sym = str(it.get("symbol", ""))
                        if not sym.endswith("USDT"):
                            continue
                        price = to_float(it.get("lastPrice"))
                        if price <= 0:
                            continue
                        out[sym] = {
                            "symbol": sym,
                            "base_coin": sym[:-4].upper(),
                            "price": price,
                            "change_24h": to_float(it.get("price24hPcnt")) * 100.0,
                            "price24hPcnt": to_float(it.get("price24hPcnt")),
                            "turnover_m": to_float(it.get("turnover24h")) / 1e6,
                            "turnover24h": to_float(it.get("turnover24h")),
                            "oi_m": to_float(it.get("openInterestValue")) / 1e6,
                            "openInterestValue": to_float(it.get("openInterestValue")),
                            "funding_pct": _normalize_funding(to_float(it.get("fundingRate")) * 100.0),
                            "fundingRate": to_float(it.get("fundingRate")),
                        }
                    if out:
                        return out
        except requests.RequestException:
            continue
    return {}


def _fetch_gate_tickers():
    out = {}
    try:
        r = requests.get(
            "https://api.gateio.ws/api/v4/futures/usdt/tickers",
            headers=BROWSER_HEADERS,
            timeout=5,
        )
        if r.status_code == 200:
            for it in r.json():
                contract = str(it.get("contract", ""))
                if not contract.endswith("_USDT"):
                    continue
                base = contract[:-5].upper()
                sym = f"{base}USDT"
                price = to_float(it.get("last"))
                if price <= 0:
                    continue
                vol_usd = to_float(it.get("volume_24h_settle") or it.get("volume_24h_quote"))
                quanto = to_float(it.get("quanto_multiplier"), 1.0)
                total_size = to_float(it.get("total_size"))
                oi_usd = (total_size * quanto * price) if quanto > 0 else 0.0
                chg_pct = to_float(it.get("change_percentage"))
                f_rate = to_float(it.get("funding_rate"))
                out[sym] = {
                    "symbol": sym,
                    "base_coin": base,
                    "price": price,
                    "change_24h": chg_pct,
                    "price24hPcnt": chg_pct / 100.0,
                    "turnover_m": vol_usd / 1e6,
                    "turnover24h": vol_usd,
                    "oi_m": oi_usd / 1e6,
                    "openInterestValue": oi_usd,
                    "funding_pct": _normalize_funding(f_rate * 100.0),
                    "fundingRate": f_rate,
                }
    except requests.RequestException as e:
        print(f"[Gate 티커 조회 경고] {e}")
    return out


def _fetch_bitget_tickers():
    out = {}
    try:
        r = requests.get(
            "https://api.bitget.com/api/v2/mix/market/tickers",
            params={"productType": "USDT-FUTURES"},
            headers=BROWSER_HEADERS,
            timeout=5,
        )
        if r.status_code == 200:
            for it in r.json().get("data", []):
                sym = str(it.get("symbol", ""))
                if not sym.endswith("USDT"):
                    continue
                base = sym[:-4].upper()
                price = to_float(it.get("lastPr"))
                if price <= 0:
                    continue
                vol_usd = to_float(it.get("usdtVolume") or it.get("quoteVolume"))
                oi_usd = to_float(it.get("holdingAmount")) * price
                chg_raw = to_float(it.get("change24h"))
                f_rate = to_float(it.get("fundingRate"))
                out[sym] = {
                    "symbol": sym,
                    "base_coin": base,
                    "price": price,
                    "change_24h": chg_raw * 100.0,
                    "price24hPcnt": chg_raw,
                    "turnover_m": vol_usd / 1e6,
                    "turnover24h": vol_usd,
                    "oi_m": oi_usd / 1e6,
                    "openInterestValue": oi_usd,
                    "funding_pct": _normalize_funding(f_rate * 100.0),
                    "fundingRate": f_rate,
                }
    except requests.RequestException as e:
        print(f"[Bitget 티커 조회 경고] {e}")
    return out


def _fetch_spot_pure_crypto_bases():
    """Binance Vision 및 OKX 현물 마켓에 존재하는 종목을 교차 확인해 RWA/주식 합성 티커 배제"""
    pure_bases = set()
    try:
        r = requests.get("https://data-api.binance.vision/api/v3/ticker/price", headers=BROWSER_HEADERS, timeout=4)
        if r.status_code == 200:
            for it in r.json():
                s = str(it.get("symbol", ""))
                if s.endswith("USDT"):
                    pure_bases.add(s[:-4].upper())
    except requests.RequestException:
        pass

    try:
        r = requests.get(
            "https://www.okx.com/api/v5/market/tickers",
            params={"instType": "SPOT"},
            headers=BROWSER_HEADERS,
            timeout=4,
        )
        if r.status_code == 200:
            for it in r.json().get("data", []):
                inst = str(it.get("instId", ""))
                if inst.endswith("-USDT"):
                    pure_bases.add(inst[:-5].upper())
    except requests.RequestException:
        pass

    return pure_bases


def get_scanner_candidates():
    """
    IDE 권고 반영: CoinGecko 단일 장애점(429 오류) 및 무료 웹 프록시를 완전 제거하고,
    다중 거래소 직통 선물 티커를 병렬 수집·교차 검증하여 후보군을 추출합니다.
    """
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_bybit = ex.submit(_fetch_bybit_tickers)
        f_gate = ex.submit(_fetch_gate_tickers)
        f_bitget = ex.submit(_fetch_bitget_tickers)
        f_spot = ex.submit(_fetch_spot_pure_crypto_bases)

        bybit_map = f_bybit.result()
        gate_map = f_gate.result()
        bitget_map = f_bitget.result()
        pure_crypto_bases = f_spot.result()

    # 2개 이상 선물 거래소에 동시 상장된 코인은 순수 크립토로 인정
    for sym in set(gate_map.keys()) & set(bitget_map.keys()):
        pure_crypto_bases.add(sym[:-4].upper())
    for sym in bybit_map.keys():
        b = sym[:-4].upper()
        if sym in gate_map or sym in bitget_map:
            pure_crypto_bases.add(b)

    primary_map = bybit_map if bybit_map else (bitget_map if bitget_map else gate_map)
    all_symbols = set(primary_map.keys()) | (set(gate_map.keys()) & set(bitget_map.keys()))

    rwa_blocked_bases = set()
    candidates = []

    for sym in all_symbols:
        base = sym[:-4].upper()
        clean_base = strip_multiplier_prefix(base)

        if is_tradfi_ticker(base) or is_tradfi_ticker(clean_base):
            rwa_blocked_bases.add(base)
            continue

        item = dict(primary_map.get(sym) or bitget_map.get(sym) or gate_map.get(sym) or {})
        if not item:
            continue

        # 거래대금 및 미결제약정(OI) 누락 시 타 거래소 데이터로 상호 보완
        g_it = gate_map.get(sym, {})
        b_it = bitget_map.get(sym, {})

        if to_float(item.get("turnover_m")) <= 0:
            item["turnover_m"] = max(to_float(b_it.get("turnover_m")), to_float(g_it.get("turnover_m")))
            item["turnover24h"] = item["turnover_m"] * 1e6

        if to_float(item.get("oi_m")) <= 0:
            item["oi_m"] = max(to_float(b_it.get("oi_m")), to_float(g_it.get("oi_m")))
            item["openInterestValue"] = item["oi_m"] * 1e6

        if to_float(item.get("funding_pct")) == 0.0:
            f_backup = to_float(b_it.get("funding_pct")) or to_float(g_it.get("funding_pct"))
            item["funding_pct"] = _normalize_funding(f_backup)

        turnover_m = to_float(item.get("turnover_m"))
        change_24h = to_float(item.get("change_24h"))

        # 최소 유동성(거래대금 1.5M 이상) 및 과도한 설거지 제외 필터
        if turnover_m < 1.5:
            continue
        if change_24h < -15.0:
            continue

        candidates.append(item)

    # 스캔 효율을 위해 상승 탄력 및 거래대금 상위 90개 핵심 후보군 정렬
    candidates.sort(
        key=lambda x: (to_float(x.get("change_24h")) * 0.7 + min(50.0, to_float(x.get("turnover_m"))) * 0.3),
        reverse=True,
    )
    return candidates[:90], pure_crypto_bases, rwa_blocked_bases


def _normalize_kline_df(raw, expected_interval_min=None):
    """오름차순(과거->최신) 강제 정렬 및 타임프레임 간격 검증 (인위적 가격 보정 배제)"""
    df = pd.DataFrame()
    if isinstance(raw, pd.DataFrame) and not raw.empty:
        df = raw.copy()
    elif isinstance(raw, list) and len(raw) > 0:
        try:
            if isinstance(raw[0], dict):
                df = pd.DataFrame(raw)
                df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
            else:
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


def fetch_klines(symbol, interval="5", limit=150):
    """프록시 없이 직통 거래소 엔드포인트에서 순수 원본 캔들 조회 (시간 오름차순 정렬 보장)"""
    iv_min = int(interval)
    base = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = strip_multiplier_prefix(base)

    # 1순위: Gate.io 선물 직통 API (미국 GitHub Actions 서버 무차단, 0.1초 응답)
    gate_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1h", 240: "4h"}.get(iv_min, "5m")
    for contract in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": contract, "interval": gate_iv, "limit": int(limit)},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if r.status_code == 200:
                df = _normalize_kline_df(r.json(), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 15:
                    return df
        except requests.RequestException:
            pass

    # 2순위: Bitget 선물 직통 API
    bg_iv = {1: "1m", 5: "5m", 15: "15m", 60: "1H", 240: "4H"}.get(iv_min, "5m")
    for bg_sym in (symbol, f"{clean_base}USDT"):
        try:
            r = requests.get(
                "https://api.bitget.com/api/v2/mix/market/candles",
                params={"symbol": bg_sym, "productType": "USDT-FUTURES", "granularity": bg_iv, "limit": str(limit)},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if r.status_code == 200:
                df = _normalize_kline_df(r.json().get("data", []), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 15:
                    return df
        except requests.RequestException:
            pass

    # 3순위: Bybit 글로벌 미러 직통 호출
    bybit_params = {"category": "linear", "symbol": symbol, "interval": str(interval), "limit": int(limit)}
    for host in BYBIT_DIRECT_HOSTS:
        try:
            r = requests.get(f"{host}/v5/market/kline", params=bybit_params, headers=BROWSER_HEADERS, timeout=4)
            if r.status_code == 200:
                df = _normalize_kline_df(r.json().get("result", {}).get("list", []), expected_interval_min=iv_min)
                if not df.empty and len(df) >= 15:
                    return df
        except requests.RequestException:
            pass

    # 4순위: MEXC 선물 직통 호출
    mexc_iv = {1: "Min1", 5: "Min5", 15: "Min15", 60: "Min60"}.get(iv_min, "Min5")
    for m_sym in (f"{base}_USDT", f"{clean_base}_USDT"):
        try:
            r = requests.get(
                f"https://contract.mexc.com/api/v1/contract/kline/{m_sym}",
                params={"interval": mexc_iv},
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if r.status_code == 200:
                d = r.json().get("data", {})
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
                    if not df.empty and len(df) >= 15:
                        return df
        except requests.RequestException:
            pass

    return pd.DataFrame()


def fetch_verified_1m_klines(symbol, entry_ref=0.0, limit=25):
    """
    IDE 권고 반영: 인위적인 1~4% 미세 가격 보정(allow_fine_scale)을 제거하여
    실제 차트의 고점/저점/지지선을 왜곡 없이 그대로 검증합니다.
    (단, 1000PEPE vs PEPE처럼 정확히 10^n배 차이 나는 단위만 정식 환산)
    """
    df = fetch_klines(symbol, "1", limit=limit)
    if df.empty:
        return pd.DataFrame()

    if entry_ref > 0:
        last_close = to_float(df["close"].iloc[-1])
        if last_close <= 0:
            return pd.DataFrame()

        ratio = entry_ref / last_close
        # 1000배/1000000배 접두어 차이인 경우에만 정확한 10의 거듭제곱 배율 적용
        for mult in (100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 0.01, 0.001, 0.0001, 0.000001):
            if 0.85 <= (ratio / mult) <= 1.15:
                for col in ("open", "high", "low", "close"):
                    df[col] = df[col] * mult
                last_close = to_float(df["close"].iloc[-1])
                break

        # 진입가 대비 35% 이상 괴리된 엉뚱한 동명이인 티커 데이터만 차단 (미세 강제 스케일링 금지)
        final_ratio = last_close / max(entry_ref, 1e-8)
        if not (0.65 <= final_ratio <= 1.45):
            return pd.DataFrame()

    return df
