import os, re, json, time, threading, urllib.parse, requests
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

import config as cfg
from config import (
    BROWSER_HEADERS,
    STATE_FILE,
    MARKET_CAP_CACHE_FILE,
    AMBIGUOUS_MC_TICKERS,
    to_float,
    normalize_funding_pct,
    is_tradfi_ticker,
    is_heavy_major_coin,
)

SCHEME = "ht" + "tps://"
BYBIT_PROXY_URL = str(getattr(cfg, "BYBIT_PROXY_URL", "") or os.getenv("BYBIT_PROXY_URL", "")).strip()

BYBIT_HOST_NAMES = (
    "api.bytick.com",
    "api.bybit.com",
    "api.bybit-tr.com",
)

# 주식 · ETF · 원자재 정적 백업 블랙리스트 (정상 크립토 단글자·일반 영단어 오탐 제거)
STOCK_TRADFI_EXACT_BLACKLIST = {
    "SAMSUNG", "HYNIX", "SKHYNIX", "SKHY", "SSNLF", "HYUNDAI", "POSCO", "NAVER", "KAKAO",
    "CELLTRION", "LGCHEM", "LGES", "COUPANG",
    "KOSPI", "KOSDAQ", "SNDK", "SOXL", "SOXS", "KORU", "CIEN",
    "COHR", "MSI", "TSLL", "NVDL", "NVDX", "TQQQ", "SQQQ", "CONL", "MSTU", "MSTX", "ANTA",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NFLX", "AMD",
    "INTC", "TSM", "ASML", "PLTR", "COIN", "MSTR", "HOOD", "BABA", "BIDU",
    "PDD", "NIO", "XPEV", "SPY", "QQQ", "IWM", "DIA", "TLT", "ARKK", "SMH",
    "GLD", "SLV", "USO", "UNG", "XAU", "XAG", "XPD", "XPT", "WTI", "BRENT", "OIL",
    "NATGAS", "COPPER", "GOLD", "SILVER", "PAXG", "XAUT", "XAUM", "KAU", "KAG", "DGX", "PMGT",
    "SPX", "NDX", "NAS100", "US500", "US30", "DJI", "VIX", "NIKKEI", "FTSE",
    "DAX", "CAC", "HSI", "BRK", "JPM", "BAC", "WFC", "UNH", "JNJ",
    "LLY", "NVO", "PFE", "MRK", "ABBV", "WMT", "COST", "SBUX", "NKE", "DIS",
    "XOM", "CVX", "LMT", "RTX", "UBER", "ABNB",
    "SNOW", "CRWD", "PANW", "SMCI", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "TXN",
    "AMAT", "LRCX", "KLAC", "MRVL", "SONY", "TOYOTA", "SOFTBANK", "TENCENT", "XIAOMI", "BYD",
    "CRCL", "RKLB", "ASTS", "IONQ", "RGTI", "OKLO", "HIMS", "RDDT", "VST", "CEG", "GEV",
    "DELL", "HPE", "WDC", "STX", "ANET", "VRT", "ALAB", "EWY", "EWJ", "FXI", "KWEB",
}

# 스테이블코인 · 각국 법정화폐(ISO 4217) · 통화 연동 토큰 영구 차단 목록
STABLECOIN_FIAT_EXACT_BLACKLIST = {
    "USD", "USDT", "USDC", "USDE", "SUSDE", "USDS", "SUSDS", "FDUSD", "DAI", "SDAI",
    "TUSD", "BUSD", "USDP", "GUSD", "PYUSD", "RLUSD", "USDD", "USDJ", "USD0", "USD1",
    "USDG", "USDL", "USDY", "USDQ", "USDR", "USDB", "USDA", "USDF", "USDM", "USDN",
    "USDX", "USDK", "USDCV", "USDCE", "USDTE", "USDTB", "STUSDT", "EUSD", "XUSD",
    "AUSD", "CUSD", "DUSD", "FUSD", "HUSD", "IUSD", "LUSD", "MUSD", "NUSD", "OUSD",
    "PUSD", "RUSD", "SUSD", "VUSD", "WUSD", "ZUSD", "FRAX", "SFRXUSD", "FRXUSD", "CRVUSD",
    "GHO", "MIM", "DOLA", "FEI", "ALUSD", "LISUSD", "DEUSD", "FXUSD", "MKUSD", "HAY",
    "USR", "UXD", "USTC", "UST", "TBILL", "OUSG", "BUIDL", "USYC", "USDZ", "USDO",
    "EUR", "EURC", "EURT", "EURS", "EURI", "AEUR", "CEUR", "EURA", "AGEUR", "EURQ", "EURR", "JEUR", "SEUR", "EUROC",
    "GBP", "GBPT", "GBPC", "EGBP", "BGBP", "VGBP",
    "JPY", "JPYC", "GYEN", "ZJPY", "JPYT",
    "KRW", "KRWUSDT", "KRWT", "KRWC", "XKRW", "DKRW", "KRT",
    "CNY", "CNH", "CNHT", "CNYT",
    "AUD", "AUDD", "AUDC", "AUDT", "AUDX", "TAUD",
    "CAD", "CADC", "CADT", "CADX", "TCAD", "QCAD",
    "CHF", "CHFT", "ZCHF", "XCHF", "VCHF",
    "HKD", "HKDT", "HKDC", "THKD",
    "SGD", "XSGD", "SGDT", "SGDC",
    "NZD", "NZDS", "NZDT",
    "TRY", "TRYB", "TRYC", "TRYT", "BIDR", "IDRT", "IDRX", "XIDR", "IDR",
    "BRL", "BRZ", "BRLA", "BRLT", "CBRL",
    "MXN", "MXNT", "MXNE", "MXNC",
    "ZAR", "ZARP", "ZART",
    "INR", "INRT", "VND", "VNDC", "THB", "THBT", "PHP", "PHPT", "MYR", "TWD",
    "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RUB", "UAH",
    "ARS", "ARST", "CLP", "COP", "PEN", "AED", "AECOIN", "SAR", "ILS", "NGN", "NGNT", "KES", "EGP",
}

STABLE_FIAT_REGEX = re.compile(
    r"^(?:"
    r"USD[A-Z0-9]{0,3}|"
    r"[A-Z]{1,4}USD|"
    r"EUR[CTSIQR]?|"
    r"[ACJSE]EUR|"
    r"(?:GBP|JPY|KRW|CNY|CNH|CHF|HKD|SGD|NZD|TRY|BRL|MXN|ZAR|IDR|VND|THB|AED|SAR)[CTSBDRQX]?"
    r")$"
)

STOCK_SUBSTRING_KEYWORDS = (
    "SAMSUNG", "HYNIX", "SKHY", "HYUNDAI", "POSCO", "NAVER", "KAKAO", "KOSPI", "NASDAQ",
    "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "NFLX", "MSTR", "COIN",
    "XAU", "XAG", "WTI", "BRENT", "SPX", "NDX", "XSTOCK",
)

TRADFI_SYMBOL_TYPE_KEYWORDS = (
    "stock", "etf", "commod", "forex", "fx", "tradfi", "equity", "metal", "index", "rwa",
)

_CACHE_LOCK = threading.RLock()
_VALID_BYBIT_SYMBOLS_CACHE = {"ts": 0, "symbols": set(), "bases": set()}
_DYNAMIC_TRADFI_CACHE = {"ts": 0, "symbols": set(), "bases": set()}
_MARKET_CAP_MEM_CACHE = {"ts": 0, "caps": {}}
_ON_DEMAND_MC_CACHE = {}
_LS_RATIO_MEM_CACHE = {}
_BYBIT_DIRECT_BLOCKED = False
_BINANCE_FAPI_DIRECT_BLOCKED = False


def clean_url(raw_url):
    s = str(raw_url or "").strip()
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", s)
    s = s.replace("[", "").replace("]", "")
    if not s.startswith("http"):
        s = SCHEME + s.lstrip("/")
    return s


def _is_bybit_direct_blocked():
    with _CACHE_LOCK:
        return _BYBIT_DIRECT_BLOCKED


def _set_bybit_direct_blocked(val=True):
    global _BYBIT_DIRECT_BLOCKED
    with _CACHE_LOCK:
        _BYBIT_DIRECT_BLOCKED = bool(val)


def _is_binance_fapi_blocked():
    with _CACHE_LOCK:
        return _BINANCE_FAPI_DIRECT_BLOCKED


def _set_binance_fapi_blocked(val=True):
    global _BINANCE_FAPI_DIRECT_BLOCKED
    with _CACHE_LOCK:
        _BINANCE_FAPI_DIRECT_BLOCKED = bool(val)


def _is_bybit_item_tradfi(it):
    if not isinstance(it, dict):
        return False
    sym_type = str(it.get("symbolType") or "").lower().strip()
    if sym_type and any(k in sym_type for k in TRADFI_SYMBOL_TYPE_KEYWORDS):
        return True
    if str(it.get("marketRegion") or "").strip():
        return True
    if str(it.get("underlyingTicker") or "").strip():
        return True
    x_mult = str(it.get("xstockMultiplier") or "").strip()
    if x_mult and x_mult not in ("1", "1.0", "0"):
        return True
    return False


def _fetch_bitget_rwa_blocklist():
    syms, bases = set(), set()
    try:
        url = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/contracts?productType=USDT-FUTURES")
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=4.0)
        if resp.status_code == 200:
            for it in resp.json().get("data", []):
                is_rwa = str(it.get("isRwa") or "").upper().strip() == "YES"
                sym_type = str(it.get("symbolType") or "").lower().strip()
                if is_rwa or any(k in sym_type for k in TRADFI_SYMBOL_TYPE_KEYWORDS):
                    sym = str(it.get("symbol") or "").upper().strip()
                    base = str(it.get("baseCoin") or (sym[:-4] if sym.endswith("USDT") else sym)).upper().strip()
                    if sym:
                        syms.add(sym)
                    if base:
                        bases.add(base)
                        syms.add(f"{base}USDT")
    except Exception:
        pass
    return syms, bases


def _fetch_okx_tradfi_blocklist():
    syms, bases = set(), set()
    try:
        url = clean_url(f"{SCHEME}www.okx.com/api/v5/public/instruments?instType=SWAP")
        resp = requests.get(url, headers=BROWSER_HEADERS, timeout=4.0)
        if resp.status_code == 200:
            for it in resp.json().get("data", []):
                cat = str(it.get("instCategory") or "").strip()
                if cat in ("3", "4"):
                    inst_id = str(it.get("instId") or "").upper().strip()
                    base = inst_id.split("-")[0] if "-" in inst_id else ""
                    if base:
                        bases.add(base)
                        syms.add(f"{base}USDT")
    except Exception:
        pass
    return syms, bases


def _ensure_dynamic_tradfi_blocklist():
    now = time.time()
    with _CACHE_LOCK:
        if now - _DYNAMIC_TRADFI_CACHE["ts"] < 900 and _DYNAMIC_TRADFI_CACHE["bases"]:
            return
        _DYNAMIC_TRADFI_CACHE["ts"] = now

    new_syms, new_bases = set(), set()
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_bg = ex.submit(_fetch_bitget_rwa_blocklist)
            f_okx = ex.submit(_fetch_okx_tradfi_blocklist)
            for fut in (f_bg, f_okx):
                try:
                    s_set, b_set = fut.result(timeout=5)
                    new_syms.update(s_set)
                    new_bases.update(b_set)
                except Exception:
                    pass
    except Exception:
        pass

    if new_syms or new_bases:
        with _CACHE_LOCK:
            _DYNAMIC_TRADFI_CACHE["symbols"].update(new_syms)
            _DYNAMIC_TRADFI_CACHE["bases"].update(new_bases)


def _register_dynamic_tradfi(sym, base=None):
    s = str(sym or "").upper().strip()
    b = str(base or (s[:-4] if s.endswith("USDT") else s)).upper().strip()
    with _CACHE_LOCK:
        if s:
            _DYNAMIC_TRADFI_CACHE["symbols"].add(s)
        if b:
            _DYNAMIC_TRADFI_CACHE["bases"].add(b)
            _DYNAMIC_TRADFI_CACHE["symbols"].add(f"{b}USDT")


def is_stablecoin_or_fiat_symbol(sym_or_base):
    s = str(sym_or_base or "").upper().strip()
    if not s:
        return True
    base = s[:-4] if s.endswith("USDT") else s
    base_clean = re.sub(r"^(1000000|100000|10000|1000|100|10)", "", base)
    if not base_clean:
        return True
    if base in STABLECOIN_FIAT_EXACT_BLACKLIST or base_clean in STABLECOIN_FIAT_EXACT_BLACKLIST:
        return True
    if STABLE_FIAT_REGEX.match(base) or STABLE_FIAT_REGEX.match(base_clean):
        return True
    return False


def is_stock_or_tradfi_symbol(sym_or_base):
    """전 모듈 단일 통합 비대상(스테이블·법정화폐·주식·RWA) 차단 필터"""
    if is_stablecoin_or_fiat_symbol(sym_or_base):
        return True
    _ensure_dynamic_tradfi_blocklist()
    s = str(sym_or_base or "").upper().strip()
    if not s:
        return True
    base = s[:-4] if s.endswith("USDT") else s
    base_clean = re.sub(r"^(1000000|100000|10000|1000|100|10)", "", base)

    with _CACHE_LOCK:
        if (
            s in _DYNAMIC_TRADFI_CACHE["symbols"]
            or base in _DYNAMIC_TRADFI_CACHE["bases"]
            or base_clean in _DYNAMIC_TRADFI_CACHE["bases"]
        ):
            return True
    if base in STOCK_TRADFI_EXACT_BLACKLIST or base_clean in STOCK_TRADFI_EXACT_BLACKLIST:
        return True
    for kw in STOCK_SUBSTRING_KEYWORDS:
        if kw in base:
            return True
    if is_tradfi_ticker(base) or is_tradfi_ticker(base_clean):
        return True
    return False


def _build_private_proxy_url(encoded_target):
    """BYBIT_PROXY_URL 말단 '/' 포함 여부와 무관하게 '?url=' 쿼리를 정확히 조합"""
    base_p = str(BYBIT_PROXY_URL or "").strip()
    if not base_p:
        return ""
    if base_p.endswith("="):
        return f"{base_p}{encoded_target}"
    if "?" in base_p:
        sep = "" if base_p.endswith("&") else "&"
        return f"{base_p}{sep}url={encoded_target}"
    return f"{base_p.rstrip('/')}?url={encoded_target}"


def _fetch_json_via_proxy(target_url, timeout=4.5):
    cleaned = clean_url(target_url)
    encoded = urllib.parse.quote(cleaned, safe="")
    proxy_urls = []
    if BYBIT_PROXY_URL:
        p_priv = _build_private_proxy_url(encoded)
        if p_priv:
            proxy_urls.append(p_priv)
    proxy_urls.extend([
        clean_url(f"{SCHEME}api.codetabs.com/v1/proxy?quest={encoded}"),
        clean_url(f"{SCHEME}api.allorigins.win/raw?url={encoded}"),
    ])
    for purl in proxy_urls:
        try:
            resp = requests.get(purl, headers=BROWSER_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            continue
    return None


def _fetch_bybit_json_via_proxy(path_and_query, timeout=4.5):
    data = _fetch_json_via_proxy(f"{SCHEME}api.bybit.com{path_and_query}", timeout=timeout)
    if isinstance(data, dict) and data.get("retCode") == 0:
        return data
    return None


def _fetch_binance_fapi_json(path_and_query, timeout=3.0):
    full_url = clean_url(f"{SCHEME}fapi.binance.com{path_and_query}")
    if not _is_binance_fapi_blocked():
        try:
            resp = requests.get(full_url, headers=BROWSER_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (403, 451):
                _set_binance_fapi_blocked(True)
        except Exception:
            _set_binance_fapi_blocked(True)

    return _fetch_json_via_proxy(full_url, timeout=max(3.5, timeout))


def _fetch_on_demand_market_cap_m(base_clean):
    if not base_clean or len(base_clean) <= 1 or base_clean in AMBIGUOUS_MC_TICKERS:
        return 0.0
    now = time.time()
    with _CACHE_LOCK:
        cached = _ON_DEMAND_MC_CACHE.get(base_clean)
        if cached and (now - cached["ts"]) < 21600:
            return cached["mc_m"]

    mc_m = 0.0
    try:
        s_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/search")
        r_s = requests.get(s_url, params={"query": base_clean}, headers=BROWSER_HEADERS, timeout=3.0)
        if r_s.status_code == 200:
            coins = r_s.json().get("coins") or []
            matched = [c for c in coins if str(c.get("symbol") or "").upper().strip() == base_clean]
            if matched:
                matched.sort(key=lambda x: int(to_float(x.get("market_cap_rank"), 999999)))
                cg_id = str(matched[0].get("id") or "").strip()
                if cg_id:
                    p_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/simple/price")
                    r_p = requests.get(
                        p_url,
                        params={"ids": cg_id, "vs_currencies": "usd", "include_market_cap": "true"},
                        headers=BROWSER_HEADERS,
                        timeout=3.0,
                    )
                    if r_p.status_code == 200:
                        mc_usd = to_float((r_p.json().get(cg_id) or {}).get("usd_market_cap"), 0.0)
                        if mc_usd > 0:
                            mc_m = round(mc_usd / 1e6, 3)
    except Exception:
        pass

    with _CACHE_LOCK:
        _ON_DEMAND_MC_CACHE[base_clean] = {"ts": now, "mc_m": mc_m}
    return mc_m


def get_market_cap_m(symbol_or_base, allow_on_demand=True, turnover_m=0.0):
    """스레드 안전 시가총액 조회 ($M) — 1글자·중복 충돌 티커 중립(0.0) 처리"""
    s = str(symbol_or_base or "").upper().strip()
    base = s[:-4] if s.endswith("USDT") else s
    base_clean = re.sub(r"^(1000000|100000|10000|1000|100|10)", "", base)

    if not base_clean or len(base_clean) <= 1 or base_clean in AMBIGUOUS_MC_TICKERS or base in AMBIGUOUS_MC_TICKERS:
        return 0.0

    now = time.time()
    with _CACHE_LOCK:
        if (now - _MARKET_CAP_MEM_CACHE["ts"]) >= 600:
            _MARKET_CAP_MEM_CACHE["ts"] = now
            if os.path.exists(MARKET_CAP_CACHE_FILE):
                try:
                    with open(MARKET_CAP_CACHE_FILE, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    caps = raw.get("caps") if isinstance(raw, dict) else {}
                    if isinstance(caps, dict):
                        _MARKET_CAP_MEM_CACHE["caps"] = dict(caps)
                except Exception:
                    pass
        caps = _MARKET_CAP_MEM_CACHE.get("caps") or {}

    mc_val = 0.0
    if base in caps:
        mc_val = round(to_float(caps.get(base), 0.0), 3)
    elif base_clean in caps:
        mc_val = round(to_float(caps.get(base_clean), 0.0), 3)

    if mc_val >= 0.5:
        return mc_val

    if allow_on_demand:
        return _fetch_on_demand_market_cap_m(base_clean)
    return 0.0


def _parse_bitget_ls_ratio(resp_json):
    if not isinstance(resp_json, dict):
        return 0.0
    data = resp_json.get("data")
    if isinstance(data, list) and data:
        row = data[-1]
        if isinstance(row, dict):
            direct_r = to_float(row.get("longShortRatio"), 0.0)
            if direct_r > 0:
                return round(direct_r, 2)
            lr = to_float(row.get("longRatio"), 0.0)
            sr = to_float(row.get("shortRatio"), 0.0)
            if lr > 0 and sr > 0:
                return round(lr / sr, 2)
    return 0.0


def fetch_smart_money_ls_ratio(symbol):
    """스레드 안전 스마트머니(고래) vs 개미 롱숏비율 4중 자동 폴백 엔진"""
    sym = str(symbol or "").upper().strip()
    if not sym.endswith("USDT") or is_stock_or_tradfi_symbol(sym):
        return {"source": "NONE", "top_ls": 0.0, "retail_ls": 0.0, "divergence": 0.0}

    now = time.time()
    with _CACHE_LOCK:
        cached = _LS_RATIO_MEM_CACHE.get(sym)
        if cached and (now - cached["ts"]) < 600:
            return dict(cached["data"])

    result = {"source": "NONE", "top_ls": 0.0, "retail_ls": 0.0, "divergence": 0.0}
    base = sym[:-4]
    base_clean = re.sub(r"^(1000000|100000|10000|1000|100|10)", "", base)

    if (not _is_binance_fapi_blocked()) or BYBIT_PROXY_URL:
        bn_syms = [sym]
        if base_clean and f"{base_clean}USDT" != sym:
            bn_syms.append(f"{base_clean}USDT")
        for b_sym in bn_syms:
            try:
                top_data = _fetch_binance_fapi_json(
                    f"/futures/data/topLongShortPositionRatio?symbol={b_sym}&period=1h&limit=1",
                    timeout=2.8,
                )
                ret_data = _fetch_binance_fapi_json(
                    f"/futures/data/globalLongShortAccountRatio?symbol={b_sym}&period=1h&limit=1",
                    timeout=2.8,
                )
                if isinstance(top_data, list) and top_data and isinstance(ret_data, list) and ret_data:
                    s_val = round(to_float(top_data[-1].get("longShortRatio"), 0.0), 2)
                    t_val = round(to_float(ret_data[-1].get("longShortRatio"), 0.0), 2)
                    if s_val > 0 and t_val > 0:
                        result = {
                            "source": "BINANCE",
                            "top_ls": s_val,
                            "retail_ls": t_val,
                            "divergence": round(s_val / max(t_val, 0.05), 2),
                        }
                        break
            except Exception:
                continue

    if result["source"] == "NONE":
        gate_contracts = list(dict.fromkeys([f"{base}_USDT", f"{base_clean}_USDT"]))
        for g_ct in gate_contracts:
            if not g_ct or g_ct.startswith("_"):
                continue
            try:
                g_url = clean_url(f"{SCHEME}api.gateio.ws/api/v4/futures/usdt/contract_stats")
                g_resp = requests.get(g_url, params={"contract": g_ct, "limit": 1}, headers=BROWSER_HEADERS, timeout=3.0)
                if g_resp.status_code == 200:
                    g_rows = g_resp.json()
                    if isinstance(g_rows, list) and g_rows:
                        row = g_rows[-1]
                        s_g = round(to_float(row.get("top_lsr_size"), 0.0), 2)
                        if s_g <= 0:
                            s_g = round(to_float(row.get("top_lsr_account"), 0.0), 2)
                        t_g = round(to_float(row.get("lsr_account"), 0.0), 2)
                        if s_g > 0 and t_g > 0:
                            result = {
                                "source": "GATE",
                                "top_ls": s_g,
                                "retail_ls": t_g,
                                "divergence": round(s_g / max(t_g, 0.05), 2),
                            }
                            break
                        elif t_g > 0:
                            result = {
                                "source": "GATE",
                                "top_ls": 0.0,
                                "retail_ls": t_g,
                                "divergence": 0.0,
                            }
                            break
            except Exception:
                continue

    if result["source"] == "NONE":
        bg_syms = list(dict.fromkeys([sym, f"{base_clean}USDT"]))
        for bg_s in bg_syms:
            if not bg_s or bg_s == "USDT":
                continue
            try:
                p_url = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/position-long-short")
                a_url = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/account-long-short")
                bg_p = {"symbol": bg_s, "productType": "USDT-FUTURES", "period": "1H"}
                r_pos = requests.get(p_url, params=bg_p, headers=BROWSER_HEADERS, timeout=3.0)
                r_acc = requests.get(a_url, params=bg_p, headers=BROWSER_HEADERS, timeout=3.0)
                s_bg = _parse_bitget_ls_ratio(r_pos.json()) if r_pos.status_code == 200 else 0.0
                t_bg = _parse_bitget_ls_ratio(r_acc.json()) if r_acc.status_code == 200 else 0.0
                if s_bg > 0 and t_bg > 0:
                    result = {
                        "source": "BITGET",
                        "top_ls": s_bg,
                        "retail_ls": t_bg,
                        "divergence": round(s_bg / max(t_bg, 0.05), 2),
                    }
                    break
                elif t_bg > 0:
                    result = {
                        "source": "BITGET",
                        "top_ls": 0.0,
                        "retail_ls": t_bg,
                        "divergence": 0.0,
                    }
                    break
            except Exception:
                continue

    if result["source"] == "NONE" and base_clean:
        try:
            u_okx = clean_url(
                f"{SCHEME}www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={base_clean}&period=1H"
            )
            r_okx = requests.get(u_okx, headers=BROWSER_HEADERS, timeout=3.0)
            if r_okx.status_code == 200:
                rows = r_okx.json().get("data") or []
                if isinstance(rows, list) and rows and isinstance(rows[0], list) and len(rows[0]) >= 2:
                    t_okx = round(to_float(rows[0][1], 0.0), 2)
                    if t_okx > 0:
                        result = {
                            "source": "OKX",
                            "top_ls": 0.0,
                            "retail_ls": t_okx,
                            "divergence": 0.0,
                        }
        except Exception:
            pass

    with _CACHE_LOCK:
        _LS_RATIO_MEM_CACHE[sym] = {"ts": now, "data": dict(result)}
    return result


def _fetch_24h_change_and_funding_backup_map():
    _ensure_dynamic_tradfi_blocklist()
    chg_map, fund_map = {}, {}
    try:
        u1 = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES")
        r1 = requests.get(u1, headers=BROWSER_HEADERS, timeout=4)
        if r1.status_code == 200:
            for it in r1.json().get("data", []):
                sym = str(it.get("symbol", "")).upper().strip()
                if not sym.endswith("USDT") or is_stock_or_tradfi_symbol(sym):
                    continue
                c24 = to_float(it.get("change24h")) * 100.0
                f_pct = normalize_funding_pct(to_float(it.get("fundingRate")) * 100.0)
                if abs(c24) > 0.001:
                    chg_map[sym] = round(c24, 2)
                fund_map[sym] = f_pct
    except Exception:
        pass

    try:
        u2 = clean_url(f"{SCHEME}api.gateio.ws/api/v4/futures/usdt/tickers")
        r2 = requests.get(u2, headers=BROWSER_HEADERS, timeout=4)
        if r2.status_code == 200:
            for it in r2.json():
                contract = str(it.get("contract", "")).upper().replace("_", "")
                if not contract.endswith("USDT") or is_stock_or_tradfi_symbol(contract):
                    continue
                c24 = to_float(it.get("change_percentage"))
                f_pct = normalize_funding_pct(to_float(it.get("funding_rate")) * 100.0)
                if contract not in chg_map and abs(c24) > 0.001:
                    chg_map[contract] = round(c24, 2)
                if contract not in fund_map:
                    fund_map[contract] = f_pct
    except Exception:
        pass

    return chg_map, fund_map


def _normalize_kline_df(rows, expected_interval_min=None, allow_historical=False):
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
            .drop_duplicates(subset=["timestamp"], keep="last")
            .sort_values("timestamp", ascending=True)
            .reset_index(drop=True)
        )

        if expected_interval_min is not None and len(df) >= 4:
            median_diff_sec = df["timestamp"].diff().median() / 1000.0
            target_sec = expected_interval_min * 60.0
            if not (target_sec * 0.5 <= median_diff_sec <= target_sec * 1.5):
                return pd.DataFrame()

        if not allow_historical:
            max_stale_sec = max(6 * 3600.0, float(expected_interval_min or 5) * 60.0 * 2.5)
            latest_ts_ms = to_float(df["timestamp"].iloc[-1], 0.0)
            if latest_ts_ms > 0 and (time.time() * 1000.0 - latest_ts_ms) > (max_stale_sec * 1000.0):
                return pd.DataFrame()

        return df
    except Exception:
        return pd.DataFrame()


def get_valid_bybit_linear_symbols():
    """상장 유효성 검증(valid_syms) — CoinGecko $500k 컷 제거 유지"""
    _ensure_dynamic_tradfi_blocklist()
    now = time.time()
    with _CACHE_LOCK:
        if now - _VALID_BYBIT_SYMBOLS_CACHE["ts"] < 900 and len(_VALID_BYBIT_SYMBOLS_CACHE["symbols"]) >= 80:
            return set(_VALID_BYBIT_SYMBOLS_CACHE["symbols"]), set(_VALID_BYBIT_SYMBOLS_CACHE["bases"])

    valid_syms, valid_bases = set(), set()

    def _parse_bybit_instruments(items):
        for it in items:
            sym = str(it.get("symbol", "")).upper().strip()
            quote = str(it.get("quoteCoin", "")).upper().strip()
            status = str(it.get("status", "")).strip()
            contract_type = str(it.get("contractType", "")).strip()
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            if _is_bybit_item_tradfi(it):
                _register_dynamic_tradfi(sym, base)
                continue
            if quote == "USDT" and status == "Trading" and contract_type == "LinearPerpetual":
                if base.isascii() and not is_stock_or_tradfi_symbol(sym):
                    valid_syms.add(sym)
                    valid_bases.add(base)

    if not _is_bybit_direct_blocked():
        for host in BYBIT_HOST_NAMES:
            try:
                cursor = ""
                fetched_ok = False
                for _ in range(4):
                    params = {"category": "linear", "status": "Trading", "limit": 1000}
                    if cursor:
                        params["cursor"] = cursor
                    url = clean_url(f"{SCHEME}{host}/v5/market/instruments-info")
                    resp = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=2.5)
                    if resp.status_code != 200:
                        break
                    res_obj = resp.json().get("result", {})
                    items = res_obj.get("list", [])
                    if not items:
                        break
                    fetched_ok = True
                    _parse_bybit_instruments(items)
                    cursor = str(res_obj.get("nextPageCursor") or "").strip()
                    if not cursor:
                        break
                if fetched_ok and len(valid_syms) >= 80:
                    with _CACHE_LOCK:
                        _VALID_BYBIT_SYMBOLS_CACHE.update({"ts": now, "symbols": set(valid_syms), "bases": set(valid_bases)})
                    return valid_syms, valid_bases
            except Exception:
                continue
        _set_bybit_direct_blocked(True)

    cursor = ""
    for _ in range(3):
        q = "/v5/market/instruments-info?category=linear&status=Trading&limit=1000"
        if cursor:
            q += f"&cursor={urllib.parse.quote(cursor, safe='')}"
        proxy_data = _fetch_bybit_json_via_proxy(q, timeout=4.5)
        if not proxy_data:
            break
        res_obj = proxy_data.get("result", {})
        items = res_obj.get("list", [])
        if not items:
            break
        _parse_bybit_instruments(items)
        cursor = str(res_obj.get("nextPageCursor") or "").strip()
        if not cursor:
            break

    if len(valid_syms) >= 80:
        with _CACHE_LOCK:
            _VALID_BYBIT_SYMBOLS_CACHE.update({"ts": now, "symbols": set(valid_syms), "bases": set(valid_bases)})
        return valid_syms, valid_bases

    try:
        cg_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all")
        cg_resp = requests.get(cg_url, headers=BROWSER_HEADERS, timeout=6)
        if cg_resp.status_code == 200:
            for t in cg_resp.json().get("tickers", []):
                sym = str(t.get("symbol", "")).upper().strip()
                c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                if sym.endswith("USDT") and c_type == "perpetual" and not t.get("expired_at"):
                    base = sym[:-4]
                    if base.isascii() and not is_stock_or_tradfi_symbol(sym):
                        valid_syms.add(sym)
                        valid_bases.add(base)
            if len(valid_syms) >= 50:
                with _CACHE_LOCK:
                    _VALID_BYBIT_SYMBOLS_CACHE.update({"ts": now, "symbols": set(valid_syms), "bases": set(valid_bases)})
                return valid_syms, valid_bases
    except Exception:
        pass

    with _CACHE_LOCK:
        return set(_VALID_BYBIT_SYMBOLS_CACHE["symbols"]), set(_VALID_BYBIT_SYMBOLS_CACHE["bases"])


def get_scanner_candidates():
    """
    [2차 교정 · 4중 후보군 수집 폴백]
    1순위: Bybit 직통
    2순위: 프라이빗/공용 프록시 경유 Bybit
    3순위: CoinGecko Bybit 파생 티커
    4순위: 미국 IP 무차단 Bitget USDT-FUTURES 직통 폴백 (CoinGecko 429 제한 시에도 스캔 무중단 보장!)
    """
    valid_syms, valid_bases = get_valid_bybit_linear_symbols()
    candidates = []
    pure_crypto_bases = set(valid_bases)
    with _CACHE_LOCK:
        rwa_blocked_bases = set(_DYNAMIC_TRADFI_CACHE["bases"])

    def _parse_bybit_ticker_items(items):
        out = []
        for it in items:
            sym = str(it.get("symbol", "")).upper().strip()
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            if is_stock_or_tradfi_symbol(sym):
                rwa_blocked_bases.add(base)
                continue
            if (valid_syms and sym not in valid_syms) or not base.isascii():
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
                "symbol": sym, "base_coin": base, "price": price,
                "turnover_m": turnover_m, "oi_m": oi_m,
                "change_24h": round(change_24h, 2), "funding_pct": funding_pct,
            })
        return out

    # 1순위: Bybit 직통
    if not _is_bybit_direct_blocked():
        for host in BYBIT_HOST_NAMES:
            try:
                url = clean_url(f"{SCHEME}{host}/v5/market/tickers")
                resp = requests.get(url, params={"category": "linear"}, headers=BROWSER_HEADERS, timeout=2.5)
                if resp.status_code == 200:
                    candidates = _parse_bybit_ticker_items(resp.json().get("result", {}).get("list", []))
                    if candidates:
                        break
            except Exception:
                continue

    # 2순위: 프록시 경유 Bybit
    if not candidates:
        proxy_data = _fetch_bybit_json_via_proxy("/v5/market/tickers?category=linear", timeout=4.5)
        if proxy_data:
            candidates = _parse_bybit_ticker_items(proxy_data.get("result", {}).get("list", []))

    # 3순위: CoinGecko Bybit 파생 티커
    if not candidates:
        chg_backup_map, fund_backup_map = _fetch_24h_change_and_funding_backup_map()
        try:
            cg_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all")
            cg_resp = requests.get(cg_url, headers=BROWSER_HEADERS, timeout=6)
            if cg_resp.status_code == 200:
                for t in cg_resp.json().get("tickers", []):
                    sym = str(t.get("symbol", "")).upper().strip()
                    c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                    if not sym.endswith("USDT") or c_type != "perpetual" or t.get("expired_at"):
                        continue
                    base = sym[:-4]
                    if is_stock_or_tradfi_symbol(sym):
                        rwa_blocked_bases.add(base)
                        continue
                    if (valid_syms and sym not in valid_syms) or not base.isascii():
                        continue
                    if chg_backup_map and sym not in chg_backup_map:
                        continue
                    if is_heavy_major_coin(base):
                        continue

                    price = to_float(t.get("last"))
                    turnover_m = to_float((t.get("converted_volume") or {}).get("usd")) / 1e6
                    oi_m = to_float(t.get("open_interest_usd")) / 1e6

                    raw_chg = t.get("h24_percentage_change")
                    if raw_chg is None or abs(to_float(raw_chg)) < 1e-5:
                        raw_chg = t.get("price_percentage_change_24h")
                    change_24h = to_float(raw_chg)
                    if abs(change_24h) < 1e-5 and sym in chg_backup_map:
                        change_24h = chg_backup_map[sym]

                    if sym in fund_backup_map:
                        funding_pct = fund_backup_map[sym]
                    else:
                        raw_f = to_float(t.get("funding_rate"))
                        if 0 < abs(raw_f) <= 0.005:
                            raw_f = raw_f * 100.0
                        funding_pct = normalize_funding_pct(raw_f)

                    if price <= 0 or turnover_m < 2.0 or oi_m <= 0.1:
                        continue

                    pure_crypto_bases.add(base)
                    candidates.append({
                        "symbol": sym, "base_coin": base, "price": price,
                        "turnover_m": turnover_m, "oi_m": oi_m,
                        "change_24h": round(change_24h, 2), "funding_pct": funding_pct,
                    })
            else:
                print(f"[후보군 CoinGecko HTTP {cg_resp.status_code}] 4순위 Bitget 직통 폴백 전환")
        except Exception as e:
            print(f"[후보군 CoinGecko 예외] {e} — 4순위 Bitget 직통 폴백 전환")

    # 4순위: 미국 IP 무차단 Bitget USDT-FUTURES 직통 폴백 (CoinGecko 429 대응)
    if not candidates:
        try:
            bg_url = clean_url(f"{SCHEME}api.bitget.com/api/v2/mix/market/tickers?productType=USDT-FUTURES")
            bg_resp = requests.get(bg_url, headers=BROWSER_HEADERS, timeout=5.0)
            if bg_resp.status_code == 200:
                for it in bg_resp.json().get("data", []):
                    sym = str(it.get("symbol", "")).upper().strip()
                    if not sym.endswith("USDT"):
                        continue
                    base = sym[:-4]
                    if is_stock_or_tradfi_symbol(sym):
                        rwa_blocked_bases.add(base)
                        continue
                    if (valid_syms and sym not in valid_syms) or not base.isascii():
                        continue
                    if is_heavy_major_coin(base):
                        continue

                    price = to_float(it.get("lastPr") or it.get("lastPrice"))
                    turnover_m = to_float(it.get("usdtVolume") or it.get("quoteVolume")) / 1e6
                    hold_amt = to_float(it.get("holdingAmount"))
                    oi_m = (hold_amt * price) / 1e6 if (hold_amt > 0 and price > 0) else 0.0
                    change_24h = to_float(it.get("change24h")) * 100.0
                    funding_pct = normalize_funding_pct(to_float(it.get("fundingRate")) * 100.0)

                    if price <= 0 or turnover_m < 2.0 or oi_m <= 0.1:
                        continue

                    pure_crypto_bases.add(base)
                    candidates.append({
                        "symbol": sym, "base_coin": base, "price": price,
                        "turnover_m": turnover_m, "oi_m": oi_m,
                        "change_24h": round(change_24h, 2), "funding_pct": funding_pct,
                    })
                if candidates:
                    print(f"[후보군 4순위 Bitget 직통 폴백 성공] {len(candidates)}종목 확보")
        except Exception as e:
            print(f"[후보군 4순위 Bitget 폴백 예외] {e}")

    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates, pure_crypto_bases, rwa_blocked_bases


def fetch_klines(symbol, interval="5", limit=150, end_ms=None, allow_historical=False):
    sym = str(symbol).upper().strip()
    if not sym.endswith("USDT") or is_stock_or_tradfi_symbol(sym):
        return pd.DataFrame()

    valid_syms, _ = get_valid_bybit_linear_symbols()
    if valid_syms and sym not in valid_syms:
        return pd.DataFrame()

    iv_str = str(interval).upper().strip()
    iv_min = {
        "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
        "60": 60, "120": 120, "240": 240, "360": 360, "720": 720, "D": 1440, "W": 10080,
    }.get(iv_str, 5)
    lim = int(min(max(limit, 2), 1000))
    hist_ok = bool(allow_historical or (end_ms is not None))
    base_min = 4 if iv_str == "W" else (5 if hist_ok else (10 if iv_str in ("D", "720") else 15))
    min_bars = max(2, min(base_min, lim))

    if not _is_bybit_direct_blocked():
        params = {"category": "linear", "symbol": sym, "interval": iv_str, "limit": lim}
        if end_ms is not None:
            params["end"] = int(end_ms)
        for host in BYBIT_HOST_NAMES:
            try:
                url = clean_url(f"{SCHEME}{host}/v5/market/kline")
                resp = requests.get(url, params=params, headers=BROWSER_HEADERS, timeout=3.0)
                if resp.status_code == 200:
                    df = _normalize_kline_df(
                        resp.json().get("result", {}).get("list", []),
                        expected_interval_min=iv_min,
                        allow_historical=hist_ok,
                    )
                    if not df.empty and len(df) >= min_bars:
                        return df
            except Exception:
                continue

    if BYBIT_PROXY_URL:
        proxy_q = f"/v5/market/kline?category=linear&symbol={sym}&interval={iv_str}&limit={lim}"
        if end_ms is not None:
            proxy_q += f"&end={int(end_ms)}"
        proxy_data = _fetch_bybit_json_via_proxy(proxy_q, timeout=4.5)
        if proxy_data:
            df = _normalize_kline_df(
                proxy_data.get("result", {}).get("list", []),
                expected_interval_min=iv_min,
                allow_historical=hist_ok,
            )
            if not df.empty and len(df) >= min_bars:
                return df

    bg_gran = {
        "1": "1m", "3": "3m", "5": "5m", "15": "15m", "30": "30m",
        "60": "1H", "120": "2H", "240": "4H", "360": "6H", "720": "12H", "D": "1D", "W": "1W",
    }.get(iv_str, "5m")

    bg_endpoints = (
        (f"{SCHEME}api.bitget.com/api/v2/mix/market/candles", min(lim, 1000)),
        (f"{SCHEME}api.bitget.com/api/v2/mix/market/history-candles", min(lim, 200)),
    )
    for bg_raw_url, bg_lim in bg_endpoints:
        try:
            bg_params = {
                "symbol": sym,
                "productType": "USDT-FUTURES",
                "granularity": bg_gran,
                "limit": str(bg_lim),
            }
            if end_ms is not None:
                bg_params["endTime"] = str(int(end_ms))
            bg_resp = requests.get(clean_url(bg_raw_url), params=bg_params, headers=BROWSER_HEADERS, timeout=4.5)
            if bg_resp.status_code == 200:
                df = _normalize_kline_df(
                    bg_resp.json().get("data", []),
                    expected_interval_min=iv_min,
                    allow_historical=hist_ok,
                )
                if not df.empty and len(df) >= min_bars:
                    return df
        except Exception:
            continue

    if not BYBIT_PROXY_URL:
        proxy_q = f"/v5/market/kline?category=linear&symbol={sym}&interval={iv_str}&limit={lim}"
        if end_ms is not None:
            proxy_q += f"&end={int(end_ms)}"
        proxy_data = _fetch_bybit_json_via_proxy(proxy_q, timeout=5.0)
        if proxy_data:
            df = _normalize_kline_df(
                proxy_data.get("result", {}).get("list", []),
                expected_interval_min=iv_min,
                allow_historical=hist_ok,
            )
            if not df.empty and len(df) >= min_bars:
                return df

    return pd.DataFrame()


def fetch_klines_window(symbol, interval="5", start_ms=0, end_ms=None, max_bars=850):
    iv_str = str(interval).upper().strip()
    iv_min = {
        "1": 1, "3": 3, "5": 5, "15": 15, "30": 30,
        "60": 60, "120": 120, "240": 240, "360": 360, "720": 720, "D": 1440, "W": 10080,
    }.get(iv_str, 5)
    iv_ms = iv_min * 60 * 1000
    cur_end = int(end_ms) if end_ms is not None else int(time.time() * 1000)
    target_start = int(start_ms) if start_ms else 0

    chunks = []
    collected = 0
    for _ in range(5):
        remain = max(50, int(max_bars) - collected)
        batch_lim = min(remain, 1000)
        df_b = fetch_klines(symbol, interval=iv_str, limit=batch_lim, end_ms=cur_end, allow_historical=True)
        if df_b.empty:
            break
        chunks.append(df_b)
        earliest = int(to_float(df_b["timestamp"].iloc[0], 0))
        collected += len(df_b)
        if target_start > 0 and earliest <= target_start:
            break
        if collected >= max_bars or len(df_b) < 20:
            break
        next_end = earliest - iv_ms
        if next_end >= cur_end:
            break
        cur_end = next_end
        time.sleep(0.12)

    if not chunks:
        return pd.DataFrame()

    merged = (
        pd.concat(chunks, ignore_index=True)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp", ascending=True)
        .reset_index(drop=True)
    )
    if end_ms is not None:
        merged = merged[merged["timestamp"] <= (int(end_ms) + iv_ms)].reset_index(drop=True)
    return merged


def fetch_verified_1m_klines(symbol, entry_ref=0.0, limit=25):
    df = fetch_klines(symbol, interval="1", limit=limit)
    if df.empty:
        return pd.DataFrame()
    if entry_ref and entry_ref > 0:
        last_close = to_float(df["close"].iloc[-1], 0.0)
        if last_close <= 0 or not (0.40 <= (last_close / entry_ref) <= 15.0):
            return pd.DataFrame()
    return df
# === END OF FILE ===
