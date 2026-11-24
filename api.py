import os, re, json, time, urllib.parse, requests
from concurrent.futures import ThreadPoolExecutor
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
    "api.bytick.com",
    "api.bybit.com",
    "api.bybit-tr.com",
)

# 주식 · ETF · 원자재 정적 백업 블랙리스트
STOCK_TRADFI_EXACT_BLACKLIST = {
    "SAMSUNG", "HYNIX", "SKHYNIX", "SKHY", "SKH", "SSNLF", "HYUNDAI", "POSCO", "NAVER", "KAKAO",
    "CELLTRION", "LGCHEM", "LGES", "KIA", "KB", "SHINHAN", "HANA", "WOORI", "COUPANG",
    "KOSPI", "KOSDAQ", "LITE", "SNDK", "SOXL", "SOXS", "KORU", "CIEN",
    "COHR", "MSI", "TSLL", "NVDL", "NVDX", "TQQQ", "SQQQ", "CONL", "MSTU", "MSTX", "ANTA",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NFLX", "AMD",
    "INTC", "TSM", "ASML", "ARM", "PLTR", "COIN", "MSTR", "HOOD", "BABA", "BIDU",
    "JD", "PDD", "NIO", "XPEV", "LI", "SPY", "QQQ", "IWM", "DIA", "TLT", "ARKK", "SMH",
    "GLD", "SLV", "USO", "UNG", "XAU", "XAG", "XPD", "XPT", "WTI", "BRENT", "OIL",
    "NATGAS", "COPPER", "GOLD", "SILVER", "PAXG", "XAUT", "XAUM", "KAU", "KAG", "DGX", "PMGT",
    "SPX", "NDX", "NAS100", "US500", "US30", "DJI", "VIX", "NIKKEI", "FTSE",
    "DAX", "CAC", "HSI", "BRK", "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "UNH", "JNJ",
    "LLY", "NVO", "PFE", "MRK", "ABBV", "WMT", "COST", "HD", "PG", "KO", "PEP", "MCD",
    "SBUX", "NKE", "DIS", "XOM", "CVX", "BA", "CAT", "GE", "LMT", "RTX", "UBER", "ABNB",
    "SNOW", "CRWD", "PANW", "SMCI", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "TXN",
    "MU", "AMAT", "LRCX", "KLAC", "MRVL", "SONY", "TOYOTA", "SOFTBANK", "TENCENT", "XIAOMI", "BYD",
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
    r"USD[A-Z0-9]*|"
    r"[A-Z0-9]*USD|"
    r"EUR[A-Z0-9]*|"
    r"[A-Z0-9]*EUR|"
    r"(?:GBP|JPY|KRW|CNY|CNH|CHF|HKD|SGD|NZD|TRY|BRL|MXN|ZAR|IDR|VND|THB|AED|SAR)(?:[CTSBDRQXENP])?"
    r")$"
)

STOCK_SUBSTRING_KEYWORDS = (
    "SAMSUNG", "HYNIX", "SKHY", "HYUNDAI", "POSCO", "NAVER", "KAKAO", "KOSPI", "NASDAQ",
    "TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "NFLX", "MSTR", "COIN",
    "XAU", "XAG", "WTI", "BRENT", "SPX", "NDX", "STOCK", "XSTOCK",
)

TRADFI_SYMBOL_TYPE_KEYWORDS = (
    "stock", "etf", "commod", "forex", "fx", "tradfi", "equity", "metal", "index", "rwa",
)

_VALID_BYBIT_SYMBOLS_CACHE = {"ts": 0, "symbols": set(), "bases": set()}
_DYNAMIC_TRADFI_CACHE = {"ts": 0, "symbols": set(), "bases": set()}
_BYBIT_DIRECT_BLOCKED = False


def clean_url(raw_url):
    s = str(raw_url or "").strip()
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", s)
    s = s.replace("[", "").replace("]", "")
    if not s.startswith("http"):
        s = SCHEME + s.lstrip("/")
    return s


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
    if now - _DYNAMIC_TRADFI_CACHE["ts"] < 900 and _DYNAMIC_TRADFI_CACHE["bases"]:
        return
    _DYNAMIC_TRADFI_CACHE["ts"] = now
    try:
        with ThreadPoolExecutor(max_workers=2) as ex:
            f_bg = ex.submit(_fetch_bitget_rwa_blocklist)
            f_okx = ex.submit(_fetch_okx_tradfi_blocklist)
            for fut in (f_bg, f_okx):
                try:
                    s_set, b_set = fut.result(timeout=5)
                    _DYNAMIC_TRADFI_CACHE["symbols"].update(s_set)
                    _DYNAMIC_TRADFI_CACHE["bases"].update(b_set)
                except Exception:
                    pass
    except Exception:
        pass


def _register_dynamic_tradfi(sym, base=None):
    s = str(sym or "").upper().strip()
    b = str(base or (s[:-4] if s.endswith("USDT") else s)).upper().strip()
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
    if is_stablecoin_or_fiat_symbol(sym_or_base):
        return True
    _ensure_dynamic_tradfi_blocklist()
    s = str(sym_or_base or "").upper().strip()
    if not s:
        return True
    base = s[:-4] if s.endswith("USDT") else s
    base_clean = re.sub(r"^(1000000|100000|10000|1000|100|10)", "", base)

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


def _fetch_bybit_json_via_proxy(path_and_query, timeout=4.5):
    target_url = clean_url(f"{SCHEME}api.bybit.com{path_and_query}")
    encoded = urllib.parse.quote(target_url, safe="")
    proxy_urls = (
        clean_url(f"{SCHEME}api.codetabs.com/v1/proxy?quest={encoded}"),
        clean_url(f"{SCHEME}api.allorigins.win/raw?url={encoded}"),
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
    global _BYBIT_DIRECT_BLOCKED
    _ensure_dynamic_tradfi_blocklist()
    now = time.time()
    if now - _VALID_BYBIT_SYMBOLS_CACHE["ts"] < 900 and len(_VALID_BYBIT_SYMBOLS_CACHE["symbols"]) >= 80:
        return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]

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

    if not _BYBIT_DIRECT_BLOCKED:
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
                    _VALID_BYBIT_SYMBOLS_CACHE.update({"ts": now, "symbols": valid_syms, "bases": valid_bases})
                    return valid_syms, valid_bases
            except Exception:
                continue
        _BYBIT_DIRECT_BLOCKED = True

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
        _VALID_BYBIT_SYMBOLS_CACHE.update({"ts": now, "symbols": valid_syms, "bases": valid_bases})
        return valid_syms, valid_bases

    try:
        cg_url = clean_url(f"{SCHEME}api.coingecko.com/api/v3/derivatives/exchanges/bybit?include_tickers=all")
        cg_resp = requests.get(cg_url, headers=BROWSER_HEADERS, timeout=6)
        if cg_resp.status_code == 200:
            for t in cg_resp.json().get("tickers", []):
                sym = str(t.get("symbol", "")).upper().strip()
                c_type = str(t.get("contract_type", "perpetual")).lower().strip()
                conv_vol = to_float((t.get("converted_volume") or {}).get("usd"))
                if sym.endswith("USDT") and c_type == "perpetual" and not t.get("expired_at") and conv_vol >= 500000:
                    base = sym[:-4]
                    if base.isascii() and not is_stock_or_tradfi_symbol(sym):
                        valid_syms.add(sym)
                        valid_bases.add(base)
            if len(valid_syms) >= 50:
                _VALID_BYBIT_SYMBOLS_CACHE.update({"ts": now, "symbols": valid_syms, "bases": valid_bases})
                return valid_syms, valid_bases
    except Exception:
        pass

    return _VALID_BYBIT_SYMBOLS_CACHE["symbols"], _VALID_BYBIT_SYMBOLS_CACHE["bases"]


def get_scanner_candidates():
    global _BYBIT_DIRECT_BLOCKED
    valid_syms, valid_bases = get_valid_bybit_linear_symbols()
    candidates = []
    pure_crypto_bases = set(valid_bases)
    rwa_blocked_bases = set(_DYNAMIC_TRADFI_CACHE["bases"])

    def _parse_bybit_ticker_items(items):
        out = []
        for it in items:
            sym = str(it.get("symbol", "")).upper().strip()
            if not sym.endswith("USDT"):
                continue
            base = sym[:-4]
            if (valid_syms and sym not in valid_syms) or not base.isascii() or is_stock_or_tradfi_symbol(sym):
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
                "symbol": sym, "base_coin": base, "price": price,
                "turnover_m": turnover_m, "oi_m": oi_m,
                "change_24h": round(change_24h, 2), "funding_pct": funding_pct,
            })
        return out

    if not _BYBIT_DIRECT_BLOCKED:
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

    if not candidates:
        proxy_data = _fetch_bybit_json_via_proxy("/v5/market/tickers?category=linear", timeout=4.5)
        if proxy_data:
            candidates = _parse_bybit_ticker_items(proxy_data.get("result", {}).get("list", []))

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
                    if (valid_syms and sym not in valid_syms) or not base.isascii() or is_stock_or_tradfi_symbol(sym):
                        rwa_blocked_bases.add(base)
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
        except Exception as e:
            print(f"[후보군 수집 백업 예외] {e}")

    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates, pure_crypto_bases, rwa_blocked_bases


def fetch_klines(symbol, interval="5", limit=150, end_ms=None, allow_historical=False):
    """
    [보강] 최대 1000봉 조회 + 과거 구간(end_ms) 조회 지원 (기존 호출부 100% 하위 호환)
    """
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
    lim = int(min(max(limit, 10), 1000))
    hist_ok = bool(allow_historical or (end_ms is not None))
    min_bars = 4 if iv_str == "W" else (5 if hist_ok else (10 if iv_str in ("D", "720") else 15))

    if not _BYBIT_DIRECT_BLOCKED:
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
    """
    [리플레이 영상 전용 역방향 페이지네이션 캔들 로더]
    - 10시간 미만 1m(최대 650봉), 20~70시간 15m(최대 330봉) 등 단일 API 한도를 넘는 구간도
      end_ms부터 start_ms까지 역방향 자동 페이지네이션으로 누락 없이 수집
    """
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
