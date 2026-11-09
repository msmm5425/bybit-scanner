import os
import json
import time
import subprocess
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

# 1. 환경변수 (TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN 모두 호환)
TELEGRAM_BOT_TOKEN = (
    os.getenv("TELEGRAM_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
    or ""
).strip()

TELEGRAM_CHAT_ID = (
    os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("CHAT_ID")
    or ""
).strip()

KST = timezone(timedelta(hours=9))
STATE_FILE = "bot_state.json"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

# 비암호화폐(미국 주식, Pre-IPO, ETF, 원자재, 외환) 영구 배제 블랙리스트
TRADFI_BLACKLIST = {
    "RKLB", "APPSTOCK", "APP", "COHR", "NBIS", "CRWV", "MRNA", "MUU", "CBRS",
    "SPCX", "SPCXX", "SNDK", "SNXX", "SKHYNIX", "KORU", "CEG", "AKAM", "USDBRL",
    "OPENAI", "ANTHROPIC", "STRIPE", "DATABRICKS", "DISCORD", "EPIC", "CANVA", "REVOLUT",
    "CL", "BZ", "WTI", "BRENT", "OIL", "USOIL", "UKOIL", "NG", "NATGAS", "NATGGAS",
    "GC", "SI", "HG", "PL", "PA", "XAU", "XAG", "XPT", "XPD", "COPPER", "GOLD", "SILVER",
    "SPX", "NDX", "DJI", "NAS100", "SPX500", "US30", "US500", "USTEC", "UK100", "GER40",
    "DE40", "JPN225", "JP225", "HK50", "CN50", "AUS200", "FRA40", "EU50", "DXY", "VIX",
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "TLT",
    "GLD", "SLV", "USO", "UNG", "UVXY", "ARKK", "SMH", "XLF",
    "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "GDX", "HYG",
    "EEM", "FXI", "KWEB", "IBIT", "ETHA", "BITO", "MSTX", "MSTU", "NVDL",
    "NVDU", "TSLL", "TSLS", "CONL", "AMDL",
    "AAPL", "GOOGL", "GOOG", "MSFT", "NVDA", "AMZN", "META", "TSLA",
    "NFLX", "AMD", "INTC", "AVGO", "TSM", "ASML", "ARM", "SMCI", "MU",
    "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "SNPS", "CDNS",
    "PLTR", "COIN", "MSTR", "HOOD", "MARA", "RIOT", "CLSK", "IREN",
    "CORZ", "HUT", "WULF", "BITF", "CRCL", "UNITREE", "BABA", "JD",
    "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME", "TCEHY", "BYDDY",
    "XIAOMI", "MEITUAN", "BRK", "JPM", "BAC", "WFC", "C", "GS", "MS",
    "V", "MA", "AXP", "PYPL", "SQ", "SHOP", "WMT", "COST", "TGT", "HD",
    "LOW", "DIS", "CMCSA", "BA", "LMT", "RTX", "GE", "CAT", "DE", "XOM",
    "CVX", "COP", "OXY", "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV",
    "AMGN", "GILD", "ISRG", "PG", "KO", "PEP", "MCD", "SBUX", "NKE",
    "LULU", "UBER", "LYFT", "ABNB", "BKNG", "DASH", "SNOW", "CRM",
    "ORCL", "IBM", "CSCO", "ADBE", "NOW", "INTU", "PANW", "CRWD", "FTNT",
    "NET", "DDOG", "ZS", "MDB", "TEAM", "U", "RBLX", "RDDT",
    "SPOT", "SNAP", "PINS", "DJT", "GME", "AMC", "CVNA", "RIVN", "LCID",
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNH", "HKD", "SGD",
    "TRY", "MXN", "ZAR", "SEK", "NOK", "PLN", "INR", "BRL", "KRW",
    "USDJPY", "USDEUR", "USDGBP", "USDCNH", "USDKRW", "USDMXN", "USDTRY", "USDZAR"
}


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def strip_multiplier_prefix(base_coin):
    clean = base_coin.upper()
    for prefix in ["1000000", "100000", "10000", "1000", "100"]:
        if clean.startswith(prefix) and len(clean) > len(prefix):
            return clean[len(prefix):]
    return clean


def strip_tokenized_suffix(base_coin):
    clean = base_coin.upper()
    if clean.endswith("STOCK") and len(clean) > 5:
        return clean[:-5]
    if len(clean) >= 4 and clean.endswith("X") and clean[:-1] in TRADFI_BLACKLIST:
        return clean[:-1]
    return clean


def is_tradfi_ticker(base_coin):
    b1 = base_coin.upper()
    if b1.endswith("STOCK"):
        return True
    b2 = strip_multiplier_prefix(b1)
    b3 = strip_tokenized_suffix(b2)
    if (b1 in TRADFI_BLACKLIST) or (b2 in TRADFI_BLACKLIST) or (b3 in TRADFI_BLACKLIST):
        return True
    if len(b1) == 6 and (b1.startswith("USD") or b1.endswith("USD")):
        return True
    return False


def fmt_pct(val):
    v = round(to_float(val), 1)
    return f"{int(v)}" if v == int(v) else f"{v:.1f}"


def send_telegram_report(text):
    print(text)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[경고] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    chunks = []
    current = ""
    for line in text.split("\n"):
        if len(current) + len(line) + 1 > 3800:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=12)
            if r.status_code != 200:
                print(f"[Telegram 응답 오류] {r.status_code}: {r.text}")
            time.sleep(0.4)
        except Exception as e:
            print(f"[Telegram 전송 예외] {e}")


def get_fast_pure_crypto_whitelist():
    """5대 거래소 직통 API로 순수 코인 화이트리스트 및 RWA/주식 차단 목록 확보"""
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


def fetch_bybit_fast(path, params=None):
    query_str = f"?{urlencode(params)}" if params else ""
    target_bybit = f"https://api.bybit.com{path}{query_str}"
    encoded_bybit = quote(target_bybit, safe="")

    routes = [
        f"https://api.codetabs.com/v1/proxy?quest={encoded_bybit}",
        f"https://api.allorigins.win/raw?url={encoded_bybit}",
    ]
    for url in routes:
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=4.5)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, dict) and data.get("retCode") == 0:
                    return data.get("result", {})
        except Exception:
            pass
    return {}


def fetch_klines_for_week(symbol, interval="15m", ref_price=0.0, limit=750):
    """
    [정밀도 개선]
    Gate.io / Binance Vision / Bybit V5는 한 번의 요청으로 최대 750~1000개(15분봉 750개 = 7.8일치)를
    반환하므로, 일주일 내내 왜곡 없는 15분봉 정밀 캔들로 익절/손절/보유를 추적합니다.
    """
    base_coin = symbol[:-4].upper() if symbol.endswith("USDT") else symbol.upper()
    clean_base = strip_multiplier_prefix(base_coin)
    rows = []

    bg_gran = "1H" if interval == "1h" else "15m"
    gt_inter = "1h" if interval == "1h" else "15m"
    bn_inter = "1h" if interval == "1h" else "15m"
    by_inter = "60" if interval == "1h" else "15"

    # 1순위: Gate.io USDT Futures (최대 750개 15분봉 = 7.8일치 한 번에 수집)
    for base_try in [base_coin, clean_base]:
        try:
            r = requests.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": f"{base_try}_USDT", "interval": gt_inter, "limit": str(min(limit, 800))},
                headers=BROWSER_HEADERS,
                timeout=4
            )
            if r.status_code == 200:
                gt_list = r.json()
                if isinstance(gt_list, list) and len(gt_list) >= 15:
                    rows = [
                        [int(float(i.get("t", 0))) * 1000, i.get("o", 0), i.get("h", 0), i.get("l", 0), i.get("c", 0), float(i.get("v", 0)) or 1.0, 0]
                        for i in gt_list
                    ]
                    break
        except Exception:
            pass

    # 2순위: Binance Vision Public Data (최대 750개 15분봉 한 번에 수집)
    if not rows:
        for sym_try in [symbol, f"{clean_base}USDT"]:
            try:
                r = requests.get(
                    "https://data-api.binance.vision/api/v3/klines",
                    params={"symbol": sym_try, "interval": bn_inter, "limit": str(min(limit, 800))},
                    headers=BROWSER_HEADERS,
                    timeout=4
                )
                if r.status_code == 200:
                    bn_list = r.json()
                    if isinstance(bn_list, list) and len(bn_list) >= 15:
                        rows = [[x[0], x[1], x[2], x[3], x[4], x[5], x[7]] for x in bn_list]
                        break
            except Exception:
                pass

    # 3순위: Bitget USDT-FUTURES
    if not rows:
        for sym_try in [symbol, f"{clean_base}USDT"]:
            try:
                r = requests.get(
                    "https://api.bitget.com/api/v2/mix/market/candles",
                    params={"symbol": sym_try, "productType": "USDT-FUTURES", "granularity": bg_gran, "limit": "200"},
                    headers=BROWSER_HEADERS,
                    timeout=4
                )
                if r.status_code == 200:
                    bg_list = r.json().get("data") or []
                    if isinstance(bg_list, list) and len(bg_list) >= 15:
                        rows = [x[:7] for x in bg_list]
                        break
            except Exception:
                pass

    # 4순위: Bybit V5 API 프록시 (최대 750개)
    if not rows:
        res = fetch_bybit_fast("/v5/market/kline", {
            "category": "linear",
            "symbol": symbol,
            "interval": by_inter,
            "limit": min(limit, 800)
        })
        bybit_list = res.get("list") or []
        if isinstance(bybit_list, list) and len(bybit_list) >= 15:
            rows = bybit_list

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    if df["timestamp"].iloc[-1] < 10_000_000_000:
        df["timestamp"] = df["timestamp"] * 1000
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # 단위 배수(1000PEPE 등)만 보정하고, 인위적인 첫 캔들 스케일링 왜곡은 절대 하지 않음!
    last_close = to_float(df["close"].iloc[-1])
    if ref_price > 0 and last_close > 0:
        ratio = ref_price / last_close
        for mult in [100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 0.001, 0.0001]:
            if 0.70 <= (ratio / mult) <= 1.40:
                for col in ["open", "high", "low", "close"]:
                    df[col] = df[col] * mult
                break

    return df


def extract_signal_episodes_by_symbol(start_ts, end_ts, pure_crypto_bases, rwa_blocked_bases):
    """
    [버그 수정 1: 과거 손절 플래그(sl_alerted)가 최신 재진입 신호를 오염시키던 현상 완벽 해결]
    동일 코인이 11/06에 손절된 뒤 11/08에 새로 진입한 경우, 두 신호를 별개의 에피소드로 분리하여
    11/06의 손절 기록이 11/08의 보유/익절 신호를 덮어쓰지 않도록 보호합니다.
    """
    snapshots = []

    try:
        out = subprocess.check_output(
            ["git", "log", "--since=14 days ago", "--reverse", "--format=%H", "--", STATE_FILE],
            stderr=subprocess.DEVNULL,
            text=True
        )
        commit_hashes = [h.strip() for h in out.splitlines() if h.strip()]
        for chash in commit_hashes:
            try:
                raw_json = subprocess.check_output(
                    ["git", "show", f"{chash}:{STATE_FILE}"],
                    stderr=subprocess.DEVNULL,
                    text=True
                )
                data = json.loads(raw_json)
                if isinstance(data, dict):
                    snapshots.append(data)
            except Exception:
                pass
    except Exception:
        pass

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    snapshots.append(data)
        except Exception:
            pass

    episodes_by_key = {}
    for snap in snapshots:
        tracked = snap.get("tracked", {})
        if not isinstance(tracked, dict):
            continue
        for sym, info in tracked.items():
            if not isinstance(info, dict):
                continue
            sym = str(sym).upper().strip()
            if not sym.endswith("USDT"):
                continue

            base_coin = sym[:-4]
            clean_base = strip_multiplier_prefix(base_coin)

            # 주식/TradFi 원천 제거
            if is_tradfi_ticker(base_coin) or base_coin in rwa_blocked_bases or clean_base in rwa_blocked_bases:
                continue
            if pure_crypto_bases and (base_coin not in pure_crypto_bases and clean_base not in pure_crypto_bases):
                continue

            created_at = int(info.get("created_at", 0))
            if not (start_ts <= created_at <= end_ts):
                continue

            # 같은 종목이라도 created_at이 2시간(7200초) 이내인 경우만 같은 에피소드로 묶음
            matched_key = None
            for k, existing in episodes_by_key.items():
                if existing["symbol"] == sym and abs(existing["created_at"] - created_at) < 7200:
                    matched_key = k
                    break

            raw_tps = [to_float(x) for x in (info.get("tps") or [])][:3]
            raw_tp_pcts = [to_float(x) for x in (info.get("tp_pcts") or [])][:3]

            if matched_key is None:
                key = f"{sym}_{created_at}"
                episodes_by_key[key] = {
                    "symbol": sym,
                    "created_at": created_at,
                    "track_type": info.get("track_type", "TRACK_A"),
                    "entry_low": to_float(info.get("entry_low")),
                    "entry_high": to_float(info.get("entry_high")),
                    "stop_price": to_float(info.get("stop_price")),
                    "sl_pct": abs(to_float(info.get("sl_pct"), default=3.0)),
                    "tps": raw_tps,
                    "tp_pcts": raw_tp_pcts,
                    "sl_alerted": bool(info.get("sl_alerted", False)),
                }
            else:
                # 해당 에피소드 내에서만 sl_alerted 상태 업데이트 (다른 날짜 에피소드 오염 방지!)
                if info.get("sl_alerted"):
                    episodes_by_key[matched_key]["sl_alerted"] = True

    grouped_by_sym = {}
    for ep in sorted(episodes_by_key.values(), key=lambda x: x["created_at"]):
        grouped_by_sym.setdefault(ep["symbol"], []).append(ep)

    return grouped_by_sym


def evaluate_symbol_episodes_to_single_result(sym_episodes, end_ts):
    """
    [버그 수정 2 & 3]
    1) 인위적인 fine_scale 가격 왜곡 제거
    2) 손절(SL) 체크는 반드시 알림 발생 시각 '이후'에 열린 캔들(c_ts >= created_ms)에서만 수행
    3) 한 종목에 여러 에피소드가 있을 경우, 리포트에는 종목명이 딱 1번만 나오도록
       [익절 성공(WIN) > 현재 보유중(HOLDING) > 손절(LOSS)] 우선순위로 대표 결과를 선정합니다.
    """
    if not sym_episodes:
        return None

    sym = sym_episodes[0]["symbol"]
    ref_entry = (sym_episodes[-1]["entry_low"] + sym_episodes[-1]["entry_high"]) / 2.0
    if ref_entry <= 0:
        ref_entry = max(sym_episodes[-1]["entry_low"], sym_episodes[-1]["entry_high"])

    # 15분봉 750개(7.8일치) + 1시간봉 200개(8.3일치) 결합
    df_15m = fetch_klines_for_week(sym, interval="15m", ref_price=ref_entry, limit=750)
    df_1h = fetch_klines_for_week(sym, interval="1h", ref_price=ref_entry, limit=200)

    frames = [d for d in (df_15m, df_1h) if not d.empty]
    if frames:
        df_all = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    else:
        df_all = pd.DataFrame()

    end_ms = end_ts * 1000
    ep_results = []

    for sig in sym_episodes:
        created_ms = sig["created_at"] * 1000
        entry_low = sig["entry_low"]
        entry_high = sig["entry_high"]
        avg_entry = (entry_low + entry_high) / 2.0 if (entry_low > 0 and entry_high > 0) else max(entry_low, entry_high)
        stop_price = sig["stop_price"]
        sl_pct = sig["sl_pct"] if sig["sl_pct"] > 0 else 3.0
        tps = [to_float(x) for x in sig["tps"][:3] if to_float(x) > 0]
        tp_pcts = [round(to_float(p), 1) for p in sig["tp_pcts"][:3] if to_float(p) > 0]

        if not tp_pcts:
            tp_pcts = [round(sl_pct * 2.0, 1), round(sl_pct * 3.5, 1), round(sl_pct * 5.0, 1)]
        while len(tp_pcts) < 3:
            tp_pcts.append(round(tp_pcts[-1] * 1.4, 1))
        if not tps and avg_entry > 0:
            tps = [avg_entry * (1.0 + p / 100.0) for p in tp_pcts]

        if not df_all.empty and avg_entry > 0:
            # 집계 기간 내 캔들 (신호 발생 직전 15분 ~ 주간 마감 시각)
            post_df = df_all[(df_all["timestamp"] >= created_ms - 900_000) & (df_all["timestamp"] <= end_ms + 3_600_000)].copy()
            # 알림 이후 현재까지 전체 최고가 계산용
            full_post_df = df_all[df_all["timestamp"] >= created_ms - 900_000].copy()
        else:
            post_df = pd.DataFrame()
            full_post_df = pd.DataFrame()

        if post_df.empty or avg_entry <= 0:
            if sig["sl_alerted"]:
                ep_results.append({
                    "symbol": sym,
                    "status": "LOSS",
                    "pct": -round(sl_pct, 1),
                    "max_surge_pct": 0.0,
                    "tp_stage": 0,
                    "note": "손절선 이탈",
                    "created_at": sig["created_at"]
                })
            else:
                ep_results.append({
                    "symbol": sym,
                    "status": "HOLDING",
                    "pct": 0.0,
                    "max_surge_pct": 0.0,
                    "tp_stage": 0,
                    "note": "보유중·1차 미달",
                    "created_at": sig["created_at"]
                })
            continue

        # 알림 이후 최고점 기준 최고수익률(%) 산출
        overall_max_high = max(to_float(full_post_df["high"].max(), default=avg_entry), avg_entry)
        overall_max_surge_pct = max(0.0, round(((overall_max_high - avg_entry) / avg_entry) * 100.0, 1))

        # 현재가 수익률 산출
        current_price = to_float(df_all["close"].iloc[-1], default=avg_entry)
        curr_pnl_pct = round(((current_price - avg_entry) / avg_entry) * 100.0, 1)

        highest_tp_idx = -1
        sl_hit_first = False

        for _, row in post_df.iterrows():
            c_ts = int(row["timestamp"])
            c_high = to_float(row["high"], default=avg_entry)
            c_low = to_float(row["low"], default=avg_entry)

            # [핵심] 손절(SL) 체크는 반드시 알림 발생 시각(created_ms) 이후에 시작된 캔들에서만 검사!
            # (알림 울리기 전의 스탑헌팅 아랫꼬리 때문에 억울하게 손절 판정나는 버그 차단)
            if c_ts >= created_ms and highest_tp_idx == -1 and stop_price > 0 and c_low <= stop_price:
                if not (tps and c_high >= tps[0] and to_float(row["close"]) >= to_float(row["open"])):
                    sl_hit_first = True
                    break

            # 1차 -> 2차 -> 3차 익절 도달 확인
            for idx in range(len(tps)):
                if tps[idx] > 0 and c_high >= tps[idx]:
                    if idx > highest_tp_idx:
                        highest_tp_idx = idx

        # 1) 🟢 익절 달성 (1차 TP 이상 도달)
        if highest_tp_idx >= 0 and highest_tp_idx < len(tp_pcts):
            tp_stage = highest_tp_idx + 1
            achieved_pct = round(to_float(tp_pcts[highest_tp_idx]), 1)
            true_max_surge = max(overall_max_surge_pct, achieved_pct)

            if tp_stage >= 3:
                stage_note = "🎯 3차 최종목표 달성!"
            elif tp_stage == 2:
                stage_note = "2차 익절 달성 (3차 미달)"
            else:
                stage_note = "1차 익절 달성 (3차 미달)"

            ep_results.append({
                "symbol": sym,
                "status": "WIN",
                "pct": achieved_pct,
                "max_surge_pct": round(true_max_surge, 1),
                "tp_stage": tp_stage,
                "note": stage_note,
                "created_at": sig["created_at"]
            })

        # 2) 🔴 손절 이탈 (1차 익절 전 손절선 도달)
        elif sl_hit_first or sig["sl_alerted"]:
            ep_results.append({
                "symbol": sym,
                "status": "LOSS",
                "pct": -round(sl_pct, 1),
                "max_surge_pct": overall_max_surge_pct,
                "tp_stage": 0,
                "note": "손절선 이탈",
                "created_at": sig["created_at"]
            })

        # 3) 🟡 보유/진행 중 종목 (1차 익절도 손절도 안 닿은 종목 -> 현재가 기준)
        else:
            ep_results.append({
                "symbol": sym,
                "status": "HOLDING",
                "pct": curr_pnl_pct,
                "max_surge_pct": overall_max_surge_pct,
                "tp_stage": 0,
                "note": "보유중·1차 미달",
                "created_at": sig["created_at"]
            })

    # [종목당 1개 대표 결과 선정]:
    # 1순위: 익절 성공(WIN) 에피소드 중 가장 높은 단계
    # 2순위: 현재 보유/진행 중(HOLDING)인 최신 에피소드 (과거 손절 기록이 최신 보유 종목을 가리지 않도록!)
    # 3순위: 손절(LOSS) 에피소드
    wins = [r for r in ep_results if r["status"] == "WIN"]
    if wins:
        wins.sort(key=lambda x: (x["tp_stage"], x["pct"], x["max_surge_pct"]), reverse=True)
        return wins[0]

    holds = [r for r in ep_results if r["status"] == "HOLDING"]
    if holds:
        holds.sort(key=lambda x: x["created_at"], reverse=True)
        return holds[0]

    losses = [r for r in ep_results if r["status"] == "LOSS"]
    losses.sort(key=lambda x: x["created_at"], reverse=True)
    return losses[0]


def get_target_week_window(now_dt):
    days_since_monday = now_dt.weekday()
    this_monday_start = (now_dt - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    last_monday_start = this_monday_start - timedelta(days=7)
    last_sunday_end = this_monday_start - timedelta(seconds=1)
    return last_monday_start, last_sunday_end, this_monday_start


def main():
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

    pure_crypto_bases, rwa_blocked_bases = get_fast_pure_crypto_whitelist()

    last_mon, last_sun, this_mon = get_target_week_window(now_dt)
    start_ts = int(last_mon.timestamp())
    end_ts = int(last_sun.timestamp())
    period_label = f"{last_mon.strftime('%Y-%m-%d(월) 00:00')} ~ {last_sun.strftime('%Y-%m-%d(일) 23:59')} KST"

    grouped_episodes = extract_signal_episodes_by_symbol(start_ts, end_ts, pure_crypto_bases, rwa_blocked_bases)

    if not grouped_episodes:
        alt_start_dt = this_mon if (now_dt - this_mon).total_seconds() >= 3600 else (now_dt - timedelta(days=7))
        start_ts = int(alt_start_dt.timestamp())
        end_ts = int(now_dt.timestamp())
        period_label = f"{alt_start_dt.strftime('%Y-%m-%d %H:%M')} ~ {now_dt.strftime('%Y-%m-%d %H:%M')} KST (실시간 누적)"
        grouped_episodes = extract_signal_episodes_by_symbol(start_ts, end_ts, pure_crypto_bases, rwa_blocked_bases)

    if not grouped_episodes:
        send_telegram_report(
            f"📅 [Bybit 순수 코인 주간 성과 결산 리포트]\n"
            f"🕒 집계 기간: {period_label}\n\n"
            f"• 해당 기간에 저장된 순수 코인 스캐너 알림 기록이 없습니다."
        )
        return

    all_results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        f_map = {
            executor.submit(evaluate_symbol_episodes_to_single_result, eps, end_ts): sym
            for sym, eps in grouped_episodes.items()
        }
        for future in as_completed(f_map):
            try:
                res = future.result()
                if res:
                    all_results.append(res)
            except Exception as e:
                print(f"[종목 평가 예외] {e}")

    win_list = [r for r in all_results if r["status"] == "WIN"]
    loss_list = [r for r in all_results if r["status"] == "LOSS"]
    hold_list = [r for r in all_results if r["status"] == "HOLDING"]

    win_list.sort(key=lambda x: (x["tp_stage"], x["pct"], x["max_surge_pct"]), reverse=True)
    loss_list.sort(key=lambda x: x["pct"])
    hold_list.sort(key=lambda x: x["pct"], reverse=True)

    total_cnt = len(all_results)
    win_cnt = len(win_list)
    loss_cnt = len(loss_list)
    hold_cnt = len(hold_list)

    tp3_cnt = sum(1 for r in win_list if r["tp_stage"] >= 3)
    tp2_cnt = sum(1 for r in win_list if r["tp_stage"] == 2)
    tp1_cnt = sum(1 for r in win_list if r["tp_stage"] == 1)

    closed_cnt = win_cnt + loss_cnt
    closed_win_rate = (win_cnt / closed_cnt * 100.0) if closed_cnt > 0 else 0.0

    realized_profit_pct = round(sum(r["pct"] for r in win_list), 1)
    realized_loss_pct = round(sum(abs(r["pct"]) for r in loss_list), 1)
    realized_net_pct = round(realized_profit_pct - realized_loss_pct, 1)

    hold_profit_pct = round(sum(r["pct"] for r in hold_list if r["pct"] >= 0), 1)
    hold_loss_pct = round(sum(abs(r["pct"]) for r in hold_list if r["pct"] < 0), 1)

    total_profit_all = round(realized_profit_pct + hold_profit_pct, 1)
    total_loss_all = round(realized_loss_pct + hold_loss_pct, 1)
    total_net_all = round(total_profit_all - total_loss_all, 1)

    # =====================================================================
    # 리포트 메시지 조립 (종목명 1회 표기 + 익절 종목만 알림후 최고수익률 표기)
    # =====================================================================
    lines = [
        "📅 [Bybit 스캐너 순수 코인 주간 성과 결산]",
        f"🕒 집계 기간: {period_label}",
        f"🕒 발송 시각: {now_str}\n",
        f"🟢 [1. 익절 달성 종목 (최종목표: 3차 익절)] ({win_cnt}종목 | 3차달성 {tp3_cnt}·2차 {tp2_cnt}·1차 {tp1_cnt})"
    ]

    if win_list:
        for w in win_list:
            short_sym = w["symbol"].replace("USDT", "")
            dt_str = datetime.fromtimestamp(w["created_at"], tz=KST).strftime("%m/%d")
            lines.append(
                f"• ${short_sym} ({dt_str}): +{fmt_pct(w['pct'])}% "
                f"({w['note']} | 알림후 최고 +{fmt_pct(w['max_surge_pct'])}%)"
            )
    else:
        lines.append("• 익절 달성 종목 없음")

    lines.append(f"\n🔴 [2. 손절 이탈 종목] ({loss_cnt}종목)")
    if loss_list:
        for l_item in loss_list:
            short_sym = l_item["symbol"].replace("USDT", "")
            dt_str = datetime.fromtimestamp(l_item["created_at"], tz=KST).strftime("%m/%d")
            lines.append(
                f"• ${short_sym} ({dt_str}): -{fmt_pct(abs(l_item['pct']))}% ({l_item['note']})"
            )
    else:
        lines.append("• 손절 이탈 종목 없음")

    lines.append(f"\n🟡 [3. 보유/진행 중 종목 (1차 익절·손절 미도달 | 현재가 기준)] ({hold_cnt}종목)")
    if hold_list:
        for h_item in hold_list:
            short_sym = h_item["symbol"].replace("USDT", "")
            dt_str = datetime.fromtimestamp(h_item["created_at"], tz=KST).strftime("%m/%d")
            sign = "+" if h_item["pct"] >= 0 else "-"
            lines.append(
                f"• ${short_sym} ({dt_str}): 현재 {sign}{fmt_pct(abs(h_item['pct']))}% ({h_item['note']})"
            )
    else:
        lines.append("• 현재 보유/진행 중인 종목 없음")

    r_sign = "+" if realized_net_pct >= 0 else ""
    t_sign = "+" if total_net_all >= 0 else ""

    lines.append(
        "\n━━━━━━━━━━━━━━━━━━\n"
        "📊 [4. 일주일 합산 최종 성적표 (순수 코인·단일 종목 기준)]\n"
        f"• 총 신호 종목 수: {total_cnt}개 (익절 {win_cnt} / 손절 {loss_cnt} / 보유중 {hold_cnt})\n"
        f"• 3차 최종목표 달성: {tp3_cnt}개 | 2차 달성: {tp2_cnt}개 | 1차 달성: {tp1_cnt}개\n"
        f"• 확정 종목 성공률(승률): {closed_win_rate:.1f}% ({win_cnt}승 {loss_cnt}패)\n"
        f"• [확정 손익 계산(익절-손절)]: +{fmt_pct(realized_profit_pct)}% - {fmt_pct(realized_loss_pct)}% = {r_sign}{fmt_pct(realized_net_pct)}%\n"
        f"• [보유 종목 현재가 합산 시]: +{fmt_pct(total_profit_all)}% - {fmt_pct(total_loss_all)}% = {t_sign}{fmt_pct(total_net_all)}%"
    )

    report_text = "\n".join(lines)
    send_telegram_report(report_text)


if __name__ == "__main__":
    main()
