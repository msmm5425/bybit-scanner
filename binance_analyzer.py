import os
import re
import time
import urllib.parse
import datetime
import requests
import pandas as pd
import numpy as np

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SKIP_PREFIXES = ("1000000", "100000", "10000", "1000")
ENGLISH_SYM_RE = re.compile(r"^[A-Z0-9]+USDT$")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
    "Accept": "application/json"
}


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("텔레그램 설정 누락")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        try:
            requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i + 3500]},
                timeout=15
            )
        except Exception as e:
            print(f"텔레그램 오류: {e}")


def to_float(val, default=0.0):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default


def fmt_price(val, ref):
    if ref >= 1000:
        return f"{val:.2f}"
    elif ref >= 10:
        return f"{val:.3f}"
    elif ref >= 1:
        return f"{val:.4f}"
    elif ref >= 0.01:
        return f"{val:.5f}"
    else:
        return f"{val:.6f}"


def fetch_bybit_via_relay(target_url):
    """유럽/글로벌 비미국 프록시 릴레이를 경유하여 미국 IP 403 차단을 우회합니다."""
    enc = urllib.parse.quote(target_url, safe="")
    relays = [
        f"https://api.codetabs.com/v1/proxy?quest={target_url}",
        f"https://api.allorigins.win/raw?url={enc}",
        f"https://corsproxy.io/?{enc}"
    ]
    for r_url in relays:
        try:
            res = requests.get(r_url, headers=HEADERS, timeout=15)
            if res.status_code == 200:
                data = res.json()
                if isinstance(data, dict) and data.get("retCode") == 0:
                    return data
        except Exception:
            continue
    return None


def parse_bybit_tickers(lst):
    valid = []
    for x in lst:
        sym = str(x.get("symbol", "")).strip()
        if not ENGLISH_SYM_RE.match(sym):
            continue
        if any(sym.startswith(p) for p in SKIP_PREFIXES):
            continue

        last_p = to_float(x.get("lastPrice"))
        if last_p <= 0:
            continue

        chg = to_float(x.get("price24hPcnt")) * 100.0
        turnover = to_float(x.get("turnover24h"))
        funding = to_float(x.get("fundingRate")) * 100.0
        oi_val = to_float(x.get("openInterestValue"))

        valid.append({
            "symbol": sym,
            "ticker": sym[:-4],
            "price": last_p,
            "chg": chg,
            "turnover": turnover,
            "funding": funding,
            "oi_val": oi_val
        })
    valid.sort(key=lambda k: k["chg"], reverse=True)
    return valid[:5]


def fetch_top5_from_coingecko_bybit():
    """릴레이 지연 시 미국 IP에서 100% 열리는 CoinGecko의 Bybit(Futures) 공식 집계 사용"""
    url = "https://api.coingecko.com/api/v3/derivatives"
    try:
        res = requests.get(url, headers=HEADERS, timeout=15)
        if res.status_code != 200:
            return []
        items = res.json()
        if not isinstance(items, list):
            return []

        valid = []
        seen = set()
        for x in items:
            mkt = str(x.get("market", "")).lower()
            if "bybit" not in mkt:
                continue
            if str(x.get("contract_type", "")).lower() != "perpetual":
                continue

            raw_sym = str(x.get("symbol", "")).upper().split(":")[0]
            clean_sym = re.sub(r"[^A-Z0-9]", "", raw_sym)
            if not ENGLISH_SYM_RE.match(clean_sym):
                continue
            if any(clean_sym.startswith(p) for p in SKIP_PREFIXES):
                continue
            if clean_sym in seen:
                continue

            last_p = to_float(x.get("price"))
            if last_p <= 0:
                continue

            seen.add(clean_sym)
            chg = to_float(x.get("price_percentage_change_24h"))
            turnover = to_float(x.get("volume_24h"))
            funding = to_float(x.get("funding_rate"))
            oi_val = to_float(x.get("open_interest"))

            valid.append({
                "symbol": clean_sym,
                "ticker": clean_sym[:-4],
                "price": last_p,
                "chg": chg,
                "turnover": turnover,
                "funding": funding,
                "oi_val": oi_val
            })

        valid.sort(key=lambda k: k["chg"], reverse=True)
        return valid[:5]
    except Exception as e:
        print(f"CoinGecko 백업 오류: {e}")
        return []


def get_top5_bybit_gainers():
    # 1차: 유럽 프록시 릴레이로 바이비트 V5 공식 티커 수집
    bybit_url = "https://api.bybit.com/v5/market/tickers?category=linear"
    data = fetch_bybit_via_relay(bybit_url)
    if data:
        lst = data.get("result", {}).get("list", [])
        top5 = parse_bybit_tickers(lst)
        if top5:
            return top5, "Bybit V5 (EU Relay)"

    # 2차: CoinGecko 공식 파생상품 API 내 Bybit (Futures) 실시간 데이터
    top5 = fetch_top5_from_coingecko_bybit()
    if top5:
        return top5, "Bybit Futures (CoinGecko Feed)"

    return [], "모든 우회 경로 응답 없음"


def normalize_df(df):
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df
    if df["ts"].iloc[-1] < 10_000_000_000:
        df["ts"] = df["ts"] * 1000.0
    return df.reset_index(drop=True)


def get_klines_5m(symbol, ticker):
    """5분봉 350개 수집 (1차: 바이비트 유럽 릴레이 -> 2차: Gate.io 선물 -> 3차: OKX 선물 -> 4차: CryptoCompare)"""
    # 1차: 바이비트 V5 공식 5분봉 (릴레이 경유)
    bybit_kline_url = (
        f"https://api.bybit.com/v5/market/kline"
        f"?category=linear&symbol={symbol}&interval=5&limit=350"
    )
    data = fetch_bybit_via_relay(bybit_kline_url)
    if data:
        rows = data.get("result", {}).get("list", [])
        if rows and len(rows) >= 40:
            df = pd.DataFrame(
                rows,
                columns=["ts", "open", "high", "low", "close", "volume", "turnover"]
            )
            df = df.iloc[::-1]
            df = normalize_df(df)
            if len(df) >= 40:
                return df

    # 2차: Gate.io USDT 무기한 선물 5분봉 (미국 IP 즉시 통과)
    try:
        g_url = (
            f"https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
            f"?contract={ticker}_USDT&interval=5m&limit=350"
        )
        res = requests.get(g_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            rows = res.json()
            if isinstance(rows, list) and len(rows) >= 40:
                parsed = []
                for r in rows:
                    parsed.append([
                        r.get("t"),
                        r.get("o"),
                        r.get("h"),
                        r.get("l"),
                        r.get("c"),
                        r.get("sum") or r.get("v")
                    ])
                df = pd.DataFrame(
                    parsed,
                    columns=["ts", "open", "high", "low", "close", "volume"]
                )
                df = normalize_df(df)
                if len(df) >= 40:
                    return df
    except Exception:
        pass

    # 3차: OKX USDT 무기한 선물 5분봉 (미국 IP 즉시 통과)
    try:
        o_url = (
            f"https://www.okx.com/api/v5/market/candles"
            f"?instId={ticker}-USDT-SWAP&bar=5m&limit=300"
        )
        res = requests.get(o_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            rows = res.json().get("data", [])
            if rows and len(rows) >= 40:
                parsed = [r[:6] for r in rows]
                df = pd.DataFrame(
                    parsed,
                    columns=["ts", "open", "high", "low", "close", "volume"]
                )
                df = df.iloc[::-1]
                df = normalize_df(df)
                if len(df) >= 40:
                    return df
    except Exception:
        pass

    # 4차: CryptoCompare 글로벌 5분봉 (미국 IP 즉시 통과)
    try:
        c_url = (
            f"https://min-api.cryptocompare.com/data/v2/histominute"
            f"?fsym={ticker}&tsym=USDT&limit=350&aggregate=5"
        )
        res = requests.get(c_url, headers=HEADERS, timeout=10)
        if res.status_code == 200:
            rows = res.json().get("Data", {}).get("Data", [])
            if rows and len(rows) >= 40:
                parsed = []
                for r in rows:
                    parsed.append([
                        r.get("time"),
                        r.get("open"),
                        r.get("high"),
                        r.get("low"),
                        r.get("close"),
                        r.get("volumeto")
                    ])
                df = pd.DataFrame(
                    parsed,
                    columns=["ts", "open", "high", "low", "close", "volume"]
                )
                df = normalize_df(df)
                if len(df) >= 40:
                    return df
    except Exception:
        pass

    return pd.DataFrame()


def analyze_pre_surge(item):
    sym = item["symbol"]
    ticker = item["ticker"]
    df = get_klines_5m(sym, ticker)
    if len(df) < 40:
        return None

    for m in [5, 20, 50, 120, 200, 300]:
        w = min(len(df), m)
        df[f"ma{m}"] = df["close"].rolling(w).mean()

    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma20"] = df["volume"].rolling(20).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()

    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100.0
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    search_end = max(25, len(df) - 2)
    search_start = max(15, len(df) - 288)
    search_zone = df.iloc[search_start:search_end].copy()
    if search_zone.empty:
        return None

    explosive = search_zone[(search_zone["ret"] >= 2.0) & (search_zone["vol_ratio"] >= 3.0)]
    if not explosive.empty:
        ig_idx = explosive.index[0]
    else:
        score = search_zone["ret"].clip(lower=0) * search_zone["vol_ratio"]
        ig_idx = score.idxmax() if score.notna().sum() > 0 else search_zone.index[-1]

    ig_candle = df.loc[ig_idx]
    pre_idx = max(0, ig_idx - 1)
    pre_candle = df.loc[pre_idx]

    kst = datetime.timezone(datetime.timedelta(hours=9))
    ig_time = datetime.datetime.fromtimestamp(
        ig_candle["ts"] / 1000.0, kst
    ).strftime("%m/%d %H:%M")

    ig_open = float(ig_candle["open"])
    post_high = float(df.loc[ig_idx:, "high"].max())
    surge_after_ig = ((post_high - ig_open) / (ig_open + 1e-9)) * 100.0

    ma_short_ok = bool(pre_candle["ma5"] > pre_candle["ma20"] > pre_candle["ma50"])
    ma_full_ok = bool(
        ma_short_ok and (pre_candle["ma50"] > pre_candle["ma120"] > pre_candle["ma200"])
    )

    box_start = max(0, pre_idx - 24)
    box_zone = df.loc[box_start:pre_idx]
    box_low = float(box_zone["low"].min())
    box_high = float(box_zone["high"].max())
    box_range_pct = ((box_high - box_low) / box_low * 100.0) if box_low > 0 else 0.0

    obv_ok = bool(
        pre_candle["obv"] > pre_candle["obv_ma10"]
        and pre_candle["obv"] > box_zone["obv"].iloc[0]
    )

    vol_mult = to_float(ig_candle["vol_ratio"], default=1.0)
    if np.isnan(vol_mult) or vol_mult <= 0:
        vol_mult = 1.0

    post_zone = df.loc[ig_idx + 1: min(len(df) - 1, ig_idx + 12)]
    pullback_str = "눌림목 없이 장대양봉 연속 직행"
    held_ma20 = False

    if not post_zone.empty:
        min_low = float(post_zone["low"].min())
        ma20_avg = float(post_zone["ma20"].mean())
        ma50_avg = float(post_zone["ma50"].mean())
        vol_dry = bool((post_zone["volume"] < post_zone["vma5"]).any())
        dry_tag = "거래량 건조 동반" if vol_dry else "수급 유지"

        if min_low >= ma20_avg * 0.992:
            pullback_str = f"5분봉 MA 20선 지지 리테스트 ({dry_tag})"
            held_ma20 = True
        elif min_low >= ma50_avg * 0.992:
            pullback_str = f"5분봉 MA 50선 눌림 후 반등 ({dry_tag})"
        else:
            pullback_str = f"발화 시작가 리테스트 후 반등 ({dry_tag})"

    return {
        "symbol": sym,
        "ticker": ticker,
        "price": item["price"],
        "chg": item["chg"],
        "turnover_m": item["turnover"] / 1_000_000.0,
        "ig_time": ig_time,
        "ig_price": ig_open,
        "surge_after_ig": surge_after_ig,
        "ma_short_ok": ma_short_ok,
        "ma_full_ok": ma_full_ok,
        "obv_ok": obv_ok,
        "box_range_pct": box_range_pct,
        "vol_mult": vol_mult,
        "pullback_str": pullback_str,
        "held_ma20": held_ma20,
        "funding": item["funding"],
        "oi_val_m": item["oi_val"] / 1_000_000.0
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5, src_label = get_top5_bybit_gainers()
    if not top5:
        send_telegram(f"⚠️ [바이비트 Top 5 수집 실패] ({now_str})\n사유: {src_label}")
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item)
            if r:
                results.append(r)
            time.sleep(0.3)
        except Exception as e:
            print(f"{item['symbol']} 분석 오류: {e}")

    if not results:
        send_telegram("⚠️ 캔들 데이터 분석 결과가 없습니다.")
        return

    lines = [
        f"📊 [바이비트 상승률 Top 5 급등 전 초동전략 역추적 리포트]\n"
        f"({now_str} | 수집경로: {src_label})\n"
    ]

    ma_short_cnt = 0
    ma_full_cnt = 0
    obv_cnt = 0
    ma20_cnt = 0
    vol_list = []

    for idx, r in enumerate(results, 1):
        if r["ma_short_ok"]:
            ma_short_cnt += 1
        if r["ma_full_ok"]:
            ma_full_cnt += 1
        if r["obv_ok"]:
            obv_cnt += 1
        if r["held_ma20"]:
            ma20_cnt += 1
        vol_list.append(r["vol_mult"])

        if r["ma_full_ok"]:
            ma_tag = "✅ 6개 이평 완전정배열"
        elif r["ma_short_ok"]:
            ma_tag = "✅ 단기 정배열(5>20>50)"
        else:
            ma_tag = "❌ 역배열 바닥돌파"

        obv_tag = "✅ 우상향 선행매집" if r["obv_ok"] else "➖ 중립/동행"
        p = r["price"]

        lines.append("━━━━━━━━━━━━━━")
        lines.append(
            f"{idx}️⃣ ${r['ticker']} ({r['symbol']} | "
            f"+{r['chg']:.1f}% | 현재가 {fmt_price(p, p)} USDT)"
        )
        lines.append(
            f"• 거래대금: {r['turnover_m']:,.1f}M | "
            f"미결제약정(OI): {r['oi_val_m']:,.1f}M | "
            f"펀딩비: {r['funding']:+.4f}%"
        )
        lines.append(
            f"• [초동 발화점]: {r['ig_time']} KST "
            f"(발화가 {fmt_price(r['ig_price'], p)} → 고점까지 +{r['surge_after_ig']:.1f}% 폭등)"
        )
        lines.append("• [발화 직전 컨펌된 초동 전략]:")
        lines.append(f"  - 5분봉 이평선: [{ma_tag}]")
        lines.append(f"  - 사전 매집: 직전 2시간 변동폭 {r['box_range_pct']:.1f}% 응축 | OBV [{obv_tag}]")
        lines.append(f"  - 초동 수급: 평소 대비 거래량 [{r['vol_mult']:.1f}배 폭발]")
        lines.append(f"  - 돌파 후 패턴: {r['pullback_str']}")

    total = len(results)
    avg_vol = sum(vol_list) / total

    lines.append("━━━━━━━━━━━━━━\n")
    lines.append("💡 [내 전략(scanner.py) 실시간 업데이트 피드백]")
    lines.append(
        f"1) 5분봉 단기정배열(5>20>50) 선행률: {total}종목 중 {ma_short_cnt}종목 "
        f"(완전정배열 {ma_full_cnt}종목)"
    )
    lines.append(f"2) 발화 전 박스권 OBV 매집 컨펌: {total}종목 중 {obv_cnt}종목")
    lines.append(
        f"3) 초동 발화 평균 거래량: 평소 대비 {avg_vol:.1f}배 폭발 "
        f"→ 스캐너 거래량 가중치 기준 {max(3.0, min(5.0, avg_vol * 0.7)):.1f}배 설정 적합"
    )
    lines.append(
        f"4) 최적 눌림목 진입 타점: {total}종목 중 {ma20_cnt}종목이 첫 돌파 후 "
        f"5분봉 MA 20선 지지를 받고 2차 급등 분출"
    )

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
