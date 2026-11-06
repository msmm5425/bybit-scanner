import os
import re
import datetime
import requests
import pandas as pd
import numpy as np

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SKIP_PREFIXES = ("1000000", "100000", "10000", "1000")
ENGLISH_RAW_RE = re.compile(r"^[A-Z0-9]+USDT$")

# 미국 GitHub Actions 서버 IP 차단을 우회하는 바이비트 공식 글로벌 도메인 목록
BYBIT_HOSTS = [
    "https://api.bytick.com",
    "https://api.bybit.com"
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
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
            print(f"텔레그램 전송 오류: {e}")


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


def fetch_top5_via_ccxt():
    """1차: CCXT 공식 라이브러리 + bytick.com 우회 호스트로 선물 틱커 수집"""
    import ccxt
    for host in ["bytick.com", "bybit.com"]:
        try:
            ex = ccxt.bybit({
                "hostname": host,
                "enableRateLimit": True,
                "options": {"defaultType": "linear"}
            })
            tickers = ex.fetch_tickers(params={"category": "linear"})
            valid = []
            for sym, t in tickers.items():
                info = t.get("info", {})
                raw_sym = str(info.get("symbol", "")).strip()
                if not raw_sym and ":" in sym:
                    raw_sym = sym.split(":")[0].replace("/", "")

                if not ENGLISH_RAW_RE.match(raw_sym):
                    continue
                if any(raw_sym.startswith(p) for p in SKIP_PREFIXES):
                    continue

                last_p = to_float(info.get("lastPrice") or t.get("last"))
                if last_p <= 0:
                    continue

                raw_pcnt = info.get("price24hPcnt")
                if raw_pcnt not in (None, ""):
                    chg = to_float(raw_pcnt) * 100.0
                else:
                    chg = to_float(t.get("percentage"))

                turnover = to_float(info.get("turnover24h") or t.get("quoteVolume"))
                funding = to_float(info.get("fundingRate")) * 100.0
                oi_val = to_float(info.get("openInterestValue"))

                valid.append({
                    "symbol": raw_sym,
                    "ccxt_sym": sym,
                    "ticker": raw_sym[:-4],
                    "price": last_p,
                    "chg": chg,
                    "turnover": turnover,
                    "funding": funding,
                    "oi_val": oi_val
                })

            if valid:
                valid.sort(key=lambda k: k["chg"], reverse=True)
                return valid[:5], ex, None
        except Exception as e:
            print(f"CCXT ({host}) 오류: {e}")
    return [], None, "CCXT 연결 실패"


def fetch_top5_via_rest():
    """2차 폴백: CCXT 실패 시 바이비트 V5 REST API 직접 우회 호출"""
    errors = []
    for base in BYBIT_HOSTS:
        try:
            url = f"{base}/v5/market/tickers"
            res = requests.get(
                url,
                params={"category": "linear"},
                headers=HEADERS,
                timeout=12
            )
            if res.status_code != 200:
                errors.append(f"{base}: HTTP {res.status_code}")
                continue
            data = res.json()
            lst = data.get("result", {}).get("list", [])
            if not lst:
                errors.append(f"{base}: empty list")
                continue

            valid = []
            for x in lst:
                raw_sym = str(x.get("symbol", "")).strip()
                if not ENGLISH_RAW_RE.match(raw_sym):
                    continue
                if any(raw_sym.startswith(p) for p in SKIP_PREFIXES):
                    continue

                last_p = to_float(x.get("lastPrice"))
                if last_p <= 0:
                    continue

                chg = to_float(x.get("price24hPcnt")) * 100.0
                turnover = to_float(x.get("turnover24h"))
                funding = to_float(x.get("fundingRate")) * 100.0
                oi_val = to_float(x.get("openInterestValue"))

                valid.append({
                    "symbol": raw_sym,
                    "ccxt_sym": None,
                    "ticker": raw_sym[:-4],
                    "price": last_p,
                    "chg": chg,
                    "turnover": turnover,
                    "funding": funding,
                    "oi_val": oi_val
                })

            if valid:
                valid.sort(key=lambda k: k["chg"], reverse=True)
                return valid[:5], " | ".join(errors)
        except Exception as e:
            errors.append(f"{base}: {type(e).__name__}")
    return [], " | ".join(errors)


def fetch_klines(item, ex=None, limit=350):
    """5분봉 350개 수집 (CCXT 우선 시도 후 REST API 폴백)"""
    raw_sym = item["symbol"]
    ccxt_sym = item.get("ccxt_sym")

    if ex is not None and ccxt_sym:
        try:
            ohlcv = ex.fetch_ohlcv(
                ccxt_sym,
                timeframe="5m",
                limit=limit,
                params={"category": "linear"}
            )
            if ohlcv and len(ohlcv) >= 30:
                df = pd.DataFrame(
                    ohlcv,
                    columns=["ts", "open", "high", "low", "close", "volume"]
                )
                for col in ["open", "high", "low", "close", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
                return df.dropna().reset_index(drop=True)
        except Exception as e:
            print(f"CCXT kline 폴백 전환 ({raw_sym}): {e}")

    for base in BYBIT_HOSTS:
        try:
            url = f"{base}/v5/market/kline"
            res = requests.get(
                url,
                params={
                    "category": "linear",
                    "symbol": raw_sym,
                    "interval": "5",
                    "limit": str(limit)
                },
                headers=HEADERS,
                timeout=12
            )
            if res.status_code != 200:
                continue
            rows = res.json().get("result", {}).get("list", [])
            if not rows:
                continue
            df = pd.DataFrame(
                rows,
                columns=["ts", "open", "high", "low", "close", "volume", "turnover"]
            )
            df = df.iloc[::-1].reset_index(drop=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            return df.dropna(subset=["ts", "open", "high", "low", "close", "volume"]).reset_index(drop=True)
        except Exception:
            continue

    return pd.DataFrame()


def analyze_pre_surge(item, ex=None):
    df = fetch_klines(item, ex=ex, limit=350)
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
        "symbol": item["symbol"],
        "ticker": item["ticker"],
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

    top5, ex, err_ccxt = fetch_top5_via_ccxt()
    err_rest = ""
    if not top5:
        top5, err_rest = fetch_top5_via_rest()

    if not top5:
        send_telegram(
            f"⚠️ [바이비트 상승률 Top 5 수집 실패]\n"
            f"• 시각: {now_str}\n"
            f"• 진단 로그: {err_ccxt} / {err_rest}"
        )
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item, ex=ex)
            if r:
                results.append(r)
        except Exception as e:
            print(f"{item['symbol']} 분석 오류: {e}")

    if not results:
        send_telegram("⚠️ 캔들 데이터 분석 결과가 없습니다.")
        return

    lines = [
        f"📊 [바이비트 상승률 Top 5 급등 전 초동전략 해부 리포트]\n"
        f"({now_str} 기준)\n"
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
        f"→ 스캐너 거래량 기준점 {max(3.0, min(5.0, avg_vol * 0.7)):.1f}배 설정 적합"
    )
    lines.append(
        f"4) 최적 눌림목 타점: {total}종목 중 {ma20_cnt}종목이 첫 돌파 후 "
        f"5분봉 MA 20선 지지를 받고 2차 급등 분출"
    )

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
