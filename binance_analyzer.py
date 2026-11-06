import os
import re
import time
import datetime
import requests
import pandas as pd
import numpy as np

# 바이비트 공식 메인 및 대체(우회) API 엔드포인트 3중 폴백 구성
BYBIT_ENDPOINTS = [
    "https://api.bytick.com",
    "https://api.bybit.com",
    "https://api.bybit.nl",
    "https://api.bybit.tr"
]

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# CloudFront WAF(방화벽) 봇 차단을 우회하기 위한 표준 브라우저 헤더
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive"
}

SKIP_PREFIXES = ("1000000", "100000", "10000", "1000")
ENGLISH_SYMBOL_RE = re.compile(r"^[A-Z0-9]+USDT$")


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("TELEGRAM_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        try:
            res = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i + 3500]},
                timeout=15
            )
            if res.status_code != 200:
                print(f"텔레그램 전송 실패: {res.status_code} {res.text}")
        except Exception as e:
            print(f"텔레그램 전송 오류: {e}")


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
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


def request_bybit_json(path, params):
    """3중 도메인 폴백을 통해 미국 IP 차단(403)을 우회하여 JSON을 안전하게 가져옵니다."""
    diag_errors = []
    for base_url in BYBIT_ENDPOINTS:
        url = f"{base_url}{path}"
        try:
            res = requests.get(url, params=params, headers=HEADERS, timeout=12)
            if res.status_code != 200:
                diag_errors.append(f"{base_url.split('//')[1]}: HTTP {res.status_code}")
                continue

            data = res.json()
            if data.get("retCode") == 0 and data.get("result"):
                return data, None
            diag_errors.append(
                f"{base_url.split('//')[1]}: retCode {data.get('retCode')} ({data.get('retMsg')})"
            )
        except Exception as e:
            diag_errors.append(f"{base_url.split('//')[1]}: {type(e).__name__}")
            time.sleep(0.3)

    return None, " | ".join(diag_errors)


def get_top5_bybit_gainers():
    data, err_msg = request_bybit_json("/v5/market/tickers", {"category": "linear"})
    if not data:
        return [], err_msg

    lst = data.get("result", {}).get("list", [])
    valid = []

    for x in lst:
        sym = str(x.get("symbol", "")).strip()

        # 순수 영문/숫자 USDT 무기한 선물만 필터링 (한자/특수 티커 및 배수 토큰 제외)
        if not ENGLISH_SYMBOL_RE.match(sym):
            continue
        if any(sym.startswith(prefix) for prefix in SKIP_PREFIXES):
            continue

        last_p = to_float(x.get("lastPrice"))
        chg = to_float(x.get("price24hPcnt")) * 100
        turnover = to_float(x.get("turnover24h"))
        funding = to_float(x.get("fundingRate")) * 100
        oi_val = to_float(x.get("openInterestValue"))

        if last_p <= 0:
            continue

        valid.append({
            "symbol": sym,
            "price": last_p,
            "chg": chg,
            "turnover": turnover,
            "funding": funding,
            "oi_val": oi_val
        })

    valid.sort(key=lambda k: k["chg"], reverse=True)
    return valid[:5], None


def get_bybit_klines(symbol, limit=360):
    data, _ = request_bybit_json(
        "/v5/market/kline",
        {"category": "linear", "symbol": symbol, "interval": "5", "limit": str(limit)}
    )
    if not data:
        return pd.DataFrame()

    rows = data.get("result", {}).get("list", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(
        rows,
        columns=["ts", "open", "high", "low", "close", "volume", "turnover"]
    )
    df = df.iloc[::-1].reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df = df.dropna(subset=["ts", "open", "high", "low", "close", "volume"])
    return df.reset_index(drop=True)


def analyze_pre_surge(item):
    sym = item["symbol"]
    df = get_bybit_klines(sym, limit=360)
    if len(df) < 60:
        return None

    # 5분봉 6개 이평선 및 거래량 이평선 계산
    for m in [5, 20, 50, 120, 200, 300]:
        df[f"ma{m}"] = df["close"].rolling(min(len(df), m)).mean()

    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma20"] = df["volume"].rolling(20).mean()

    # OBV 및 매집 추세 계산
    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()

    # 캔들 수익률 및 평소 대비 거래량 폭발 배수 계산
    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    # 최근 24시간(288개 5분봉) 내 초동 발화(Ignition) 캔들 역추적
    search_end = max(30, len(df) - 3)
    search_start = max(20, len(df) - 288)
    search_zone = df.iloc[search_start:search_end].copy()
    if search_zone.empty:
        return None

    # 양봉(+2% 이상)이면서 거래량이 터진(3배 이상) 첫 발화점 우선 탐색
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
        ig_candle["ts"] / 1000, kst
    ).strftime("%m/%d %H:%M")

    ig_open = float(ig_candle["open"])
    post_high = float(df.loc[ig_idx:, "high"].max())
    surge_after_ig = ((post_high - ig_open) / (ig_open + 1e-9)) * 100

    # 1. 발화 직전 5분봉 이평선 정배열 단계 판별
    ma_short_ok = bool(pre_candle["ma5"] > pre_candle["ma20"] > pre_candle["ma50"])
    ma_full_ok = bool(
        ma_short_ok
        and (pre_candle["ma50"] > pre_candle["ma120"] > pre_candle["ma200"])
    )

    # 2. 발화 직전 2시간(24개 봉) 박스권 응축률 및 OBV 선행 매집 여부
    box_start = max(0, pre_idx - 24)
    box_zone = df.loc[box_start:pre_idx]
    box_low = float(box_zone["low"].min())
    box_high = float(box_zone["high"].max())
    box_range_pct = ((box_high - box_low) / box_low * 100) if box_low > 0 else 0.0

    obv_ok = bool(
        pre_candle["obv"] > pre_candle["obv_ma10"]
        and pre_candle["obv"] > box_zone["obv"].iloc[0]
    )

    # 3. 발화 순간 거래량 폭발 배수
    vol_mult = to_float(ig_candle["vol_ratio"], default=1.0)
    if np.isnan(vol_mult) or vol_mult <= 0:
        vol_mult = 1.0

    # 4. 발화 직후 1시간(12개 봉) 내 눌림목 지지선(MA20 vs MA50) 및 거래량 건조 확인
    post_zone = df.loc[ig_idx + 1: min(len(df) - 1, ig_idx + 12)]
    pullback_str = "눌림목 없이 연속 장대양봉 직행"
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
            pullback_str = f"5분봉 MA 50선 깊게 눌림 후 반등 ({dry_tag})"
        else:
            pullback_str = f"발화 시작가 부근 리테스트 후 반등 ({dry_tag})"

    return {
        "symbol": sym,
        "ticker": sym.replace("USDT", ""),
        "price": item["price"],
        "chg": item["chg"],
        "turnover_m": item["turnover"] / 1_000_000,
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
        "oi_val_m": item["oi_val"] / 1_000_000
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5, err_detail = get_top5_bybit_gainers()
    if not top5:
        send_telegram(
            f"⚠️ [바이비트 상승률 Top 5 수집 실패]\n"
            f"• 시각: {now_str}\n"
            f"• 원인 진단: {err_detail}"
        )
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item)
            if r:
                results.append(r)
        except Exception as e:
            print(f"{item.get('symbol')} 분석 오류: {e}")

    if not results:
        send_telegram("⚠️ 분석 가능한 종목 캔들 데이터가 부족합니다.")
        return

    lines = [
        f"📊 [바이비트 상승률 Top 5 급등 전 초동전략 역추적 리포트]\n"
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
        lines.append("• [발화 직전 컨펌된 초동 시그널]:")
        lines.append(f"  - 5분봉 이평선: [{ma_tag}]")
        lines.append(f"  - 사전 매집: 직전 2시간 변동폭 {r['box_range_pct']:.1f}% 응축 | OBV [{obv_tag}]")
        lines.append(f"  - 초동 수급: 평소 대비 거래량 [{r['vol_mult']:.1f}배 폭발]")
        lines.append(f"  - 돌파 후 패턴: {r['pullback_str']}")

    total = len(results)
    avg_vol = sum(vol_list) / total

    lines.append("━━━━━━━━━━━━━━\n")
    lines.append("💡 [내 스캐너(scanner.py) 전략 튜닝 인사이트]")
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
