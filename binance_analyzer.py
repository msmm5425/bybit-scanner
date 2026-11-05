import os
import datetime
import requests
import pandas as pd
import numpy as np

BINANCE_URL = "https://fapi.binance.com"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: 텔레그램 설정 누락")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        chunk = text[i:i + 3500]
        try:
            res = requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk},
                timeout=15
            )
            print(f"Telegram status: {res.status_code}")
        except Exception as e:
            print(f"Telegram error: {e}")


def fmt_price(val):
    if val >= 1000:
        return f"{val:.2f}"
    elif val >= 10:
        return f"{val:.3f}"
    elif val >= 1:
        return f"{val:.4f}"
    elif val >= 0.01:
        return f"{val:.5f}"
    else:
        return f"{val:.6f}"


def get_top5_binance_gainers():
    try:
        res = requests.get(
            f"{BINANCE_URL}/fapi/v1/ticker/24hr",
            headers=HEADERS,
            timeout=10
        )
        data = res.json()
        valid = []
        for x in data:
            sym = x.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            q_vol = float(x.get("quoteVolume", 0))
            chg = float(x.get("priceChangePercent", 0))
            last_p = float(x.get("lastPrice", 0))
            if q_vol < 10_000_000 or last_p <= 0:
                continue
            valid.append({
                "symbol": sym,
                "price": last_p,
                "chg": chg,
                "volume_usdt": q_vol
            })
        valid.sort(key=lambda k: k["chg"], reverse=True)
        return valid[:5]
    except Exception as e:
        print(f"Top5 fetch error: {e}")
        return []


def get_binance_klines(symbol, interval="5m", limit=350):
    try:
        url = (
            f"{BINANCE_URL}/fapi/v1/klines?symbol={symbol}"
            f"&interval={interval}&limit={limit}"
        )
        res = requests.get(url, headers=HEADERS, timeout=10)
        rows = res.json()
        if not rows or not isinstance(rows, list):
            return pd.DataFrame()
        cols = [
            "ts", "open", "high", "low", "close", "volume",
            "close_ts", "turnover", "trades", "taker_buy_vol",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(rows, columns=cols)
        for c in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
            df[c] = df[c].astype(float)
        df["ts"] = df["ts"].astype(int)
        return df
    except Exception:
        return pd.DataFrame()


def get_funding_and_oi(symbol):
    funding = 0.0
    oi_chg = 0.0
    try:
        r1 = requests.get(
            f"{BINANCE_URL}/fapi/v1/premiumIndex?symbol={symbol}",
            headers=HEADERS,
            timeout=10
        ).json()
        funding = float(r1.get("lastFundingRate", 0)) * 100
    except Exception:
        pass

    try:
        r2 = requests.get(
            f"{BINANCE_URL}/futures/data/openInterestHist"
            f"?symbol={symbol}&period=1h&limit=25",
            headers=HEADERS,
            timeout=10
        ).json()
        if isinstance(r2, list) and len(r2) >= 2:
            oi_start = float(r2[0].get("sumOpenInterest", 0))
            oi_now = float(r2[-1].get("sumOpenInterest", 0))
            if oi_start > 0:
                oi_chg = ((oi_now - oi_start) / oi_start) * 100
    except Exception:
        pass
    return funding, oi_chg


def analyze_pre_surge(item):
    sym = item["symbol"]
    df = get_binance_klines(sym, "5m", 350)
    if len(df) < 300:
        return None

    for m in [5, 20, 50, 120, 200]:
        df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma20"] = df["volume"].rolling(20).mean()
    df["vma5"] = df["volume"].rolling(5).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()

    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    # 최근 24시간(288개 5분봉) 중 본격 급등을 시작한 초동 발화 캔들 탐색
    search_zone = df.iloc[-288:-6].copy()
    explosive = search_zone[
        (search_zone["ret"] >= 2.0) & (search_zone["vol_ratio"] >= 3.0)
    ]
    if len(explosive) > 0:
        ig_idx = explosive.index[0]
    else:
        ig_idx = (search_zone["ret"] * search_zone["vol_ratio"]).idxmax()

    ig_candle = df.loc[ig_idx]
    pre_idx = max(205, ig_idx - 1)
    pre_candle = df.loc[pre_idx]

    kst = datetime.timezone(datetime.timedelta(hours=9))
    ig_time = datetime.datetime.fromtimestamp(
        ig_candle["ts"] / 1000, kst
    ).strftime("%m/%d %H:%M")

    # 1. 발화 직전 5분봉 정배열 컨펌 여부
    ma_short_ok = pre_candle["ma5"] > pre_candle["ma20"] > pre_candle["ma50"]
    ma_long_ok = pre_candle["ma50"] > pre_candle["ma120"] > pre_candle["ma200"]

    # 2. 발화 직전 박스권 OBV 매집 여부
    obv_ok = (
        pre_candle["obv"] > pre_candle["obv_ma10"]
        and pre_candle["obv"] > df.loc[max(0, pre_idx - 12), "obv"]
    )

    # 3. 발화 순간 거래량 폭발 배수 및 매수 체결 우위(Taker Buy Ratio)
    vol_mult = float(ig_candle["vol_ratio"])
    taker_ratio = (
        float(ig_candle["taker_buy_vol"]) / (float(ig_candle["volume"]) + 1e-9)
    ) * 100

    # 4. 첫 돌파 직후 1시간(12개 봉) 내 눌림목 지지선 및 거래량 건조 검증
    post_zone = df.loc[ig_idx + 1: min(len(df) - 1, ig_idx + 12)]
    pullback_desc = "눌림목 없이 직행 돌파"
    held_ma20 = False
    if len(post_zone) > 0:
        min_low = post_zone["low"].min()
        ma20_post = post_zone["ma20"].mean()
        ma50_post = post_zone["ma50"].mean()
        vol_dry_ok = (post_zone["volume"] < post_zone["vma5"]).any()
        dry_str = "거래량 건조 동반" if vol_dry_ok else "거래량 유지"

        if min_low >= ma20_post * 0.992:
            pullback_desc = f"5분봉 MA 20선 지지 리테스트 ({dry_str})"
            held_ma20 = True
        elif min_low >= ma50_post * 0.992:
            pullback_desc = f"5분봉 MA 50선 깊게 눌림 후 반등 ({dry_str})"
        else:
            pullback_desc = f"전고점 돌파 시작점 리테스트 ({dry_str})"

    funding, oi_chg = get_funding_and_oi(sym)

    return {
        "symbol": sym,
        "price": item["price"],
        "chg": item["chg"],
        "ig_time": ig_time,
        "ig_price": float(ig_candle["open"]),
        "ma_short_ok": bool(ma_short_ok),
        "ma_long_ok": bool(ma_long_ok),
        "obv_ok": bool(obv_ok),
        "vol_mult": vol_mult,
        "taker_ratio": taker_ratio,
        "pullback_desc": pullback_desc,
        "held_ma20": held_ma20,
        "funding": funding,
        "oi_chg": oi_chg
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5 = get_top5_binance_gainers()
    if not top5:
        print("바이낸스 데이터 수집 실패")
        return

    results = []
    for item in top5:
        r = analyze_pre_surge(item)
        if r:
            results.append(r)

    if not results:
        return

    lines = [
        f"📊 [바이낸스 급등 Top 5 발화점 역추적 & 전략 검증] ({now_str})\n"
    ]

    ma_short_cnt = 0
    obv_cnt = 0
    ma20_hold_cnt = 0
    vol_mults = []

    for idx, r in enumerate(results, 1):
        if r["ma_short_ok"]:
            ma_short_cnt += 1
        if r["obv_ok"]:
            obv_cnt += 1
        if r["held_ma20"]:
            ma20_hold_cnt += 1
        vol_mults.append(r["vol_mult"])

        s_tag = "✅컨펌" if r["ma_short_ok"] else "❌미완성(바닥역배열돌파)"
        l_tag = "✅완성" if r["ma_long_ok"] else "⏳수렴중"
        obv_tag = "✅우상향 매집" if r["obv_ok"] else "➖중립"

        lines.append(
            f"{idx}. ${r['symbol']} (+{r['chg']:.1f}% | 현재가 {fmt_price(r['price'])})"
        )
        lines.append(
            f"- 초동 발화 시점: {r['ig_time']} (발화가 {fmt_price(r['ig_price'])})"
        )
        lines.append(
            f"- 발화 직전 5분봉 정배열: 단기(5>20>50) [{s_tag}] / 장기 [{l_tag}]"
        )
        lines.append(
            f"- 발화 직전 수급: OBV [{obv_tag}] | 24h OI {r['oi_chg']:+.1f}% "
            f"| 펀딩비 {r['funding']:+.4f}%"
        )
        lines.append(
            f"- 발화 캔들 폭발력: 평소 거래량의 {r['vol_mult']:.1f}배 "
            f"(시장가 매수체결 비중 {r['taker_ratio']:.1f}%)"
        )
        lines.append(f"- 돌파 후 눌림목 패턴: {r['pullback_desc']}\n")

    avg_vol = sum(vol_mults) / len(vol_mults)
    total = len(results)

    lines.append("💡 [내 전략(scanner.py) 업데이트 핵심 피드백]")
    lines.append(
        f"1) 5분봉 단기 정배열 선행률: {total}종목 중 {ma_short_cnt}종목 컨펌 "
        f"-> {'현재 5분봉 중심 세팅 적중!' if ma_short_cnt >= 3 else '바닥 첫 장대양봉 거래량 가중치 상향 권장'}"
    )
    lines.append(
        f"2) 박스권 OBV 선행 매집률: {total}종목 중 {obv_cnt}종목 컨펌"
    )
    lines.append(
        f"3) 초동 발화 평균 거래량: 평소 대비 약 {avg_vol:.1f}배 폭증 "
        f"(스캐너 거래량 기준점 {max(3.0, min(5.0, avg_vol * 0.7)):.1f}배 권장)"
    )
    lines.append(
        f"4) 최적 눌림목 타점: {total}종목 중 {ma20_hold_cnt}종목이 첫 돌파 후 "
        f"5분봉 MA 20선을 지지하고 2차 급등함"
    )

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
