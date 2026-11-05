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


def get_binance_klines(symbol, interval="5m", limit=400):
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
    oi_chg_24h = 0.0
    oi_pre_surge = 0.0
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
        if isinstance(r2, list) and len(r2) >= 6:
            oi_start = float(r2[0].get("sumOpenInterest", 0))
            oi_mid = float(r2[len(r2) // 2].get("sumOpenInterest", 0))
            oi_now = float(r2[-1].get("sumOpenInterest", 0))
            if oi_start > 0:
                oi_chg_24h = ((oi_now - oi_start) / oi_start) * 100
                oi_pre_surge = ((oi_mid - oi_start) / oi_start) * 100
    except Exception:
        pass
    return funding, oi_chg_24h, oi_pre_surge


def analyze_pre_surge(item):
    sym = item["symbol"]
    df = get_binance_klines(sym, "5m", 400)
    if len(df) < 320:
        return None

    for m in [5, 20, 50, 120, 200, 300]:
        df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma20"] = df["volume"].rolling(20).mean()
    df["vma5"] = df["volume"].rolling(5).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100
    df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

    # 최근 24시간(288개 봉) 내 본격 상승이 시작된 첫 초동 발화(Ignition) 캔들 탐색
    search_zone = df.iloc[-288:-8].copy()
    explosive = search_zone[
        (search_zone["ret"] >= 2.2) & (search_zone["vol_ratio"] >= 3.5)
    ]
    if len(explosive) > 0:
        ig_idx = explosive.index[0]
    else:
        ig_idx = (search_zone["ret"] * search_zone["vol_ratio"]).idxmax()

    ig_candle = df.loc[ig_idx]
    pre_idx = max(305, ig_idx - 1)
    pre_candle = df.loc[pre_idx]

    kst = datetime.timezone(datetime.timedelta(hours=9))
    ig_time = datetime.datetime.fromtimestamp(
        ig_candle["ts"] / 1000, kst
    ).strftime("%m/%d %H:%M")

    ig_open = float(ig_candle["open"])
    post_high = float(df.loc[ig_idx:, "high"].max())
    surge_after_ig = ((post_high - ig_open) / (ig_open + 1e-9)) * 100

    # 1. 발화 직전 5분봉 정배열 상태 점검
    ma_short_ok = pre_candle["ma5"] > pre_candle["ma20"] > pre_candle["ma50"]
    ma_long_ok = (
        pre_candle["ma50"] > pre_candle["ma120"] > pre_candle["ma200"]
    )

    # 2. 발화 직전 3시간(36개 봉) 박스권 가격 변동폭 및 OBV 매집 분석
    box_zone = df.loc[max(0, pre_idx - 36): pre_idx]
    box_high = float(box_zone["high"].max())
    box_low = float(box_zone["low"].min())
    box_range_pct = ((box_high - box_low) / (box_low + 1e-9)) * 100

    obv_start = float(box_zone["obv"].iloc[0])
    obv_end = float(pre_candle["obv"])
    obv_ok = (
        obv_end > float(pre_candle["obv_ma10"])
        and obv_end > obv_start
    )

    # 3. 발화 캔들 거래량 폭발 배수 및 시장가 매수(Taker Buy) 체결 비중
    vol_mult = float(ig_candle["vol_ratio"])
    taker_ratio = (
        float(ig_candle["taker_buy_vol"]) / (float(ig_candle["volume"]) + 1e-9)
    ) * 100

    # 4. 첫 돌파 직후 90분(18개 봉) 이내 리테스트 눌림목 및 거래량 건조 정밀 역산
    post_zone = df.loc[ig_idx + 1: min(len(df) - 1, ig_idx + 18)]
    pullback_desc = "눌림목 없이 장대양봉 연속 직행"
    mae_desc = "눌림목 진입 구간 없음"
    held_ma20 = False

    if len(post_zone) > 0:
        min_idx = post_zone["low"].idxmin()
        min_low = float(post_zone.loc[min_idx, "low"])
        mins_elapsed = (min_idx - ig_idx) * 5

        ma20_at_pb = float(post_zone.loc[min_idx, "ma20"])
        ma50_at_pb = float(post_zone.loc[min_idx, "ma50"])
        atr_at_pb = float(post_zone.loc[min_idx, "atr14"])

        vol_at_pb = float(post_zone.loc[min_idx, "volume"])
        ig_vol = float(ig_candle["volume"])
        vol_drop_pct = max(0.0, (1.0 - (vol_at_pb / (ig_vol + 1e-9))) * 100)

        if min_low >= ma20_at_pb * 0.990:
            held_ma20 = True
            pullback_desc = (
                f"돌파 {mins_elapsed}분 후 [5분봉 MA 20선({fmt_price(ma20_at_pb)}) 지지] "
                f"+ 돌파봉 대비 거래량 {vol_drop_pct:.0f}% 건조 후 2차 급등"
            )
            mae_pct = ((min_low - ma20_at_pb) / (ma20_at_pb + 1e-9)) * 100
            rec_sl = max(2, int(round((1.5 * atr_at_pb / ma20_at_pb) * 100)))
            mae_desc = (
                f"MA 20선 진입 시 최대 하방 오차 {mae_pct:.2f}% "
                f"(1.5×ATR 동적 손절폭 -{rec_sl}% 내 완벽 방어)"
            )
        elif min_low >= ma50_at_pb * 0.990:
            pullback_desc = (
                f"돌파 {mins_elapsed}분 후 [5분봉 MA 50선({fmt_price(ma50_at_pb)}) 지지] "
                f"+ 거래량 {vol_drop_pct:.0f}% 감소 후 반등"
            )
            mae_pct = ((min_low - ma50_at_pb) / (ma50_at_pb + 1e-9)) * 100
            mae_desc = f"MA 20선 이탈 후 MA 50선({fmt_price(ma50_at_pb)})에서 지지 성공"
        else:
            pullback_desc = (
                f"돌파 {mins_elapsed}분 후 발화 시가({fmt_price(ig_open)}) 부근까지 "
                f"깊은 되돌림(거래량 {vol_drop_pct:.0f}% 감소) 후 반등"
            )
            mae_desc = "이평선 하향 돌파 후 발화 시작점 지지 확인 필요"

    funding, oi_chg_24h, oi_pre = get_funding_and_oi(sym)

    return {
        "symbol": sym,
        "price": item["price"],
        "chg": item["chg"],
        "vol_usdt_m": item["volume_usdt"] / 1_000_000,
        "ig_time": ig_time,
        "ig_price": ig_open,
        "surge_after_ig": surge_after_ig,
        "ma_short_ok": bool(ma_short_ok),
        "ma_long_ok": bool(ma_long_ok),
        "box_range_pct": box_range_pct,
        "obv_ok": bool(obv_ok),
        "vol_mult": vol_mult,
        "taker_ratio": taker_ratio,
        "pullback_desc": pullback_desc,
        "mae_desc": mae_desc,
        "held_ma20": held_ma20,
        "funding": funding,
        "oi_chg_24h": oi_chg_24h,
        "oi_pre": oi_pre
    }


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    top5 = get_top5_binance_gainers()
    if not top5:
        print("바이낸스 Top 5 데이터를 가져오지 못했습니다.")
        return

    results = []
    for item in top5:
        try:
            r = analyze_pre_surge(item)
            if r:
                results.append(r)
        except Exception as e:
            print(f"Analyze error: {e}")

    if not results:
        return

    lines = [
        f"📊 [바이낸스 급등 Top 5 발화점(Pre-Surge) 정밀 해부 리포트]\n"
        f"({now_str} 기준)\n"
    ]

    ma_short_cnt = 0
    obv_cnt = 0
    ma20_hold_cnt = 0
    neg_fund_cnt = 0
    vol_mults = []

    for idx, r in enumerate(results, 1):
        if r["ma_short_ok"]:
            ma_short_cnt += 1
        if r["obv_ok"]:
            obv_cnt += 1
        if r["held_ma20"]:
            ma20_hold_cnt += 1
        if r["funding"] <= 0.01:
            neg_fund_cnt += 1
        vol_mults.append(r["vol_mult"])

        s_tag = "✅정배열 완성" if r["ma_short_ok"] else "❌역배열/혼조 돌파"
        l_tag = "✅완성" if r["ma_long_ok"] else "⏳수렴중"
        obv_tag = "✅우상향 선행매집" if r["obv_ok"] else "➖중립/동행"

        lines.append("━━━━━━━━━━━━━━")
        lines.append(
            f"{idx}️⃣ ${r['symbol']} (+{r['chg']:.1f}% | 현재가 {fmt_price(r['price'])} USDT)"
        )
        lines.append(
            f"• 24h 거래대금: {r['vol_usdt_m']:,.1f}M USDT | "
            f"펀딩비: {r['funding']:+.4f}% | 24h OI: {r['oi_chg_24h']:+.1f}%"
        )
        lines.append(
            f"• [초동 발화 시점]: {r['ig_time']} KST "
            f"(발화가 {fmt_price(r['ig_price'])} → 발화 후 고점까지 +{r['surge_after_ig']:.1f}% 폭발)"
        )
        lines.append("• [급등 직전 3시간 컨펌 내역]:")
        lines.append(
            f"  - 5분봉 이평선: 단기(5>20>50) [{s_tag}] / 장기(50>120>200) [{l_tag}]"
        )
        lines.append(
            f"  - 박스권 매집: 발화 전 3시간 변동폭 {r['box_range_pct']:.1f}% 응축 중 "
            f"OBV [{obv_tag}]"
        )
        lines.append(
            f"  - 발화 캔들 수급: 평소 거래량의 [{r['vol_mult']:.1f}배 폭증] / "
            f"시장가 매수(Taker Buy) 비중 [{r['taker_ratio']:.1f}%]"
        )
        lines.append("• [돌파 후 눌림목 & 손절폭 역산]:")
        lines.append(f"  - 리테스트 타점: {r['pullback_desc']}")
        lines.append(f"  - 리스크 검증: {r['mae_desc']}")

    avg_vol = sum(vol_mults) / len(vol_mults)
    total = len(results)

    lines.append("━━━━━━━━━━━━━━\n")
    lines.append("💡 [내 전략(scanner.py) 업데이트 핵심 피드백]")
    lines.append(
        f"1) 5분봉 단기 정배열 선행률: {total}종목 중 {ma_short_cnt}종목 ({ma_short_cnt/total*100:.0f}%) 컨펌"
    )
    lines.append(
        f"2) 박스권 OBV 우상향 매집률: {total}종목 중 {obv_cnt}종목 ({obv_cnt/total*100:.0f}%) 컨펌"
    )
    lines.append(
        f"3) 음수/중립 펀딩비 비율: {total}종목 중 {neg_fund_cnt}종목 ({neg_fund_cnt/total*100:.0f}%) 해당"
    )
    lines.append(
        f"4) 초동 발화 평균 거래량: 평소 대비 {avg_vol:.1f}배 폭발 "
        f"(스캐너 기준점 {max(3.0, min(5.5, avg_vol * 0.75)):.1f}배 권장)"
    )
    lines.append(
        f"5) 최적 눌림목 지지선: {total}종목 중 {ma20_hold_cnt}종목이 첫 돌파 후 "
        f"5분봉 MA 20선을 지지하고 2차 본파동 분출"
    )

    send_telegram("\n".join(lines))


if __name__ == "__main__":
    main()
