import os
import json
import datetime
import requests
import pandas as pd
import numpy as np

BYBIT_URL = "https://api.bybit.com"
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
STATE_FILE = "bot_state.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


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


def get_bybit_tickers():
    try:
        url = f"{BYBIT_URL}/v5/market/tickers?category=linear"
        res = requests.get(url, headers=HEADERS, timeout=10)
        raw_list = res.json().get("result", {}).get("list", [])
        tickers = []
        for x in raw_list:
            sym = x.get("symbol", "")
            if not sym.endswith("USDT") or sym.startswith("1000000"):
                continue
            turnover = float(x.get("turnover24h", 0))
            oi_val = float(x.get("openInterestValue", 0))
            price = float(x.get("lastPrice", 0))
            low = float(x.get("lowPrice24h", 0))
            high = float(x.get("highPrice24h", 0))
            if turnover < 1_500_000 or oi_val < 400_000:
                continue
            if price <= 0 or low <= 0:
                continue
            bounce = ((price - low) / low) * 100
            if bounce < 1.0:
                continue
            surge_ratio = turnover / (oi_val + 1.0)
            priority = (surge_ratio * 10) + bounce
            tickers.append({
                "symbol": sym,
                "lastPrice": price,
                "price24hPcnt": float(x.get("price24hPcnt", 0)) * 100,
                "fundingRate": float(x.get("fundingRate", 0)) * 100,
                "highPrice24h": high,
                "lowPrice24h": low,
                "turnover24h": turnover,
                "priority": priority
            })
        tickers.sort(key=lambda k: k["priority"], reverse=True)
        return tickers[:80], {t["symbol"]: t for t in tickers}
    except Exception:
        return [], {}


def get_klines(symbol, interval="5", limit=350):
    try:
        url = (
            f"{BYBIT_URL}/v5/market/kline?category=linear"
            f"&symbol={symbol}&interval={interval}&limit={limit}"
        )
        res = requests.get(url, headers=HEADERS, timeout=10)
        rows = res.json().get("result", {}).get("list", [])
        if not rows:
            return pd.DataFrame()
        cols = ["ts", "open", "high", "low", "close", "volume", "turnover"]
        df = pd.DataFrame(rows, columns=cols)
        df = df.iloc[::-1].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            df[c] = df[c].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


def get_oi_metrics(symbol):
    try:
        url = (
            f"{BYBIT_URL}/v5/market/open-interest?category=linear"
            f"&symbol={symbol}&intervalTime=1h&limit=25"
        )
        res = requests.get(url, headers=HEADERS, timeout=10)
        lst = res.json().get("result", {}).get("list", [])
        if len(lst) >= 24:
            oi_now = float(lst[0]["openInterest"])
            oi_1h = float(lst[1]["openInterest"])
            oi_24h = float(lst[-1]["openInterest"])
            chg_24h = ((oi_now - oi_24h) / oi_24h * 100) if oi_24h > 0 else 0.0
            chg_1h = ((oi_now - oi_1h) / oi_1h * 100) if oi_1h > 0 else 0.0
            return chg_24h, chg_1h
    except Exception:
        pass
    return 0.0, 0.0


def get_orderbook_delta(symbol):
    try:
        url = (
            f"{BYBIT_URL}/v5/market/orderbook?category=linear"
            f"&symbol={symbol}&limit=50"
        )
        res = requests.get(url, headers=HEADERS, timeout=10)
        res_data = res.json().get("result", {})
        bids = sum(float(b[1]) for b in res_data.get("b", []))
        asks = sum(float(a[1]) for a in res_data.get("a", []))
        return bids / (asks + 1e-9)
    except Exception:
        return 1.0


def calc_indicators(df):
    for m in [5, 20, 50, 120, 200, 300]:
        df[f"ma{m}"] = df["close"].rolling(m).mean()
    df["vma5"] = df["volume"].rolling(5).mean()
    df["vma10"] = df["volume"].rolling(10).mean()
    df["vma20"] = df["volume"].rolling(20).mean()

    diff = df["close"].diff()
    direction = np.where(diff > 0, 1, np.where(diff < 0, -1, 0))
    df["obv"] = (direction * df["volume"]).cumsum()
    df["obv_ma10"] = df["obv"].rolling(10).mean()

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    gain = diff.clip(lower=0).rolling(12).mean()
    loss = (-diff.clip(upper=0)).rolling(12).mean()
    df["rsi12"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))

    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    return df


def analyze_symbol(t):
    sym = t["symbol"]
    df5 = get_klines(sym, "5", 350)
    df15 = get_klines(sym, "15", 150)
    if len(df5) < 310 or len(df15) < 60:
        return None

    df5 = calc_indicators(df5)
    df15 = calc_indicators(df15)

    # 직전 완성봉(iloc[-2]) 기준 판별
    c5 = df5.iloc[-2]
    c15 = df15.iloc[-2]
    price = t["lastPrice"]

    oi_24h, oi_1h = get_oi_metrics(sym)
    ob_ratio = get_orderbook_delta(sym)

    score = 50
    # 1. 5분봉 중심 정배열 검증 (핵심 가중치)
    if c5["ma5"] > c5["ma20"] > c5["ma50"]:
        score += 10
    if c5["ma50"] > c5["ma120"] > c5["ma200"] > c5["ma300"]:
        score += 8
    elif c5["ma50"] > c5["ma120"] > c5["ma200"]:
        score += 5

    # 15분봉은 후행성을 배제하고 단기 추세 지지 여부만 확인
    if c15["ma5"] > c15["ma20"]:
        score += 4

    # 2. 5분봉 박스권 OBV 우상향 선행 매집
    if c5["obv"] > c5["obv_ma10"] and c5["obv"] > df5["obv"].iloc[-15]:
        score += 6

    # 3. OI 급증 (24h +5% 이상 또는 최근 1시간 단기 급증)
    if oi_24h >= 5.0:
        score += 7
    elif oi_24h >= 2.0:
        score += 3
    if oi_1h >= 1.5:
        score += 4

    # 4. 펀딩비 중립/음수 역행
    if t["fundingRate"] <= 0.01:
        score += 5
    if t["fundingRate"] < 0:
        score += 4

    # 5. 5분봉 거래량 폭발 전조 & 눌림목 거래량 건조
    recent_5m_vol = df5["volume"].iloc[-20:-2]
    recent_5m_vma = df5["vma20"].iloc[-20:-2]
    if (recent_5m_vol >= (recent_5m_vma * 5.0)).any():
        score += 7
    elif (recent_5m_vol >= (recent_5m_vma * 3.0)).any():
        score += 4

    vol_dry = (c5["volume"] < c5["vma5"]) or (c5["volume"] < c5["vma10"])
    if vol_dry:
        score += 5

    # 6. 호가창 매수벽(Bid Delta) 및 모멘텀
    if ob_ratio >= 1.15:
        score += 5
    elif ob_ratio >= 1.05:
        score += 2
    if c5["macd"] > 0 and 48 <= c5["rsi12"] <= 78:
        score += 3

    score = min(int(round(score)), 98)
    # 급등 전 초동 포착을 위해 커트라인 85점으로 하향
    if score < 85:
        return None

    # 5분봉 MA 20선 및 직전 스윙고점 리테스트 기준 설정
    ma20_5 = float(c5["ma20"])
    swing_high = float(df5["high"].iloc[-30:-3].max())
    support_ref = ma20_5
    support_str = fmt_price(support_ref, price)

    dist_to_support = abs(price - support_ref) / support_ref
    is_pullback_ready = (
        (dist_to_support <= 0.012)
        and vol_dry
        and (price >= support_ref * 0.993)
    )

    grade_tag = "핵심 수급과 5분봉 정배열 완성" if score >= 90 else "급등 전 초동 수급·5분봉 정배열 포착"

    if is_pullback_ready:
        status = (
            f"{grade_tag} / 즉시 분할 진입 구간"
            f"(5분봉 MA 20선 및 직전 스윙고점 돌파 리테스트 구간인 "
            f"{support_str} USDT 지지·거래량 건조 컨펌 완료)"
        )
    else:
        status = (
            f"{grade_tag} / 눌림목 컨펌대기중"
            f"(5분봉 MA 20선 및 직전 스윙고점 리테스트 구간인 "
            f"{support_str} USDT에 눌림목이 완성되면 진입)"
        )

    entry_low = min(price, support_ref) * 0.996
    entry_high = max(price, support_ref) * 1.001
    avg_entry = (entry_low + entry_high) / 2.0

    atr15 = float(c15["atr14"]) if not np.isnan(c15["atr14"]) else (price * 0.02)
    sl_pct = max(2, min(6, int(round((1.5 * atr15 / avg_entry) * 100))))
    stop_price = avg_entry * (1.0 - sl_pct / 100.0)

    min_tp1_pct = sl_pct * 2
    swing_range = max(
        t["highPrice24h"] - t["lowPrice24h"],
        swing_high - float(df5["low"].iloc[-30:-3].min()),
        avg_entry * (min_tp1_pct / 100.0)
    )
    fib_levels = [
        (1.272, "전고점 매물대 및 피보나치 1.272 확장"),
        (1.618, "피보나치 1.618 확장"),
        (2.000, "피보나치 2.0 확장"),
        (2.618, "피보나치 2.618 확장")
    ]
    tps = []
    prev_pct = min_tp1_pct - 1
    for idx, (fib, reason) in enumerate(fib_levels):
        raw_target = avg_entry + swing_range * (fib - 0.5)
        pct = int(round(((raw_target - avg_entry) / avg_entry) * 100))
        if idx == 0:
            pct = max(pct, min_tp1_pct)
        else:
            pct = max(pct, prev_pct + max(3, sl_pct))
        prev_pct = pct
        tps.append((avg_entry * (1.0 + pct / 100.0), pct, reason))

    return {
        "symbol": sym,
        "ticker": sym.replace("USDT", ""),
        "score": score,
        "price": price,
        "chg": t["price24hPcnt"],
        "funding": t["fundingRate"],
        "oi_chg": oi_24h,
        "status": status,
        "is_pullback": is_pullback_ready,
        "support_ref": support_ref,
        "entry_low": entry_low,
        "entry_high": entry_high,
        "avg_entry": avg_entry,
        "stop_price": stop_price,
        "sl_pct": sl_pct,
        "tps": tps
    }


def check_existing_tracked(state, all_ticker_map):
    updated_state = {}
    for sym, info in state.items():
        t = all_ticker_map.get(sym)
        if not t:
            continue
        price = t["lastPrice"]
        stop_price = float(info.get("stop_price", 0))
        support_ref = float(info.get("support_ref", 0))
        tp1_price = float(info.get("tp1_price", price * 1.5))
        notified_pullback = info.get("notified_pullback", False)
        notified_sl = info.get("notified_sl", False)
        ticker = sym.replace("USDT", "")

        if not notified_sl and stop_price > 0 and price <= stop_price:
            msg = (
                f"🛑 [손절가 도달 경고]\n"
                f"${ticker} ({sym}) 현재가 {fmt_price(price, price)} USDT가 "
                f"동적 손절가({fmt_price(stop_price, price)} USDT)에 도달했습니다. "
                f"즉시 대응하세요."
            )
            send_telegram(msg)
            continue

        if not notified_pullback and support_ref > 0:
            dist = abs(price - support_ref) / support_ref
            if dist <= 0.012 and price >= stop_price * 1.005:
                msg = (
                    f"🎯 [눌림목 컨펌 완료 알림]\n"
                    f"${ticker} ({sym}) 5분봉 MA 20선 및 리테스트 지지 구간"
                    f"({fmt_price(support_ref, price)} USDT) 눌림목이 진행 중입니다!\n"
                    f"- 현재가: {fmt_price(price, price)} USDT\n"
                    f"- 동적 손절가: {fmt_price(stop_price, price)} USDT"
                )
                send_telegram(msg)
                notified_pullback = True

        if price >= tp1_price:
            continue

        info["notified_pullback"] = notified_pullback
        info["notified_sl"] = notified_sl
        updated_state[sym] = info

    return updated_state


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_dt = datetime.datetime.now(kst)
    date_dot = now_dt.strftime("%Y.%m.%d")
    time_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

    state = load_state()
    candidates, all_ticker_map = get_bybit_tickers()
    new_state = check_existing_tracked(state, all_ticker_map)

    new_lines = [f"🚨 [바이비트 85점+ 급등 전 포착 브리핑] ({time_str})\n"]
    has_new = False

    for t in candidates:
        sym = t["symbol"]
        if sym in new_state:
            continue
        try:
            r = analyze_symbol(t)
            if not r:
                continue
            has_new = True
            p = r["price"]
            new_lines.append(f"●{date_dot} 추천●")
            new_lines.append(f"${r['ticker']} ({sym}) — [전략 점수: {r['score']}점]")
            new_lines.append(f"* 현황 진단: {r['status']}")
            new_lines.append(
                f"분할 진입가 구간: {fmt_price(r['entry_low'], p)}"
                f"~{fmt_price(r['entry_high'], p)} USDT"
            )
            new_lines.append(
                f"동적 손절가: {fmt_price(r['stop_price'], p)} USDT "
                f"(예상 약 -{r['sl_pct']}%)"
            )
            for idx, (tp_val, tp_pct, _) in enumerate(r["tps"], 1):
                new_lines.append(
                    f"{idx}차 목표가: {fmt_price(tp_val, p)} USDT "
                    f"(예상 약 +{tp_pct}%)"
                )
            new_lines.append("")

            new_state[sym] = {
                "notified_initial": True,
                "notified_pullback": r["is_pullback"],
                "notified_sl": False,
                "support_ref": r["support_ref"],
                "stop_price": r["stop_price"],
                "tp1_price": r["tps"][0][0]
            }
        except Exception:
            continue

    if has_new:
        send_telegram("\n".join(new_lines))
    else:
        print("신규 85점 이상 포착 종목 없음.")

    save_state(new_state)


if __name__ == "__main__":
    main()
