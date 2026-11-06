import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# 텔레그램 환경변수
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# 바이비트 다중 엔드포인트 (GitHub Actions IP 차단 우회용)
BYBIT_ENDPOINTS = [
    "https://api.bybit.com",
    "https://api.bytick.com"
]

KST = timezone(timedelta(hours=9))

# 바이비트 TradFi Perps (주식, ETF, 주가지수, 원자재, 외환) 블랙리스트
TRADFI_BLACKLIST = {
    # 레버리지/인버스 및 주요 ETF
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "TLT",
    "GLD", "SLV", "USO", "UNG", "UVXY", "VIX", "ARKK", "SMH", "XLF",
    "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "GDX", "HYG",
    "EEM", "FXI", "KWEB", "IBIT", "ETHA", "BITO", "MSTX", "MSTU", "NVDL",
    "NVDU", "TSLL", "TSLS", "CONL", "AMDL",
    # 미국 빅테크 및 주요 주식 / 중국 기술주 (TradFi Perps)
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
    "NET", "DDOG", "ZS", "MDB", "TEAM", "PLTR", "U", "RBLX", "RDDT",
    "SPOT", "SNAP", "PINS", "DJT", "GME", "AMC", "CVNA", "RIVN", "LCID",
    # 원자재 및 외환(Forex)
    "XAU", "XAG", "XPT", "XPD", "WTI", "BRENT", "NATGGAS", "COPPER",
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNH", "HKD", "SGD"
}


def request_bybit(path, params=None):
    """바이비트 V5 API 호출 (다중 엔드포인트 폴백)"""
    headers = {"User-Agent": "Mozilla/5.0"}
    for base_url in BYBIT_ENDPOINTS:
        try:
            url = f"{base_url}{path}"
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("retCode") == 0:
                    return data.get("result", {})
        except Exception:
            continue
    return {}


def get_valid_crypto_symbols():
    """
    /v5/market/instruments-info를 조회하여
    정식 거래 중인 순수 USDT 무기한 선물 심볼 집합 반환 (TradFi 속성 제외)
    """
    valid_symbols = set()
    cursor = ""

    for _ in range(5):
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        res = request_bybit("/v5/market/instruments-info", params)
        items = res.get("list", [])
        if not items:
            break

        for item in items:
            symbol = item.get("symbol", "")
            contract_type = item.get("contractType", "")
            status = item.get("status", "")
            quote_coin = item.get("quoteCoin", "")
            base_coin = item.get("baseCoin", "")

            # 순수 USDT 무기한 선물만 허용
            if (
                contract_type != "LinearPerpetual"
                or status != "Trading"
                or quote_coin != "USDT"
                or not symbol.endswith("USDT")
            ):
                continue

            # TradFi 블랙리스트 티커 제외
            if base_coin.upper() in TRADFI_BLACKLIST:
                continue

            # API 메타데이터 내 TradFi/Stock/ETF 관련 문자열 포함 시 제외
            raw_dump = str(item).lower()
            if any(k in raw_dump for k in ["tradfi", "stock", "equity", "forex", "commodity"]):
                continue

            valid_symbols.add(symbol)

        cursor = res.get("nextPageCursor", "")
        if not cursor:
            break

    return valid_symbols


def get_sorted_gainer_candidates():
    """바이비트 Futures 탭과 동일하게 24시간 상승률 순으로 후보군 정렬"""
    valid_symbols = get_valid_crypto_symbols()
    res = request_bybit("/v5/market/tickers", {"category": "linear"})
    tickers = res.get("list", [])
    candidates = []

    for t in tickers:
        symbol = t.get("symbol", "")
        if not symbol.endswith("USDT") or "-" in symbol:
            continue

        base_coin = symbol[:-4].upper()
        if base_coin in TRADFI_BLACKLIST:
            continue

        # instruments-info 조회 성공 시 해당 화이트리스트 검증 적용
        if valid_symbols and symbol not in valid_symbols:
            continue

        try:
            change_24h = float(t.get("price24hPcnt", 0)) * 100
            turnover_24h = float(t.get("turnover24h", 0))
            oi_val = float(t.get("openInterestValue", 0))
            funding_rate = float(t.get("fundingRate", 0)) * 100
            last_price = float(t.get("lastPrice", 0))

            if turnover_24h > 0 and last_price > 0:
                candidates.append({
                    "symbol": symbol,
                    "change_24h": change_24h,
                    "turnover_24h": turnover_24h,
                    "oi_value": oi_val,
                    "funding_rate": funding_rate,
                    "last_price": last_price
                })
        except (ValueError, TypeError):
            continue

    # 바이비트 Gainers 탭과 동일하게 24h 상승률 내림차순 정렬
    candidates.sort(key=lambda x: x["change_24h"], reverse=True)
    return candidates


def get_klines_5m(symbol, limit=200):
    """5분봉 캔들 데이터 수집 및 보조지표 계산"""
    res = request_bybit("/v5/market/kline", {
        "category": "linear",
        "symbol": symbol,
        "interval": "5",
        "limit": limit
    })
    rows = res.get("list", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "turnover"
    ])
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 이동평균선
    for ma in [5, 20, 50, 100, 200]:
        df[f"ma{ma}"] = df["close"].rolling(ma).mean()

    # 거래량 이동평균 (직전 20캔들 기준)
    df["vol_ma20"] = df["volume"].shift(1).rolling(20).mean()
    df["vol_mult"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # OBV
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    # 볼린저밴드 (20, 2) 및 BBW (밴드 폭)
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bbw"] = ((df["bb_upper"] - df["bb_lower"]) / df["ma20"].replace(0, np.nan)) * 100

    # 롤링 4시간(48캔들) VWAP 근사치
    tp = (df["high"] + df["low"] + df["close"]) / 3
    df["vwap48"] = (tp * df["volume"]).rolling(48).sum() / df["volume"].rolling(48).sum().replace(0, np.nan)

    # 캔들 구조 (몸통 비율 및 아랫꼬리 비율)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / candle_range).fillna(0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(upper=10.0)

    return df


def is_24_7_crypto_kline(df, coin):
    """
    [2단계 검증] 5분봉 연속성을 검사하여 주식/ETF(TradFi Perps) 종목을 100% 배제
    - 증시 휴장 시간이 있는 TradFi 선물은 5분봉 사이 타임스탬프 단절(>15분)이 발생하거나,
      장 마감 구간 동안 거래량 0인 더미 캔들이 대량 발생함.
    """
    if len(df) < 60:
        return False

    # 1) 캔들 간 시간 공백 체크 (5분 = 300,000ms, 15분 초과 공백 존재 시 증시 휴장형 TradFi 종목)
    max_time_gap = df["timestamp"].diff().max()
    if pd.notna(max_time_gap) and max_time_gap > 900_000:
        return False

    # 2) 최근 100캔들(약 8시간) 중 거래량 0인 캔들이 10개 이상이면 휴장 시간대 보유 종목으로 판정
    recent_100 = df.tail(100)
    zero_vol_count = int((recent_100["volume"] <= 0).sum())
    if zero_vol_count >= 10:
        return False

    # 3) 펀딩비가 정확히 0.0000%이면서 5분봉 고가==저가(변동 0) 캔들이 과다한 경우 제외
    flat_candles = int((recent_100["high"] == recent_100["low"]).sum())
    if abs(coin["funding_rate"]) < 1e-7 and flat_candles >= 15:
        return False

    return True


def get_oi_history_5m(symbol, limit=200):
    """5분봉 미결제약정(OI) 히스토리 조회"""
    res = request_bybit("/v5/market/open-interest", {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": "5min",
        "limit": limit
    })
    rows = res.get("list", [])
    if not rows:
        return pd.DataFrame()

    oi_df = pd.DataFrame(rows)
    oi_df["timestamp"] = oi_df["timestamp"].astype(np.int64)
    oi_df["openInterest"] = oi_df["openInterest"].astype(float)
    oi_df = oi_df.sort_values("timestamp").reset_index(drop=True)
    return oi_df


def analyze_coin(coin):
    """개별 순수 암호화폐 종목의 기존 전략 + 신규 선행/돌파 전략 정밀 역추적"""
    symbol = coin["symbol"]
    df = get_klines_5m(symbol, limit=200)

    # TradFi Perps(주식/ETF) 여부 캔들 연속성 최종 검증
    if not is_24_7_crypto_kline(df, coin):
        return None

    oi_df = get_oi_history_5m(symbol, limit=200)
    if not oi_df.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            oi_df[["timestamp", "openInterest"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest"
        )
    else:
        df["openInterest"] = np.nan

    # 분석 구간에서 초동 발화 캔들 탐색
    search_df = df.iloc[30:-3].copy()
    candidates = search_df[
        (search_df["close"] > search_df["open"]) &
        (search_df["vol_mult"] >= 3.0) &
        ((search_df["close"] - search_df["open"]) / search_df["open"] >= 0.015)
    ]

    if not candidates.empty:
        best_idx = None
        for idx in candidates.index:
            future_max = df.loc[idx:, "high"].max()
            gain_after = (future_max - df.loc[idx, "close"]) / df.loc[idx, "close"]
            if gain_after >= 0.08:
                best_idx = idx
                break
        if best_idx is None:
            best_idx = candidates["vol_mult"].idxmax()
        ign_idx = best_idx
    else:
        ign_idx = search_df["vol_mult"].idxmax()

    ign_row = df.loc[ign_idx]
    pre_df = df.loc[max(0, ign_idx - 24):ign_idx - 1]  # 직전 2시간(24캔들)
    post_df = df.loc[ign_idx + 1:]                     # 발화 이후 구간

    # 1. 발화 기본 정보
    ign_time_kst = datetime.fromtimestamp(ign_row["timestamp"] / 1000, tz=KST).strftime("%m/%d %H:%M KST")
    ign_price = ign_row["close"]
    post_high = df.loc[ign_idx:, "high"].max()
    pump_pct = ((post_high - ign_price) / ign_price) * 100
    vol_mult = float(ign_row["vol_mult"]) if not np.isnan(ign_row["vol_mult"]) else 1.0

    # 2. 기존 전략 체크
    short_align = bool(ign_row["ma5"] > ign_row["ma20"] > ign_row["ma50"])
    full_align = bool(
        short_align
        and not np.isnan(ign_row["ma200"])
        and (ign_row["ma50"] > ign_row["ma100"] > ign_row["ma200"])
    )

    box_high = pre_df["high"].max()
    box_low = pre_df["low"].min()
    box_range_pct = ((box_high - box_low) / box_low) * 100 if box_low > 0 else 0.0

    if len(pre_df) >= 5:
        obv_slope = np.polyfit(np.arange(len(pre_df)), pre_df["obv"].values, 1)[0]
        obv_accum = obv_slope > 0
    else:
        obv_accum = False

    # 3. 신규 전략 역추적 (New Strategy Discovery)
    new_strategies = []

    # [신규 A] OI 선행 매집 다이버전스
    oi_pre_change = 0.0
    oi_ignition_change = 0.0
    if not df["openInterest"].isna().all() and ign_idx >= 12:
        oi_1h_ago = df.loc[ign_idx - 12, "openInterest"]
        oi_1candle_ago = df.loc[ign_idx - 1, "openInterest"]
        oi_at_ign = df.loc[ign_idx, "openInterest"]
        if oi_1h_ago > 0:
            oi_pre_change = ((oi_1candle_ago - oi_1h_ago) / oi_1h_ago) * 100
        if oi_1candle_ago > 0:
            oi_ignition_change = ((oi_at_ign - oi_1candle_ago) / oi_1candle_ago) * 100

    stealth_oi = (oi_pre_change >= 3.0) and (box_range_pct <= 8.0)
    if stealth_oi:
        new_strategies.append(f"OI 선행매집(발화전 1시간 OI +{oi_pre_change:.1f}%)")

    # [신규 B] BB 스퀴즈 + VWAP 장악 장대양봉
    pre_bbw = pre_df["bbw"].iloc[-1] if not pre_df.empty else 10.0
    bbw_percentile = (df.loc[max(0, ign_idx - 48):ign_idx, "bbw"] < pre_bbw).mean() * 100
    vwap_break = bool(ign_row["close"] > ign_row["vwap48"]) and (ign_row["body_ratio"] >= 0.65)
    bb_squeeze_break = (bbw_percentile <= 35.0) and vwap_break
    if bb_squeeze_break:
        new_strategies.append(
            f"BB스퀴즈(하위 {bbw_percentile:.0f}%)+VWAP 장악(몸통 {ign_row['body_ratio']*100:.0f}%)"
        )

    # [신규 C] 비정배열 바닥권 유동성 스윕 (Spring / 스탑헌팅 후 급반등)
    recent_6_df = df.loc[max(0, ign_idx - 6):ign_idx]
    had_wick_sweep = bool((recent_6_df["lower_wick_ratio"] >= 1.5).any())
    spring_reversal = (not short_align) and had_wick_sweep and (ign_row["close"] > ign_row["ma20"])
    if spring_reversal:
        new_strategies.append("유동성 스윕(Spring 아랫꼬리 스탑헌팅 후 MA20 돌파)")

    # [신규 D & 돌파 후 눌림목 정밀 분석] 전고점 S/R Flip 리테스트 vs MA20/50 지지 & 거래량 건조율
    pullback_type = "눌림 없이 직행(Runaway)"
    dryup_ratio = 1.0
    retest_sl_pct = 3
    sr_flip_hit = False
    ma20_hit = False

    if len(post_df) >= 3:
        pb_window = post_df.iloc[:15]
        min_pb_idx = pb_window["low"].idxmin()
        min_pb_row = pb_window.loc[min_pb_idx]
        pb_low = min_pb_row["low"]

        dryup_ratio = float(pb_window["volume"].min() / max(ign_row["volume"], 1e-8))

        sr_diff_pct = abs(pb_low - box_high) / box_high * 100 if box_high > 0 else 99.0
        ma20_diff_pct = abs(pb_low - min_pb_row["ma20"]) / min_pb_row["ma20"] * 100 if min_pb_row["ma20"] > 0 else 99.0
        ma50_diff_pct = abs(pb_low - min_pb_row["ma50"]) / min_pb_row["ma50"] * 100 if min_pb_row["ma50"] > 0 else 99.0

        if sr_diff_pct <= 1.2 and pb_low >= box_high * 0.988:
            sr_flip_hit = True
            pullback_type = f"전고점 S/R Flip 칼지지(오차 {sr_diff_pct:.2f}%)"
            new_strategies.append(f"돌파후 박스상단 S/R Flip 리테스트(거래량 {dryup_ratio*100:.0f}%로 건조)")
        elif ma20_diff_pct <= 1.2:
            ma20_hit = True
            pullback_type = "5분봉 MA 20선 지지 리테스트"
        elif ma50_diff_pct <= 1.5:
            pullback_type = "5분봉 MA 50선 눌림 후 반등"
        else:
            pullback_type = f"단기 {((ign_row['high'] - pb_low)/ign_row['high']*100):.1f}% 자율 눌림 후 반등"

        raw_sl = ((min_pb_row["close"] - (pb_low * 0.992)) / min_pb_row["close"]) * 100
        retest_sl_pct = max(2, min(6, int(round(raw_sl))))

    # [신규 E] 숏스퀴즈 발화 (펀딩비 <= +0.01% + 발화 캔들 OI 동반 급증)
    short_squeeze = (coin["funding_rate"] <= 0.01) and (oi_ignition_change >= 1.5 or oi_pre_change >= 3.0)
    if short_squeeze:
        new_strategies.append(f"숏스퀴즈 발화(펀딩 {coin['funding_rate']:+.4f}% + 발화OI +{oi_ignition_change:.1f}%)")

    if not new_strategies:
        new_strategies.append("단독 대량거래량 모멘텀 돌파(평균 대비 고배수 수급 집중)")

    return {
        "symbol": symbol,
        "change_24h": coin["change_24h"],
        "last_price": coin["last_price"],
        "turnover_m": coin["turnover_24h"] / 1_000_000,
        "oi_m": coin["oi_value"] / 1_000_000,
        "funding_rate": coin["funding_rate"],
        "ign_time_kst": ign_time_kst,
        "ign_price": ign_price,
        "pump_pct": pump_pct,
        "short_align": short_align,
        "full_align": full_align,
        "box_range_pct": box_range_pct,
        "obv_accum": obv_accum,
        "vol_mult": vol_mult,
        "pullback_type": pullback_type,
        "dryup_ratio": dryup_ratio,
        "oi_pre_change": oi_pre_change,
        "oi_ignition_change": oi_ignition_change,
        "stealth_oi": stealth_oi,
        "bb_squeeze_break": bb_squeeze_break,
        "bbw_val": pre_bbw,
        "body_ratio": float(ign_row["body_ratio"]),
        "spring_reversal": spring_reversal,
        "sr_flip_hit": sr_flip_hit,
        "ma20_hit": ma20_hit,
        "short_squeeze": short_squeeze,
        "retest_sl_pct": retest_sl_pct,
        "new_strategies": new_strategies
    }


def build_report(results):
    """텔레그램 전송용 브리핑 및 최하단 제미나이 복붙용 스캐너 업데이트 명세서 생성"""
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    lines = [
        f"📊 [Bybit Futures 급등 Top 5 초동 역추적 & 신규 전략 발굴]",
        f"🕒 기준 시각: {now_str} (TradFi 주식/ETF 선물 제외)\n"
    ]

    num_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    for idx, r in enumerate(results):
        emoji = num_emojis[idx] if idx < len(num_emojis) else f"{idx+1}."
        ticker_short = r["symbol"].replace("USDT", "")
        align_str = "✅ 완전 정배열(5>20>50>100>200)" if r["full_align"] else (
            "✅ 단기 정배열(5>20>50)" if r["short_align"] else "⚠️ 비정배열(역배열/혼조 돌파)"
        )
        obv_str = "✅ 우상향 선행매집" if r["obv_accum"] else "➖ 중립/횡보"

        lines.append(
            f"{emoji} ${ticker_short} ({r['symbol']} | +{r['change_24h']:.2f}% | 현재가 {r['last_price']:g} USDT)\n"
            f"• 거래대금: {r['turnover_m']:.2f}M | 미결제약정(OI): {r['oi_m']:.2f}M | 펀딩비: {r['funding_rate']:+.4f}%\n"
            f"• [초동 발화점]: {r['ign_time_kst']} (발화가 {r['ign_price']:g} → 고점까지 +{r['pump_pct']:.1f}% 폭등)\n"
            f"• [기존 스캐너 전략 체크]:\n"
            f"  - 5분봉 이평선: [{align_str}]\n"
            f"  - 사전 매집: 직전 2시간 변동폭 {r['box_range_pct']:.1f}% 응축 | OBV [{obv_str}]\n"
            f"  - 초동 수급: 평소 대비 거래량 [{r['vol_mult']:.1f}배 폭발]\n"
            f"  - 돌파 후 패턴: {r['pullback_type']} (눌림 최저거래량 {r['dryup_ratio']*100:.0f}% 수준)\n"
            f"• [🔍 포착된 신규 알파 전략]:\n"
            f"  - {' / '.join(r['new_strategies'])}\n"
        )

    # 통계 집계 (최하단 스캐너 업데이트 명세서용)
    n = len(results)
    cnt_short_align = sum(1 for r in results if r["short_align"])
    cnt_full_align = sum(1 for r in results if r["full_align"])
    cnt_obv = sum(1 for r in results if r["obv_accum"])
    cnt_stealth_oi = sum(1 for r in results if r["stealth_oi"])
    cnt_bb_vwap = sum(1 for r in results if r["bb_squeeze_break"])
    cnt_spring = sum(1 for r in results if r["spring_reversal"])
    cnt_sr_flip = sum(1 for r in results if r["sr_flip_hit"])
    cnt_ma20 = sum(1 for r in results if r["ma20_hit"])
    cnt_squeeze = sum(1 for r in results if r["short_squeeze"])

    avg_vol_mult = float(np.mean([r["vol_mult"] for r in results]))
    min_vol_mult = float(np.min([r["vol_mult"] for r in results]))
    rec_vol_mult = round(max(3.5, min(min_vol_mult * 0.85, avg_vol_mult * 0.6)), 1)

    med_box_range = float(np.median([r["box_range_pct"] for r in results]))
    rec_box_max = round(max(4.5, med_box_range * 1.15), 1)

    med_dryup = float(np.median([r["dryup_ratio"] for r in results]))
    rec_dryup_max = round(min(0.45, max(0.15, med_dryup * 1.2)), 2)

    avg_oi_pre = float(np.mean([r["oi_pre_change"] for r in results]))
    rec_sl_pct = int(round(np.median([r["retest_sl_pct"] for r in results])))
    avg_pump = float(np.mean([r["pump_pct"] for r in results]))

    tp1_pct = max(rec_sl_pct * 2, 6)
    tp2_pct = int(round(max(tp1_pct * 1.6, avg_pump * 0.45)))
    tp3_pct = int(round(max(tp2_pct * 1.4, avg_pump * 0.70)))
    tp4_pct = int(round(max(tp3_pct * 1.3, avg_pump * 0.95)))

    w_ma = 20 if cnt_short_align >= 3 else 10
    w_obv = 20 if cnt_obv >= 3 else 15
    w_oi = 25 if (cnt_stealth_oi >= 2 or cnt_squeeze >= 2) else 15
    w_bb_vwap = 20 if cnt_bb_vwap >= 2 else 15
    w_retest = 100 - (w_ma + w_obv + w_oi + w_bb_vwap)

    lines.append(
        "━━━━━━━━━━━━━━━━━━\n"
        "🛠️ [scanner.py 업데이트용 엔지니어링 명세서 (제미나이 복붙용)]\n\n"
        "[1. 기존 vs 신규 전략 적중률 비교]\n"
        f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {cnt_short_align}종목 (완전정배열 {cnt_full_align}종목)\n"
        f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {cnt_obv}종목\n"
        f"• [신규] OI 선행매집(발화전 1h OI 급증): {n}종목 중 {cnt_stealth_oi}종목 (평균 {avg_oi_pre:+.1f}%)\n"
        f"• [신규] BB스퀴즈 + VWAP 장대양봉 돌파: {n}종목 중 {cnt_bb_vwap}종목\n"
        f"• [신규] 비정배열 유동성 스윕(Spring) 반전: {n}종목 중 {cnt_spring}종목\n"
        f"• [신규] 숏스퀴즈(펀딩<=0.01% & OI급증): {n}종목 중 {cnt_squeeze}종목\n"
        f"• [신규] 돌파 후 타점: S/R Flip(전고점 지지) {cnt_sr_flip}종목 | MA20 지지 {cnt_ma20}종목\n\n"
        "[2. scanner.py 파라미터 수치 튜닝안]\n"
        f"• VOL_MULTIPLIER_MIN: {rec_vol_mult} (오늘 발화 평균 {avg_vol_mult:.1f}x / 최소 {min_vol_mult:.1f}x)\n"
        f"• PRE_BOX_RANGE_MAX_PCT: {rec_box_max}% (오늘 중앙값 {med_box_range:.1f}%)\n"
        f"• PULLBACK_DRYUP_MAX_RATIO: {rec_dryup_max} (발화 거래량 대비 {int(rec_dryup_max*100)}% 이하 감소 시 컨펌)\n"
        f"• OI_PRE_SURGE_MIN_PCT: +3.0% (5분봉 12캔들 OI 증가율 기준)\n"
        f"• FUNDING_NEUTRAL_MAX: +0.0100% (숏스퀴즈 가산점 기준)\n\n"
        "[3. scanner.py 추가할 신규 조건식 (Python Logic)]\n"
        "• cond_stealth_oi = (oi_change_1h_pct >= 3.0) and (box_range_2h_pct <= "
        f"{rec_box_max})\n"
        "• cond_bb_vwap = (bbw_percentile_4h <= 35) and (close > vwap48) and (body_ratio >= 0.65)\n"
        "• cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.5) and (close > ma20)\n"
        "• cond_sr_flip_retest = (abs(pullback_low - prev_box_high) / prev_box_high <= 0.012) and "
        f"(pullback_vol_ratio <= {rec_dryup_max})\n\n"
        "[4. 100점 만점 스코어링 배점 재조정안]\n"
        f"• 이평선 정배열(5>20>50) 또는 Spring 반전: {w_ma}점\n"
        f"• 박스권 응축 + OBV 우상향 선행매집: {w_obv}점\n"
        f"• 선물 OI 선행 급증(+3%↑) & 중립/음수 펀딩비: {w_oi}점\n"
        f"• BB 스퀴즈 돌파 + VWAP 안착 + 거래량({rec_vol_mult}x↑): {w_bb_vwap}점\n"
        f"• S/R Flip(전고점) 또는 MA20 리테스트 + 거래량 건조: {w_retest}점 (총합 100점, 85점 이상 알림)\n\n"
        "[5. 타점 / 손절(SL) / 4단계 익절(TP) 표준 파라미터]\n"
        f"• 진입 타점: 돌파 후 직전 박스 상단(S/R Flip ±1.0%) 또는 5분봉 MA20 리테스트 양봉 확인\n"
        f"• 권장 손절폭(SL): -{rec_sl_pct}% (소수점 제외 정수, 리테스트 스윙 저점 이탈 기준)\n"
        f"• 권장 4단계 익절(TP): 1차 +{tp1_pct}% (R:R 1:2 보장) / 2차 +{tp2_pct}% / 3차 +{tp3_pct}% / 4차 +{tp4_pct}%"
    )

    return "\n".join(lines)


def send_telegram_message(text):
    """텔레그램 메시지 전송 (4096자 제한 시 자동 분할 전송)"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    max_len = 3900
    chunks = []
    current = ""

    for line in text.split("\n"):
        if len(current) + len(line) + 1 > max_len:
            chunks.append(current)
            current = line + "\n"
        else:
            current += line + "\n"
    if current:
        chunks.append(current)

    for chunk in chunks:
        try:
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": chunk
            }, timeout=10)
            time.sleep(0.5)
        except Exception as e:
            print(f"Telegram send error: {e}")


def main():
    candidates = get_sorted_gainer_candidates()
    if not candidates:
        send_telegram_message("⚠️ 바이비트 상승률 상위 종목 데이터를 가져오지 못했습니다.")
        return

    results = []
    # 상위 후보를 순서대로 검사하여 순수 24/7 암호화폐 선물 상위 5개가 채워질 때까지 수집
    for coin in candidates[:25]:
        res = analyze_coin(coin)
        if res:
            results.append(res)
        if len(results) >= 5:
            break
        time.sleep(0.25)

    if not results:
        send_telegram_message("⚠️ 5분봉 상세 분석 데이터가 부족합니다.")
        return

    report_text = build_report(results)
    send_telegram_message(report_text)


if __name__ == "__main__":
    main()
