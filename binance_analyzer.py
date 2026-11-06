import os
import time
import json
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlencode, quote
from datetime import datetime, timedelta, timezone

# 텔레그램 환경변수 (TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN 모두 자동 인식)
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

# 바이비트 TradFi Perps (주식/ETF/원자재/외환) 제외 목록 -> 순수 코인만 남김
TRADFI_BLACKLIST = {
    "SOXL", "SOXS", "TQQQ", "SQQQ", "SPY", "QQQ", "IWM", "DIA", "TLT",
    "GLD", "SLV", "USO", "UNG", "UVXY", "VIX", "ARKK", "SMH", "XLF",
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
    "XAU", "XAG", "XPT", "XPD", "WTI", "BRENT", "NATGGAS", "COPPER",
    "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNH", "HKD", "SGD"
}


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def fetch_bybit_via_relay(path, params=None):
    """
    오전에 성공했던 유럽 프록시 릴레이 방식 (api.bybit.com 기준)
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    query_str = f"?{urlencode(params)}" if params else ""
    target_url = f"https://api.bybit.com{path}{query_str}"
    encoded_url = quote(target_url, safe="")

    # 1순위: 유럽 프록시 릴레이 (인코딩 / 비인코딩 / allorigins 폴백)
    relay_candidates = [
        ("raw", f"https://api.codetabs.com/v1/proxy?quest={encoded_url}"),
        ("raw", f"https://api.codetabs.com/v1/proxy?quest={target_url}"),
        ("raw", f"https://api.allorigins.win/raw?url={encoded_url}"),
        ("wrapped", f"https://api.allorigins.win/get?url={encoded_url}"),
        ("raw", f"https://api.bytick.com{path}{query_str}"),
        ("raw", target_url)
    ]

    for mode, url in relay_candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue

            if mode == "wrapped":
                outer = resp.json()
                raw_text = outer.get("contents", "")
                data = json.loads(raw_text) if raw_text else {}
            else:
                data = resp.json()
                if isinstance(data, str):
                    data = json.loads(data)

            if isinstance(data, dict) and data.get("retCode") == 0:
                return data.get("result", {})
        except Exception:
            continue

    return {}


def get_top5_bybit_gainers():
    """
    바이비트 홈페이지 [Futures -> Gainers] 탭과 동일하게
    주식/ETF 코인(SOXS 등)만 빼고 순수 암호화폐 상승률 상위 5개를 즉시 반환
    """
    res = fetch_bybit_via_relay("/v5/market/tickers", {"category": "linear"})
    tickers = res.get("list", [])
    gainers = []

    for t in tickers:
        symbol = str(t.get("symbol", ""))
        # USDT 무기한 선물만 선택 (만기 선물 '-' 제외)
        if not symbol.endswith("USDT") or "-" in symbol:
            continue

        # 주식/ETF/원자재(TradFi) 블랙리스트 즉시 제외
        base_coin = symbol[:-4].upper()
        if base_coin in TRADFI_BLACKLIST:
            continue

        last_price = to_float(t.get("lastPrice"))
        turnover_24h = to_float(t.get("turnover24h"))
        if last_price <= 0 or turnover_24h <= 0:
            continue

        gainers.append({
            "symbol": symbol,
            "change_24h": to_float(t.get("price24hPcnt")) * 100.0,
            "turnover_24h": turnover_24h,
            "oi_value": to_float(t.get("openInterestValue")),
            "funding_rate": to_float(t.get("fundingRate")) * 100.0,
            "last_price": last_price
        })

    # 홈페이지와 동일하게 24시간 상승률 순으로 정렬 후 상위 5개만 추출
    gainers.sort(key=lambda x: x["change_24h"], reverse=True)
    return gainers[:5]


def get_klines_5m(symbol, limit=200):
    """상위 5개 종목의 5분봉 데이터 조회 및 지표 계산"""
    res = fetch_bybit_via_relay("/v5/market/kline", {
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
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"]).copy()
    if df.empty:
        return pd.DataFrame()

    df["timestamp"] = df["timestamp"].astype(np.int64)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 이동평균선 (초기 구간에서도 계산되도록 min_periods 적용)
    df["ma5"] = df["close"].rolling(5, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=10).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=20).mean()
    df["ma100"] = df["close"].rolling(100, min_periods=40).mean()
    df["ma200"] = df["close"].rolling(200, min_periods=60).mean()

    # 거래량 배수
    df["vol_ma20"] = df["volume"].shift(1).rolling(20, min_periods=5).mean()
    df["vol_mult"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)

    # OBV
    direction = np.sign(df["close"].diff()).fillna(0)
    df["obv"] = (direction * df["volume"]).cumsum()

    # 볼린저밴드 폭(BBW)
    std20 = df["close"].rolling(20, min_periods=10).std()
    df["bb_upper"] = df["ma20"] + 2 * std20
    df["bb_lower"] = df["ma20"] - 2 * std20
    df["bbw"] = ((df["bb_upper"] - df["bb_lower"]) / df["ma20"].replace(0, np.nan)) * 100.0

    # 4시간(48캔들) 롤링 VWAP
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap48"] = (
        (tp * df["volume"]).rolling(48, min_periods=10).sum()
        / df["volume"].rolling(48, min_periods=10).sum().replace(0, np.nan)
    )

    # 캔들 몸통 비율 및 아랫꼬리 비율
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = ((df["close"] - df["open"]).abs() / candle_range).fillna(0.0)
    lower_wick = np.minimum(df["open"], df["close"]) - df["low"]
    body_size = (df["close"] - df["open"]).abs().replace(0, 1e-8)
    df["lower_wick_ratio"] = (lower_wick / body_size).clip(lower=0.0, upper=10.0)

    return df


def get_oi_5m(symbol, limit=200):
    """5분봉 미결제약정(OI) 조회"""
    res = fetch_bybit_via_relay("/v5/market/open-interest", {
        "category": "linear",
        "symbol": symbol,
        "intervalTime": "5min",
        "limit": limit
    })
    rows = res.get("list", [])
    if not rows:
        return pd.DataFrame()

    oi_df = pd.DataFrame(rows)
    if "timestamp" not in oi_df.columns or "openInterest" not in oi_df.columns:
        return pd.DataFrame()

    oi_df["timestamp"] = pd.to_numeric(oi_df["timestamp"], errors="coerce")
    oi_df["openInterest"] = pd.to_numeric(oi_df["openInterest"], errors="coerce")
    oi_df = oi_df.dropna(subset=["timestamp", "openInterest"]).copy()
    if oi_df.empty:
        return pd.DataFrame()

    oi_df["timestamp"] = oi_df["timestamp"].astype(np.int64)
    return oi_df.sort_values("timestamp").reset_index(drop=True)


def analyze_coin(coin):
    """상위 5개 각 종목의 기존 전략 + 신규 전략 분석"""
    symbol = coin["symbol"]
    df = get_klines_5m(symbol, limit=200)
    if len(df) < 30:
        return None

    oi_df = get_oi_5m(symbol, limit=200)
    if not oi_df.empty:
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            oi_df[["timestamp", "openInterest"]].sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=300_000
        )
    else:
        df["openInterest"] = np.nan

    # 초동 발화 캔들 찾기
    search_df = df.iloc[24:-2].copy()
    candidates = search_df[
        (search_df["close"] > search_df["open"]) &
        (search_df["vol_mult"] >= 3.0) &
        ((search_df["close"] - search_df["open"]) / search_df["open"].replace(0, np.nan) >= 0.015)
    ]

    ign_idx = None
    if not candidates.empty:
        for idx in candidates.index:
            future_max = to_float(df.loc[idx:, "high"].max())
            ign_close = to_float(df.loc[idx, "close"])
            if ign_close > 0 and (future_max - ign_close) / ign_close >= 0.08:
                ign_idx = idx
                break
        if ign_idx is None:
            ign_idx = candidates["vol_mult"].dropna().idxmax()
    else:
        valid_vol = search_df["vol_mult"].dropna()
        ign_idx = valid_vol.idxmax() if not valid_vol.empty else search_df.index[0]

    ign_row = df.loc[ign_idx]
    pre_df = df.loc[max(0, ign_idx - 24):ign_idx - 1]
    post_df = df.loc[ign_idx + 1:]

    ign_time_kst = datetime.fromtimestamp(ign_row["timestamp"] / 1000.0, tz=KST).strftime("%m/%d %H:%M KST")
    ign_price = to_float(ign_row["close"])
    post_high = to_float(df.loc[ign_idx:, "high"].max())
    pump_pct = ((post_high - ign_price) / ign_price) * 100.0 if ign_price > 0 else 0.0
    vol_mult = to_float(ign_row["vol_mult"], default=1.0)

    # 기존 전략 체크
    short_align = bool(ign_row["ma5"] > ign_row["ma20"] > ign_row["ma50"])
    full_align = bool(short_align and (ign_row["ma50"] > ign_row["ma100"] > ign_row["ma200"]))

    box_high = to_float(pre_df["high"].max()) if not pre_df.empty else ign_price
    box_low = to_float(pre_df["low"].min()) if not pre_df.empty else ign_price
    box_range_pct = ((box_high - box_low) / box_low) * 100.0 if box_low > 0 else 0.0

    obv_accum = False
    if len(pre_df) >= 5:
        try:
            obv_slope = np.polyfit(np.arange(len(pre_df)), pre_df["obv"].values, 1)[0]
            obv_accum = bool(obv_slope > 0)
        except Exception:
            obv_accum = False

    # 신규 전략 포착 리스트
    new_strategies = []

    # [신규 1] OI 선행매집 (발화 전 1시간 OI 변화율)
    oi_pre_change = 0.0
    oi_ignition_change = 0.0
    if df["openInterest"].notna().any() and ign_idx >= 12:
        oi_1h_ago = to_float(df.loc[ign_idx - 12, "openInterest"])
        oi_1c_ago = to_float(df.loc[ign_idx - 1, "openInterest"])
        oi_now = to_float(df.loc[ign_idx, "openInterest"])
        if oi_1h_ago > 0 and oi_1c_ago > 0:
            oi_pre_change = ((oi_1c_ago - oi_1h_ago) / oi_1h_ago) * 100.0
        if oi_1c_ago > 0 and oi_now > 0:
            oi_ignition_change = ((oi_now - oi_1c_ago) / oi_1c_ago) * 100.0

    stealth_oi = (oi_pre_change >= 3.0) and (box_range_pct <= 8.0)
    if stealth_oi:
        new_strategies.append(f"OI 선행매집(발화전 1시간 OI +{oi_pre_change:.1f}%)")

    # [신규 2] BB스퀴즈 + VWAP 장악 돌파
    pre_bbw = to_float(pre_df["bbw"].iloc[-1], default=-1.0) if not pre_df.empty else -1.0
    bb_squeeze_break = False
    if pre_bbw > 0:
        bbw_win = df.loc[max(0, ign_idx - 48):ign_idx, "bbw"].dropna()
        if not bbw_win.empty:
            bbw_pct = float((bbw_win < pre_bbw).mean() * 100.0)
            vwap_ok = bool(pd.notna(ign_row["vwap48"]) and ign_row["close"] > ign_row["vwap48"] and ign_row["body_ratio"] >= 0.65)
            bb_squeeze_break = (bbw_pct <= 35.0) and vwap_ok
            if bb_squeeze_break:
                new_strategies.append(f"BB스퀴즈(하위 {bbw_pct:.0f}%)+VWAP 돌파(양봉몸통 {ign_row['body_ratio']*100:.0f}%)")

    # [신규 3] 비정배열 아랫꼬리 스윕(Spring) 반전
    recent_6 = df.loc[max(0, ign_idx - 6):ign_idx]
    had_wick = bool((recent_6["lower_wick_ratio"] >= 1.5).any()) if not recent_6.empty else False
    spring_reversal = bool((not short_align) and had_wick and ign_row["close"] > ign_row["ma20"])
    if spring_reversal:
        new_strategies.append("유동성 스윕(Spring 아랫꼬리 스탑헌팅 후 MA20 돌파)")

    # [신규 4] 발화 직전 거래량 극건조 포켓 (평소 대비 35% 이하 수축)
    pre_5 = df.loc[max(0, ign_idx - 5):ign_idx - 1]
    pre_vol_pocket = False
    if not pre_5.empty and to_float(ign_row["vol_ma20"]) > 0:
        min_pre_vol_ratio = to_float(pre_5["volume"].min() / ign_row["vol_ma20"], default=1.0)
        if min_pre_vol_ratio <= 0.35:
            pre_vol_pocket = True
            new_strategies.append(f"발화 직전 거래량 극건조 포켓(평소 대비 {min_pre_vol_ratio*100:.0f}% 수축 후 폭발)")

    # [신규 5] 돌파 후 눌림목 패턴 (전고점 S/R Flip vs MA20/50 지지)
    pullback_type = "눌림 없이 직행(Runaway)"
    dryup_ratio = 1.0
    retest_sl_pct = 3
    sr_flip_hit = False
    ma20_hit = False

    if len(post_df) >= 3:
        pb_window = post_df.iloc[:15]
        min_pb_idx = pb_window["low"].idxmin()
        min_pb_row = pb_window.loc[min_pb_idx]
        pb_low = to_float(min_pb_row["low"])

        ign_vol = max(to_float(ign_row["volume"]), 1e-8)
        dryup_ratio = to_float(pb_window["volume"].min() / ign_vol, default=1.0)

        sr_diff_pct = abs(pb_low - box_high) / box_high * 100.0 if box_high > 0 else 99.0
        ma20_val = to_float(min_pb_row["ma20"])
        ma50_val = to_float(min_pb_row["ma50"])
        ma20_diff = abs(pb_low - ma20_val) / ma20_val * 100.0 if ma20_val > 0 else 99.0
        ma50_diff = abs(pb_low - ma50_val) / ma50_val * 100.0 if ma50_val > 0 else 99.0

        if sr_diff_pct <= 1.2 and pb_low >= box_high * 0.988:
            sr_flip_hit = True
            pullback_type = f"전고점 S/R Flip 칼지지(오차 {sr_diff_pct:.2f}%)"
            new_strategies.append(f"돌파후 박스상단 S/R Flip 리테스트(거래량 {dryup_ratio*100:.0f}%로 건조)")
        elif ma20_diff <= 1.2:
            ma20_hit = True
            pullback_type = "5분봉 MA 20선 지지 리테스트"
        elif ma50_diff <= 1.5:
            pullback_type = "5분봉 MA 50선 눌림 후 반등"
        else:
            pullback_type = "단기 자율 눌림 후 반등"

        pb_close = to_float(min_pb_row["close"])
        raw_sl = ((pb_close - (pb_low * 0.992)) / pb_close * 100.0) if pb_close > 0 else 3.0
        retest_sl_pct = max(2, min(6, int(round(raw_sl))))

    # [신규 6] 숏스퀴즈 발화
    short_squeeze = bool((coin["funding_rate"] <= 0.01) and (oi_ignition_change >= 1.5 or oi_pre_change >= 3.0))
    if short_squeeze:
        new_strategies.append(f"숏스퀴즈 발화(펀딩 {coin['funding_rate']:+.4f}% + 발화OI +{oi_ignition_change:.1f}%)")

    if not new_strategies:
        new_strategies.append("단독 대량거래량 모멘텀 돌파(수급 집중)")

    return {
        "symbol": symbol,
        "change_24h": coin["change_24h"],
        "last_price": coin["last_price"],
        "turnover_m": coin["turnover_24h"] / 1_000_000.0,
        "oi_m": coin["oi_value"] / 1_000_000.0,
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
        "stealth_oi": stealth_oi,
        "bb_squeeze_break": bb_squeeze_break,
        "spring_reversal": spring_reversal,
        "pre_vol_pocket": pre_vol_pocket,
        "sr_flip_hit": sr_flip_hit,
        "ma20_hit": ma20_hit,
        "short_squeeze": short_squeeze,
        "retest_sl_pct": retest_sl_pct,
        "new_strategies": new_strategies
    }


def build_report(results):
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")
    lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 신규 전략 발굴]",
        f"🕒 기준 시각: {now_str} (순수 코인 상위 5개)\n"
    ]

    num_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    for idx, r in enumerate(results):
        emoji = num_emojis[idx] if idx < len(num_emojis) else f"{idx + 1}."
        ticker_short = r["symbol"].replace("USDT", "")
        align_str = "✅ 완전 정배열(5>20>50>100>200)" if r["full_align"] else (
            "✅ 단기 정배열(5>20>50)" if r["short_align"] else "⚠️ 비정배열(역배열/혼조 돌파)"
        )
        obv_str = "✅ 우상향 선행매집" if r["obv_accum"] else "➖ 중립/횡보"

        lines.append(
            f"{emoji} ${ticker_short} ({r['symbol']} | +{r['change_24h']:.2f}% | 현재가 {r['last_price']:g} USDT)\n"
            f"• 거래대금: {r['turnover_m']:.1f}M | 미결제약정(OI): {r['oi_m']:.1f}M | 펀딩비: {r['funding_rate']:+.4f}%\n"
            f"• [초동 발화점]: {r['ign_time_kst']} (발화가 {r['ign_price']:g} → 고점까지 +{r['pump_pct']:.1f}% 폭등)\n"
            f"• [발화 직전 컨펌된 초동 전략]:\n"
            f"  - 5분봉 이평선: [{align_str}]\n"
            f"  - 사전 매집: 직전 2시간 변동폭 {r['box_range_pct']:.1f}% 응축 | OBV [{obv_str}]\n"
            f"  - 초동 수급: 평소 대비 거래량 [{r['vol_mult']:.1f}배 폭발]\n"
            f"  - 돌파 후 패턴: {r['pullback_type']} (거래량 {r['dryup_ratio']*100:.0f}% 수준 건조)\n"
            f"• [🔍 포착된 신규 알파 전략]:\n"
            f"  - {' / '.join(r['new_strategies'])}\n"
        )

    n = len(results)
    cnt_short_align = sum(1 for r in results if r["short_align"])
    cnt_full_align = sum(1 for r in results if r["full_align"])
    cnt_obv = sum(1 for r in results if r["obv_accum"])
    cnt_stealth_oi = sum(1 for r in results if r["stealth_oi"])
    cnt_bb_vwap = sum(1 for r in results if r["bb_squeeze_break"])
    cnt_vol_pocket = sum(1 for r in results if r["pre_vol_pocket"])
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
    w_obv = 20 if (cnt_obv >= 3 or cnt_vol_pocket >= 2) else 15
    w_oi = 25 if (cnt_stealth_oi >= 2 or cnt_squeeze >= 2) else 15
    w_bb_vwap = 20 if cnt_bb_vwap >= 2 else 15
    w_retest = max(0, 100 - (w_ma + w_obv + w_oi + w_bb_vwap))

    lines.append(
        "━━━━━━━━━━━━━━━━━━\n"
        "💡 [내 전략(scanner.py) 업데이트용 상세 명세서 (제미나이 복붙용)]\n\n"
        "[1. 기존 vs 신규 전략 적중률 비교]\n"
        f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {cnt_short_align}종목 (완전정배열 {cnt_full_align}종목)\n"
        f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {cnt_obv}종목\n"
        f"• [신규] OI 선행매집(발화전 1h OI 급증): {n}종목 중 {cnt_stealth_oi}종목 (평균 {avg_oi_pre:+.1f}%)\n"
        f"• [신규] BB스퀴즈 + VWAP 장대양봉 돌파: {n}종목 중 {cnt_bb_vwap}종목\n"
        f"• [신규] 발화 직전 거래량 극건조 포켓(<=35%): {n}종목 중 {cnt_vol_pocket}종목\n"
        f"• [신규] 비정배열 유동성 스윕(Spring) 반전: {n}종목 중 {cnt_spring}종목\n"
        f"• [신규] 숏스퀴즈(펀딩<=0.01% & OI급증): {n}종목 중 {cnt_squeeze}종목\n"
        f"• [신규] 돌파 후 타점: S/R Flip(전고점 지지) {cnt_sr_flip}종목 | MA20 지지 {cnt_ma20}종목\n\n"
        "[2. scanner.py 파라미터 튜닝 권장값]\n"
        f"• VOL_MULTIPLIER_MIN = {rec_vol_mult} (초동 평균 {avg_vol_mult:.1f}x / 최소 {min_vol_mult:.1f}x)\n"
        f"• PRE_BOX_RANGE_MAX_PCT = {rec_box_max}% (직전 2시간 변동폭 중앙값 {med_box_range:.1f}%)\n"
        f"• PULLBACK_DRYUP_MAX_RATIO = {rec_dryup_max} (발화봉 대비 눌림 거래량 {int(rec_dryup_max*100)}% 이하)\n"
        "• PRE_VOL_POCKET_MAX_RATIO = 0.35 (발화 직전 5캔들 최저거래량 / vol_ma20)\n"
        "• OI_PRE_SURGE_MIN_PCT = 3.0% (발화 전 1시간 OI 증가율)\n\n"
        "[3. scanner.py 추가할 신규 파이썬 조건식]\n"
        f"• cond_stealth_oi = (oi_change_1h_pct >= 3.0) and (box_range_2h_pct <= {rec_box_max})\n"
        "• cond_bb_vwap = (bbw_percentile_4h <= 35) and (close > vwap48) and (body_ratio >= 0.65)\n"
        f"• cond_vol_pocket = (pre5_min_vol / vol_ma20 <= 0.35) and (vol_mult >= {rec_vol_mult})\n"
        "• cond_spring = (not ma_short_aligned) and (lower_wick_ratio_max6 >= 1.5) and (close > ma20)\n"
        f"• cond_sr_flip_retest = (abs(pullback_low - prev_box_high) / prev_box_high <= 0.012) and (pullback_vol_ratio <= {rec_dryup_max})\n\n"
        "[4. 100점 만점 스코어링 배점 조정안]\n"
        f"• 이평선 정배열(5>20>50) 또는 Spring 반전: {w_ma}점\n"
        f"• 박스권 응축 + OBV 우상향 + 발화전 거래량 포켓: {w_obv}점\n"
        f"• 선물 OI 선행 급증(+3%↑) & 중립/음수 펀딩비: {w_oi}점\n"
        f"• BB 스퀴즈 돌파 + VWAP 안착 + 거래량({rec_vol_mult}x↑): {w_bb_vwap}점\n"
        f"• S/R Flip(전고점) 또는 MA20 리테스트 + 거래량 건조: {w_retest}점 (총 100점, 85점↑ 알림)\n\n"
        "[5. 타점 / 손절(SL) / 4단계 익절(TP) 기준]\n"
        f"• 권장 손절폭(SL): -{rec_sl_pct}% (정수형, 리테스트 저점 이탈 기준)\n"
        f"• 권장 4단계 익절(TP): 1차 +{tp1_pct}% (1:2 R:R) / 2차 +{tp2_pct}% / 3차 +{tp3_pct}% / 4차 +{tp4_pct}%"
    )

    return "\n".join(lines)


def send_telegram_message(text):
    print(text)
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("텔레그램 환경변수가 설정되지 않았습니다.")
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
            requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": chunk}, timeout=10)
            time.sleep(0.4)
        except Exception as e:
            print(f"Telegram 전송 오류: {e}")


def main():
    # 1. 바이비트 Futures 상승률 상위 5개 순수 코인 즉시 추출
    top5 = get_top5_bybit_gainers()
    if not top5:
        send_telegram_message("⚠️ 바이비트 상승률 상위 5개 종목 조회에 실패했습니다.")
        return

    # 2. 추출된 5개 종목만 심층 분석
    results = []
    for coin in top5:
        try:
            res = analyze_coin(coin)
            if res:
                results.append(res)
        except Exception as e:
            print(f"{coin.get('symbol')} 분석 오류: {e}")
        time.sleep(0.3)

    if not results:
        send_telegram_message("⚠️ 상위 5개 종목의 5분봉 데이터를 불러오지 못했습니다.")
        return

    # 3. 브리핑 및 스캐너 업데이트 명세서 전송
    report = build_report(results)
    send_telegram_message(report)


if __name__ == "__main__":
    main()
