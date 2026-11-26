import os
import math
from datetime import timedelta, timezone

# =============================================================================
# [단일 설정 원천 · Single Source of Truth]
# =============================================================================
KST = timezone(timedelta(hours=9))

STATE_FILE = "bot_state.json"
ANALYZER_HISTORY_FILE = "analyzer_history.json"
MARKET_CAP_CACHE_FILE = "market_cap_cache.json"
DNA_PROFILE_FILE = "top5_dna_profile.json"
POSTMORTEM_DB_FILE = "postmortem_db.json"

# 진입 커트라인 (신규 88점 · 손절 후 재진입 90점 고정)
SCORE_THRESHOLD = 88
REENTRY_SCORE_THRESHOLD = 90

# 기존 리스크/손절 정책 유지 (단기 -6% / 장기 -20%)
FIXED_SL_PCT = 6
LONGTERM_SL_PCT = 20

# 실전 원장 정산용 왕복 테이커 수수료 + 슬리피지 보정치 (%)
ROUNDTRIP_FEE_SLIPPAGE_PCT = 0.15

# [1안 · 균형 완화] BTC 3단계 마켓 레짐 게이트 임계치 (%)
# - 2단계(역추세 E·F + 장기 D 허용): 15m <= -1.20% OR 1H <= -2.20%
# - 3단계(신규 전면 차단): 15m <= -2.20% OR 1H <= -3.50%
BTC_DUMP_DROP_15M_PCT = -1.20
BTC_DUMP_DROP_1H_PCT = -2.20
BTC_CRASH_15M_PCT = -2.20
BTC_CRASH_1H_PCT = -3.50

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

# 선택 사항: 본인 소유 Cloudflare Worker 등 프라이빗 Bybit 릴레이 URL (Secrets 등록 시 최우선 사용)
BYBIT_PROXY_URL = (os.getenv("BYBIT_PROXY_URL") or "").strip()

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

# 초고시총 메이저 코인 제외 목록 (알트코인 급등 포착 집중)
HEAVY_MAJOR_COINS = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "TRX",
    "BCH", "LTC", "WBTC", "WETH", "STETH", "WSTETH",
}

# 타 프로젝트 시총 충돌 빈발 단글자·중복 티커 (시총 캐시·조회 중립 처리 대상)
AMBIGUOUS_MC_TICKERS = {
    "H", "B", "M", "S", "Q", "W", "G", "T", "F", "C", "V", "K", "LIT", "RAIN",
}

# 정밀화된 TradFi / 주식 / ETF / 원자재 블랙리스트 (단글자·일반 크립토 충돌 티커 제거)
TRADFI_BLACKLIST = {
    "SAMSUNG", "HYNIX", "SKHYNIX", "SKHY", "SSNLF", "HYUNDAI", "POSCO", "NAVER", "KAKAO",
    "CELLTRION", "LGCHEM", "LGES", "COUPANG", "KOSPI", "KOSDAQ",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOG", "GOOGL", "AMZN", "META", "NFLX", "AMD",
    "INTC", "TSM", "ASML", "PLTR", "COIN", "MSTR", "HOOD", "BABA", "BIDU",
    "PDD", "NIO", "XPEV", "SPY", "QQQ", "IWM", "DIA", "TLT", "ARKK", "SMH",
    "SOXL", "SOXS", "KORU", "TSLL", "NVDL", "NVDX", "TQQQ", "SQQQ", "CONL", "MSTU", "MSTX",
    "GLD", "SLV", "USO", "UNG", "XAU", "XAG", "XPD", "XPT", "WTI", "BRENT", "OIL",
    "NATGAS", "COPPER", "GOLD", "SILVER", "PAXG", "XAUT", "XAUM",
    "SPX", "NDX", "NAS100", "US500", "US30", "DJI", "VIX", "NIKKEI", "FTSE",
    "DAX", "CAC", "HSI", "BRK", "JPM", "BAC", "WFC", "UNH", "JNJ",
    "LLY", "NVO", "PFE", "MRK", "ABBV", "WMT", "COST", "SBUX", "NKE", "DIS",
    "XOM", "CVX", "LMT", "RTX", "UBER", "ABNB", "SNOW", "CRWD", "PANW", "SMCI",
    "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MRVL",
    "SONY", "TOYOTA", "SOFTBANK", "TENCENT", "XIAOMI", "BYD",
    "CRCL", "RKLB", "ASTS", "IONQ", "RGTI", "OKLO", "HIMS", "RDDT", "VST", "CEG", "GEV",
    "DELL", "HPE", "WDC", "STX", "ANET", "VRT", "ALAB", "EWY", "EWJ", "FXI", "KWEB",
}


def to_float(value, default=0.0):
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def normalize_funding_pct(raw_pct):
    """펀딩비(%) 이상치 방어 및 소수점 4자리 정규화"""
    v = to_float(raw_pct, 0.0)
    if v > 5.0:
        v = 5.0
    elif v < -5.0:
        v = -5.0
    return round(v, 4)


def is_tradfi_ticker(base):
    b = str(base or "").upper().strip()
    if not b:
        return True
    return b in TRADFI_BLACKLIST


def is_heavy_major_coin(base):
    b = str(base or "").upper().strip()
    return b in HEAVY_MAJOR_COINS
# === END OF FILE ===
