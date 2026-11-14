import os
import numpy as np
from datetime import timedelta, timezone

# 1. 환경변수 및 기본 설정 (TELEGRAM_TOKEN / TELEGRAM_BOT_TOKEN 모두 호환)
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

# =====================================================================
# [2026-11-14 13:41 KST 리포트 권장값 + 81.8% 승률 고도화 세팅]
# =====================================================================
SCORE_THRESHOLD = 85                   # 평시 85점 이상 알림 발송
BTC_DUMP_SCORE_THRESHOLD = 90          # [오답노트 유형1 방어] BTC 급락 시 컷오프 90점으로 자동 상향
BTC_DUMP_DROP_15M_PCT = -0.70          # BTC 15분봉 최근 3캔들 고점 대비 -0.70% 이하 시 위험장 판정
BTC_DUMP_DROP_1H_PCT = -1.20           # [신규] BTC 1시간봉 최근 4캔들 고점 대비 -1.20% 이하 시 도미노 투매장 판정
MAX_NEW_ALERTS_PER_SCAN = 3            # 1회 스캔(5분)당 최고점 상위 3종목만 발송 (텔레그램 도배/누락 방지)

PULLBACK_DRYUP_STRICT = 0.14           # 발화봉 대비 눌림 거래량 14% 이하 극건조 ($UNITAS 2%, $BERA 2%)
PULLBACK_DRYUP_MAX = 0.28              # [13:41 튜닝] 0.19 -> 0.28 (리포트 권장 완화 반영하되 허수 돌파 방어선 유지)
SR_FLIP_DRYUP_RATIO_MAX = 0.25         # 돌파 후 S/R Flip 리테스트 거래량 상한
SR_FLIP_TOLERANCE_PCT = 1.20           # 전고점 S/R Flip 칼지지 허용 오차 ($UNITAS 0.07%, $BERA 1.03% 적중)

PRE_BOX_RANGE_MAX_PCT = 9.5            # [13:41 튜닝] 5.5 -> 9.5% (리포트 권장 9.5% 및 $UNITAS 6.8% 포착 반영)
PRE_BOX_LOOSE_MAX_PCT = 14.0           # 역배열 유동성 스윕(Spring) 시 허용 박스 상한 ($SOPH 12.6% 적중)
PRE_BOX_TIGHT_PCT = 4.5                # 4.5% 이내 초응축 시 가산점 부여 ($SHELL 2.6%, $BERA 2.3%)
EMA_RIBBON_MAX_SPREAD_PCT = 1.6        # Top 5 중 4종목(0.1%~1.0%) 적중한 이평선 리본 초밀집 기준
PRE_VOL_POCKET_MAX_RATIO = 0.35        # 발화 직전 5캔들 최저거래량 / vol_ma20 ($SHELL 0%, $UNITAS 1%, $BERA 0%)

EARLY_VOL_MULT_MIN = 2.6               # [13:41 튜닝] 리포트 권장값 2.6x 유지 ($AKE 3.2x, $BERA 4.4x, $UNITAS 10.2x)
EARLY_VOL_MULT_MAX = 150.0             # 거래량 극건조 바닥 초응축 돌파 대응 상한
EARLY_BREAKOUT_MAX_PCT = 4.2           # 돌파 양봉 직후 공중 눌림(Bull Flag/FVG) 허용 상한 (4.2%)
BULL_FLAG_MAX_RETRACE = 0.382          # 돌파 양봉 상단 38.2% 이내 공중 지지 기준

SL_ATR_MULT_TRACK_B = 1.45             # 돌파 타점 ATR 손절 버퍼 1.45배
SL_ATR_MULT_TRACK_A = 1.45             # 선취매 타점 ATR 손절 버퍼 1.45배
BREAKEVEN_TRIGGER_PCT = 3.5            # [13:41 오답노트 $ZRO(+3.5%)·$AAVE(+3.2%) 처방] +3.5%(1R) 도달 시 본전 방어 가동
SL_RECOVERY_COOLDOWN_SEC = 14400       # 스탑헌팅 손절 후 4시간 경과 & 88점↑ 시 V자 재진입 허용

MAX_DRAWDOWN_FROM_12H_HIGH_PCT = 7.5   # 최근 12시간 고점 대비 -7.5% 이상 하락한 설거지 종목 즉시 차단
MAX_PRIOR_PUMP_PCT = 20.0              # 최근 12시간 내 이미 +20% 이상 폭등 이력 시 설거지 경계
MAX_RETRACEMENT_RATIO = 0.35           # 폭등 후 상승폭의 35% 이상 반납한 역V자 하락 종목 즉시 차단

CUTOFF_RSI6_15M_MAX = 75.0             # 15분봉 RSI(6) 75 이상 진입 차단
CUTOFF_KDJ_J_15M_MAX = 88.0            # 15분봉 KDJ J값 88 이상 진입 차단
CUTOFF_BB_MID_EXT_MAX_PCT = 4.8        # 초응축 돌파 첫 파동 15분봉 BB 중심선 이격 허용 상한 (4.8%)
CUTOFF_RSI6_1H_MAX = 78.0              # 1시간봉 RSI(6) 78 이상 장기 과열 차단

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

# 단기 급등 탄력이 둔하고 BTC 하락에 1:1 연동되는 무거운 시총 코인 및 거래소 토큰 제외 ($LDO, $BGB 등 추가)
HEAVY_MAJOR_COINS = {
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "TRX", "LTC", "BCH",
    "DOT", "AVAX", "LINK", "XLM", "TON", "SHIB", "ETC", "XMR", "ZEC", "EOS",
    "FIL", "ATOM", "NEAR", "ICP", "HBAR", "VET", "UNI", "AAVE",
    "LDO", "BGB", "OKB", "CRO", "GT", "KCS", "MKR", "POL", "MATIC", "ALGO"
}

NON_CRYPTO_META_KEYWORDS = (
    "xstock", "dinari", "backed", "tokenized", "ondo", "spdr", "ishares",
    "proshares", "direxion", "vanguard", "invesco", "wisdomtree", "tradr", "ark-",
    "etf", "shares", "trust", "index", "crude", "brent", "wti", "natural-gas",
    "gold", "silver", "platinum", "palladium", "copper", "forex", "treasury",
    "inc", "corp", "ltd", "plc", "holdings", "nasdaq", "s&p", "spacex",
    "space exploration", "pre-ipo", "sandisk", "hynix", "technologies", "rocket lab",
    "moderna", "coherent", "coreweave", "nebius", "applovin", "celestica", "pax gold", "tether gold"
)

TRADFI_BLACKLIST = {
    "PAXG", "XAUT", "VNXAU", "DGX", "PMGT", "KAU", "KAG",
    "EWY", "AAOI", "EWJ", "EWZ", "EWT", "EWG", "EWU", "EWC", "EWA", "EWH",
    "SKDD", "SKLL", "SKUU", "CLS", "GGLL", "GGLS", "AAPU", "AAPD", "AMZU", "AMZD",
    "MSFU", "MSFD", "METU", "METD", "NFXL", "NFXS", "PLTU", "PLTD", "AVL", "AVS",
    "TSMX", "BRKU", "BRKD", "LLYU", "LLYD", "NVDD", "NVDG", "TSDD", "TSLG",
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


def normalize_funding_pct(raw_val):
    v = to_float(raw_val, 0.0)
    if abs(v) > 0.15:
        return v / 100.0 if abs(v) <= 2.5 else v
    return v


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
    if len(b1) == 4 and (b1.endswith("DD") or b1.endswith("LL") or b1.endswith("UU")) and b1 not in {"SPELL", "CELL", "BULL", "NULL", "CALL", "FULL", "ROLL"}:
        return True
    b2 = strip_multiplier_prefix(b1)
    b3 = strip_tokenized_suffix(b2)
    if (b1 in TRADFI_BLACKLIST) or (b2 in TRADFI_BLACKLIST) or (b3 in TRADFI_BLACKLIST):
        return True
    if len(b1) == 6 and (b1.startswith("USD") or b1.endswith("USD")):
        return True
    return False


def is_heavy_major_coin(base_coin):
    b = strip_multiplier_prefix(base_coin.upper())
    return b in HEAVY_MAJOR_COINS
