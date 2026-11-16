import os
import numpy as np
from datetime import timedelta, timezone

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
HISTORY_FILE = "analyzer_history.json"

SCORE_THRESHOLD = 85
BTC_DUMP_SCORE_THRESHOLD = 90
BTC_DUMP_DROP_15M_PCT = -0.70
BTC_DUMP_DROP_1H_PCT = -1.20
MAX_NEW_ALERTS_PER_SCAN = 3

# [-6% 고정 손절 & 1R(+6%) 본전 방어 원칙 고정]
FIXED_SL_PCT = 6
BREAKEVEN_TRIGGER_PCT = 6.0
SL_RECOVERY_COOLDOWN_SEC = 14400

TP1_MIN_PCT = 8.5
TP2_MIN_PCT = 16.0
TP3_MIN_PCT = 24.0
TP3_MAX_CAP_PCT = 50.0

TP1_FIB_MULT = 1.85
TP2_FIB_MULT = 2.95
TP3_FIB_MULT = 4.10

LEADER_MIN_TURNOVER_M = 25.0
LEADER_MIN_PRIOR_PUMP_PCT = 35.0
LEADER_FIB_RETRACE_MIN = 0.38
LEADER_FIB_RETRACE_MAX = 0.62
LEADER_MAX_FUNDING_PCT = 0.0150

PULLBACK_DRYUP_STRICT = 0.14
PULLBACK_DRYUP_MAX = 0.19
SR_FLIP_DRYUP_RATIO_MAX = 0.22
SR_FLIP_TOLERANCE_PCT = 1.25

PRE_BOX_RANGE_MAX_PCT = 5.5
PRE_BOX_LOOSE_MAX_PCT = 14.0
PRE_BOX_TIGHT_PCT = 4.0
EMA_RIBBON_MAX_SPREAD_PCT = 1.6
PRE_VOL_POCKET_MAX_RATIO = 0.35

EARLY_VOL_MULT_MIN = 2.6
EARLY_VOL_MULT_MAX = 150.0
EARLY_BREAKOUT_MAX_PCT = 4.2
BULL_FLAG_MAX_RETRACE = 0.382

MAX_DRAWDOWN_FROM_12H_HIGH_PCT = 7.5
MAX_PRIOR_PUMP_PCT = 22.0
MAX_CHANGE_24H_FOR_EARLY_TRACK = 26.0
MAX_MA112_15M_EXT_PCT = 13.5
MAX_RETRACEMENT_RATIO = 0.35

CUTOFF_RSI6_15M_MAX = 75.0
CUTOFF_KDJ_J_15M_MAX = 88.0
CUTOFF_BB_MID_EXT_MAX_PCT = 4.8
CUTOFF_RSI6_1H_MAX = 78.0

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
}

# [$PEPE 등 시총이 무거운 준메이저·대형 밈코인 제외 목록]
HEAVY_MAJOR_COINS = {
    "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "TRX", "LTC", "BCH",
    "DOT", "AVAX", "LINK", "XLM", "TON", "SHIB", "ETC", "XMR", "ZEC", "EOS",
    "FIL", "ATOM", "NEAR", "ICP", "HBAR", "VET", "UNI", "AAVE",
    "LDO", "BGB", "OKB", "CRO", "GT", "KCS", "MKR", "POL", "MATIC", "ALGO",
    "RENDER", "RNDR", "JUP", "ONE", "ZRO", "ARB", "OP", "SUI", "APT", "SEI",
    "TIA", "INJ", "FET", "TAO", "WLD", "PENDLE", "ENA", "STRK", "ONDO", "QNT",
    "CRV", "SNX", "COMP", "SUSHI", "1INCH", "DYDX", "GMX", "GRT", "SAND", "MANA",
    "AXS", "APE", "GALA", "IMX", "FLOW", "THETA", "EGLD", "XTZ", "NEO", "IOTA", "KAS",
    "WIF", "PEPE", "BONK", "FLOKI", "ORDI", "SATS", "NOT", "BOME", "MEW", "MEME"
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
    b1 = str(base_coin).upper()
    if not b1.isascii():
        return True
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
    b = strip_multiplier_prefix(str(base_coin).upper())
    return b in HEAVY_MAJOR_COINS
