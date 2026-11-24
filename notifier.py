import os, re, io, sys, time, json, shutil, tempfile, subprocess, urllib.parse, requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

try:
    import config as cfg
    TELEGRAM_BOT_TOKEN = getattr(cfg, "TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = getattr(cfg, "TELEGRAM_CHAT_ID", "")
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    BROWSER_HEADERS = getattr(cfg, "BROWSER_HEADERS", {"User-Agent": "Mozilla/5.0"})
except Exception:
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    FIXED_SL_PCT = 6
    KST = timezone(timedelta(hours=9))
    BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0"}

try:
    import api as api_mod
except Exception:
    api_mod = None

if not TELEGRAM_BOT_TOKEN:
    TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
if not TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

SCHEME = "ht" + "tps://"
WATERMARK_HANDLE = "@DnaPyung"
_PLAYWRIGHT_READY = False
LAST_VIDEO_ERROR = ""


def get_last_video_error():
    return LAST_VIDEO_ERROR


def clean_url(raw_url):
    s = str(raw_url or "").strip()
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", s)
    s = s.replace("[", "").replace("]", "")
    if not s.startswith("http"):
        s = SCHEME + s.lstrip("/")
    return s


def to_float(value, default=0.0):
    try:
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except Exception:
        return default


def fmt_price(price):
    p = to_float(price, 0.0)
    if p <= 0:
        return "0.0000"
    return f"{p:,.2f}" if p >= 1000 else (f"{p:.3f}" if p >= 10 else (f"{p:.4f}" if p >= 1 else (f"{p:.5f}" if p >= 0.01 else f"{p:.6f}")))


def _fmt_hold_duration(hold_minutes):
    m = max(1, int(to_float(hold_minutes, 1)))
    if m < 60:
        return f"{m}분"
    h, rem_m = divmod(m, 60)
    if h < 24:
        return f"{h}시간 {rem_m}분" if rem_m > 0 else f"{h}시간"
    d, rem_h = divmod(h, 24)
    return f"{d}일 {rem_h}시간" if rem_h > 0 else f"{d}일"


def _ensure_playwright():
    global _PLAYWRIGHT_READY
    if _PLAYWRIGHT_READY:
        return True
    try:
        import playwright
        _PLAYWRIGHT_READY = True
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "playwright"])
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"])
            _PLAYWRIGHT_READY = True
            return True
        except Exception:
            return False


def fetch_exact_bybit_klines(symbol, interval="60", limit=130, min_rows=10):
    if api_mod is not None and hasattr(api_mod, "fetch_klines"):
        for attempt in range(2):
            df = api_mod.fetch_klines(symbol, str(interval), limit=int(limit))
            if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= min_rows:
                df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", ascending=True).reset_index(drop=True)
                if len(df) >= min_rows:
                    return df
            if attempt == 0:
                time.sleep(0.5)
    return pd.DataFrame()


def _resample_4h_to_12h(df_4h):
    """12H 직접 조회가 일시 실패할 경우 4H 캔들을 12H UTC 버킷(00:00/12:00)으로 자동 리샘플링"""
    if df_4h is None or df_4h.empty or len(df_4h) < 30:
        return pd.DataFrame()
    try:
        tmp = df_4h.copy()
        bucket_ms = 12 * 3600 * 1000
        tmp["bucket"] = (tmp["timestamp"].astype("int64") // bucket_ms) * bucket_ms
        grouped = tmp.groupby("bucket", as_index=False).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        grouped = grouped.rename(columns={"bucket": "timestamp"})
        return grouped.sort_values("timestamp", ascending=True).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def build_briefing_message(res, now_str):
    base = res.get("base_coin") or res.get("base") or ""
    symbol = res.get("symbol", "")
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
    title = res.get("track_title", "🐋 [장기 포지션 · 세력 대시세 매집 초입]" if is_long else "🚀 [타점 포착]")
    horizon = res.get("position_horizon", "장기 포지션 (1개월 이상 홀딩 예상)" if is_long else "단기·스윙 포지션")
    score = res.get("score", 90)
    price = to_float(res.get("price"))
    chg = to_float(res.get("change_24h"))
    turn_m = to_float(res.get("turnover_m"))
    oi_m = to_float(res.get("oi_m"))
    oi_ratio = to_float(res.get("oi_ratio"))
    fund = to_float(res.get("funding_pct"))
    htf = res.get("htf_status", "1H 중심선지지")
    rsi15 = to_float(res.get("rsi6_15m"))
    kdj15 = to_float(res.get("kdj_j_15m"))
    ppo15 = res.get("ppo_15m")
    ppo_str = f" · PPO {to_float(ppo15):+.2f}%" if ppo15 is not None else ""

    e_low = to_float(res.get("entry_low"), price * 0.9970)
    e_high = to_float(res.get("entry_high"), price * 1.0015)
    sl_p = to_float(res.get("stop_price"))
    sl_pct = int(round(to_float(res.get("sl_pct"), 20 if is_long else FIXED_SL_PCT)))
    sl_note = res.get("sl_note", f"장기 고정손절(-{sl_pct}%)" if is_long else f"고정손절(-{sl_pct}%)")

    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]
    tp_labels = res.get("tp_labels") or ["단기 매물대 저항", "1H 주요 저항구간", "Fib 1.618 확장목표"]
    reasons = res.get("reasons") or []
    trail_drop = 10 if is_long else 4

    lines = [
        f"{title} ${base} ({symbol})",
        f"📌 포지션 구분: 【{horizon}】 | 종합점수: {score}점",
        f"🕒 포착 시각: {now_str}",
        f"• 현재가: {fmt_price(price)} USDT (24h {chg:+.2f}%)",
        f"• 수급: 거래대금 {turn_m:.1f}M | OI {oi_m:.1f}M (비중 {oi_ratio:.0f}%) | 펀딩 {fund:+.4f}%",
        f"• 멀티프레임: {htf} | 15m RSI6 {rsi15:.1f} · J {kdj15:.1f}{ppo_str}",
        "",
        f"🎯 [{'장기 포지션 매매 전략 가이드' if is_long else '매매 전략 가이드'}]",
        f"• 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT",
        f"• 손절가(SL): {fmt_price(sl_p)} USDT (-{sl_pct}% | {sl_note})",
        f"• 1차 구간(TP1·전량 홀딩): {fmt_price(tps[0])} USDT (+{to_float(tp_pcts[0]):.1f}% | {tp_labels[0]} · 익절 없이 100% 홀딩)",
        f"• 2차 구간(TP2·전량 홀딩): {fmt_price(tps[1])} USDT (+{to_float(tp_pcts[1]):.1f}% | {tp_labels[1]} · 돌파시 방어선 TP1({fmt_price(tps[0])}) 상향 고정)",
        f"• 3차 목표(TP3·50% 익절): {fmt_price(tps[2])} USDT (+{to_float(tp_pcts[2]):.1f}% | {tp_labels[2]} · 방어선 TP2 상향 & 잔여 50% 고점 -{trail_drop}% 트레일링)",
    ]

    if reasons:
        lines.append("\n🔍 [포착 근거]")
        for r in reasons:
            lines.append(f"• {r}")

    if is_long:
        lines.append(
            "\n⚠️ [장기 진입 전 10초 체크]: 30일 내 대형 클리프 언락(Tokenomist) · 바이낸스 유의종목(Monitoring Tag) 여부 확인 권장"
        )

    return "\n".join(lines)


def build_x_hook_line(res):
    track_type = str(res.get("track_type", "TRACK_A"))
    if track_type == "TRACK_D" or res.get("is_long_term"):
        sub_s = str(res.get("sub_strategy", "")).upper()
        if "D6" in sub_s or "스프링" in sub_s:
            return " 신저가 개미털기(스프링) 후 다개월 바닥선 거래량 동반 재탈환 포착"
        return " 세력 4H·1D 매집 후 대시세 1파동 초입 포착"
    r_text = " ".join(str(x) for x in (res.get("reasons") or []))
    if "재진입" in str(res.get("track_title", "")) or "재탈환" in r_text:
        return "개미털기 바닥 확인 후 거래량 실리며 V자 반등 시도중"
    if track_type == "TRACK_H":
        return "최근 급등 대장주 Top 5 발사 직전 DNA 패턴과 일치하는 초입 포착"
    if track_type == "TRACK_E":
        return "24H 저점 유동성 스윕(개미털기) 직후 강한 반전 매수세 유입중"
    if track_type == "TRACK_F":
        return "비트코인 흔들림 속에서도 역행 강세 보이는 상대강도 대장주 포착"
    if track_type == "TRACK_G":
        return "깊은 음수 펀딩비와 OI 집중 속 숏스퀴즈 강제청산 초입 포착"
    if track_type == "TRACK_C":
        return "급등 대장주 핵심 눌림목 지지 후 2차 파동 시도중"
    if track_type == "TRACK_B":
        return "거래량이 살아나면서 24H 전고점 돌파 시도중"
    if "응축" in r_text or "스퀴즈" in r_text:
        return "박스권 에너지 응축 후 거래량 살아나며 추세 전환중"
    if "음수 펀딩" in r_text:
        return "숏 스퀴즈 수급 유입되며 단기 추세 반등 시도중"
    return "눌림목 거래량 건조 후 수급 살아나며 반등 시도중"


def build_x_post_message(res):
    base = res.get("base_coin") or res.get("base") or ""
    symbol = res.get("symbol", "")
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
    price = to_float(res.get("price"))
    e_low = to_float(res.get("entry_low"), price * 0.9970)
    e_high = to_float(res.get("entry_high"), price * 1.0015)
    sl_p = to_float(res.get("stop_price"))
    sl_pct = int(round(to_float(res.get("sl_pct"), 20 if is_long else FIXED_SL_PCT)))

    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]

    p1 = round(to_float(tp_pcts[0], 8.0), 1)
    p2 = round(to_float(tp_pcts[1], 15.0), 1)
    p3 = round(to_float(tp_pcts[2], 22.0), 1)
    hook_line = build_x_hook_line(res)
    footer_tags = "#코인시황 #코인신호 #코인 #재테크"

    if is_long:
        return (
            "【장기 포지션】1개월이상 홀딩예상\n"
            "🔺️상승 예상 코인 검색기에 포착된 종목😎\n"
            f"⚠️{hook_line}⚠️\n\n"
            f"${base} ({symbol})\n\n"
            f"- 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT\n"
            f"- SL: {fmt_price(sl_p)} (-{sl_pct}%)\n"
            f"- tp1: {fmt_price(tps[0])}(+{p1:g}%)\n"
            f"- tp2: {fmt_price(tps[1])}(+{p2:g}%)\n"
            f"- tp3: {fmt_price(tps[2])}(+{p3:g}%)\n\n"
            f"#{base} 어디까지 오를거 같나요?\n"
            "보유중이신 분들 계신가요?\n\n"
            f"{footer_tags}"
        )

    return (
        "🔺️상승 예상 코인 검색기에 포착된 종목😎\n"
        f"⚠️{hook_line}⚠️\n\n"
        f"${base} ({symbol})\n\n"
        f"- 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT\n"
        f"- SL: {fmt_price(sl_p)} (-{sl_pct}%)\n"
        f"- tp1: {fmt_price(tps[0])}(+{p1:g}%)\n"
        f"- tp2: {fmt_price(tps[1])}(+{p2:g}%)\n"
        f"- tp3: {fmt_price(tps[2])}(+{p3:g}%)\n\n"
        f"#{base} 어디까지 오를거 같나요?\n"
        "보유중이신 분들 계신가요?\n\n"
        f"{footer_tags}"
    )


def build_x_intent_reply_markup(x_post_text, button_label="🚀 X(트위터) 앱에 본문 자동 입력하기"):
    encoded_text = urllib.parse.quote(str(x_post_text or ""), safe="")
    intent_url = clean_url(f"{SCHEME}x.com/intent/post?text={encoded_text}")
    return {"inline_keyboard": [[{"text": button_label, "url": intent_url}]]}


def _render_tradingview_lightweight_chart(df, res, chart_tf="60"):
    if not _ensure_playwright():
        return None
    symbol = res.get("symbol", "")
    base = res.get("base_coin") or res.get("base") or symbol.replace("USDT", "")
    score = res.get("score", 90)
    track_type = res.get("track_type", "TRACK_A")
    is_long = bool(res.get("is_long_term") or track_type == "TRACK_D")
    sub_s = str(res.get("sub_strategy", "")).upper()

    latest_row = df.iloc[-1]
    price = to_float(latest_row["close"], to_float(res.get("price")))
    open_last, high_last, low_last = (
        to_float(latest_row["open"], price),
        to_float(latest_row["high"], price),
        to_float(latest_row["low"], price),
    )

    orig_entry = to_float(res.get("price"), price)
    ratio_adj = price / orig_entry if orig_entry > 0 else 1.0
    sl_pct = int(round(to_float(res.get("sl_pct"), 20 if is_long else FIXED_SL_PCT)))
    sl_p = to_float(res.get("stop_price"), price * (1.0 - sl_pct / 100.0)) * ratio_adj
    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]

    tp1_p, tp2_p, tp3_p = to_float(tps[0]) * ratio_adj, to_float(tps[1]) * ratio_adj, to_float(tps[2]) * ratio_adj
    p1, p2, p3 = round(to_float(tp_pcts[0], 8.0), 1), round(to_float(tp_pcts[1], 15.0), 1), round(to_float(tp_pcts[2], 22.0), 1)

    candle_min_low = to_float(df["low"].min(), price * 0.97)
    candle_max_high = to_float(df["high"].max(), price * 1.02)
    low_piv = min(candle_min_low, price * 0.988)
    high_piv = max(candle_max_high, price * 1.010)
    if high_piv <= low_piv:
        low_piv, high_piv = price * 0.985, price * 1.015

    core_span = max(high_piv - low_piv, price * 0.018)
    up_comp = (
        float(np.clip((core_span * 0.38) / max(tp3_p - high_piv, price * 0.02), 0.04, 1.0))
        if tp3_p > high_piv
        else 1.0
    )
    down_comp = (
        float(np.clip((core_span * 0.14) / max(low_piv - sl_p, price * 0.008), 0.06, 1.0))
        if sl_p < low_piv
        else 1.0
    )

    def _fwd(val):
        v = to_float(val, price)
        return (high_piv + (v - high_piv) * up_comp) if v > high_piv else ((low_piv - (low_piv - v) * down_comp) if v < low_piv else v)

    candles_js, vol_js, ma5_js, ma9_js, ma20_js = [], [], [], [], []
    for _, row in df.iterrows():
        ts_sec = int(to_float(row["timestamp"]) // 1000) + (9 * 3600)
        o_v, h_v, l_v, c_v, v_v = to_float(row["open"]), to_float(row["high"]), to_float(row["low"]), to_float(row["close"]), to_float(row["volume"])
        candles_js.append({"time": ts_sec, "open": _fwd(o_v), "high": _fwd(h_v), "low": _fwd(l_v), "close": _fwd(c_v)})
        vol_js.append({"time": ts_sec, "value": v_v, "color": "rgba(38, 166, 154, 0.38)" if c_v >= o_v else "rgba(239, 83, 80, 0.38)"})
        if pd.notna(row.get("ma5")):
            ma5_js.append({"time": ts_sec, "value": _fwd(row["ma5"])})
        if pd.notna(row.get("ma9")):
            ma9_js.append({"time": ts_sec, "value": _fwd(row["ma9"])})
        if pd.notna(row.get("ma20")):
            ma20_js.append({"time": ts_sec, "value": _fwd(row["ma20"])})

    tf_str = str(chart_tf).upper().strip()
    step_sec = {"720": 43200, "240": 14400, "60": 3600, "15": 900}.get(tf_str, 43200 if is_long else 3600)
    whitespace_js = [{"time": candles_js[-1]["time"] + (i * step_sec)} for i in range(1, 6)]

    high_idx = int(df["high"].astype(float).idxmax())
    low_idx = int(df["low"].astype(float).idxmin())
    fib_high = to_float(df["high"].iloc[high_idx], candle_max_high)
    fib_low = to_float(df["low"].iloc[low_idx], candle_min_low)
    fib_span = fib_high - fib_low

    fib_levels_js = []
    fib_start_time = candles_js[0]["time"]
    fib_end_time = whitespace_js[-1]["time"]
    fib_dir_tag = "FIB"

    if fib_span > price * 0.005 and len(candles_js) >= 10:
        start_idx = min(min(high_idx, low_idx), max(0, len(candles_js) // 2))
        fib_start_time = candles_js[start_idx]["time"]
        is_up_impulse = low_idx <= high_idx
        fib_dir_tag = "FIB (L→H)" if is_up_impulse else "FIB (H→L)"

        fib_specs = [
            (0.0,   "0",     "#787b86", 1, 1, 1),
            (0.236, "0.236", "#9598a1", 1, 3, 2),
            (0.382, "0.382", "#4dd0e1", 1, 3, 4),
            (0.5,   "0.5",   "#ffd54f", 1.2, 3, 5),
            (0.618, "0.618", "#ff9800", 1.3, 3, 6),
            (0.786, "0.786", "#ce93d8", 1, 3, 3),
            (1.0,   "1",     "#787b86", 1, 1, 1),
        ]
        for ratio, r_lbl, color, l_width, l_style, prio in fib_specs:
            real_p = (fib_high - fib_span * ratio) if is_up_impulse else (fib_low + fib_span * ratio)
            fib_levels_js.append({
                "ratio": ratio,
                "label": r_lbl,
                "priceStr": fmt_price(real_p),
                "yVal": _fwd(real_p),
                "color": color,
                "lineWidth": l_width,
                "lineStyle": l_style,
                "priority": prio,
            })

    y_bot_bound = min(_fwd(sl_p), _fwd(candle_min_low)) - core_span * 0.04
    y_top_bound = max(_fwd(tp3_p), _fwd(candle_max_high)) + core_span * 0.05

    kst_str = datetime.fromtimestamp(to_float(latest_row["timestamp"]) / 1000.0, tz=KST).strftime("%Y-%m-%d %H:%M KST")
    tf_name = {"720": "12H", "240": "4H", "60": "1H", "15": "15m"}.get(tf_str, "12H" if is_long else "1H")
    tf_label = f"{tf_name} · LONG-TERM" if is_long else f"{tf_name} · BYBIT"
    track_badge = {
        "TRACK_A": "TRACK A · TTM Squeeze",
        "TRACK_B": "TRACK B · 24H Breakout",
        "TRACK_C": "TRACK C · Deep Pullback",
        "TRACK_D": "LONG-TERM · Track D6 Spring" if ("D6" in sub_s or "스프링" in sub_s) else "LONG-TERM · Track D Macro",
        "TRACK_E": "TRACK E · SFP Reversal",
        "TRACK_F": "TRACK F · BTC RS Leader",
        "TRACK_G": "TRACK G · Short Squeeze",
        "TRACK_H": "TRACK H · Top5 DNA",
    }.get(track_type, track_type)
    decimals = 6 if price < 0.01 else (5 if price < 1 else (4 if price < 10 else (3 if price < 1000 else 2)))
    min_move = 10 ** (-decimals)
    lw_script_url = clean_url(f"{SCHEME}unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js")
    lw_fallback_url = clean_url(f"{SCHEME}cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js")

    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="{lw_script_url}"></script>
<script>if(typeof LightweightCharts==='undefined'){{document.write('<scr'+'ipt src="{lw_fallback_url}"></scr'+'ipt>');}}</script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#131722;font-family:-apple-system,BlinkMacSystemFont,Roboto,sans-serif;color:#d1d4dc;width:1160px;height:680px;overflow:hidden}}
.tv-wrap{{width:1160px;height:680px;background:#131722;border:1px solid #2a2e39;display:flex;flex-direction:column}}
.tv-topbar{{height:42px;background:#181c27;border-bottom:1px solid #2a2e39;display:flex;align-items:center;justify-content:space-between;padding:0 16px}}
.tv-left,.tv-right{{display:flex;align-items:center;gap:10px}}
.tv-symbol{{font-size:16px;font-weight:800;color:#fff}}
.tv-tf-badge{{background:#2a2e39;color:#d1d4dc;font-size:11.5px;font-weight:700;padding:3px 8px;border-radius:4px}}
.tv-ohlc{{display:flex;gap:10px;font-size:12px;font-weight:600}}
.tv-val-up{{color:#26a69a}}
.tv-score-pill{{background:#2962ff;color:#fff;font-size:12px;font-weight:700;padding:4px 11px;border-radius:5px}}
#chart-box{{position:relative;flex:1;width:100%;height:638px}}
.tv-ma-legend{{position:absolute;top:10px;left:14px;z-index:20;display:flex;gap:14px;font-size:12px;font-weight:700;background:rgba(19,23,34,0.86);padding:5px 11px;border-radius:4px;border:1px solid #2a2e39}}
#fib-overlay{{position:absolute;inset:0;pointer-events:none;z-index:12;overflow:hidden}}
.fib-band{{position:absolute;pointer-events:none}}
.fib-lbl{{position:absolute;font-size:10.5px;font-weight:700;line-height:1;padding:2px 6px;border-radius:3px;background:rgba(19,23,34,0.84);border:1px solid rgba(255,255,255,0.08);white-space:nowrap;letter-spacing:0.2px}}
</style></head>
<body>
<div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tv-left">
      <span class="tv-symbol">${base}USDT.P</span>
      <span class="tv-tf-badge">{tf_label}</span>
      <div class="tv-ohlc">
        <span>O <span class="tv-val-up">{fmt_price(open_last)}</span></span>
        <span>H <span class="tv-val-up">{fmt_price(high_last)}</span></span>
        <span>L <span class="tv-val-up">{fmt_price(low_last)}</span></span>
        <span>C <span class="tv-val-up">{fmt_price(price)}</span></span>
      </div>
    </div>
    <div class="tv-right">
      <span>{kst_str}</span>
      <span class="tv-score-pill">{track_badge} · {score} pts</span>
    </div>
  </div>
  <div id="chart-box">
    <div class="tv-ma-legend" id="ma-legend">
      <span style="color:#00e5ff">● MA(5) {fmt_price(df["ma5"].iloc[-1])}</span>
      <span style="color:#ffd166">● MA(9) {fmt_price(df["ma9"].iloc[-1])}</span>
      <span style="color:#2962ff">● MA(20) {fmt_price(df["ma20"].iloc[-1])}</span>
      <span style="color:#ff9800">▤ {fib_dir_tag}</span>
    </div>
    <div id="fib-overlay"></div>
  </div>
</div>
<script>
const lowPiv={low_piv},highPiv={high_piv},upComp={up_comp},downComp={down_comp},decimals={decimals};
function invScale(y){{if(y>highPiv)return highPiv+(y-highPiv)/Math.max(upComp,1e-6);if(y<lowPiv)return lowPiv-(lowPiv-y)/Math.max(downComp,1e-6);return y;}}
function fmtReal(val){{const p=invScale(val);if(p<=0)return '0.0000';if(p>=1000)return p.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});return p.toFixed(decimals);}}
const chart=LightweightCharts.createChart(document.getElementById('chart-box'),{{
  width:1160,height:638,
  layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},
  grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},
  rightPriceScale:{{borderColor:'#2a2e39',scaleMargins:{{top:0.07,bottom:0.12}},entireTextOnly:true}},
  timeScale:{{borderColor:'#2a2e39',timeVisible:true,rightOffset:5,barSpacing:9.8}},
  localization:{{priceFormatter:fmtReal}}
}});
const candleSeries=chart.addCandlestickSeries({{
  upColor:'#26a69a',downColor:'#ef5350',
  borderUpColor:'#26a69a',borderDownColor:'#ef5350',
  wickUpColor:'#26a69a',wickDownColor:'#ef5350',
  priceLineVisible:false,lastValueVisible:false,
  priceFormat:{{type:'custom',minMove:{min_move},formatter:fmtReal}}
}});
const cData={json.dumps(candles_js)};
candleSeries.setData([...cData,...{json.dumps(whitespace_js)}]);
candleSeries.setMarkers([{{time:cData[cData.length-1].time,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'ENTRY',size:2}}]);

const fibLevels={json.dumps(fib_levels_js)};
const fibStartTime={fib_start_time}, fibEndTime={fib_end_time};
fibLevels.forEach(fl=>{{
  const fs=chart.addLineSeries({{
    color:fl.color,
    lineWidth:fl.lineWidth,
    lineStyle:fl.lineStyle,
    priceLineVisible:false,
    lastValueVisible:false,
    crosshairMarkerVisible:false
  }});
  fs.setData([{{time:fibStartTime,value:fl.yVal}},{{time:fibEndTime,value:fl.yVal}}]);
}});

chart.addLineSeries({{color:'#00e5ff',lineWidth:1.6,priceLineVisible:false,lastValueVisible:false}}).setData({json.dumps(ma5_js)});
chart.addLineSeries({{color:'#ffd166',lineWidth:1.6,priceLineVisible:false,lastValueVisible:false}}).setData({json.dumps(ma9_js)});
chart.addLineSeries({{color:'#2962ff',lineWidth:2.0,priceLineVisible:false,lastValueVisible:false}}).setData({json.dumps(ma20_js)});

const volSeries=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'vol_scale',priceLineVisible:false,lastValueVisible:false}});
chart.priceScale('vol_scale').applyOptions({{scaleMargins:{{top:0.90,bottom:0.0}}}});
volSeries.setData({json.dumps(vol_js)});

[
  {{price:{_fwd(tp3_p)},color:'#ffd166',title:'TP3 (+{p3:g}%)',style:2,width:2}},
  {{price:{_fwd(tp2_p)},color:'#06d6a0',title:'TP2 (+{p2:g}%)',style:2,width:2}},
  {{price:{_fwd(tp1_p)},color:'#4cc9f0',title:'TP1 (+{p1:g}%)',style:2,width:2}},
  {{price:{_fwd(sl_p)},color:'#ef476f',title:'SL (-{sl_pct}%)',style:2,width:2}}
].forEach(t=>candleSeries.createPriceLine({{price:t.price,color:t.color,lineWidth:t.width,lineStyle:t.style,axisLabelVisible:true,title:t.title}}));

chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,priceLineVisible:false,lastValueVisible:false}}).setData([
  {{time:cData[0].time,value:{y_bot_bound}}},
  {{time:cData[cData.length-1].time,value:{y_top_bound}}}
]);
chart.timeScale().fitContent();

requestAnimationFrame(()=>requestAnimationFrame(()=>{{
  try{{
    const ov=document.getElementById('fib-overlay');
    const maBox=document.getElementById('ma-legend');
    const maRight=maBox ? (maBox.offsetLeft + maBox.offsetWidth + 10) : 390;
    const maBottom=maBox ? (maBox.offsetTop + maBox.offsetHeight + 6) : 44;

    if(ov && fibLevels.length>0){{
      const psWidth=(chart.priceScale('right') && chart.priceScale('right').width) ? chart.priceScale('right').width() : 88;
      const plotRight=Math.max(200, 1160 - (psWidth || 88));
      const rawX=chart.timeScale().timeToCoordinate(fibStartTime);
      const xStart=(rawX!==null && isFinite(rawX)) ? Math.max(8, Math.min(plotRight - 180, rawX)) : 12;

      const byRatio={{}};
      fibLevels.forEach(fl=>{{
        const y=candleSeries.priceToCoordinate(fl.yVal);
        if(y!==null && isFinite(y)) byRatio[fl.label]=y;
      }});

      const drawBand=(r1,r2,bg)=>{{
        if(byRatio[r1]!==undefined && byRatio[r2]!==undefined){{
          const topY=Math.min(byRatio[r1],byRatio[r2]);
          const hY=Math.abs(byRatio[r2]-byRatio[r1]);
          if(hY>=2){{
            const b=document.createElement('div');
            b.className='fib-band';
            b.style.left=xStart+'px';
            b.style.width=Math.max(20, plotRight - xStart)+'px';
            b.style.top=topY+'px';
            b.style.height=hY+'px';
            b.style.background=bg;
            ov.appendChild(b);
          }}
        }}
      }};
      drawBand('0.5','0.618','rgba(255, 152, 0, 0.09)');
      drawBand('0.382','0.5','rgba(77, 208, 225, 0.045)');

      const placedY=[];
      const sortedFibs=[...fibLevels].sort((a,b)=>b.priority-a.priority);
      sortedFibs.forEach(fl=>{{
        const y=byRatio[fl.label];
        if(y===undefined || y<12 || y>592) return;
        if(placedY.some(py=>Math.abs(py-y)<14)) return;
        placedY.push(y);
        const tag=document.createElement('div');
        tag.className='fib-lbl';
        let lblLeft = xStart + 6;
        if(y < maBottom && lblLeft < maRight){{
          lblLeft = Math.min(plotRight - 135, maRight);
        }}
        tag.style.left=lblLeft+'px';
        tag.style.top=Math.max(8, y-8)+'px';
        tag.style.color=fl.color;
        tag.textContent=fl.label+' ('+fl.priceStr+')';
        ov.appendChild(tag);
      }});
    }}
  }}catch(e){{}}
  window.__CHART_DONE=true;
}}));
</script></body></html>"""

    try:
        from playwright.sync_api import sync_playwright
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            tmp_html_path = f.name
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(viewport={"width": 1160, "height": 680}, device_scale_factor=1.5)
            page.goto(f"file://{tmp_html_path}", wait_until="load", timeout=25000)
            page.wait_for_function("window.__CHART_DONE === true", timeout=10000)
            page.wait_for_timeout(200)
            png_bytes = page.screenshot(type="png")
            browser.close()
        try:
            os.remove(tmp_html_path)
        except Exception:
            pass
        return png_bytes
    except Exception:
        return None


def generate_signal_chart_bytes(res):
    symbol, price = res.get("symbol", ""), to_float(res.get("price"))
    if not symbol or price <= 0:
        return None
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")

    actual_tf = "720" if is_long else "60"
    df = pd.DataFrame()

    if is_long:
        df = fetch_exact_bybit_klines(symbol, interval="720", limit=130, min_rows=20)
        if df.empty:
            df_4h_raw = fetch_exact_bybit_klines(symbol, interval="240", limit=300, min_rows=30)
            if not df_4h_raw.empty:
                df_resamp = _resample_4h_to_12h(df_4h_raw)
                if not df_resamp.empty and len(df_resamp) >= 20:
                    df = df_resamp
                    actual_tf = "720"
                else:
                    df = df_4h_raw
                    actual_tf = "240"
        if df.empty:
            df = fetch_exact_bybit_klines(symbol, interval="60", limit=130, min_rows=20)
            actual_tf = "60"
    else:
        df = fetch_exact_bybit_klines(symbol, interval="60", limit=130, min_rows=20)
        if df.empty:
            df = fetch_exact_bybit_klines(symbol, interval="15", limit=130, min_rows=20)
            actual_tf = "15"

    if df is None or df.empty or len(df) < 20:
        return None

    df = df.copy()
    df["ma5"] = df["close"].rolling(5, min_periods=2).mean()
    df["ma9"] = df["close"].rolling(9, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()
    return _render_tradingview_lightweight_chart(df.iloc[-96:].reset_index(drop=True), res, chart_tf=actual_tf)


def generate_profit_replay_video_bytes(close_info):
    """
    [최종 전량 익절 18초 HD MP4 리플레이 영상 자동 제작 엔진 v4]
    - 규격: X 최적화 1280×720 (16:9), 12fps × 216프레임 = 정확히 18.0초, H.264/yuv420p/+faststart
    - 타임라인:
      * 0.0초 ~ 4.0초 (0~47): 썸네일 완성 이미지 4초 고정 (진입봉→익절봉 파란 곡선 화살표 + 진입/익절 날짜·가격 + 최종 수익률%)
      * 4.0초 ~ 5.0초 (48~59): 진입 봉 시점(ENTRY) 리셋 및 출발 준비 (1초)
      * 5.0초 ~ 15.0초 (60~179): 진입봉→익절봉 실시간 캔들 상승 리플레이 (10초)
      * 15.0초 ~ 18.0초 (180~215): 마지막 완성 엔딩 3초 (화살표 드로잉 0.8초 + 완성 고정 2.2초)
    - 정지 구간(썸네일/대기/엔딩) 고속 복사 최적화 → 18초 영상도 기존 10초와 동일한 속도로 렌더링
    - 차트 및 EXIT 블루 콜아웃 우측 정렬 보정 + TP/SL 라벨 수직 자동 회피
    """
    global LAST_VIDEO_ERROR
    LAST_VIDEO_ERROR = ""

    if not shutil.which("ffmpeg"):
        LAST_VIDEO_ERROR = "ffmpeg 미설치 (scan.yml 확인 필요)"
        print(f"[리플레이 영상 실패] {LAST_VIDEO_ERROR}")
        return None
    if not _ensure_playwright():
        LAST_VIDEO_ERROR = "Playwright/Chromium 준비 실패"
        print(f"[리플레이 영상 실패] {LAST_VIDEO_ERROR}")
        return None

    symbol = str(close_info.get("symbol", "")).upper().strip()
    base = close_info.get("base_coin") or symbol.replace("USDT", "")
    entry_p = to_float(close_info.get("entry_price"))
    if not symbol or entry_p <= 0:
        LAST_VIDEO_ERROR = f"유효하지 않은 심볼/진입가({symbol}, {entry_p})"
        return None

    created_at = int(to_float(close_info.get("created_at"), 0))
    closed_at = int(to_float(close_info.get("closed_at"), time.time()))
    if created_at <= 0 or closed_at <= created_at:
        created_at = closed_at - 3600

    hold_minutes = max(1, int(to_float(close_info.get("hold_minutes"), (closed_at - created_at) // 60)))
    hold_hours = (closed_at - created_at) / 3600.0

    # 보유 기간별 타임프레임 규칙 엄격 적용
    if hold_hours < 12.0:
        tf_code, tf_label, step_sec, fetch_lim = "15", "15m", 900, 180
    elif hold_hours < 72.0:
        tf_code, tf_label, step_sec, fetch_lim = "60", "1H", 3600, 180
    else:
        tf_code, tf_label, step_sec, fetch_lim = "240", "4H", 14400, 260

    df_raw = fetch_exact_bybit_klines(symbol, interval=tf_code, limit=fetch_lim, min_rows=10)
    if df_raw.empty and tf_code != "15":
        tf_code, tf_label, step_sec = "15", "15m", 900
        df_raw = fetch_exact_bybit_klines(symbol, interval="15", limit=200, min_rows=10)
    if df_raw.empty:
        LAST_VIDEO_ERROR = f"{symbol} {tf_label} 캔들 데이터 API 조회 실패"
        print(f"[리플레이 영상 실패] {LAST_VIDEO_ERROR}")
        return None

    df = df_raw.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", ascending=True).reset_index(drop=True)
    df["ma5"] = df["close"].rolling(5, min_periods=2).mean()
    df["ma9"] = df["close"].rolling(9, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()

    created_ms = created_at * 1000
    closed_ms = closed_at * 1000
    ts_arr = df["timestamp"].astype("int64").values

    entry_matches = np.where(ts_arr <= created_ms)[0]
    entry_abs = int(entry_matches[-1]) if len(entry_matches) > 0 else max(0, len(df) - 20)

    exit_matches = np.where(ts_arr <= closed_ms)[0]
    exit_abs = int(exit_matches[-1]) if len(exit_matches) > 0 else (len(df) - 1)
    if exit_abs < entry_abs:
        exit_abs = min(len(df) - 1, entry_abs)

    post_count = (exit_abs - entry_abs) + 1
    pre_bars = 26 if post_count <= 90 else 18
    slice_start = max(0, entry_abs - pre_bars)
    df_win = df.iloc[slice_start : exit_abs + 1].reset_index(drop=True)
    if len(df_win) < 4:
        LAST_VIDEO_ERROR = f"{symbol} 리플레이 구간 캔들 부족({len(df_win)}개)"
        print(f"[리플레이 영상 실패] {LAST_VIDEO_ERROR}")
        return None

    entry_idx = max(0, entry_abs - slice_start)
    exit_idx = len(df_win) - 1

    is_long = bool(close_info.get("is_long_term") or close_info.get("track_type") == "TRACK_D")
    track_type = str(close_info.get("track_type", "TRACK_A"))
    close_type = str(close_info.get("close_type", "TP3_TRAILING"))
    sl_pct = int(round(to_float(close_info.get("sl_pct"), 20 if is_long else FIXED_SL_PCT)))
    sl_p = to_float(close_info.get("initial_stop_price"), entry_p * (1.0 - sl_pct / 100.0))
    tps = close_info.get("tps") or [entry_p * 1.08, entry_p * 1.15, entry_p * 1.22]
    tp1_p = to_float(tps[0], entry_p * 1.08)
    tp2_p = to_float(tps[1], entry_p * 1.15)
    tp3_p = to_float(tps[2], entry_p * 1.22)
    p1 = round((tp1_p - entry_p) / entry_p * 100.0, 1)
    p2 = round((tp2_p - entry_p) / entry_p * 100.0, 1)
    p3 = round((tp3_p - entry_p) / entry_p * 100.0, 1)

    max_high_val = max(to_float(close_info.get("max_high_price"), entry_p), to_float(df_win["high"].max(), entry_p))
    exit_p = to_float(close_info.get("exit_price"), to_float(df_win["close"].iloc[-1], entry_p))
    total_realized_pct = round(to_float(close_info.get("realized_weighted_pct"), max(p1, (exit_p - entry_p) / entry_p * 100.0)), 2)
    hold_str = _fmt_hold_duration(hold_minutes)

    entry_dt_str = datetime.fromtimestamp(created_at, tz=KST).strftime("%Y-%m-%d %H:%M")
    exit_dt_str = datetime.fromtimestamp(closed_at, tz=KST).strftime("%Y-%m-%d %H:%M")

    win_low = min(to_float(df_win["low"].min(), entry_p * 0.94), sl_p, entry_p)
    win_high = max(to_float(df_win["high"].max(), entry_p * 1.15), tp2_p, exit_p)
    if close_type == "TP3_TRAILING" or max_high_val >= tp3_p * 0.92:
        win_high = max(win_high, tp3_p)
    y_span = max(win_high - win_low, entry_p * 0.05)
    y_min_anchor = win_low - (y_span * 0.16)
    y_max_anchor = win_high + (y_span * 0.24)

    candles_js, vol_js, ma5_js, ma9_js, ma20_js, kst_strs = [], [], [], [], [], []
    last_ts_sec = -1
    for _, row in df_win.iterrows():
        ts_ms = to_float(row["timestamp"])
        ts_sec = int(ts_ms // 1000) + (9 * 3600)
        if ts_sec <= last_ts_sec:
            ts_sec = last_ts_sec + step_sec
        last_ts_sec = ts_sec
        o_v, h_v, l_v, c_v, v_v = (
            to_float(row["open"]), to_float(row["high"]),
            to_float(row["low"]), to_float(row["close"]), to_float(row["volume"]),
        )
        candles_js.append({"time": ts_sec, "open": o_v, "high": h_v, "low": l_v, "close": c_v})
        vol_js.append({"time": ts_sec, "value": v_v, "color": "rgba(38, 166, 154, 0.38)" if c_v >= o_v else "rgba(239, 83, 80, 0.38)"})
        ma5_js.append({"time": ts_sec, "value": to_float(row["ma5"])} if pd.notna(row.get("ma5")) else None)
        ma9_js.append({"time": ts_sec, "value": to_float(row["ma9"])} if pd.notna(row.get("ma9")) else None)
        ma20_js.append({"time": ts_sec, "value": to_float(row["ma20"])} if pd.notna(row.get("ma20")) else None)
        kst_strs.append(datetime.fromtimestamp(ts_ms / 1000.0, tz=KST).strftime("%Y-%m-%d %H:%M KST"))

    # 우측 여백을 5캔들로 압축해 캔들과 EXIT 지점을 조금 더 우측으로 배치
    ws_tail = [{"time": candles_js[-1]["time"] + (i * step_sec)} for i in range(1, 6)]
    decimals = 6 if entry_p < 0.01 else (5 if entry_p < 1 else (4 if entry_p < 10 else (3 if entry_p < 1000 else 2)))
    min_move = 10 ** (-decimals)
    lw_script_url = clean_url(f"{SCHEME}unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js")
    lw_fallback_url = clean_url(f"{SCHEME}cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js")

    pos_badge = "LONG-TERM REPLAY" if is_long else f"{track_type.replace('_', ' ')} REPLAY"
    price_diff_abs = abs(exit_p - entry_p)
    price_diff_prefix = "+" if exit_p >= entry_p else "-"
    exit_headline = f"{price_diff_prefix}{fmt_price(price_diff_abs)} (+{total_realized_pct:.2f}%) in {hold_str}"

    # 12fps × 216프레임 = 정확히 18.0초
    fps = 12
    total_frames = 216
    thumb_end = 48    # 0.0s ~ 4.0s (4.0초 썸네일 고정)
    play_start = 60   # 4.0s ~ 5.0s (1.0초 진입 시점 준비)
    play_end = 179    # 5.0s ~ 15.0s (10.0초 본격 상승 리플레이)
    outro_static_start = 190  # 15.0s ~ 15.8s 화살표 드로잉 / 15.8s ~ 18.0s 완성 고정

    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<script src="{lw_script_url}"></script>
<script>if(typeof LightweightCharts==='undefined'){{document.write('<scr'+'ipt src="{lw_fallback_url}"></scr'+'ipt>');}}</script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#131722;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans KR',sans-serif;color:#d1d4dc;width:1280px;height:720px;overflow:hidden}}
.tv-wrap{{width:1280px;height:720px;background:#131722;display:flex;flex-direction:column}}
.tv-topbar{{height:46px;background:#181c27;border-bottom:1px solid #2a2e39;display:flex;align-items:center;justify-content:space-between;padding:0 18px}}
.tv-left,.tv-right{{display:flex;align-items:center;gap:11px}}
.tv-symbol{{font-size:18px;font-weight:800;color:#fff;letter-spacing:0.2px}}
.tv-tf-badge{{background:#2a2e39;color:#e0e3eb;font-size:12px;font-weight:700;padding:3px 9px;border-radius:4px}}
.tv-ohlc{{display:flex;gap:10px;font-size:12.5px;font-weight:600;color:#9598a1}}
.tv-val-up{{color:#26a69a}}
.tv-val-dn{{color:#ef5350}}
.tv-replay-pill{{background:#2962ff;color:#fff;font-size:12px;font-weight:800;padding:4px 11px;border-radius:5px}}
.tv-handle-pill{{background:rgba(255,255,255,0.07);border:1px solid #363a45;color:#f0f3fa;font-size:12.5px;font-weight:800;padding:4px 11px;border-radius:5px;letter-spacing:0.3px}}
#chart-box{{position:relative;flex:1;width:1280px;height:650px;overflow:hidden}}
.tv-ma-legend{{position:absolute;top:12px;left:16px;z-index:20;display:flex;gap:14px;font-size:12.5px;font-weight:700;background:rgba(19,23,34,0.86);padding:5px 12px;border-radius:5px;border:1px solid #2a2e39}}
.live-hud{{position:absolute;top:12px;right:110px;z-index:25;display:flex;align-items:center;gap:10px}}
.pnl-pill{{background:rgba(19,23,34,0.90);border:1.5px solid #26a69a;border-radius:8px;padding:7px 14px;display:flex;flex-direction:column;align-items:flex-end;min-width:142px;box-shadow:0 6px 18px rgba(0,0,0,0.4)}}
.pnl-label{{font-size:10.5px;font-weight:700;color:#9598a1;letter-spacing:0.5px}}
.pnl-val{{font-size:22px;font-weight:900;color:#00e676;line-height:1.15}}
.stage-pill{{background:rgba(41,98,255,0.22);border:1px solid #4cc9f0;color:#e0f7fa;font-size:12px;font-weight:800;padding:7px 12px;border-radius:7px;display:none}}

#result-overlay{{position:absolute;inset:0;pointer-events:none;z-index:30;opacity:0}}
#arc-svg{{position:absolute;inset:0;width:1280px;height:650px;overflow:visible}}
.tv-callout{{
  position:absolute;
  background:linear-gradient(135deg,#2962ff 0%,#1e53e5 100%);
  color:#ffffff;
  border-radius:8px;
  padding:8px 13px;
  box-shadow:0 8px 24px rgba(0,0,0,0.52), 0 0 0 1px rgba(255,255,255,0.14) inset;
  white-space:nowrap;
}}
.tv-callout-exit .co-main{{font-size:16px;font-weight:900;letter-spacing:-0.2px;line-height:1.25;color:#ffffff}}
.tv-callout-exit .co-sub{{font-size:12.5px;font-weight:700;color:#e3ecff;margin-top:3px;display:flex;align-items:center;gap:8px}}
.tv-callout-entry .co-main{{font-size:14px;font-weight:900;line-height:1.2;color:#ffffff}}
.tv-callout-entry .co-sub{{font-size:12px;font-weight:600;color:#dbeafe;margin-top:2px}}
.co-caret-down{{
  position:absolute;bottom:-7px;width:0;height:0;
  border-left:7px solid transparent;border-right:7px solid transparent;border-top:7px solid #2258ee;
}}
.co-caret-up{{
  position:absolute;top:-7px;width:0;height:0;
  border-left:7px solid transparent;border-right:7px solid transparent;border-bottom:7px solid #2962ff;
}}

.tv-footer{{height:24px;background:#0e121b;border-top:1px solid #222632;display:flex;align-items:center;justify-content:space-between;padding:0 16px;font-size:10.5px;color:#787b86;font-weight:500}}
.prog-wrap{{width:160px;height:4px;background:#222632;border-radius:2px;overflow:hidden}}
.prog-fill{{height:100%;width:0%;background:linear-gradient(90deg,#2962ff,#00e676)}}
</style></head>
<body>
<div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tv-left">
      <span class="tv-symbol">${base}USDT.P</span>
      <span class="tv-tf-badge">{tf_label} · REPLAY</span>
      <div class="tv-ohlc">
        <span>O <span id="oh-o" class="tv-val-up">{fmt_price(entry_p)}</span></span>
        <span>H <span id="oh-h" class="tv-val-up">{fmt_price(entry_p)}</span></span>
        <span>L <span id="oh-l" class="tv-val-up">{fmt_price(entry_p)}</span></span>
        <span>C <span id="oh-c" class="tv-val-up">{fmt_price(entry_p)}</span></span>
      </div>
    </div>
    <div class="tv-right">
      <span id="top-time" style="font-size:12px;color:#b2b5be">{kst_strs[exit_idx]}</span>
      <span class="tv-replay-pill">{pos_badge}</span>
      <span class="tv-handle-pill">{WATERMARK_HANDLE}</span>
    </div>
  </div>
  <div id="chart-box">
    <div class="tv-ma-legend">
      <span style="color:#00e5ff">● MA(5) <span id="lg-m5">-</span></span>
      <span style="color:#ffd166">● MA(9) <span id="lg-m9">-</span></span>
      <span style="color:#2962ff">● MA(20) <span id="lg-m20">-</span></span>
    </div>
    <div class="live-hud" id="live-hud">
      <div id="stage-pill" class="stage-pill"></div>
      <div class="pnl-pill" id="pnl-box">
        <span class="pnl-label">LIVE RETURN</span>
        <span class="pnl-val" id="pnl-val">+0.0%</span>
      </div>
    </div>

    <div id="result-overlay">
      <svg id="arc-svg" viewBox="0 0 1280 650">
        <defs>
          <filter id="arc-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#2962ff" flood-opacity="0.55"/>
          </filter>
        </defs>
        <line id="exit-guide" x1="0" y1="0" x2="0" y2="0" stroke="#3b82f6" stroke-width="1.6" stroke-dasharray="4,3" opacity="0"/>
        <path id="arc-path" d="" fill="none" stroke="#2962ff" stroke-width="3.6" stroke-linecap="round" filter="url(#arc-glow)"/>
        <circle id="dot-entry" cx="0" cy="0" r="5.5" fill="#2962ff" stroke="#ffffff" stroke-width="2"/>
        <circle id="dot-exit" cx="0" cy="0" r="5.5" fill="#2962ff" stroke="#ffffff" stroke-width="2"/>
        <polygon id="arc-head" points="0,0 -13,-6.5 -10,0 -13,6.5" fill="#3b82f6" stroke="#ffffff" stroke-width="0.6"/>
      </svg>

      <div id="callout-exit" class="tv-callout tv-callout-exit">
        <div class="co-main">{exit_headline}</div>
        <div class="co-sub"><span>EXIT {fmt_price(exit_p)}</span><span>🕒 {exit_dt_str}</span></div>
        <div id="caret-exit" class="co-caret-down"></div>
      </div>

      <div id="callout-entry" class="tv-callout tv-callout-entry">
        <div id="caret-entry" class="co-caret-up"></div>
        <div class="co-main">ENTRY {fmt_price(entry_p)}</div>
        <div class="co-sub">🕒 {entry_dt_str}</div>
      </div>
    </div>
  </div>
  <div class="tv-footer">
    <span>※ 본 영상은 스캐너 포착 진입가·TP·방어선/트레일링 기준을 활용한 시뮬레이션 리플레이 영상이며, 실제 체결 결과와 다를 수 있습니다.</span>
    <div class="prog-wrap"><div class="prog-fill" id="prog-fill"></div></div>
  </div>
</div>
<script>
const cAll={json.dumps(candles_js)};
const vAll={json.dumps(vol_js)};
const m5All={json.dumps(ma5_js)};
const m9All={json.dumps(ma9_js)};
const m20All={json.dumps(ma20_js)};
const kstAll={json.dumps(kst_strs)};
const wsTail={json.dumps(ws_tail)};
const entryIdx={entry_idx}, exitIdx={exit_idx}, entryP={entry_p}, exitP={exit_p};
const tp1P={tp1_p}, tp2P={tp2_p}, tp3P={tp3_p}, slP={sl_p};
const decimals={decimals};
const yMinAnchor={y_min_anchor}, yMaxAnchor={y_max_anchor};

function fmtP(v){{
  if(!v || v<=0) return '0.0000';
  if(v>=1000) return v.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}});
  return v.toFixed(decimals);
}}

const chart=LightweightCharts.createChart(document.getElementById('chart-box'),{{
  width:1280, height:650,
  layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},
  grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},
  watermark:{{
    visible:true,
    fontSize:82,
    horzAlign:'center',
    vertAlign:'center',
    color:'rgba(209, 212, 220, 0.16)',
    text:'{WATERMARK_HANDLE}',
    fontFamily:'-apple-system,BlinkMacSystemFont,Roboto,sans-serif',
    fontStyle:'bold'
  }},
  rightPriceScale:{{borderColor:'#2a2e39',scaleMargins:{{top:0.05,bottom:0.12}},entireTextOnly:true}},
  timeScale:{{borderColor:'#2a2e39',timeVisible:true,rightOffset:3}},
  localization:{{priceFormatter:fmtP}}
}});

const anchorSeries=chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}});
anchorSeries.setData([{{time:cAll[0].time,value:yMinAnchor}},{{time:wsTail[wsTail.length-1].time,value:yMaxAnchor}}]);

const candleSeries=chart.addCandlestickSeries({{
  upColor:'#26a69a',downColor:'#ef5350',
  borderUpColor:'#26a69a',borderDownColor:'#ef5350',
  wickUpColor:'#26a69a',wickDownColor:'#ef5350',
  priceLineVisible:false,lastValueVisible:false,
  priceFormat:{{type:'custom',minMove:{min_move},formatter:fmtP}}
}});
const ma5Series=chart.addLineSeries({{color:'#00e5ff',lineWidth:1.6,priceLineVisible:false,lastValueVisible:false}});
const ma9Series=chart.addLineSeries({{color:'#ffd166',lineWidth:1.6,priceLineVisible:false,lastValueVisible:false}});
const ma20Series=chart.addLineSeries({{color:'#2962ff',lineWidth:2.0,priceLineVisible:false,lastValueVisible:false}});
const volSeries=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'vol_scale',priceLineVisible:false,lastValueVisible:false}});
chart.priceScale('vol_scale').applyOptions({{scaleMargins:{{top:0.89,bottom:0.0}}}});

candleSeries.createPriceLine({{price:tp3P,color:'#ffd166',lineWidth:2,lineStyle:2,axisLabelVisible:true,title:'TP3 (+{p3:g}%)'}});
candleSeries.createPriceLine({{price:tp2P,color:'#06d6a0',lineWidth:2,lineStyle:2,axisLabelVisible:true,title:'TP2 (+{p2:g}%)'}});
candleSeries.createPriceLine({{price:tp1P,color:'#4cc9f0',lineWidth:2,lineStyle:2,axisLabelVisible:true,title:'TP1 (+{p1:g}%)'}});
candleSeries.createPriceLine({{price:slP,color:'#ef476f',lineWidth:2,lineStyle:2,axisLabelVisible:true,title:'SL (-{sl_pct}%)'}});

candleSeries.setData(cAll.concat(wsTail));
chart.timeScale().fitContent();

const elO=document.getElementById('oh-o'), elH=document.getElementById('oh-h'), elL=document.getElementById('oh-l'), elC=document.getElementById('oh-c');
const elTime=document.getElementById('top-time');
const elM5=document.getElementById('lg-m5'), elM9=document.getElementById('lg-m9'), elM20=document.getElementById('lg-m20');
const elHud=document.getElementById('live-hud'), elPnl=document.getElementById('pnl-val'), elPnlBox=document.getElementById('pnl-box');
const elStage=document.getElementById('stage-pill'), elProg=document.getElementById('prog-fill');
const elOverlay=document.getElementById('result-overlay');
const elArcPath=document.getElementById('arc-path'), elArcHead=document.getElementById('arc-head');
const elExitGuide=document.getElementById('exit-guide');
const elDotEntry=document.getElementById('dot-entry'), elDotExit=document.getElementById('dot-exit');
const elCoExit=document.getElementById('callout-exit'), elCoEntry=document.getElementById('callout-entry');
const elCaretExit=document.getElementById('caret-exit'), elCaretEntry=document.getElementById('caret-entry');

let arcTotalLen=1;
let geometryReady=false;
let showExitGuide=false;

function setupResultOverlayGeometry(){{
  const psW=(chart.priceScale('right') && chart.priceScale('right').width) ? chart.priceScale('right').width() : 88;
  const plotRight=Math.max(400, 1280 - (psW || 88) - 8);
  const totalSlots=cAll.length + wsTail.length;

  let x1=chart.timeScale().timeToCoordinate(cAll[entryIdx].time);
  let x2=chart.timeScale().timeToCoordinate(cAll[exitIdx].time);
  if(x1===null || !isFinite(x1)) x1=40 + (entryIdx / Math.max(1, totalSlots)) * (plotRight - 80);
  if(x2===null || !isFinite(x2)) x2=40 + (exitIdx / Math.max(1, totalSlots)) * (plotRight - 80);

  let y1=candleSeries.priceToCoordinate(entryP);
  let y1Low=candleSeries.priceToCoordinate(cAll[entryIdx].low);
  let y2=candleSeries.priceToCoordinate(exitP);
  let y2High=candleSeries.priceToCoordinate(cAll[exitIdx].high);
  if(y1===null || !isFinite(y1)) y1=480;
  if(y1Low===null || !isFinite(y1Low)) y1Low=y1;
  if(y2===null || !isFinite(y2)) y2=160;
  if(y2High===null || !isFinite(y2High)) y2High=y2;

  y1=Math.max(60, Math.min(585, y1));
  y1Low=Math.max(y1, Math.min(595, y1Low));
  y2=Math.max(76, Math.min(560, y2));
  y2High=Math.max(54, Math.min(y2, y2High));

  const dx=x2 - x1;
  const dy=y1 - y2;
  let cx, cy;
  if(dx >= 45){{
    cx = Math.min(plotRight - 14, x1 + dx * 0.90 + 18);
    cy = Math.min(592, y1 - dy * 0.04 + 12);
  }} else {{
    cx = Math.min(plotRight - 14, Math.max(x1, x2) + 72);
    cy = Math.min(592, y1 - dy * 0.25 + 14);
  }}

  const dStr='M ' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q ' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ' ' + x2.toFixed(1) + ' ' + y2.toFixed(1);
  elArcPath.setAttribute('d', dStr);
  elDotEntry.setAttribute('cx', x1.toFixed(1));
  elDotEntry.setAttribute('cy', y1.toFixed(1));
  elDotExit.setAttribute('cx', x2.toFixed(1));
  elDotExit.setAttribute('cy', y2.toFixed(1));

  arcTotalLen=Math.max(1, elArcPath.getTotalLength());
  elArcPath.style.strokeDasharray=arcTotalLen.toFixed(1);

  // [우측 정렬 보정] EXIT 콜아웃을 우측으로 더 자연스럽게 배치 + 우측 TP/SL 라벨 수직 겹침 자동 회피
  const exitW=elCoExit.offsetWidth || 275;
  const exitH=elCoExit.offsetHeight || 56;
  let exitLeft=Math.max(16, Math.min(plotRight - exitW - 6, x2 - exitW * 0.28));
  let exitTop=Math.min(y2, y2High) - exitH - 18;
  if(exitTop < 48) exitTop = 48;

  const badgeZoneLeft = plotRight - 145;
  const tpYs = [tp3P, tp2P, tp1P, slP]
    .map(p => candleSeries.priceToCoordinate(p))
    .filter(v => v !== null && isFinite(v));

  if(exitLeft + exitW > badgeZoneLeft){{
    for(let pass=0; pass<3; pass++){{
      let hit=false;
      for(const ty of tpYs){{
        if(exitTop < ty + 16 && (exitTop + exitH) > ty - 16){{
          const candUp = ty - 16 - exitH - 6;
          if(candUp >= 48){{
            exitTop = candUp;
            hit = true;
          }}
        }}
      }}
      if(!hit) break;
    }}
  }}

  elCoExit.style.left=exitLeft.toFixed(1) + 'px';
  elCoExit.style.top=exitTop.toFixed(1) + 'px';
  const exitCaretX=Math.max(14, Math.min(exitW - 24, x2 - exitLeft - 7));
  elCaretExit.style.left=exitCaretX.toFixed(1) + 'px';

  // EXIT 박스가 TP 라벨을 피해 위로 올라간 경우 점선 가이드로 EXIT 지점과 깔끔하게 연결
  const caretTipX = exitLeft + exitCaretX + 7;
  const caretTipY = exitTop + exitH + 7;
  if((y2 - caretTipY) > 18){{
    elExitGuide.setAttribute('x1', caretTipX.toFixed(1));
    elExitGuide.setAttribute('y1', caretTipY.toFixed(1));
    elExitGuide.setAttribute('x2', x2.toFixed(1));
    elExitGuide.setAttribute('y2', (y2 - 6).toFixed(1));
    showExitGuide = true;
  }} else {{
    showExitGuide = false;
  }}

  const entryW=elCoEntry.offsetWidth || 185;
  const entryH=elCoEntry.offsetHeight || 50;
  let entryLeft=Math.max(16, Math.min(plotRight - entryW - 6, x1 - entryW * 0.35));
  let entryTop=y1Low + 14;
  if(entryTop + entryH > 634) entryTop = 634 - entryH;
  if(entryTop < exitTop + exitH + 12){{
    entryTop = Math.min(634 - entryH, exitTop + exitH + 14);
  }}
  elCoEntry.style.left=entryLeft.toFixed(1) + 'px';
  elCoEntry.style.top=entryTop.toFixed(1) + 'px';
  const entryCaretX=Math.max(14, Math.min(entryW - 24, x1 - entryLeft - 7));
  elCaretEntry.style.left=entryCaretX.toFixed(1) + 'px';

  geometryReady=true;
}}

function setResultOverlayVisual(overlayOpacity, arrowProgress, showCallouts){{
  if(!geometryReady) setupResultOverlayGeometry();
  elOverlay.style.opacity=overlayOpacity.toFixed(2);
  if(overlayOpacity <= 0.01) return;

  const p=Math.max(0.02, Math.min(1.0, arrowProgress));
  const curLen=arcTotalLen * p;
  elArcPath.style.strokeDashoffset=(arcTotalLen - curLen).toFixed(1);

  const pt=elArcPath.getPointAtLength(curLen);
  const ptPrev=elArcPath.getPointAtLength(Math.max(0, curLen - 3));
  const ang=Math.atan2(pt.y - ptPrev.y, pt.x - ptPrev.x) * 180 / Math.PI;
  elArcHead.setAttribute('transform', 'translate(' + pt.x.toFixed(1) + ',' + pt.y.toFixed(1) + ') rotate(' + ang.toFixed(1) + ')');
  elDotExit.style.opacity=(p >= 0.95 ? '1' : '0');

  const boxAlpha=showCallouts ? Math.min(1.0, overlayOpacity) : 0;
  const exitReady=(p >= 0.85 ? boxAlpha : 0);
  elCoEntry.style.opacity=boxAlpha.toFixed(2);
  elCoExit.style.opacity=exitReady.toFixed(2);
  elExitGuide.setAttribute('opacity', (showExitGuide && exitReady > 0.5) ? '0.72' : '0');
}}

// 18초 타임라인 (12fps × 216프레임):
// [0..47]    0.0s ~ 4.0s  : 썸네일 완성 이미지 고정 (4초)
// [48..59]   4.0s ~ 5.0s  : 진입 봉 시점 리셋 및 출발 준비 (1초)
// [60..179]  5.0s ~ 15.0s : 본격 캔들 상승 리플레이 재생 (10초)
// [180..215] 15.0s ~ 18.0s: 마지막 완성 엔딩 (화살표 드로잉 0.8초 + 완성 고정 2.2초)
const THUMB_END={thumb_end};
const PLAY_START={play_start};
const PLAY_END={play_end};
const postSteps=Math.max(1, exitIdx - entryIdx);

window.__renderFrame=function(f){{
  let showIdx=entryIdx;
  let frac=1.0;
  let isThumb=(f < THUMB_END);
  let isPrep=(f >= THUMB_END && f < PLAY_START);
  let isPlay=(f >= PLAY_START && f <= PLAY_END);
  let isOutro=(f > PLAY_END);

  if(isThumb){{
    showIdx=exitIdx;
    frac=1.0;
    elProg.style.width='0.0%';
  }} else if(isPrep){{
    showIdx=entryIdx;
    frac=1.0;
    elProg.style.width='0.0%';
  }} else if(isPlay){{
    const progress=(f - PLAY_START) / (PLAY_END - PLAY_START);
    elProg.style.width=(progress * 100).toFixed(1) + '%';
    const pos=entryIdx + progress * postSteps;
    showIdx=Math.min(exitIdx, Math.floor(pos));
    frac=Math.min(1.0, Math.max(0.15, pos - Math.floor(pos)));
    if(f === PLAY_END){{ showIdx=exitIdx; frac=1.0; }}
  }} else {{
    showIdx=exitIdx;
    frac=1.0;
    elProg.style.width='100.0%';
  }}

  const dispC=[];
  for(let i=0;i<=showIdx;i++){{
    const src=cAll[i];
    if(i === showIdx && i > entryIdx && frac < 0.999){{
      const cNow=src.open + (src.close - src.open) * frac;
      const hNow=Math.max(src.open, cNow, src.open + (src.high - src.open) * frac);
      const lNow=Math.min(src.open, cNow, src.open + (src.low - src.open) * frac);
      dispC.push({{time:src.time,open:src.open,high:hNow,low:lNow,close:cNow}});
    }} else {{
      dispC.push(src);
    }}
  }}
  for(let i=showIdx+1;i<cAll.length;i++) dispC.push({{time:cAll[i].time}});
  for(let i=0;i<wsTail.length;i++) dispC.push(wsTail[i]);
  candleSeries.setData(dispC);

  volSeries.setData(vAll.slice(0, showIdx + 1));
  ma5Series.setData(m5All.slice(0, showIdx + 1).filter(Boolean));
  ma9Series.setData(m9All.slice(0, showIdx + 1).filter(Boolean));
  ma20Series.setData(m20All.slice(0, showIdx + 1).filter(Boolean));

  const lastBar=dispC[showIdx];
  const isUp=lastBar.close >= lastBar.open;
  const cls=isUp ? 'tv-val-up' : 'tv-val-dn';
  elO.textContent=fmtP(lastBar.open); elO.className=cls;
  elH.textContent=fmtP(lastBar.high); elH.className=cls;
  elL.textContent=fmtP(lastBar.low);  elL.className=cls;
  elC.textContent=fmtP(lastBar.close); elC.className=cls;
  elTime.textContent=kstAll[showIdx];

  const m5v=m5All[showIdx], m9v=m9All[showIdx], m20v=m20All[showIdx];
  elM5.textContent=m5v ? fmtP(m5v.value) : '-';
  elM9.textContent=m9v ? fmtP(m9v.value) : '-';
  elM20.textContent=m20v ? fmtP(m20v.value) : '-';

  const markers=[{{time:cAll[entryIdx].time,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'ENTRY',size:2}}];
  if(isThumb || isOutro){{
    markers.push({{time:cAll[exitIdx].time,position:'aboveBar',color:'#ffd166',shape:'arrowDown',text:'EXIT',size:2}});
  }}
  candleSeries.setMarkers(markers);

  if(isThumb){{
    elHud.style.opacity='0';
    setResultOverlayVisual(1.0, 1.0, true);
  }} else if(isPrep || isPlay){{
    elHud.style.opacity='1';
    setResultOverlayVisual(0.0, 0.0, false);

    let hitTp1=false, hitTp2=false, hitTp3=false;
    for(let i=entryIdx;i<=showIdx;i++){{
      const b=dispC[i];
      if(b.high >= tp1P) hitTp1=true;
      if(b.close >= tp2P || b.high >= tp3P) hitTp2=true;
      if(b.high >= tp3P) hitTp3=true;
    }}
    const livePnl=isPrep ? 0.0 : ((lastBar.close - entryP) / entryP * 100.0);
    elPnl.textContent=(livePnl >= 0 ? '+' : '') + livePnl.toFixed(1) + '%';
    elPnl.style.color=livePnl >= 0 ? '#00e676' : '#ef5350';
    elPnlBox.style.borderColor=livePnl >= 0 ? '#26a69a' : '#ef5350';

    if(isPrep){{
      elStage.style.display='block';
      elStage.style.borderColor='#00e676';
      elStage.style.background='rgba(0,230,118,0.16)';
      elStage.textContent='🟢 ENTRY 포착 완료 · 리플레이 시작';
    }} else if(hitTp3){{
      elStage.style.display='block';
      elStage.style.borderColor='#ffd166';
      elStage.style.background='rgba(255,209,102,0.18)';
      elStage.textContent='🚀 TP3 50% 익절 · 트레일링 가동';
    }} else if(hitTp2){{
      elStage.style.display='block';
      elStage.style.borderColor='#06d6a0';
      elStage.style.background='rgba(6,214,160,0.18)';
      elStage.textContent='🛡️ TP2 안착 · TP1 방어선 고정';
    }} else if(hitTp1){{
      elStage.style.display='block';
      elStage.style.borderColor='#4cc9f0';
      elStage.style.background='rgba(76,201,240,0.18)';
      elStage.textContent='✅ TP1 돌파 · 100% 홀딩';
    }} else {{
      elStage.style.display='none';
    }}
  }} else {{
    elHud.style.opacity='0';
    const outroStep=f - PLAY_END;
    const arrowP=Math.min(1.0, outroStep / 10.0);
    setResultOverlayVisual(1.0, arrowP, true);
  }}
}};

requestAnimationFrame(()=>requestAnimationFrame(()=>{{
  setupResultOverlayGeometry();
  window.__renderFrame(0);
  window.__REPLAY_READY=true;
}}));
</script></body></html>"""

    js_errors = []
    try:
        from playwright.sync_api import sync_playwright
        with tempfile.TemporaryDirectory() as tmpdir:
            html_path = os.path.join(tmpdir, "replay.html")
            out_mp4 = os.path.join(tmpdir, "replay.mp4")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                )
                page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=1)
                page.on("pageerror", lambda err: js_errors.append(str(err)))
                page.goto(f"file://{html_path}", wait_until="load", timeout=25000)
                page.wait_for_function("window.__REPLAY_READY === true", timeout=15000)
                page.wait_for_timeout(120)

                # 정지 구간(썸네일 4초 · 대기 1초 · 엔딩 고정)은 1장 촬영 후 고속 복사로 렌더링 시간 단축
                for fi in range(total_frames):
                    dst_jpg = os.path.join(tmpdir, f"f_{fi:04d}.jpg")
                    if 1 <= fi < thumb_end:
                        shutil.copyfile(os.path.join(tmpdir, "f_0000.jpg"), dst_jpg)
                        continue
                    if (thumb_end + 1) <= fi < play_start:
                        shutil.copyfile(os.path.join(tmpdir, f"f_{thumb_end:04d}.jpg"), dst_jpg)
                        continue
                    if fi > outro_static_start:
                        shutil.copyfile(os.path.join(tmpdir, f"f_{outro_static_start:04d}.jpg"), dst_jpg)
                        continue

                    page.evaluate(f"window.__renderFrame({fi})")
                    page.screenshot(path=dst_jpg, type="jpeg", quality=88)
                browser.close()

            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", os.path.join(tmpdir, "f_%04d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "21",
                "-movflags", "+faststart",
                out_mp4,
            ]
            res_ff = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if res_ff.returncode != 0 or not os.path.exists(out_mp4):
                LAST_VIDEO_ERROR = f"ffmpeg 인코딩 오류: {(res_ff.stderr or '')[:180]}"
                print(f"[리플레이 영상 실패] {LAST_VIDEO_ERROR}")
                return None

            with open(out_mp4, "rb") as vf:
                return vf.read()
    except Exception as e:
        err_extra = f" | JS: {js_errors[0][:120]}" if js_errors else ""
        LAST_VIDEO_ERROR = f"렌더링 예외: {str(e)[:150]}{err_extra}"
        print(f"[리플레이 영상 예외] {symbol}: {LAST_VIDEO_ERROR}")
        return None


def send_telegram_single(text, reply_markup=None):
    token = (TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    if not text or not token or not chat_id:
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendMessage")
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    try:
        return requests.post(url, json=payload, timeout=12).status_code == 200
    except Exception:
        return False


def send_telegram_photo_with_caption(photo_bytes, caption_text="", reply_markup=None):
    token = (TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    if not photo_bytes or not token or not chat_id:
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendPhoto")
    for _ in range(3):
        try:
            files = {"photo": ("tv_signal_chart.png", photo_bytes, "image/png")}
            data = {"chat_id": chat_id}
            if caption_text:
                data["caption"] = caption_text
            if reply_markup:
                data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False) if isinstance(reply_markup, dict) else str(reply_markup)
            resp = requests.post(url, data=data, files=files, timeout=18)
            if resp.status_code == 200:
                time.sleep(0.7)
                return True
        except Exception:
            pass
        time.sleep(1.2)
    return False


def send_telegram_video_with_caption(video_bytes, caption_text="", reply_markup=None, filename="profit_replay.mp4"):
    """18초 HD MP4 리플레이 영상을 텔레그램 sendVideo로 전송"""
    global LAST_VIDEO_ERROR
    token = (TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    if not video_bytes or not token or not chat_id:
        LAST_VIDEO_ERROR = "영상 데이터 또는 텔레그램 토큰/챗ID 누락"
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendVideo")
    last_resp_err = ""
    for _ in range(3):
        try:
            files = {"video": (filename, video_bytes, "video/mp4")}
            data = {
                "chat_id": chat_id,
                "width": "1280",
                "height": "720",
                "duration": "18",
                "supports_streaming": "true",
            }
            if caption_text:
                data["caption"] = caption_text
            if reply_markup:
                data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False) if isinstance(reply_markup, dict) else str(reply_markup)
            resp = requests.post(url, data=data, files=files, timeout=45)
            if resp.status_code == 200:
                time.sleep(0.8)
                return True
            last_resp_err = f"Telegram HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as e:
            last_resp_err = f"Telegram 전송 예외: {str(e)[:150]}"
        time.sleep(1.5)
    LAST_VIDEO_ERROR = last_resp_err or "Telegram sendVideo 3회 실패"
    print(f"[sendVideo 실패] {LAST_VIDEO_ERROR}")
    return False
# === END OF FILE ===
