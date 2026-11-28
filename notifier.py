import os, re, io, sys, html, time, json, shutil, asyncio, tempfile, threading, subprocess, urllib.parse, requests
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import config as cfg
from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    FIXED_SL_PCT,
    LONGTERM_SL_PCT,
    KST,
    BROWSER_HEADERS,
    to_float,
)

try:
    import api as api_mod
    from api import clean_url
except Exception:
    api_mod = None

    def clean_url(raw_url):
        s = str(raw_url or "").strip()
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\2", s)
        s = s.replace("[", "").replace("]", "")
        if not s.startswith("http"):
            s = ("ht" + "tps://") + s.lstrip("/")
        return s

SCHEME = "ht" + "tps://"
WATERMARK_HANDLE = "@DnaPyung"
JP_TTS_VOICE = "ja-JP-NanamiNeural"
JP_TTS_RATE = "+10%"
EN_TTS_VOICE = "en-US-AriaNeural"
EN_TTS_RATE = "+10%"

VIDEO_FOOTER_NFA_KR = "*본 게시물은 단순 스캐너 데이터 참고용이며 투자 권유가 아닙니다. (NFA)"
VIDEO_FOOTER_NFA_EN = "*For algorithmic scanner data reference only. Not financial advice. (NFA)"
X_POST_DISCLAIMER = (
    "본 계정의 모든 차트와 알림은 알고리즘 스캐너의 단순 데이터 기록이며, "
    "특정 자산의 매수·매도를 권유하거나 수익을 보장하지 않습니다. "
    "모든 투자 판단과 책임은 본인에게 있습니다."
)
RUMBLE_DISCLAIMER_EN = (
    "All charts and alerts on this account are merely data logs from an algorithmic scanner "
    "and do not constitute a recommendation to buy or sell any specific asset, nor do they guarantee returns. "
    "You are solely responsible for all investment decisions."
)

CHART_CSS_W = 960
CHART_CSS_H = 1280
VIDEO_W = 1080
VIDEO_H = 1440

LOCAL_LW_CHARTS_JS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets",
    "lightweight-charts.standalone.production.js",
)
_PLAYWRIGHT_READY = False
LAST_VIDEO_ERROR = ""


def get_last_video_error():
    return LAST_VIDEO_ERROR


def _get_lw_charts_script_tags():
    """로컬 JS 번들(OS 공용 file URI) 우선 로드 + unpkg/jsdelivr 이중 CDN 폴백"""
    lw_unpkg = clean_url(f"{SCHEME}unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js")
    lw_jsdelivr = clean_url(f"{SCHEME}cdn.jsdelivr.net/npm/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js")
    if os.path.exists(LOCAL_LW_CHARTS_JS):
        local_uri = Path(LOCAL_LW_CHARTS_JS).resolve().as_uri()
        return (
            f'<script src="{local_uri}"></script>\n'
            f'<script>if(typeof LightweightCharts==="undefined"){{document.write(\'<scr\'+\'ipt src="{lw_unpkg}"></scr\'+\'ipt>\');}}</script>'
        )
    return (
        f'<script src="{lw_unpkg}"></script>\n'
        f'<script>if(typeof LightweightCharts==="undefined"){{document.write(\'<scr\'+\'ipt src="{lw_jsdelivr}"></scr\'+\'ipt>\');}}</script>'
    )


def fmt_price(price):
    p = to_float(price, 0.0)
    if p <= 0:
        return "0.0000"
    return f"{p:,.2f}" if p >= 1000 else (f"{p:.3f}" if p >= 10 else (f"{p:.4f}" if p >= 1 else (f"{p:.5f}" if p >= 0.01 else f"{p:.6f}")))


def _fmt_market_cap(mc_m):
    v = to_float(mc_m, 0.0)
    if v >= 1000.0:
        return f"{v / 1000.0:.2f}B"
    if v >= 100.0:
        return f"{v:.0f}M"
    return f"{v:.1f}M"


def _fmt_hold_duration(hold_minutes, lang="kr_jp"):
    m = max(1, int(to_float(hold_minutes, 1)))
    if lang == "en":
        if m < 60:
            return f"{m}m"
        h, rem_m = divmod(m, 60)
        if h < 24:
            return f"{h}h {rem_m}m" if rem_m > 0 else f"{h}h"
        d, rem_h = divmod(h, 24)
        return f"{d}d {rem_h}h" if rem_h > 0 else f"{d}d"
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


def fetch_exact_bybit_klines(symbol, interval="60", limit=150, min_rows=10):
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


def _fetch_replay_klines_window(symbol, interval, start_ms, end_ms, max_bars=860):
    if api_mod is not None and hasattr(api_mod, "fetch_klines_window"):
        df = api_mod.fetch_klines_window(symbol, interval=str(interval), start_ms=start_ms, end_ms=end_ms, max_bars=max_bars)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    return fetch_exact_bybit_klines(symbol, interval=str(interval), limit=min(max_bars, 300), min_rows=10)


def _resample_4h_to_12h(df_4h):
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


def _load_signal_chart_df(symbol, is_long):
    """포착 차트(정적 PNG / 포착 영상 공용) 캔들 로더 (120봉 + MA5/9/20)"""
    actual_tf = "720" if is_long else "60"
    df = pd.DataFrame()
    if is_long:
        df = fetch_exact_bybit_klines(symbol, interval="720", limit=155, min_rows=20)
        if df.empty:
            df_4h_raw = fetch_exact_bybit_klines(symbol, interval="240", limit=380, min_rows=30)
            if not df_4h_raw.empty:
                df_resamp = _resample_4h_to_12h(df_4h_raw)
                if not df_resamp.empty and len(df_resamp) >= 20:
                    df = df_resamp
                    actual_tf = "720"
                else:
                    df = df_4h_raw
                    actual_tf = "240"
        if df.empty:
            df = fetch_exact_bybit_klines(symbol, interval="60", limit=155, min_rows=20)
            actual_tf = "60"
    else:
        df = fetch_exact_bybit_klines(symbol, interval="60", limit=155, min_rows=20)
        if df.empty:
            df = fetch_exact_bybit_klines(symbol, interval="15", limit=155, min_rows=20)
            actual_tf = "15"

    if df is None or df.empty or len(df) < 20:
        return pd.DataFrame(), actual_tf

    df = df.copy()
    df["ma5"] = df["close"].rolling(5, min_periods=2).mean()
    df["ma9"] = df["close"].rolling(9, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()
    return df.iloc[-120:].reset_index(drop=True), actual_tf


def _build_supply_overlay_line(res):
    parts = []
    mc_m = to_float(res.get("market_cap_m"), 0.0)
    oi_mc_pct = to_float(res.get("oi_mc_pct"), 0.0)
    turn_mc_pct = to_float(res.get("turn_mc_pct"), 0.0)

    if mc_m >= 1.0:
        parts.append(f"시총 {_fmt_market_cap(mc_m)}")
        parts.append(f"Bybit OI/시총 {oi_mc_pct:.1f}% · 회전 {turn_mc_pct:.1f}%")

    ls_src = str(res.get("ls_source") or "NONE").upper()
    s_val = to_float(res.get("top_ls"), 0.0)
    t_val = to_float(res.get("retail_ls"), 0.0)
    div = to_float(res.get("ls_divergence"), 0.0)
    if div <= 0 and s_val > 0 and t_val > 0:
        div = round(s_val / max(t_val, 0.05), 2)

    src_lbl = {"BINANCE": "BN", "GATE": "GATE", "BITGET": "BG", "BYBIT": "BYBIT", "OKX": "OKX"}.get(ls_src, ls_src)

    if ls_src in ("BINANCE", "GATE", "BITGET") and s_val > 0 and t_val > 0:
        if s_val >= 1.30 and div >= 1.35:
            ls_badge = "🟢고래우위"
        elif (s_val <= 1.05 and t_val >= 2.20) or (s_val < 0.90 and t_val >= 1.80) or (t_val >= 3.00 and div <= 0.65):
            ls_badge = "🔴개미쏠림"
        elif s_val <= 1.10 and t_val >= 2.10 and div <= 0.60:
            ls_badge = "🟠개미편중"
        else:
            ls_badge = "⚪중립"
        parts.append(f"L/S({src_lbl}) 고래 {s_val:.2f} vs 개미 {t_val:.2f} ({div:.2f}x · {ls_badge})")
    elif ls_src in ("BINANCE", "GATE", "BITGET", "BYBIT", "OKX") and t_val > 0:
        ls_badge = "🟢숏쏠림" if t_val <= 0.75 else ("🔴롱과밀" if t_val >= 2.50 else "⚪중립")
        parts.append(f"L/S({src_lbl}) 리테일 {t_val:.2f} ({ls_badge})")

    return f"• 수급심층: {' | '.join(parts)}" if parts else ""


def build_briefing_message(res, now_str):
    base = res.get("base_coin") or res.get("base") or ""
    symbol = res.get("symbol", "")
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
    title = res.get("track_title", "🐋 [장기 포지션 · 세력 대시세 매집 초입]" if is_long else "🚀 [타점 포착]")
    horizon = res.get("position_horizon", "장기 포지션 (1개월 이상 홀딩 예상)" if is_long else "단기·스윙 포지션")
    score = res.get("score", 88)
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
    sl_pct = int(round(to_float(res.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
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
    ]
    overlay_line = _build_supply_overlay_line(res)
    if overlay_line:
        lines.append(overlay_line)

    lines.extend([
        f"• 멀티프레임: {htf} | 15m RSI6 {rsi15:.1f} · J {kdj15:.1f}{ppo_str}",
        "",
        f"🎯 [{'장기 포지션 매매 전략 가이드' if is_long else '매매 전략 가이드'}]",
        f"• 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT",
        f"• 손절가(SL): {fmt_price(sl_p)} USDT (-{sl_pct}% | {sl_note})",
        f"• 1차 구간(TP1·전량 홀딩): {fmt_price(tps[0])} USDT (+{to_float(tp_pcts[0]):.1f}% | {tp_labels[0]} · 익절 없이 100% 홀딩)",
        f"• 2차 구간(TP2·전량 홀딩): {fmt_price(tps[1])} USDT (+{to_float(tp_pcts[1]):.1f}% | {tp_labels[1]} · 돌파시 방어선 TP1({fmt_price(tps[0])}) 상향 고정)",
        f"• 3차 목표(TP3·50% 익절): {fmt_price(tps[2])} USDT (+{to_float(tp_pcts[2]):.1f}% | {tp_labels[2]} · 방어선 TP2 상향 & 잔여 50% 고점 -{trail_drop}% 트레일링)",
    ])
    if reasons:
        lines.append("\n🔍 [포착 근거]")
        for r in reasons:
            lines.append(f"• {r}")
    if is_long:
        lines.append("\n⚠️ [장기 진입 전 10초 체크]: 30일 내 대형 클리프 언락(Tokenomist) · 바이낸스 유의종목(Monitoring Tag) 여부 확인 권장")
    return "\n".join(lines)


def _pick_triplet(triplets, symbol):
    if not triplets:
        return "", "", ""
    seed = sum(ord(ch) for ch in str(symbol or "BTC"))
    return triplets[seed % len(triplets)]


def build_x_hook_triplet(res):
    """[36종 전략별 한·일·영 네이티브 훅(Hook) 1:1:1 매핑 엔진]"""
    symbol = str(res.get("symbol", ""))
    track_type = str(res.get("track_type", "TRACK_A")).upper()
    r_text = " ".join(str(x) for x in (res.get("reasons") or []))
    title_str = str(res.get("track_title", ""))

    # 1) 장기 포지션 (Track D1 ~ D6)
    if track_type == "TRACK_D" or res.get("is_long_term"):
        sub_s = str(res.get("sub_strategy", "")).upper()
        if "D6" in sub_s or "스프링" in sub_s:
            return _pick_triplet([
                ("신저가 개미털기(스프링) 후 다개월 바닥선 거래량 동반 재탈환 포착",
                 "安値更新のふるい落とし後、出来高を伴う底値ライン奪還シグナル",
                 "Wyckoff spring sweep below lows followed by high-volume floor reclaim"),
                ("공포 투매 물량 흡수 완료! 바닥선 강하게 재탈환하는 역발상 대시세 초입",
                 "パニック売りを吸収し、底値ラインを力強く奪還する逆張り初動シグナル",
                 "panic sell absorption and strong contrarian base reclaim"),
                ("개미 손절 털어낸 뒤 거래량 터지며 바닥 위로 올라탄 와이코프 스프링 포착",
                 "損切りを巻き込んだ直後に出来高急増で反発するワイコフスプリング",
                 "liquidity shakeout below support with surging volume recovery"),
            ], symbol)
        if "D1" in sub_s:
            return _pick_triplet([
                ("가격 횡보 속 세력 OI·OBV 선행 매집 포착! 조용히 물량 잠기는 대시세 초입",
                 "レンジ推移の中で大口の建玉とOBVが先行して積み上がる静かな仕込み初動",
                 "stealth whale OI and OBV accumulation inside a quiet base"),
                ("티 안 나게 세력 수급이 먼저 뚫린 스텔스 바닥 매집 구간 포착",
                 "目立たない形で大口需給が先行突破したステルス底値仕込みシグナル",
                 "early institutional OBV breakout during silent bottom accumulation"),
            ], symbol)
        if "D2" in sub_s:
            return _pick_triplet([
                ("세력 매집봉 출현 후 거래량 건조 눌림 마무리! 2차 대시세 출발 직전",
                 "大口の買い集め後に出来高が細る押し目を完了した上昇出発直前シグナル",
                 "accumulation candle followed by dry-volume pullback reversal"),
                ("매집봉 지지선 방어 후 거래량 말리며 반등 턴 나오는 세력 눌림목 포착",
                 "大口サポートを維持し、出来高減少から反発に転じた押し目シグナル",
                 "whale demand defense with volume dry-up rebound turn"),
            ], symbol)
        if "D3" in sub_s:
            return _pick_triplet([
                ("30일 장기 박스권 상단 거래량 폭발 첫 돌파! 큰 추세 파동 개시 포착",
                 "30日間の長期レンジ上限を出来高急増で初突破したトレンド開始シグナル",
                 "30-day macro range breakout backed by explosive volume"),
                ("한 달간 응축된 장기 박스권 저항을 거래량 싣고 뚫어낸 대시세 초입",
                 "1か月間凝縮された長期抵抗帯を出来高を伴って突破した大相場初動",
                 "month-long consolidation breakout kicking off a macro trend"),
            ], symbol)
        if "D4" in sub_s:
            return _pick_triplet([
                ("장기 하락추세 탈출 및 4H 50MA 거래량 돌파 안착! 추세 턴어라운드 초입",
                 "長期下落トレンドからの脱出と4時間足50MA突破定着の転換初動",
                 "long-term downtrend breakout above the 4H 50MA with strong volume"),
                ("이중바닥 완성 후 주요 추세선 돌파! 중장기 상승 1파동 시작점 포착",
                 "ダブルボトム完成後に主要トレンドラインを突破した上昇第1波の起点",
                 "double-bottom trendline breakout starting macro wave one"),
            ], symbol)
        if "D5" in sub_s:
            return _pick_triplet([
                ("바닥권 4H PPO·RSI·OBV 3중 강세 다이버전스 완성! 세력 바닥 흡수 포착",
                 "底値圏でのPPO・RSI・OBV三重強気ダイバージェンス完成シグナル",
                 "4H triple bullish divergence across PPO, RSI, and OBV at the bottom"),
                ("하락 에너지는 멈추고 세력 매수세는 유입되는 매크로 바닥 반전 초입",
                 "下落圧力が止まり大口の買いが流入するマクロ底値反転初動",
                 "exhausted selling pressure with macro bottom accumulation"),
            ], symbol)
        return _pick_triplet([
            ("세력 4H·1D 매집 후 대시세 1파동 초입 포착",
             "4時間足と日足での大口仕込み完了後の上昇第1波初動シグナル",
             "4H and daily institutional accumulation launching wave one"),
            ("상위 프레임 바닥 숙성 완료! 세력 수급 결집되는 중장기 대시세 초입",
             "上位足の底固めが完了し大口需給が集中する中長期上昇初動",
             "higher-timeframe base maturity with fresh smart-money inflows"),
        ], symbol)

    # 2) 스탑헌팅 후 재진입
    if "재진입" in title_str or "재탈환" in r_text:
        return _pick_triplet([
            ("개미털기 바닥 확인 후 거래량 실리며 V자 반등 시도중",
             "ふるい落としの底打ち確認後、出来高を伴うV字反発シグナル",
             "stop-hunt bottom confirmation with a V-shaped volume rebound"),
            ("스탑헌팅 악성 매물 소화 완료! 비과열 구간에서 V자 재탈환 시도중",
             "ストップ狩りの売りを消化し、非過熱圏からV字奪還を狙うシグナル",
             "post-shakeout V-recovery from a non-overheated zone"),
            ("밑꼬리 개미털기 후 매수세 재유입되며 단기 추세 재가동중",
             "下ヒゲのふるい落とし後に買いが再流入する短期トレンド再始動シグナル",
             "lower-wick liquidity flush with renewed buying momentum"),
        ], symbol)

    # 3) 수급심층 특화 훅
    oi_mc_pct = to_float(res.get("oi_mc_pct"), 0.0)
    top_ls = to_float(res.get("top_ls"), 0.0)
    retail_ls = to_float(res.get("retail_ls"), 0.0)
    ls_div = to_float(res.get("ls_divergence"), 0.0)
    if top_ls >= 1.50 and 0 < retail_ls <= 1.05 and ls_div >= 1.60:
        return _pick_triplet([
            ("고래는 롱 매집, 개미는 숏 쏠림! 스마트머니 포지션 괴리 폭발 초입",
             "クジラはロング、個人はショート偏重のスマートマネー乖離シグナル",
             "whales loading longs against crowded retail shorts"),
            ("탑트레이더 롱 베팅 vs 일반투자자 숏 쏠림! 수급 비대칭 급등 후보 포착",
             "トップトレーダーのロング優位と個人ショート偏重の需給非対称シグナル",
             "smart-money long divergence versus heavy retail shorting"),
        ], symbol)
    if oi_mc_pct >= 15.0:
        return _pick_triplet([
            ("시총 대비 선물 OI 압축 임계치 도달! 레버리지 화약고 점화 초입 포착",
             "時価総額に対する先物建玉の圧縮が臨界点に達したレバレッジ点火初動",
             "high futures OI-to-market-cap compression nearing ignition"),
            ("과적재된 선물 OI에 매수 거래량 점화! 강한 변동성 분출 직전 타점",
             "積み上がった先物建玉に買い出来高が点火したボラティリティ噴出直前",
             "loaded futures open interest igniting with buy-side volume"),
        ], symbol)

    # 4) 단기·스윙 트랙별 맞춤 훅
    if track_type == "TRACK_H":
        return _pick_triplet([
            ("최근 급등 대장주 Top 5 발사 직전 DNA 패턴과 일치하는 초입 포착",
             "直近急騰トップ5銘柄の発射直前チャートDNAに一致した初動シグナル",
             "pre-breakout DNA matching recent top 5 gainer launches"),
            ("최근 시장을 주도한 급등주들의 발사 직전 차트 공식을 그대로 닮은 초입",
             "市場を牽引した急騰銘柄の発射直前パターンを再現した初動シグナル",
             "high-momentum launchpad pattern mirrored from market leaders"),
            ("급등주 역추적 알고리즘이 포착한 에너지 응축·거래량 점화 초입 타점",
             "急騰銘柄アルゴリズムが捉えたエネルギー凝縮と出来高点火の初動",
             "algorithmic compression and volume ignition launchpad setup"),
        ], symbol)
    if track_type == "TRACK_E":
        return _pick_triplet([
            ("24H 저점 유동성 스윕(개미털기) 직후 강한 반전 매수세 유입중",
             "24時間安値の流動性スイープ直後に強い反転買いが流入したシグナル",
             "24H low liquidity sweep followed by strong reversal buying"),
            ("전저점 스탑헌팅 아래꼬리 흡수 후 강세 다이버전스와 함께 급반전 시도중",
             "直近安値のストップ狩りを吸収し強気ダイバージェンスで反転するシグナル",
             "stop-hunt wick absorption with bullish divergence reversal"),
            ("저점 이탈 공포 물량을 세력이 받아먹고 지지선 탈환하는 SFP 반전 타점",
             "安値割れの投げ売りを大口が吸収しサポートを奪還するSFP反転シグナル",
             "swing failure pattern reclaiming key support after a flush"),
        ], symbol)
    if track_type == "TRACK_F":
        return _pick_triplet([
            ("비트코인 흔들림 속에서도 역행 강세 보이는 상대강도 대장주 포착",
             "ビットコイン調整局面でも逆行高を見せる相対強度リーダー銘柄シグナル",
             "strong relative strength outperforming Bitcoin weakness"),
            ("시장 조정장에서도 고점권 버티며 독주하는 상대강도(RS) 1티어 주도주",
             "相場全体の調整をものともせず高値圏を維持する相対強度トップ銘柄",
             "tier-1 relative strength leader holding near highs"),
            ("비트코인 약세를 무시하고 OBV 신고가와 함께 치고 나가는 역행 대장주",
             "ビットコイン弱含みを無視しOBV高値更新で上昇する逆行リーダー銘柄",
             "counter-trend leader breaking OBV highs despite BTC dip"),
        ], symbol)
    if track_type == "TRACK_G":
        return _pick_triplet([
            ("깊은 음수 펀딩비와 OI 집중 속 숏스퀴즈 강제청산 초입 포착",
             "深いマイナス金利と建玉集中によるショートスクイーズ初動シグナル",
             "deep negative funding and high OI triggering a short squeeze"),
            ("과도하게 쌓인 숏 포지션 위에 매수 폭발! 연쇄 숏스퀴즈 발사 직전",
             "積み上がったショートに買いが殺到する連鎖ショートスクイーズ直前",
             "crowded short positions facing an explosive liquidation squeeze"),
            ("음수 펀딩비 연료 가득 찬 상태에서 저항선 돌파하는 숏스퀴즈 초입 타점",
             "マイナス金利を燃料に抵抗線を突破するショートスクイーズ初動シグナル",
             "resistance breakout fueled by heavy negative funding shorts"),
        ], symbol)
    if track_type == "TRACK_C":
        return _pick_triplet([
            ("급등 대장주 핵심 눌림목 지지 후 2차 파동 시도중",
             "急騰リーダー銘柄の重要押し目サポートからの第2波上昇シグナル",
             "healthy leader pullback bouncing into a second upward wave"),
            ("1파 급등 후 거래량 건조 조정 마무리! 피보나치 핵심 지지선 반등 타점",
             "第1波急騰後の出来高減少調整を終えフィボナッチ支持線で反発するシグナル",
             "volume-dry Fibonacci retracement bounce after wave one"),
            ("대장주 건강한 눌림목 소화 완료! 5분봉 추세 전환과 함께 2차 상승 시동",
             "健全な押し目を消化し5分足トレンド転換とともに第2波へ移行するシグナル",
             "5m market structure shift launching wave two from support"),
        ], symbol)
    if track_type == "TRACK_B":
        return _pick_triplet([
            ("거래량이 살아나면서 24H 전고점 돌파 시도중",
             "出来高の急増を伴う24時間高値ブレイクアウトシグナル",
             "24H structural high breakout backed by surging volume"),
            ("24시간 주요 저항선을 건전한 양봉과 거래량으로 뚫어낸 구조 돌파 초입",
             "24時間主要抵抗線を力強い陽線と出来高で突破した構造ブレイク初動",
             "clean bullish candle breaking key 24H resistance with volume"),
            ("돌파 후 지지 전환(S-R Flip) 확인! 매물벽 소화하고 상방 확장 시도중",
             "ブレイク後のロールリバーサルを確認し上値拡大を狙うシグナル",
             "confirmed support-resistance flip ready for upside expansion"),
        ], symbol)
    if track_type == "TRACK_A" or "응축" in r_text or "스퀴즈" in r_text:
        return _pick_triplet([
            ("박스권 에너지 응축 후 거래량 살아나며 추세 전환중",
             "レンジでのエネルギー凝縮後、出来高回復とともにトレンド転換するシグナル",
             "tight volatility squeeze firing upward with fresh volume"),
            ("볼린저·켈트너 초응축(TTM 스퀴즈) 상방 발사! 매수 델타 유입 초입",
             "TTMスクイーズの極限収縮から上放れし買いデルタが流入した初動",
             "TTM squeeze breakout with strong positive buy-side delta"),
            ("좁아진 변동성 박스를 거래량 싣고 뚫어내는 에너지 폭발 초입 포착",
             "狭まったボラティリティレンジを出来高を伴って上抜ける爆発初動",
             "compressed range breakout with expanding buy momentum"),
        ], symbol)
    if "음수 펀딩" in r_text:
        return (
            "숏 스퀴즈 수급 유입되며 단기 추세 반등 시도중",
            "ショートスクイーズ需給の流入による短期トレンド反発シグナル",
            "short-squeeze buying flow driving a bullish reversal",
        )
    return (
        "눌림목 거래량 건조 후 수급 살아나며 반등 시도중",
        "押し目での出来高減少後、需給回復とともに反発を狙うシグナル",
        "dry-volume pullback followed by fresh bullish demand",
    )


def build_x_hook_pair(res):
    ko, jp, _ = build_x_hook_triplet(res)
    return ko, jp


def build_x_hook_line(res):
    return build_x_hook_triplet(res)[0]


def build_x_post_message(res):
    """[X 포스팅 양식] 포착근거 제외 · 최하단 면책 문구 포함 (기존 X 자동 입력 버튼용)"""
    base = res.get("base_coin") or res.get("base") or ""
    symbol = res.get("symbol", "")
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
    price = to_float(res.get("price"))
    e_low = to_float(res.get("entry_low"), price * 0.9970)
    e_high = to_float(res.get("entry_high"), price * 1.0015)
    sl_p = to_float(res.get("stop_price"))
    sl_pct = int(round(to_float(res.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))

    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]
    p1, p2, p3 = round(to_float(tp_pcts[0], 8.0), 1), round(to_float(tp_pcts[1], 15.0), 1), round(to_float(tp_pcts[2], 22.0), 1)

    jp_header = "上昇可能性の高いコインをスキャンしました\n#仮想通貨 #コインシグナル\n\n"
    footer_tags = "#코인시황 #코인신호 #코인 #재테크"
    long_line = "【장기 포지션】1개월이상 홀딩예상\n" if is_long else ""

    return (
        f"{jp_header}"
        f"{long_line}"
        "🔺️상승 예상 코인 검색기에 포착된 종목😎\n\n"
        f"${base} ({symbol})\n\n"
        f"- 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT\n"
        f"- SL: {fmt_price(sl_p)} (-{sl_pct}%)\n"
        f"- tp1: {fmt_price(tps[0])}(+{p1:g}%)\n"
        f"- tp2: {fmt_price(tps[1])}(+{p2:g}%)\n"
        f"- tp3: {fmt_price(tps[2])}(+{p3:g}%)\n\n"
        f"#{base} 어디까지 오를거 같나요?\n"
        "보유중이신 분들 계신가요?\n\n"
        f"{footer_tags}\n\n"
        f"{X_POST_DISCLAIMER}"
    )


def build_rumble_signal_post_message(res):
    """[럼블(Rumble) 포착 업로드용 1탭 복사 양식 — Title + Description(해시태그 본문 통합)]"""
    base = str(res.get("base_coin") or res.get("base") or str(res.get("symbol", "")).replace("USDT", "")).upper()
    symbol = str(res.get("symbol", "")).upper()
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
    price = to_float(res.get("price"))
    e_low = to_float(res.get("entry_low"), price * 0.9970)
    e_high = to_float(res.get("entry_high"), price * 1.0015)
    sl_p = to_float(res.get("stop_price"))
    sl_pct = int(round(to_float(res.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]
    p1, p2, p3 = round(to_float(tp_pcts[0], 8.0), 1), round(to_float(tp_pcts[1], 15.0), 1), round(to_float(tp_pcts[2], 22.0), 1)

    long_prefix = "【Long-Term Position】Expected Holding: 1+ Month\n" if is_long else ""
    title_prefix = "[Long-Term] " if is_long else ""
    rumble_title = f"{title_prefix}${base} ({symbol}) Bullish Scanner Alert | TP3 +{p3:g}% Setup"
    hashtags_en = f"#{base} #Crypto #Altcoin #CryptoSignals #Trading #Bybit"
    rumble_desc = (
        f"{long_prefix}"
        "🔺️Spotted on the Bullish Coin Scanner😎\n"
        f"${base} ({symbol})\n"
        f"Entry Zone: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT\n"
        f"SL: {fmt_price(sl_p)} (-{sl_pct}%)\n"
        f"TP1: {fmt_price(tps[0])} (+{p1:g}%)\n"
        f"TP2: {fmt_price(tps[1])} (+{p2:g}%)\n"
        f"TP3: {fmt_price(tps[2])} (+{p3:g}%)\n\n"
        f"How high do you think #{base} will go?\n"
        "Anyone currently holding this?\n\n"
        f"{hashtags_en}\n\n"
        f"{RUMBLE_DISCLAIMER_EN}"
    )

    return (
        f"<b>📺 [Rumble 영문 포착 복사 양식 · ${html.escape(base)}]</b>\n"
        f"🔗 업로드: https://rumble.com/upload.php\n\n"
        f"<b>1) Title (터치 복사)</b>\n<code>{html.escape(rumble_title)}</code>\n\n"
        f"<b>2) Description (해시태그 포함 · 터치 복사)</b>\n<pre>{html.escape(rumble_desc)}</pre>"
    )


def build_rumble_profit_post_message(vp):
    """[럼블(Rumble) 익절 리플레이 업로드용 1탭 복사 양식 — Title + Description(해시태그 본문 통합)]"""
    base = str(vp.get("base_coin") or str(vp.get("symbol", "")).replace("USDT", "")).upper()
    symbol = str(vp.get("symbol", "")).upper()
    is_long = bool(vp.get("is_long_term") or vp.get("track_type") == "TRACK_D")
    entry_p = to_float(vp.get("entry_price"))
    exit_p = to_float(vp.get("exit_price"), entry_p)
    tot_pct = to_float(vp.get("realized_weighted_pct"))
    max_fav = to_float(vp.get("max_favorable_pct"), tot_pct)
    hold_en = _fmt_hold_duration(vp.get("hold_minutes", 60), lang="en")

    long_prefix = "【Long-Term Position】\n" if is_long else ""
    title_prefix = "[Long-Term] " if is_long else ""
    rumble_title = f"{title_prefix}${base} ({symbol}) +{tot_pct:.2f}% Profit Replay | Bullish Coin Scanner"
    hashtags_en = f"#{base} #Crypto #CryptoProfit #Altcoin #TradingReplay #Bybit"
    rumble_desc = (
        f"{long_prefix}"
        "🎯 Bullish Coin Scanner Profit Replay 😎\n"
        f"${base} ({symbol})\n"
        f"Entry Price: {fmt_price(entry_p)} USDT\n"
        f"Exit Price: {fmt_price(exit_p)} USDT\n"
        f"Peak Gain: +{max_fav:.1f}%\n"
        f"Final Return: +{tot_pct:.2f}% (Held {hold_en})\n\n"
        f"Did you catch this #{base} move?\n"
        "Follow and turn on notifications for real-time scanner alerts!\n\n"
        f"{hashtags_en}\n\n"
        f"{RUMBLE_DISCLAIMER_EN}"
    )

    return (
        f"<b>📺 [Rumble 영문 익절 복사 양식 · ${html.escape(base)}]</b>\n"
        f"🔗 업로드: https://rumble.com/upload.php\n\n"
        f"<b>1) Title (터치 복사)</b>\n<code>{html.escape(rumble_title)}</code>\n\n"
        f"<b>2) Description (해시태그 포함 · 터치 복사)</b>\n<pre>{html.escape(rumble_desc)}</pre>"
    )


def build_x_intent_reply_markup(x_post_text, button_label="🚀 X(트위터) 앱에 본문 자동 입력하기"):
    encoded_text = urllib.parse.quote(str(x_post_text or ""), safe="")
    intent_url = clean_url(f"{SCHEME}x.com/intent/post?text={encoded_text}")
    return {"inline_keyboard": [[{"text": button_label, "url": intent_url}]]}


def _probe_audio_duration_sec(audio_path, default_sec=3.0):
    if not audio_path or not os.path.exists(audio_path):
        return default_sec
    if shutil.which("ffprobe"):
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                d = to_float(r.stdout.strip(), 0.0)
                if d > 0.1:
                    return d
        except Exception:
            pass
    return default_sec


def _synth_voice_segment(text, out_mp3, voice=JP_TTS_VOICE, rate=JP_TTS_RATE, gtts_lang="ja"):
    """Edge Neural TTS (1.1배속) 전용 스레드 실행 + gTTS atempo=1.1 자동 폴백"""
    err_box = []

    def _run_edge():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            import edge_tts

            async def _go():
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate)
                await comm.save(out_mp3)

            loop.run_until_complete(_go())
        except Exception as e:
            err_box.append(str(e))
        finally:
            try:
                loop.close()
            except Exception:
                pass

    th = threading.Thread(target=_run_edge, daemon=True)
    th.start()
    th.join(timeout=22.0)

    if os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 500:
        return True, ""

    try:
        from gtts import gTTS
        raw_mp3 = out_mp3 + ".raw.mp3"
        gTTS(text=text, lang=gtts_lang).save(raw_mp3)
        if os.path.exists(raw_mp3) and os.path.getsize(raw_mp3) > 300:
            r = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", raw_mp3, "-filter:a", "atempo=1.1", "-b:a", "192k", out_mp3],
                capture_output=True, text=True, timeout=20,
            )
            try:
                os.remove(raw_mp3)
            except Exception:
                pass
            if r.returncode == 0 and os.path.exists(out_mp3) and os.path.getsize(out_mp3) > 300:
                return True, ""
    except Exception as ge:
        err_box.append(f"gTTS:{ge}")

    return False, ("; ".join(err_box)[:160] or "TTS 생성 실패")


def _encode_hd_mp4(img_pattern, out_mp4, in_fps, out_fps, segments=None, total_sec=20.0):
    """X · Rumble 풀HD(1080x1440 · CRF15 · BT.709 · 48kHz 192k) 공용 인코더"""
    v_filter_hd = f"scale={VIDEO_W}:{VIDEO_H}:flags=lanczos+accurate_rnd+full_chroma_int,unsharp=3:3:0.35:3:3:0.0"
    v_flags = [
        "-c:v", "libx264", "-profile:v", "high", "-level:v", "4.1",
        "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "15",
        "-maxrate", "8000k", "-bufsize", "16000k",
        "-r", str(out_fps), "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart",
    ]
    has_audio = bool(segments) and all(seg.get("audio_path") and os.path.exists(seg["audio_path"]) for seg in segments)
    if has_audio:
        audio_inputs, fc_parts, concat_labels, cursor = [], [f"[0:v]{v_filter_hd}[vout]"], [], 0.0
        for idx, seg in enumerate(segments):
            audio_inputs.extend(["-i", seg["audio_path"]])
            gap = max(0.05, seg["audio_start"] - cursor)
            fc_parts.append(f"aevalsrc=0:s=48000:c=stereo:d={gap:.3f}[g{idx}]")
            fc_parts.append(f"[{idx + 1}:a]aresample=48000,aformat=channel_layouts=stereo,volume=1.18[s{idx}]")
            concat_labels.extend([f"[g{idx}]", f"[s{idx}]"])
            cursor = seg["audio_start"] + seg["audio_dur"]
        fc_parts.append(f"aevalsrc=0:s=48000:c=stereo:d={max(0.20, total_sec - cursor):.3f}[gend]")
        concat_labels.append("[gend]")
        fc_parts.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=0:a=1[aout]")
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(in_fps), "-i", img_pattern,
            *audio_inputs, "-filter_complex", ";".join(fc_parts),
            "-map", "[vout]", "-map", "[aout]",
            *v_flags, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", out_mp4,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(in_fps), "-i", img_pattern,
            "-vf", v_filter_hd, *v_flags, out_mp4,
        ]

    res_ff = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if res_ff.returncode != 0 or not os.path.exists(out_mp4):
        return None, (res_ff.stderr or "ffmpeg error")[:180]
    with open(out_mp4, "rb") as vf:
        return vf.read(), ""


def _unified_opening_spec(lang="kr_jp", step_str="1 / 4"):
    if lang == "en":
        return {
            "step": step_str, "tag": "INTRO",
            "l1": "Hello, this is Danpyung 👋",
            "l2": f"{WATERMARK_HANDLE} Bullish Coin Scanner Briefing",
            "sub": "",
            "tts_text": "Hello, this is Danpyung.",
        }
    return {
        "step": step_str, "tag": "INTRO",
        "l1": "안녕하세요, 단평입니다 👋",
        "l2": f"X · {WATERMARK_HANDLE} 상승 코인 검색기 브리핑",
        "sub": "こんにちは、ダンピョンです！仮想通貨スキャナーブリーフィング",
        "tts_text": "こんにちは、ダンピョンです。",
    }


def _unified_ending_spec(lang="kr_jp", step_str="4 / 4"):
    if lang == "en":
        return {
            "step": step_str, "tag": "FOLLOW",
            "l1": "Please Follow + Turn on Notifications! 🔔",
            "l2": f"{WATERMARK_HANDLE} Real-Time Crypto Scanner Alerts",
            "sub": "",
            "tts_text": "Please follow and turn on notifications!",
        }
    return {
        "step": step_str, "tag": "FOLLOW",
        "l1": "팔로우 + 알림설정 부탁드립니다! 🔔",
        "l2": f"X · {WATERMARK_HANDLE} 실시간 상승 코인 알림",
        "sub": f"フォロー＋通知設定よろしくお願いします！ · {WATERMARK_HANDLE}",
        "tts_text": "フォローと通知設定、よろしくお願いします！",
    }


def _build_signal_video_narration(res, tp1_p, tp2_p, tp3_p, p1, p2, p3, tmpdir, lang="kr_jp"):
    """포착 차트 영상 4단계 자막 + 음성 싱크 (kr_jp: 한/일자막+일본어음성 | en: 100% 영문자막+영어음성)"""
    base = str(res.get("base_coin") or res.get("base") or str(res.get("symbol", "")).replace("USDT", "")).upper()
    hook_ko = str(res.get("x_hook_ko") or "").strip()
    hook_jp = str(res.get("x_hook_jp") or "").strip()
    hook_en = str(res.get("x_hook_en") or "").strip()
    if not hook_ko or not hook_jp or not hook_en:
        t_ko, t_jp, t_en = build_x_hook_triplet(res)
        hook_ko = hook_ko or t_ko
        hook_jp = hook_jp or t_jp
        hook_en = hook_en or t_en

    t1_s, t2_s, t3_s = fmt_price(tp1_p), fmt_price(tp2_p), fmt_price(tp3_p)
    t1_num, t2_num, t3_num = t1_s.replace(",", ""), t2_s.replace(",", ""), t3_s.replace(",", "")

    if lang == "en":
        voice, rate, gtts_lang = EN_TTS_VOICE, EN_TTS_RATE, "en"
        raw_specs = [
            _unified_opening_spec("en", step_str="1 / 4"),
            {
                "step": "2 / 4", "tag": "SIGNAL",
                "l1": f'${base} spotted on "{hook_en}"',
                "l2": "High-potential entry setup zone to watch",
                "sub": "",
                "tts_text": f"{base} is a setup worth watching, based on {hook_en}.",
            },
            {
                "step": "3 / 4", "tag": "TARGETS",
                "l1": f'TP1 "${t1_s}" , TP2 "${t2_s}" , TP3 "${t3_s}"',
                "l2": f"Projected Targets (+{p1:g}% / +{p2:g}% / +{p3:g}%)",
                "sub": "",
                "tts_text": f"Projected targets are TP1 at {t1_num} dollars, TP2 at {t2_num} dollars, and TP3 at {t3_num} dollars.",
            },
            _unified_ending_spec("en", step_str="4 / 4"),
        ]
    else:
        voice, rate, gtts_lang = JP_TTS_VOICE, JP_TTS_RATE, "ja"
        raw_specs = [
            _unified_opening_spec("kr_jp", step_str="1 / 4"),
            {
                "step": "2 / 4", "tag": "SIGNAL",
                "l1": f'${base} 종목은 "{hook_ko}" 근거로',
                "l2": "진입해볼만한 자리입니다",
                "sub": f"{base}：{hook_jp}（エントリー注目ポイント）",
                "tts_text": f"{base}は、{hook_jp}の根拠で、エントリーを狙えるポイントです。",
            },
            {
                "step": "3 / 4", "tag": "TARGETS",
                "l1": f'TP1 "${t1_s}" , TP2 "${t2_s}" , TP3 "${t3_s}"',
                "l2": f"예상하고 있습니다! (+{p1:g}% / +{p2:g}% / +{p3:g}%)",
                "sub": f"目標予想：TP1 ${t1_s} · TP2 ${t2_s} · TP3 ${t3_s}",
                "tts_text": f"TP1、{t1_num}ドル。TP2、{t2_num}ドル。TP3、{t3_num}ドルを予想しています。",
            },
            _unified_ending_spec("kr_jp", step_str="4 / 4"),
        ]

    durs, paths = [], []
    tts_ok, tts_err = True, ""
    for idx, sp in enumerate(raw_specs):
        mp3_p = os.path.join(tmpdir, f"sig_tts_{lang}_{idx}.mp3")
        ok, err = _synth_voice_segment(sp["tts_text"], mp3_p, voice=voice, rate=rate, gtts_lang=gtts_lang)
        if not ok:
            tts_ok, tts_err = False, err
            break
        durs.append(_probe_audio_duration_sec(mp3_p, default_sec=3.5))
        paths.append(mp3_p)

    if not tts_ok or len(durs) != 4:
        durs, paths = [2.0, 5.8, 6.2, 2.8], ["", "", "", ""]

    s0 = 0.25
    e0 = s0 + durs[0]
    s1 = max(e0 + 0.45, 2.20)
    e1 = s1 + durs[1]
    arrow_start_sec = max(e1 + 0.45, 8.20)
    s2 = arrow_start_sec
    e2 = s2 + durs[2]
    arrow_end_sec = max(e2 + 0.25, arrow_start_sec + 6.00)
    s3 = max(e2 + 0.45, arrow_end_sec - 0.15)
    e3 = s3 + durs[3]
    final_total_sec = max(18.0, e3 + 0.85)

    starts, ends = [s0, s1, s2, s3], [e0, e1, e2, e3]
    segments = []
    for i, sp in enumerate(raw_specs):
        segments.append({
            "step": sp["step"], "tag": sp["tag"],
            "l1": sp["l1"], "l2": sp["l2"], "sub": sp["sub"],
            "audio_path": paths[i], "audio_start": round(starts[i], 3),
            "audio_dur": round(durs[i], 3), "audio_end": round(ends[i], 3),
            "sub_start": round(0.0 if i == 0 else starts[i] - 0.12, 3),
            "sub_end": round(final_total_sec if i == len(raw_specs) - 1 else starts[i + 1] - 0.12, 3),
        })

    return {
        "segments": segments,
        "arrow_start_sec": round(arrow_start_sec, 3),
        "arrow_end_sec": round(arrow_end_sec, 3),
        "total_sec": round(final_total_sec, 3),
        "tts_ok": tts_ok,
        "tts_err": tts_err,
    }


def generate_signal_chart_video_bytes(res, lang="kr_jp"):
    """
    [<포착알림차트영상> 풀HD 세로형 1080x1440 엔진]
    - lang="kr_jp": X용 (한/일 자막 · 일본어 음성)
    - lang="en": 럼블 전용 (차트·자막·음성 100% 영문 · 한글/일어 0%)
    """
    global LAST_VIDEO_ERROR
    LAST_VIDEO_ERROR = ""
    is_en = (str(lang).lower() == "en")

    if not shutil.which("ffmpeg"):
        LAST_VIDEO_ERROR = "ffmpeg 미설치"
        return None
    if not _ensure_playwright():
        LAST_VIDEO_ERROR = "Playwright 준비 실패"
        return None

    symbol = str(res.get("symbol", "")).upper().strip()
    base = res.get("base_coin") or res.get("base") or symbol.replace("USDT", "")
    orig_entry = to_float(res.get("price"))
    if not symbol or orig_entry <= 0:
        LAST_VIDEO_ERROR = f"유효하지 않은 포착 심볼/가격({symbol}, {orig_entry})"
        return None

    track_type = str(res.get("track_type", "TRACK_A"))
    is_long = bool(res.get("is_long_term") or track_type == "TRACK_D")
    sub_s = str(res.get("sub_strategy", "")).upper()
    score = int(to_float(res.get("score"), 88))

    df, chart_tf = _load_signal_chart_df(symbol, is_long)
    if df.empty or len(df) < 20:
        LAST_VIDEO_ERROR = f"{symbol} 포착 차트 캔들 데이터 부족"
        return None

    latest_row = df.iloc[-1]
    price = to_float(latest_row["close"], orig_entry)
    ratio_adj = price / orig_entry if orig_entry > 0 else 1.0
    sl_pct = int(round(to_float(res.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
    sl_p = to_float(res.get("stop_price"), price * (1.0 - sl_pct / 100.0)) * ratio_adj
    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]
    tp1_p, tp2_p, tp3_p = to_float(tps[0], price * 1.08) * ratio_adj, to_float(tps[1], price * 1.15) * ratio_adj, to_float(tps[2], price * 1.22) * ratio_adj
    p1, p2, p3 = round(to_float(tp_pcts[0], 8.0), 1), round(to_float(tp_pcts[1], 15.0), 1), round(to_float(tp_pcts[2], 22.0), 1)

    candle_min_low, candle_max_high = to_float(df["low"].min(), price * 0.97), to_float(df["high"].max(), price * 1.02)
    low_piv, high_piv = min(candle_min_low, price * 0.988), max(candle_max_high, price * 1.010)
    if high_piv <= low_piv:
        low_piv, high_piv = price * 0.985, price * 1.015

    core_span = max(high_piv - low_piv, price * 0.018)
    up_comp = float(np.clip((core_span * 0.44) / max(tp3_p - high_piv, price * 0.02), 0.05, 1.0)) if tp3_p > high_piv else 1.0
    down_comp = float(np.clip((core_span * 0.14) / max(low_piv - sl_p, price * 0.008), 0.06, 1.0)) if sl_p < low_piv else 1.0

    def _fwd(val):
        v = to_float(val, price)
        return (high_piv + (v - high_piv) * up_comp) if v > high_piv else ((low_piv - (low_piv - v) * down_comp) if v < low_piv else v)

    tz_use = timezone.utc if is_en else KST
    tz_offset_sec = 0 if is_en else (9 * 3600)
    tz_suffix = "UTC" if is_en else "KST"

    candles_js, vol_js, ma5_js, ma9_js, ma20_js, time_strs = [], [], [], [], [], []
    tf_str = str(chart_tf).upper().strip()
    step_sec = {"720": 43200, "240": 14400, "60": 3600, "15": 900}.get(tf_str, 43200 if is_long else 3600)

    last_ts_sec = -1
    for _, row in df.iterrows():
        ts_ms = to_float(row["timestamp"])
        ts_sec = int(ts_ms // 1000) + tz_offset_sec
        if ts_sec <= last_ts_sec:
            ts_sec = last_ts_sec + step_sec
        last_ts_sec = ts_sec
        o_v, h_v, l_v, c_v, v_v = to_float(row["open"]), to_float(row["high"]), to_float(row["low"]), to_float(row["close"]), to_float(row["volume"])
        candles_js.append({"time": ts_sec, "open": _fwd(o_v), "high": _fwd(h_v), "low": _fwd(l_v), "close": _fwd(c_v)})
        vol_js.append({"time": ts_sec, "value": v_v, "color": "rgba(38, 166, 154, 0.38)" if c_v >= o_v else "rgba(239, 83, 80, 0.38)"})
        ma5_js.append({"time": ts_sec, "value": _fwd(row["ma5"])} if pd.notna(row.get("ma5")) else None)
        ma9_js.append({"time": ts_sec, "value": _fwd(row["ma9"])} if pd.notna(row.get("ma9")) else None)
        ma20_js.append({"time": ts_sec, "value": _fwd(row["ma20"])} if pd.notna(row.get("ma20")) else None)
        time_strs.append(datetime.fromtimestamp(ts_ms / 1000.0, tz=tz_use).strftime(f"%Y-%m-%d %H:%M {tz_suffix}"))

    ws_tail = [{"time": candles_js[-1]["time"] + (i * step_sec)} for i in range(1, 31)]
    entry_idx = len(candles_js) - 1
    replay_start_idx = max(12, int(len(candles_js) * 0.25))

    high_idx = int(df["high"].astype(float).idxmax())
    low_idx = int(df["low"].astype(float).idxmin())
    fib_high, fib_low = to_float(df["high"].iloc[high_idx], candle_max_high), to_float(df["low"].iloc[low_idx], candle_min_low)
    fib_span = fib_high - fib_low
    fib_levels_js, fib_start_time, fib_end_time, fib_dir_tag = [], candles_js[0]["time"], ws_tail[-1]["time"], "FIB"
    if fib_span > price * 0.005 and len(candles_js) >= 10:
        fib_start_time = candles_js[min(min(high_idx, low_idx), max(0, len(candles_js) // 2))]["time"]
        is_up_imp = low_idx <= high_idx
        fib_dir_tag = "FIB (L→H)" if is_up_imp else "FIB (H→L)"
        for ratio, r_lbl, color, l_w, l_s, prio in [
            (0.0, "0", "#787b86", 1, 1, 1), (0.236, "0.236", "#9598a1", 1, 3, 2),
            (0.382, "0.382", "#4dd0e1", 1, 3, 4), (0.5, "0.5", "#ffd54f", 1.2, 3, 5),
            (0.618, "0.618", "#ff9800", 1.3, 3, 6), (0.786, "0.786", "#ce93d8", 1, 3, 3),
            (1.0, "1", "#787b86", 1, 1, 1),
        ]:
            real_p = (fib_high - fib_span * ratio) if is_up_imp else (fib_low + fib_span * ratio)
            fib_levels_js.append({"ratio": ratio, "label": r_lbl, "priceStr": fmt_price(real_p), "yVal": _fwd(real_p), "color": color, "lineWidth": l_w, "lineStyle": l_s, "priority": prio})

    y_bot_bound = min(_fwd(sl_p), _fwd(candle_min_low)) - core_span * 0.06
    y_top_bound = max(_fwd(tp3_p), _fwd(candle_max_high)) + core_span * 0.14
    tf_name = {"720": "12H", "240": "4H", "60": "1H", "15": "15m"}.get(tf_str, "12H" if is_long else "1H")
    tf_label = f"{tf_name} · LONG-TERM" if is_long else f"{tf_name} · BYBIT"
    track_badge = {
        "TRACK_A": "TRACK A · TTM Squeeze", "TRACK_B": "TRACK B · 24H Breakout",
        "TRACK_C": "TRACK C · Deep Pullback",
        "TRACK_D": "LONG-TERM · Track D6 Spring" if ("D6" in sub_s or "스프링" in sub_s) else "LONG-TERM · Track D Macro",
        "TRACK_E": "TRACK E · SFP Reversal", "TRACK_F": "TRACK F · BTC RS Leader",
        "TRACK_G": "TRACK G · Short Squeeze", "TRACK_H": "TRACK H · Top5 DNA",
    }.get(track_type, track_type)

    decimals = 6 if price < 0.01 else (5 if price < 1 else (4 if price < 10 else (3 if price < 1000 else 2)))
    min_move = 10 ** (-decimals)
    lw_script_tags = _get_lw_charts_script_tags()

    ui_badge = "🎙️ DANPYUNG SIGNAL BRIEFING · EN" if is_en else "🎙️ 단평 포착 브리핑 · KR / JP"
    ui_footer = VIDEO_FOOTER_NFA_EN if is_en else VIDEO_FOOTER_NFA_KR
    ui_phases = {
        "thumb": "🎯 ENTRY → TP1 · TP2 · TP3 Scenario" if is_en else "🎯 ENTRY → TP1 · TP2 · TP3 예상 시나리오",
        "scan": "📈 Scanning Bullish Setup..." if is_en else "📈 실시간 포착 차트 스캔 중...",
        "entry": "🟢 ENTRY Trigger Reached!" if is_en else "🟢 포착 진입 타점(ENTRY) 도달!",
        "tp1": f"🎯 Target 1 · TP1 (+{p1:g}%)" if is_en else f"🎯 1차 예상 목표가 TP1 (+{p1:g}%)",
        "tp2": f"🚀 Target 2 · TP2 (+{p2:g}%)" if is_en else f"🚀 2차 예상 목표가 TP2 (+{p2:g}%)",
        "tp3": f"🏆 Final Target · TP3 (+{p3:g}%)" if is_en else f"🏆 최종 예상 목표가 TP3 (+{p3:g}%)",
        "outro": f"🔔 Follow + Notifications · {WATERMARK_HANDLE}" if is_en else f"🔔 팔로우 + 알림설정 · {WATERMARK_HANDLE}",
    }

    in_fps, out_fps = 10, 30
    js_errors = []

    try:
        from playwright.sync_api import sync_playwright
        with tempfile.TemporaryDirectory() as tmpdir:
            plan = _build_signal_video_narration(res, tp1_p, tp2_p, tp3_p, p1, p2, p3, tmpdir, lang=lang)
            segments, total_sec = plan["segments"], plan["total_sec"]
            arrow_start_sec, arrow_end_sec, tts_ok = plan["arrow_start_sec"], plan["arrow_end_sec"], plan["tts_ok"]

            total_frames = max(180, int(round(total_sec * in_fps)))
            thumb_end_f = 12
            candle_start_f = thumb_end_f
            candle_end_f = max(candle_start_f + 40, int(round(arrow_start_sec * in_fps)))
            arrow_start_f = candle_end_f
            arrow_end_f = min(total_frames - 8, max(arrow_start_f + 45, int(round(arrow_end_sec * in_fps))))

            sub_meta_js = [{"step": s["step"], "tag": s["tag"], "l1": s["l1"], "l2": s["l2"], "sub": s["sub"]} for s in segments]
            sub_jp_display = "none" if is_en else "block"

            html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{lw_script_tags}
<style>
*{{box-sizing:border-box;margin:0;padding:0;-webkit-font-smoothing:antialiased}}
body{{background:#131722;font-family:'Noto Sans CJK KR','Noto Sans KR','Noto Sans CJK JP','Segoe UI',-apple-system,Roboto,sans-serif;color:#d1d4dc;width:960px;height:1280px;overflow:hidden}}
.tv-wrap{{width:960px;height:1280px;background:#131722;display:flex;flex-direction:column}}
.tv-topbar{{height:112px;background:#181c27;border-bottom:1px solid #2a2e39;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:10px 20px}}
.tb-row{{display:flex;align-items:center;justify-content:space-between}}
.tv-left,.tv-right{{display:flex;align-items:center;gap:10px}}
.tv-symbol{{font-size:26px;font-weight:900;color:#fff;letter-spacing:0.2px}}
.tv-tf-badge{{background:#2a2e39;color:#e0e3eb;font-size:13.5px;font-weight:800;padding:4px 10px;border-radius:5px}}
.tv-score-pill{{background:#2962ff;color:#fff;font-size:13px;font-weight:800;padding:5px 12px;border-radius:6px}}
.tv-handle-pill{{background:rgba(255,255,255,0.08);border:1px solid #3d4354;color:#f0f3fa;font-size:13.5px;font-weight:800;padding:5px 11px;border-radius:6px}}
.tv-ohlc{{display:flex;gap:12px;font-size:14px;font-weight:700;color:#9598a1}}
.tv-val-up{{color:#26a69a}}.tv-val-dn{{color:#ef5350}}
#chart-box{{position:relative;width:960px;height:836px;overflow:hidden}}
.tv-ma-legend{{position:absolute;top:12px;left:14px;z-index:20;display:flex;flex-wrap:wrap;gap:12px;font-size:12.5px;font-weight:700;background:rgba(19,23,34,0.88);padding:6px 12px;border-radius:6px;border:1px solid #2a2e39}}
#fib-overlay{{position:absolute;inset:0;pointer-events:none;z-index:12;overflow:hidden}}
.fib-band{{position:absolute;pointer-events:none}}
.fib-lbl{{position:absolute;font-size:11px;font-weight:700;line-height:1;padding:2px 6px;border-radius:3px;background:rgba(19,23,34,0.86);border:1px solid rgba(255,255,255,0.10);white-space:nowrap}}
#arrow-overlay{{position:absolute;inset:0;pointer-events:none;z-index:26;opacity:0}}
#tp-svg{{position:absolute;inset:0;width:960px;height:836px;overflow:visible}}
.tp-tag{{position:absolute;border-radius:8px;padding:6px 12px;font-size:14px;font-weight:900;color:#0b111e;white-space:nowrap;box-shadow:0 6px 18px rgba(0,0,0,0.55);opacity:0;transform:translate(-50%,-100%)}}
.tp-tag-1{{background:#4cc9f0;border:1.5px solid #fff}}.tp-tag-2{{background:#06d6a0;border:1.5px solid #fff}}.tp-tag-3{{background:#ffd166;border:1.5px solid #fff;font-size:15px}}
.phase-pill{{position:absolute;top:52px;left:50%;transform:translateX(-50%);z-index:28;background:rgba(15,20,34,0.90);border:1.8px solid #4cc9f0;color:#eef6ff;font-size:17px;font-weight:900;padding:8px 20px;border-radius:999px;box-shadow:0 8px 22px rgba(0,0,0,0.50);white-space:nowrap}}
#bottom-panel{{height:300px;width:960px;background:linear-gradient(180deg,#121725 0%,#0c101a 100%);border-top:2px solid #272e40;padding:16px 24px;display:flex;flex-direction:column;justify-content:center;gap:12px}}
.sub-head{{display:flex;align-items:center;justify-content:space-between}}
.sub-badge{{background:#2962ff;color:#fff;font-size:13.5px;font-weight:800;padding:5px 14px;border-radius:999px;letter-spacing:0.3px}}
.sub-step{{background:rgba(255,255,255,0.08);border:1px solid #3a4258;color:#cbd5e1;font-size:13px;font-weight:800;padding:4px 12px;border-radius:999px}}
.sub-box{{width:100%;background:rgba(24,30,46,0.96);border:2px solid rgba(76,201,240,0.48);border-radius:16px;padding:18px 22px;text-align:center;box-shadow:0 10px 26px rgba(0,0,0,0.45);display:flex;flex-direction:column;gap:10px;justify-content:center;min-height:212px}}
.sub-l1{{font-size:27px;font-weight:900;color:#fff;line-height:1.34;word-break:keep-all;text-shadow:0 2px 6px rgba(0,0,0,0.55)}}
.sub-l2{{font-size:25px;font-weight:900;color:#ffd166;line-height:1.32;word-break:keep-all;text-shadow:0 2px 6px rgba(0,0,0,0.55)}}
.sub-jp{{display:{sub_jp_display};font-size:17px;font-weight:700;color:#93c5fd;line-height:1.32;border-top:1px solid rgba(255,255,255,0.10);padding-top:8px;margin-top:2px;word-break:keep-all}}
.tv-footer{{height:32px;background:#0a0d14;border-top:1px solid #1f2430;display:flex;align-items:center;justify-content:space-between;padding:0 18px;font-size:11.5px;color:#8e94a4;font-weight:600}}
.prog-wrap{{width:150px;height:5px;background:#222632;border-radius:3px;overflow:hidden}}.prog-fill{{height:100%;width:0%;background:linear-gradient(90deg,#2962ff,#00e676)}}
</style></head><body><div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tb-row"><div class="tv-left"><span class="tv-symbol">${base}USDT.P</span><span class="tv-tf-badge">{tf_label}</span></div><div class="tv-right"><span class="tv-score-pill">{track_badge} · {score} pts</span><span class="tv-handle-pill">{WATERMARK_HANDLE}</span></div></div>
    <div class="tb-row"><div class="tv-ohlc"><span>O <span id="oh-o" class="tv-val-up">{fmt_price(price)}</span></span><span>H <span id="oh-h" class="tv-val-up">{fmt_price(price)}</span></span><span>L <span id="oh-l" class="tv-val-up">{fmt_price(price)}</span></span><span>C <span id="oh-c" class="tv-val-up">{fmt_price(price)}</span></span></div><span id="top-time" style="font-size:13px;font-weight:700;color:#b2b5be">{time_strs[-1]}</span></div>
  </div>
  <div id="chart-box">
    <div class="tv-ma-legend" id="ma-legend"><span style="color:#00e5ff">● MA(5) <span id="lg-m5">-</span></span><span style="color:#ffd166">● MA(9) <span id="lg-m9">-</span></span><span style="color:#2962ff">● MA(20) <span id="lg-m20">-</span></span><span style="color:#ff9800">▤ {fib_dir_tag}</span></div>
    <div id="fib-overlay"></div>
    <div class="phase-pill" id="phase-pill">{ui_phases['thumb']}</div>
    <div id="arrow-overlay">
      <svg id="tp-svg" viewBox="0 0 960 836">
        <defs>
          <linearGradient id="tp-grad" x1="0%" y1="100%" x2="100%" y2="0%"><stop offset="0%" stop-color="#00e676"/><stop offset="45%" stop-color="#4cc9f0"/><stop offset="100%" stop-color="#ffd166"/></linearGradient>
          <filter id="tp-glow" x="-25%" y="-25%" width="150%" height="150%"><feDropShadow dx="0" dy="0" stdDeviation="4.5" flood-color="#4cc9f0" flood-opacity="0.75"/></filter>
        </defs>
        <path id="tp-path-1" d="" fill="none" stroke="url(#tp-grad)" stroke-width="5.2" stroke-linecap="round" filter="url(#tp-glow)"/>
        <path id="tp-path-2" d="" fill="none" stroke="url(#tp-grad)" stroke-width="5.2" stroke-linecap="round" filter="url(#tp-glow)"/>
        <path id="tp-path-3" d="" fill="none" stroke="url(#tp-grad)" stroke-width="5.4" stroke-linecap="round" filter="url(#tp-glow)"/>
        <circle id="pt-entry" cx="0" cy="0" r="7" fill="#00e676" stroke="#fff" stroke-width="2.4"/>
        <circle id="pt-tp1" cx="0" cy="0" r="6.5" fill="#4cc9f0" stroke="#fff" stroke-width="2.2" opacity="0"/>
        <circle id="pt-tp2" cx="0" cy="0" r="6.5" fill="#06d6a0" stroke="#fff" stroke-width="2.2" opacity="0"/>
        <circle id="pt-tp3" cx="0" cy="0" r="7.5" fill="#ffd166" stroke="#fff" stroke-width="2.4" opacity="0"/>
        <polygon id="tp-head" points="0,0 -20,-10 -14,0 -20,10" fill="#ffd166" stroke="#fff" stroke-width="1.2"/>
      </svg>
      <div id="tag-tp1" class="tp-tag tp-tag-1">TP1 ${fmt_price(tp1_p)} (+{p1:g}%)</div>
      <div id="tag-tp2" class="tp-tag tp-tag-2">TP2 ${fmt_price(tp2_p)} (+{p2:g}%)</div>
      <div id="tag-tp3" class="tp-tag tp-tag-3">🎯 TP3 ${fmt_price(tp3_p)} (+{p3:g}%)</div>
    </div>
  </div>
  <div id="bottom-panel">
    <div class="sub-head"><span class="sub-badge">{ui_badge}</span><span class="sub-step" id="sub-step">1 / 4</span></div>
    <div class="sub-box"><div class="sub-l1" id="sub-l1"></div><div class="sub-l2" id="sub-l2"></div><div class="sub-jp" id="sub-jp"></div></div>
  </div>
  <div class="tv-footer"><span>{ui_footer}</span><div class="prog-wrap"><div class="prog-fill" id="prog-fill"></div></div></div>
</div><script>
const lowPiv={low_piv},highPiv={high_piv},upComp={up_comp},downComp={down_comp},decimals={decimals};
function invScale(y){{if(y>highPiv)return highPiv+(y-highPiv)/Math.max(upComp,1e-6);if(y<lowPiv)return lowPiv-(lowPiv-y)/Math.max(downComp,1e-6);return y;}}
function fmtReal(val){{const p=invScale(val);if(p<=0)return '0.0000';return p>=1000?p.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}}):p.toFixed(decimals);}}
const cAll={json.dumps(candles_js)},vAll={json.dumps(vol_js)},m5All={json.dumps(ma5_js)},m9All={json.dumps(ma9_js)},m20All={json.dumps(ma20_js)},tAll={json.dumps(time_strs)},wsTail={json.dumps(ws_tail)},fibLevels={json.dumps(fib_levels_js)},subMeta={json.dumps(sub_meta_js, ensure_ascii=False)},uiP={json.dumps(ui_phases, ensure_ascii=False)};
const fibStartTime={fib_start_time},fibEndTime={fib_end_time},entryIdx={entry_idx},replayStartIdx={replay_start_idx};
const entryYVal={_fwd(price)},tp1YVal={_fwd(tp1_p)},tp2YVal={_fwd(tp2_p)},tp3YVal={_fwd(tp3_p)};

const chart=LightweightCharts.createChart(document.getElementById('chart-box'),{{width:960,height:836,layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},watermark:{{visible:true,fontSize:66,horzAlign:'center',vertAlign:'center',color:'rgba(209,212,220,0.14)',text:'{WATERMARK_HANDLE}',fontStyle:'bold'}},rightPriceScale:{{borderColor:'#2a2e39',scaleMargins:{{top:0.06,bottom:0.09}},entireTextOnly:true}},timeScale:{{borderColor:'#2a2e39',timeVisible:true,rightOffset:2}},localization:{{priceFormatter:fmtReal}}}});
chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}}).setData([{{time:cAll[0].time,value:{y_bot_bound}}},{{time:wsTail[wsTail.length-1].time,value:{y_top_bound}}}]);
const candleSeries=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',priceLineVisible:false,lastValueVisible:false,priceFormat:{{type:'custom',minMove:{min_move},formatter:fmtReal}}}});
fibLevels.forEach(fl=>chart.addLineSeries({{color:fl.color,lineWidth:fl.lineWidth,lineStyle:fl.lineStyle,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}}).setData([{{time:fibStartTime,value:fl.yVal}},{{time:fibEndTime,value:fl.yVal}}]));
const ma5Series=chart.addLineSeries({{color:'#00e5ff',lineWidth:1.8,priceLineVisible:false,lastValueVisible:false}});
const ma9Series=chart.addLineSeries({{color:'#ffd166',lineWidth:1.8,priceLineVisible:false,lastValueVisible:false}});
const ma20Series=chart.addLineSeries({{color:'#2962ff',lineWidth:2.2,priceLineVisible:false,lastValueVisible:false}});
const volSeries=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'vol_scale',priceLineVisible:false,lastValueVisible:false}});
chart.priceScale('vol_scale').applyOptions({{scaleMargins:{{top:0.90,bottom:0.0}}}});
[{{p:tp3YVal,c:'#ffd166',t:'TP3 (+{p3:g}%)'}},{{p:tp2YVal,c:'#06d6a0',t:'TP2 (+{p2:g}%)'}},{{p:tp1YVal,c:'#4cc9f0',t:'TP1 (+{p1:g}%)'}},{{p:{_fwd(sl_p)},c:'#ef476f',t:'SL (-{sl_pct}%)'}}].forEach(x=>candleSeries.createPriceLine({{price:x.p,color:x.c,lineWidth:2,lineStyle:2,axisLabelVisible:true,title:x.t}}));
candleSeries.setData(cAll.concat(wsTail));chart.timeScale().fitContent();

const elO=document.getElementById('oh-o'),elH=document.getElementById('oh-h'),elL=document.getElementById('oh-l'),elC=document.getElementById('oh-c'),elTime=document.getElementById('top-time'),elM5=document.getElementById('lg-m5'),elM9=document.getElementById('lg-m9'),elM20=document.getElementById('lg-m20');
const elProg=document.getElementById('prog-fill'),elPhase=document.getElementById('phase-pill'),elArrowOv=document.getElementById('arrow-overlay');
const elP1=document.getElementById('tp-path-1'),elP2=document.getElementById('tp-path-2'),elP3=document.getElementById('tp-path-3');
const elPtE=document.getElementById('pt-entry'),elPt1=document.getElementById('pt-tp1'),elPt2=document.getElementById('pt-tp2'),elPt3=document.getElementById('pt-tp3'),elHead=document.getElementById('tp-head');
const elTag1=document.getElementById('tag-tp1'),elTag2=document.getElementById('tag-tp2'),elTag3=document.getElementById('tag-tp3');
const elSubStep=document.getElementById('sub-step'),elSubL1=document.getElementById('sub-l1'),elSubL2=document.getElementById('sub-l2'),elSubJp=document.getElementById('sub-jp');
let len1=1,len2=1,len3=1,geoReady=false;

function buildFibAndArrowGeometry(){{
  const psW=(chart.priceScale('right')&&chart.priceScale('right').width)?chart.priceScale('right').width():86;
  const plotRight=Math.max(360,960-(psW||86)-6);
  try{{
    const ov=document.getElementById('fib-overlay'),maBox=document.getElementById('ma-legend');
    const maRight=maBox?(maBox.offsetLeft+maBox.offsetWidth+10):410,maBottom=maBox?(maBox.offsetTop+maBox.offsetHeight+6):48;
    if(ov&&fibLevels.length>0){{
      ov.innerHTML='';
      const rawX=chart.timeScale().timeToCoordinate(fibStartTime);
      const xStart=(rawX!==null&&isFinite(rawX))?Math.max(8,Math.min(plotRight-220,rawX)):12;
      const byR={{}}; fibLevels.forEach(fl=>{{const y=candleSeries.priceToCoordinate(fl.yVal);if(y!==null&&isFinite(y))byR[fl.label]=y;}});
      const drawB=(r1,r2,bg)=>{{if(byR[r1]!==undefined&&byR[r2]!==undefined){{const hY=Math.abs(byR[r2]-byR[r1]);if(hY>=2){{const b=document.createElement('div');b.className='fib-band';b.style.left=xStart+'px';b.style.width=Math.max(20,plotRight-xStart)+'px';b.style.top=Math.min(byR[r1],byR[r2])+'px';b.style.height=hY+'px';b.style.background=bg;ov.appendChild(b);}}}}}};
      drawB('0.5','0.618','rgba(255,152,0,0.09)'); drawB('0.382','0.5','rgba(77,208,225,0.045)');
      const placedY=[];
      [...fibLevels].sort((a,b)=>b.priority-a.priority).forEach(fl=>{{
        const y=byR[fl.label]; if(y===undefined||y<12||y>785||placedY.some(py=>Math.abs(py-y)<15))return;
        placedY.push(y); const tag=document.createElement('div'); tag.className='fib-lbl';
        let lX=xStart+6; if(y<maBottom&&lX<maRight)lX=Math.min(plotRight-140,maRight);
        tag.style.left=lX+'px';tag.style.top=Math.max(8,y-8)+'px';tag.style.color=fl.color;tag.textContent=fl.label+' ('+fl.priceStr+')'; ov.appendChild(tag);
      }});
    }}
  }}catch(e){{}}

  let x0=chart.timeScale().timeToCoordinate(cAll[entryIdx].time); if(x0===null||!isFinite(x0))x0=plotRight*0.72;
  const aw=Math.max(90,plotRight-x0-12),x1=x0+aw*0.32,x2=x0+aw*0.64,x3=x0+aw*0.94;
  let y0=candleSeries.priceToCoordinate(entryYVal),y1=candleSeries.priceToCoordinate(tp1YVal),y2=candleSeries.priceToCoordinate(tp2YVal),y3=candleSeries.priceToCoordinate(tp3YVal);
  if(y0===null||!isFinite(y0))y0=500;if(y1===null||!isFinite(y1))y1=y0-95;if(y2===null||!isFinite(y2))y2=y0-190;if(y3===null||!isFinite(y3))y3=y0-290;
  const mkQ=(ax,ay,bx,by)=>'M '+ax.toFixed(1)+' '+ay.toFixed(1)+' Q '+(ax+(bx-ax)*0.58).toFixed(1)+' '+(ay-(ay-by)*0.28).toFixed(1)+' '+bx.toFixed(1)+' '+by.toFixed(1);
  elP1.setAttribute('d',mkQ(x0,y0,x1,y1));elP2.setAttribute('d',mkQ(x0,y0,x2,y2));elP3.setAttribute('d',mkQ(x0,y0,x3,y3));
  len1=Math.max(1,elP1.getTotalLength());len2=Math.max(1,elP2.getTotalLength());len3=Math.max(1,elP3.getTotalLength());
  elP1.style.strokeDasharray=len1.toFixed(1);elP2.style.strokeDasharray=len2.toFixed(1);elP3.style.strokeDasharray=len3.toFixed(1);
  elPtE.setAttribute('cx',x0.toFixed(1));elPtE.setAttribute('cy',y0.toFixed(1));elPt1.setAttribute('cx',x1.toFixed(1));elPt1.setAttribute('cy',y1.toFixed(1));
  elPt2.setAttribute('cx',x2.toFixed(1));elPt2.setAttribute('cy',y2.toFixed(1));elPt3.setAttribute('cx',x3.toFixed(1));elPt3.setAttribute('cy',y3.toFixed(1));
  const pT=(el,tx,ty)=>{{el.style.left=Math.max(95,Math.min(plotRight-95,tx)).toFixed(1)+'px';el.style.top=Math.max(98,ty-14).toFixed(1)+'px';}};
  pT(elTag1,x1,y1);pT(elTag2,x2,y2);pT(elTag3,x3,y3); geoReady=true;
}}

function setTpArrowStage(stage,frac){{
  if(!geoReady)buildFibAndArrowGeometry();
  const ease=1-Math.pow(1-Math.max(0.02,Math.min(1.0,frac)),2.2);
  [elP1,elP2,elP3,elPt1,elPt2,elPt3,elTag1,elTag2,elTag3].forEach(x=>x.style.opacity='0');
  if(stage<=0){{elArrowOv.style.opacity='0';return;}}
  elArrowOv.style.opacity='1';
  let ap=elP1,al=len1;
  if(stage===1){{elP1.style.opacity='1';if(ease>=0.88){{elPt1.style.opacity='1';elTag1.style.opacity='1';}}}}
  else if(stage===2){{elP1.style.opacity='0.30';elP1.style.strokeDashoffset='0';elPt1.style.opacity='0.85';elTag1.style.opacity='0.78';elP2.style.opacity='1';ap=elP2;al=len2;if(ease>=0.88){{elPt2.style.opacity='1';elTag2.style.opacity='1';}}}}
  else{{elP1.style.opacity='0.24';elP1.style.strokeDashoffset='0';elP2.style.opacity='0.28';elP2.style.strokeDashoffset='0';elPt1.style.opacity='0.80';elPt2.style.opacity='0.85';elTag1.style.opacity='0.72';elTag2.style.opacity='0.78';elP3.style.opacity='1';ap=elP3;al=len3;if(ease>=0.88){{elPt3.style.opacity='1';elTag3.style.opacity='1';}}}}
  const curL=al*ease; ap.style.strokeDashoffset=(al-curL).toFixed(1);
  const pt=ap.getPointAtLength(curL),pp=ap.getPointAtLength(Math.max(0,curL-3.5));
  elHead.setAttribute('transform','translate('+pt.x.toFixed(1)+','+pt.y.toFixed(1)+') rotate('+(Math.atan2(pt.y-pp.y,pt.x-pp.x)*180/Math.PI).toFixed(1)+')');
}}

function setSubtitle(idx){{
  if(!subMeta.length)return;
  const seg=subMeta[Math.max(0,Math.min(subMeta.length-1,idx||0))];
  elSubStep.textContent=seg.step;elSubL1.textContent=seg.l1;elSubL2.textContent=seg.l2;elSubJp.textContent=seg.sub||'';
  elSubL2.style.color=(seg.tag==='TARGETS')?'#00e676':'#ffd166';
}}

const THUMB_END={thumb_end_f},CANDLE_START={candle_start_f},CANDLE_END={candle_end_f},ARROW_START={arrow_start_f},ARROW_END={arrow_end_f};
window.__renderFrame=function(f,subIdx){{
  setSubtitle(subIdx);
  const isThumb=(f<THUMB_END),isCandle=(f>=CANDLE_START&&f<CANDLE_END),isArrow=(f>=ARROW_START&&f<=ARROW_END);
  let showIdx=entryIdx,frac=1.0;
  if(isThumb){{elProg.style.width='0.0%';elPhase.textContent=uiP.thumb;elPhase.style.borderColor='#ffd166';}}
  else if(isCandle){{const p=(f-CANDLE_START)/Math.max(1,CANDLE_END-CANDLE_START);elProg.style.width=(p*50).toFixed(1)+'%';const pos=replayStartIdx+p*(entryIdx-replayStartIdx);showIdx=Math.min(entryIdx,Math.floor(pos));frac=Math.min(1,Math.max(0.2,pos-Math.floor(pos)));if(f>=CANDLE_END-1){{showIdx=entryIdx;frac=1;}}elPhase.textContent=(showIdx>=entryIdx)?uiP.entry:uiP.scan;elPhase.style.borderColor='#00e676';}}
  else if(isArrow){{const p=(f-ARROW_START)/Math.max(1,ARROW_END-ARROW_START);elProg.style.width=(50+p*50).toFixed(1)+'%';if(p<0.333){{elPhase.textContent=uiP.tp1;elPhase.style.borderColor='#4cc9f0';}}else if(p<0.666){{elPhase.textContent=uiP.tp2;elPhase.style.borderColor='#06d6a0';}}else{{elPhase.textContent=uiP.tp3;elPhase.style.borderColor='#ffd166';}}}}
  else{{elProg.style.width='100.0%';elPhase.textContent=uiP.outro;elPhase.style.borderColor='#4cc9f0';}}
  const dispC=[];
  for(let i=0;i<=showIdx;i++){{const s=cAll[i];if(i===showIdx&&isCandle&&frac<0.999){{const cN=s.open+(s.close-s.open)*frac;dispC.push({{time:s.time,open:s.open,high:Math.max(s.open,cN,s.open+(s.high-s.open)*frac),low:Math.min(s.open,cN,s.open+(s.low-s.open)*frac),close:cN}});}}else dispC.push(s);}}
  for(let i=showIdx+1;i<cAll.length;i++)dispC.push({{time:cAll[i].time}});
  for(let i=0;i<wsTail.length;i++)dispC.push(wsTail[i]);
  candleSeries.setData(dispC);volSeries.setData(vAll.slice(0,showIdx+1));ma5Series.setData(m5All.slice(0,showIdx+1).filter(Boolean));ma9Series.setData(m9All.slice(0,showIdx+1).filter(Boolean));ma20Series.setData(m20All.slice(0,showIdx+1).filter(Boolean));
  const lb=dispC[showIdx],cls=(lb.close>=lb.open)?'tv-val-up':'tv-val-dn';
  elO.textContent=fmtReal(lb.open);elO.className=cls;elH.textContent=fmtReal(lb.high);elH.className=cls;elL.textContent=fmtReal(lb.low);elL.className=cls;elC.textContent=fmtReal(lb.close);elC.className=cls;elTime.textContent=tAll[showIdx];
  elM5.textContent=m5All[showIdx]?fmtReal(m5All[showIdx].value):'-';elM9.textContent=m9All[showIdx]?fmtReal(m9All[showIdx].value):'-';elM20.textContent=m20All[showIdx]?fmtReal(m20All[showIdx].value):'-';
  candleSeries.setMarkers(showIdx>=entryIdx?[{{time:cAll[entryIdx].time,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'ENTRY',size:2}}]:[]);
  if(isThumb||f>ARROW_END)setTpArrowStage(3,1.0);
  else if(isCandle)setTpArrowStage(0,0.0);
  else{{const p=(f-ARROW_START)/Math.max(1,ARROW_END-ARROW_START);if(p<0.333)setTpArrowStage(1,p/0.333);else if(p<0.666)setTpArrowStage(2,(p-0.333)/0.333);else setTpArrowStage(3,(p-0.666)/0.334);}}
}};
requestAnimationFrame(()=>requestAnimationFrame(()=>{{buildFibAndArrowGeometry();window.__renderFrame(0,0);window.__SIG_VID_READY=true;}}));
</script></body></html>"""

            html_path = os.path.join(tmpdir, f"signal_video_{lang}.html")
            out_mp4 = os.path.join(tmpdir, f"signal_video_{lang}.mp4")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            def _sub_idx_for_f(fi_num):
                t_s = fi_num / float(in_fps)
                for si, seg in enumerate(segments):
                    if seg["sub_start"] <= t_s < seg["sub_end"]:
                        return si
                return len(segments) - 1

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"])
                try:
                    page = browser.new_page(viewport={"width": CHART_CSS_W, "height": CHART_CSS_H}, device_scale_factor=1.5)
                    page.on("pageerror", lambda err: js_errors.append(str(err)))
                    page.goto(Path(html_path).resolve().as_uri(), wait_until="load", timeout=25000)
                    page.wait_for_function("window.__SIG_VID_READY === true", timeout=15000)
                    page.wait_for_timeout(140)

                    prev_sub = -1
                    for fi in range(total_frames):
                        dst_png = os.path.join(tmpdir, f"sf_{fi:04d}.png")
                        cur_sub = _sub_idx_for_f(fi)
                        if ((1 <= fi < thumb_end_f) or (fi > arrow_end_f + 1)) and cur_sub == prev_sub:
                            shutil.copyfile(os.path.join(tmpdir, f"sf_{fi - 1:04d}.png"), dst_png)
                        else:
                            page.evaluate(f"window.__renderFrame({fi}, {cur_sub})")
                            page.screenshot(path=dst_png, type="png")
                            prev_sub = cur_sub
                finally:
                    browser.close()

            video_bytes, enc_err = _encode_hd_mp4(
                os.path.join(tmpdir, "sf_%04d.png"), out_mp4, in_fps, out_fps,
                segments=segments if tts_ok else None, total_sec=total_sec,
            )
            if not video_bytes:
                LAST_VIDEO_ERROR = f"포착 영상({lang}) ffmpeg 오류: {enc_err}"
                print(f"[포착 차트 영상 실패] {LAST_VIDEO_ERROR}")
                return None

            print(f"[포착 차트 영상 완료({lang})] {symbol} ({tf_name}, {len(df)}봉 · {VIDEO_W}x{VIDEO_H}) | {total_sec:.1f}초 · TTS={'OK' if tts_ok else 'OFF'}")
            return {"video": video_bytes, "width": VIDEO_W, "height": VIDEO_H, "duration": round(total_sec, 1)}
    except Exception as e:
        err_extra = f" | JS: {js_errors[0][:120]}" if js_errors else ""
        LAST_VIDEO_ERROR = f"포착 영상({lang}) 예외: {str(e)[:150]}{err_extra}"
        print(f"[포착 차트 영상 예외] {symbol}: {LAST_VIDEO_ERROR}")
        return None


def _render_tradingview_lightweight_chart(df, res, chart_tf="60"):
    """[정적 초고화질 PNG 폴백 렌더러 · 1920x2560]"""
    if not _ensure_playwright():
        return None
    symbol = res.get("symbol", "")
    base = res.get("base_coin") or res.get("base") or symbol.replace("USDT", "")
    score = res.get("score", 88)
    track_type = res.get("track_type", "TRACK_A")
    is_long = bool(res.get("is_long_term") or track_type == "TRACK_D")
    sub_s = str(res.get("sub_strategy", "")).upper()

    latest_row = df.iloc[-1]
    price = to_float(latest_row["close"], to_float(res.get("price")))
    open_last, high_last, low_last = to_float(latest_row["open"], price), to_float(latest_row["high"], price), to_float(latest_row["low"], price)
    orig_entry = to_float(res.get("price"), price)
    ratio_adj = price / orig_entry if orig_entry > 0 else 1.0
    sl_pct = int(round(to_float(res.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
    sl_p = to_float(res.get("stop_price"), price * (1.0 - sl_pct / 100.0)) * ratio_adj
    tps = res.get("tps") or [price * 1.08, price * 1.15, price * 1.22]
    tp_pcts = res.get("tp_pcts") or [8.0, 15.0, 22.0]
    tp1_p, tp2_p, tp3_p = to_float(tps[0]) * ratio_adj, to_float(tps[1]) * ratio_adj, to_float(tps[2]) * ratio_adj
    p1, p2, p3 = round(to_float(tp_pcts[0], 8.0), 1), round(to_float(tp_pcts[1], 15.0), 1), round(to_float(tp_pcts[2], 22.0), 1)

    candle_min_low, candle_max_high = to_float(df["low"].min(), price * 0.97), to_float(df["high"].max(), price * 1.02)
    low_piv, high_piv = min(candle_min_low, price * 0.988), max(candle_max_high, price * 1.010)
    if high_piv <= low_piv:
        low_piv, high_piv = price * 0.985, price * 1.015
    core_span = max(high_piv - low_piv, price * 0.018)
    up_comp = float(np.clip((core_span * 0.36) / max(tp3_p - high_piv, price * 0.02), 0.04, 1.0)) if tp3_p > high_piv else 1.0
    down_comp = float(np.clip((core_span * 0.14) / max(low_piv - sl_p, price * 0.008), 0.06, 1.0)) if sl_p < low_piv else 1.0

    def _fwd(val):
        v = to_float(val, price)
        return (high_piv + (v - high_piv) * up_comp) if v > high_piv else ((low_piv - (low_piv - v) * down_comp) if v < low_piv else v)

    candles_js, ma5_js, ma9_js, ma20_js = [], [], [], []
    for _, row in df.iterrows():
        ts_sec = int(to_float(row["timestamp"]) // 1000) + (9 * 3600)
        candles_js.append({"time": ts_sec, "open": _fwd(row["open"]), "high": _fwd(row["high"]), "low": _fwd(row["low"]), "close": _fwd(row["close"])})
        if pd.notna(row.get("ma5")):
            ma5_js.append({"time": ts_sec, "value": _fwd(row["ma5"])})
        if pd.notna(row.get("ma9")):
            ma9_js.append({"time": ts_sec, "value": _fwd(row["ma9"])})
        if pd.notna(row.get("ma20")):
            ma20_js.append({"time": ts_sec, "value": _fwd(row["ma20"])})

    tf_str = str(chart_tf).upper().strip()
    step_sec = {"720": 43200, "240": 14400, "60": 3600, "15": 900}.get(tf_str, 43200 if is_long else 3600)
    whitespace_js = [{"time": candles_js[-1]["time"] + (i * step_sec)} for i in range(1, 7)]
    y_bot_bound, y_top_bound = min(_fwd(sl_p), _fwd(candle_min_low)) - core_span * 0.05, max(_fwd(tp3_p), _fwd(candle_max_high)) + core_span * 0.06
    kst_str = datetime.fromtimestamp(to_float(latest_row["timestamp"]) / 1000.0, tz=KST).strftime("%Y-%m-%d %H:%M KST")
    tf_name = {"720": "12H", "240": "4H", "60": "1H", "15": "15m"}.get(tf_str, "12H" if is_long else "1H")
    tf_label = f"{tf_name} · LONG-TERM" if is_long else f"{tf_name} · BYBIT"
    track_badge = {
        "TRACK_A": "TRACK A · TTM Squeeze", "TRACK_B": "TRACK B · 24H Breakout",
        "TRACK_C": "TRACK C · Deep Pullback",
        "TRACK_D": "LONG-TERM · Track D6 Spring" if ("D6" in sub_s or "스프링" in sub_s) else "LONG-TERM · Track D Macro",
        "TRACK_E": "TRACK E · SFP Reversal", "TRACK_F": "TRACK F · BTC RS Leader",
        "TRACK_G": "TRACK G · Short Squeeze", "TRACK_H": "TRACK H · Top5 DNA",
    }.get(track_type, track_type)
    decimals = 6 if price < 0.01 else (5 if price < 1 else (4 if price < 10 else (3 if price < 1000 else 2)))
    min_move = 10 ** (-decimals)
    lw_script_tags = _get_lw_charts_script_tags()

    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{lw_script_tags}
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#131722;font-family:'Noto Sans CJK KR','Noto Sans KR',-apple-system,sans-serif;color:#d1d4dc;width:960px;height:1280px;overflow:hidden}}
.tv-wrap{{width:960px;height:1280px;display:flex;flex-direction:column}}
.tv-topbar{{height:112px;background:#181c27;border-bottom:1px solid #2a2e39;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:10px 20px}}
.tb-row{{display:flex;align-items:center;justify-content:space-between}}
.tv-symbol{{font-size:26px;font-weight:900;color:#fff}}.tv-tf-badge{{background:#2a2e39;color:#e0e3eb;font-size:13.5px;font-weight:800;padding:4px 10px;border-radius:5px;margin-left:8px}}
.tv-score-pill{{background:#2962ff;color:#fff;font-size:13px;font-weight:800;padding:5px 12px;border-radius:6px}}
#chart-box{{position:relative;width:960px;height:836px}}
#bottom-panel{{height:300px;background:#0f1422;border-top:2px solid #272e40;padding:18px 24px;display:flex;flex-direction:column;justify-content:center;gap:14px}}
.cc-cta{{background:rgba(41,98,255,0.18);border:1.5px solid rgba(76,201,240,0.45);border-radius:12px;padding:14px 20px;font-size:23px;font-weight:900;color:#fff;text-align:center}}
.cc-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.cc-box{{background:#181e2e;border:1px solid #2a3144;border-radius:12px;padding:15px 10px;text-align:center}}
.cc-k{{font-size:13px;color:#9598a1;display:block}}.cc-v{{font-size:21px;font-weight:900;color:#fff;display:block;margin-top:4px}}
.tv-footer{{height:32px;background:#0a0d14;display:flex;align-items:center;justify-content:space-between;padding:0 18px;font-size:11.5px;color:#8e94a4;font-weight:600}}
</style></head><body><div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tb-row"><div><span class="tv-symbol">${base}USDT.P</span><span class="tv-tf-badge">{tf_label}</span></div><span class="tv-score-pill">{track_badge} · {score} pts</span></div>
    <div class="tb-row"><span style="font-size:14px;color:#26a69a">O {fmt_price(open_last)} H {fmt_price(high_last)} L {fmt_price(low_last)} C {fmt_price(price)}</span><span style="font-size:13px;color:#b2b5be">{kst_str}</span></div>
  </div>
  <div id="chart-box"></div>
  <div id="bottom-panel">
    <div class="cc-cta">{WATERMARK_HANDLE} 팔로우+알림설정 해주세요 ♥</div>
    <div class="cc-grid">
      <div class="cc-box"><span class="cc-k">포착 현재가</span><span class="cc-v">${fmt_price(price)}</span></div>
      <div class="cc-box"><span class="cc-k">손절가 (SL)</span><span class="cc-v" style="color:#ef476f">${fmt_price(sl_p)}</span></div>
      <div class="cc-box"><span class="cc-k">1차 · 2차 목표</span><span class="cc-v" style="color:#4cc9f0">+{p1:g}% / +{p2:g}%</span></div>
      <div class="cc-box"><span class="cc-k">최종 목표 (TP3)</span><span class="cc-v" style="color:#ffd166">${fmt_price(tp3_p)}</span></div>
    </div>
  </div>
  <div class="tv-footer"><span>{VIDEO_FOOTER_NFA_KR}</span><span>{WATERMARK_HANDLE}</span></div>
</div><script>
const lowPiv={low_piv},highPiv={high_piv},upComp={up_comp},downComp={down_comp},decimals={decimals};
function invScale(y){{if(y>highPiv)return highPiv+(y-highPiv)/Math.max(upComp,1e-6);if(y<lowPiv)return lowPiv-(lowPiv-y)/Math.max(downComp,1e-6);return y;}}
function fmtReal(val){{const p=invScale(val);if(p<=0)return '0.0000';return p>=1000?p.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}}):p.toFixed(decimals);}}
const chart=LightweightCharts.createChart(document.getElementById('chart-box'),{{width:960,height:836,layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc'}},grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},rightPriceScale:{{borderColor:'#2a2e39'}},timeScale:{{borderColor:'#2a2e39',timeVisible:true}},localization:{{priceFormatter:fmtReal}}}});
const cs=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',priceLineVisible:false,lastValueVisible:false,priceFormat:{{type:'custom',minMove:{min_move},formatter:fmtReal}}}});
const cData={json.dumps(candles_js)};
cs.setData([...cData,...{json.dumps(whitespace_js)}]);
cs.setMarkers([{{time:cData[cData.length-1].time,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'ENTRY',size:2}}]);
chart.addLineSeries({{color:'#00e5ff',lineWidth:1.8,priceLineVisible:false,lastValueVisible:false}}).setData({json.dumps(ma5_js)});
chart.addLineSeries({{color:'#ffd166',lineWidth:1.8,priceLineVisible:false,lastValueVisible:false}}).setData({json.dumps(ma9_js)});
chart.addLineSeries({{color:'#2962ff',lineWidth:2.2,priceLineVisible:false,lastValueVisible:false}}).setData({json.dumps(ma20_js)});
[{{p:{_fwd(tp3_p)},c:'#ffd166',t:'TP3 (+{p3:g}%)'}},{{p:{_fwd(tp2_p)},c:'#06d6a0',t:'TP2 (+{p2:g}%)'}},{{p:{_fwd(tp1_p)},c:'#4cc9f0',t:'TP1 (+{p1:g}%)'}},{{p:{_fwd(sl_p)},c:'#ef476f',t:'SL (-{sl_pct}%)'}}].forEach(x=>cs.createPriceLine({{price:x.p,color:x.c,lineWidth:2,lineStyle:2,axisLabelVisible:true,title:x.t}}));
chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,priceLineVisible:false,lastValueVisible:false}}).setData([{{time:cData[0].time,value:{y_bot_bound}}},{{time:cData[cData.length-1].time,value:{y_top_bound}}}]);
chart.timeScale().fitContent();
requestAnimationFrame(()=>{{window.__CHART_DONE=true;}});
</script></body></html>"""

    tmp_html_path = None
    try:
        from playwright.sync_api import sync_playwright
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            tmp_html_path = f.name
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"])
            try:
                page = browser.new_page(viewport={"width": CHART_CSS_W, "height": CHART_CSS_H}, device_scale_factor=2.0)
                page.goto(Path(tmp_html_path).resolve().as_uri(), wait_until="load", timeout=25000)
                page.wait_for_function("window.__CHART_DONE === true", timeout=10000)
                page.wait_for_timeout(180)
                return page.screenshot(type="png")
            finally:
                browser.close()
    except Exception as e:
        print(f"[정적 차트 폴백 예외] {symbol}: {e}")
        return None
    finally:
        if tmp_html_path and os.path.exists(tmp_html_path):
            try:
                os.remove(tmp_html_path)
            except Exception:
                pass


def generate_signal_chart_bytes(res):
    symbol, price = res.get("symbol", ""), to_float(res.get("price"))
    if not symbol or price <= 0:
        return None
    is_long = bool(res.get("is_long_term") or res.get("track_type") == "TRACK_D")
    df, actual_tf = _load_signal_chart_df(symbol, is_long)
    if df.empty or len(df) < 20:
        return None
    return _render_tradingview_lightweight_chart(df, res, chart_tf=actual_tf)


def _build_profit_narration_plan(close_info, entry_p, total_realized_pct, created_at, tmpdir, base_total_sec=30.5, lang="kr_jp"):
    """
    [익절 리플레이 영상 5단계 자막 + 음성 싱크]
    - 1/5: 공통 오프닝
    - 2/5: 포착 근거
    - 3/5: 진입가 & 최종 수익률 마감
    - 4/5: 엔딩 전 후원 안내 (X용 한/일: 프로필 링크 Rumble | 럼블 영어: 우측 하단 TIP 럼블월렛)
    - 5/5: 공통 엔딩 (팔로우 + 알림설정)
    """
    base = str(close_info.get("base_coin") or str(close_info.get("symbol", "")).replace("USDT", "")).upper()
    hook_ko = str(close_info.get("x_hook_ko") or "").strip()
    hook_jp = str(close_info.get("x_hook_jp") or "").strip()
    hook_en = str(close_info.get("x_hook_en") or "").strip()
    if not hook_ko or not hook_jp or not hook_en:
        t_ko, t_jp, t_en = build_x_hook_triplet(close_info)
        hook_ko = hook_ko or t_ko
        hook_jp = hook_jp or t_jp
        hook_en = hook_en or t_en

    entry_str = fmt_price(entry_p)
    entry_num = entry_str.replace(",", "")
    pct_str = f"{total_realized_pct:.2f}"
    pct_spoken = f"{total_realized_pct:.1f}"

    if lang == "en":
        voice, rate, gtts_lang = EN_TTS_VOICE, EN_TTS_RATE, "en"
        dt_utc = datetime.fromtimestamp(created_at, tz=timezone.utc)
        date_en = dt_utc.strftime("%b %d")
        date_en_spoken = dt_utc.strftime("%B %d")
        raw_specs = [
            _unified_opening_spec("en", step_str="1 / 5"),
            {
                "step": "2 / 5", "tag": "SIGNAL",
                "l1": f'{date_en}, spotted on "{hook_en}"',
                "l2": f"${base} triggered on the bullish scanner",
                "sub": "",
                "tts_text": f"On {date_en_spoken}, {base} was spotted on the scanner based on {hook_en}.",
            },
            {
                "step": "3 / 5", "tag": "PROFIT",
                "l1": f"Entry ${entry_str} · Final Return +{pct_str}%",
                "l2": "Position closed in profit ❤️",
                "sub": "",
                "tts_text": f"Entry price {entry_num} dollars. Closed with a final return of plus {pct_spoken} percent.",
            },
            {
                "step": "4 / 5", "tag": "SUPPORT",
                "l1": "If this helped you take profit, a coffee or server tip",
                "l2": "via TIP (Rumble Wallet) at the bottom right is a huge help ☕",
                "sub": "",
                "tts_text": "If this helped you take profit, supporting server costs or a cup of coffee using the TIP button with Rumble Wallet at the bottom right would be a huge help.",
            },
            _unified_ending_spec("en", step_str="5 / 5"),
        ]
    else:
        voice, rate, gtts_lang = JP_TTS_VOICE, JP_TTS_RATE, "ja"
        dt_kst = datetime.fromtimestamp(created_at, tz=KST)
        date_ko, date_jp = f"{dt_kst.month}월 {dt_kst.day}일", f"{dt_kst.month}月{dt_kst.day}日"
        raw_specs = [
            _unified_opening_spec("kr_jp", step_str="1 / 5"),
            {
                "step": "2 / 5", "tag": "SIGNAL",
                "l1": f'{date_ko}, "{hook_ko}" 근거로',
                "l2": f"${base} 종목이 검색기에 포착되었습니다",
                "sub": f"{date_jp}、「{hook_jp}」で{base}を捕捉",
                "tts_text": f"{date_jp}、「{hook_jp}」の根拠で、{base}がスキャナーに捕捉されました。",
            },
            {
                "step": "3 / 5", "tag": "PROFIT",
                "l1": f"진입가격 {entry_str} 달러 · 최종 수익률 +{pct_str}%",
                "l2": "마감되었습니다 ❤️",
                "sub": f"エントリー ${entry_str} · 最終収益率 +{pct_str}% で決済されました ❤️",
                "tts_text": f"エントリー価格、{entry_num}ドル。最終収益率、プラス{pct_spoken}パーセントで決済されました。",
            },
            {
                "step": "4 / 5", "tag": "SUPPORT",
                "l1": "수익실현의 도움이 되셨다면, 프로필 링크(Rumble)에서",
                "l2": "서버비·커피 한 잔 후원해주시면 큰 힘이 됩니다 ☕",
                "sub": "利益確定のお役に立てましたら、プロフィールのRumbleリンクからサーバー代やコーヒー1杯のご支援をいただけると大きな励みになります ☕",
                "tts_text": "利益確定のお役に立てましたら、プロフィールのランブルリンクから、サーバー代やコーヒー一杯のご支援をいただけると、大きな励みになります。",
            },
            _unified_ending_spec("kr_jp", step_str="5 / 5"),
        ]

    durs, paths = [], []
    tts_ok, tts_err = True, ""
    for idx, sp in enumerate(raw_specs):
        mp3_p = os.path.join(tmpdir, f"tts_seg_{lang}_{idx}.mp3")
        ok, err = _synth_voice_segment(sp["tts_text"], mp3_p, voice=voice, rate=rate, gtts_lang=gtts_lang)
        if not ok:
            tts_ok, tts_err = False, err
            break
        durs.append(_probe_audio_duration_sec(mp3_p, default_sec=3.5))
        paths.append(mp3_p)

    if not tts_ok or len(durs) != len(raw_specs):
        return [], base_total_sec, False, tts_err

    s0 = 0.25
    e0 = s0 + durs[0]
    s1 = max(e0 + 0.50, 2.30)
    e1 = s1 + durs[1]
    s2 = max(e1 + 1.00, 15.00)
    e2 = s2 + durs[2]
    s3 = max(e2 + 0.70, 22.00)
    e3 = s3 + durs[3]
    s4 = max(e3 + 0.70, 28.00)
    e4 = s4 + durs[4]
    final_total_sec = max(base_total_sec, e4 + 0.90)

    starts = [s0, s1, s2, s3, s4]
    ends = [e0, e1, e2, e3, e4]
    segments = []
    for i, sp in enumerate(raw_specs):
        segments.append({
            "step": sp["step"], "tag": sp["tag"],
            "l1": sp["l1"], "l2": sp["l2"], "sub": sp["sub"],
            "audio_path": paths[i], "audio_start": round(starts[i], 3),
            "audio_dur": round(durs[i], 3), "audio_end": round(ends[i], 3),
            "sub_start": round(0.0 if i == 0 else starts[i] - 0.15, 3),
            "sub_end": round(final_total_sec if i == len(raw_specs) - 1 else starts[i + 1] - 0.15, 3),
        })
    return segments, final_total_sec, True, ""


def generate_profit_replay_videos_bundle(close_info, include_clean=True, lang="kr_jp"):
    """
    [익절 리플레이 풀HD 세로형 1080x1440 엔진]
    - lang="kr_jp": X용 (5단계 한/일 자막 · 일본어 음성 · 4/5 Rumble 프로필 링크 후원 안내 포함)
    - lang="en": 럼블 전용 (5단계 100% 영문 차트·자막·음성 · 4/5 우측 하단 TIP 럼블월렛 후원 안내 포함)
    """
    global LAST_VIDEO_ERROR
    LAST_VIDEO_ERROR = ""
    is_en = (str(lang).lower() == "en")

    if not shutil.which("ffmpeg"):
        LAST_VIDEO_ERROR = "ffmpeg 미설치"
        return None
    if not _ensure_playwright():
        LAST_VIDEO_ERROR = "Playwright 준비 실패"
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
    tf_ladder = [("1", "1m", 60, 860), ("5", "5m", 300, 480), ("15", "15m", 900, 480), ("60", "1H", 3600, 300), ("240", "4H", 14400, 300)]
    start_tier = 0 if hold_hours < 10.0 else (1 if hold_hours < 20.0 else (2 if hold_hours < 70.0 else (3 if hold_hours < 90.0 else 4)))

    created_ms, closed_ms = created_at * 1000, closed_at * 1000
    df_raw, tf_code, tf_label, step_sec, best_fallback = pd.DataFrame(), *tf_ladder[start_tier][:3], None
    for idx_t in range(start_tier, len(tf_ladder)):
        cand_code, cand_lbl, cand_step, cand_max = tf_ladder[idx_t]
        cand_df = _fetch_replay_klines_window(symbol, interval=cand_code, start_ms=created_ms - (85 * cand_step * 1000), end_ms=closed_ms + (2 * cand_step * 1000), max_bars=cand_max)
        if cand_df.empty or len(cand_df) < 10:
            continue
        if best_fallback is None:
            best_fallback = (cand_df, cand_code, cand_lbl, cand_step)
        if int(to_float(cand_df["timestamp"].iloc[0], 0)) <= (created_ms + cand_step * 1000 * 3):
            df_raw, tf_code, tf_label, step_sec = cand_df, cand_code, cand_lbl, cand_step
            break
    if df_raw.empty and best_fallback is not None:
        df_raw, tf_code, tf_label, step_sec = best_fallback
    if df_raw.empty:
        LAST_VIDEO_ERROR = f"{symbol} {tf_label} 캔들 데이터 API 조회 실패"
        return None

    df = df_raw.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", ascending=True).reset_index(drop=True)
    df["ma5"] = df["close"].rolling(5, min_periods=2).mean()
    df["ma9"] = df["close"].rolling(9, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()

    ts_arr = df["timestamp"].astype("int64").values
    entry_matches, exit_matches = np.where(ts_arr <= created_ms)[0], np.where(ts_arr <= closed_ms)[0]
    entry_abs = int(entry_matches[-1]) if len(entry_matches) > 0 else 0
    exit_abs = int(exit_matches[-1]) if len(exit_matches) > 0 else (len(df) - 1)
    if exit_abs < entry_abs:
        exit_abs = min(len(df) - 1, entry_abs)

    post_count = (exit_abs - entry_abs) + 1
    pre_bars = 44 if post_count <= 90 else (34 if post_count <= 220 else 26)
    slice_start = max(0, entry_abs - pre_bars)
    df_win = df.iloc[slice_start : exit_abs + 1].reset_index(drop=True)
    if len(df_win) < 4:
        LAST_VIDEO_ERROR = f"{symbol} 리플레이 구간 캔들 부족({len(df_win)}개)"
        return None

    entry_idx, exit_idx = max(0, entry_abs - slice_start), len(df_win) - 1
    is_long = bool(close_info.get("is_long_term") or close_info.get("track_type") == "TRACK_D")
    track_type = str(close_info.get("track_type", "TRACK_A"))
    close_type = str(close_info.get("close_type", "TP3_TRAILING"))
    sl_pct = int(round(to_float(close_info.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
    sl_p = to_float(close_info.get("initial_stop_price"), entry_p * (1.0 - sl_pct / 100.0))
    tps = close_info.get("tps") or [entry_p * 1.08, entry_p * 1.15, entry_p * 1.22]
    tp1_p, tp2_p, tp3_p = to_float(tps[0], entry_p * 1.08), to_float(tps[1], entry_p * 1.15), to_float(tps[2], entry_p * 1.22)
    p1, p2, p3 = round((tp1_p - entry_p) / entry_p * 100.0, 1), round((tp2_p - entry_p) / entry_p * 100.0, 1), round((tp3_p - entry_p) / entry_p * 100.0, 1)

    max_high_val = max(to_float(close_info.get("max_high_price"), entry_p), to_float(df_win["high"].max(), entry_p))
    exit_p = to_float(close_info.get("exit_price"), to_float(df_win["close"].iloc[-1], entry_p))
    total_realized_pct = round(to_float(close_info.get("realized_weighted_pct"), max(p1, (exit_p - entry_p) / entry_p * 100.0)), 2)
    max_fav_val = round(max(to_float(close_info.get("max_favorable_pct"), total_realized_pct), (max_high_val - entry_p) / entry_p * 100.0), 1)

    tz_use = timezone.utc if is_en else KST
    tz_offset_sec = 0 if is_en else (9 * 3600)
    tz_suffix = "UTC" if is_en else "KST"

    hold_str = _fmt_hold_duration(hold_minutes, lang="en" if is_en else "kr_jp")
    entry_dt_str = datetime.fromtimestamp(created_at, tz=tz_use).strftime(f"%Y-%m-%d %H:%M{' UTC' if is_en else ''}")
    exit_dt_str = datetime.fromtimestamp(closed_at, tz=tz_use).strftime(f"%Y-%m-%d %H:%M{' UTC' if is_en else ''}")

    win_low = min(to_float(df_win["low"].min(), entry_p * 0.94), sl_p, entry_p)
    win_high = max(to_float(df_win["high"].max(), entry_p * 1.15), tp2_p, exit_p)
    if close_type == "TP3_TRAILING" or max_high_val >= tp3_p * 0.92:
        win_high = max(win_high, tp3_p)
    y_span = max(win_high - win_low, entry_p * 0.05)
    y_min_anchor, y_max_anchor = win_low - (y_span * 0.28), win_high + (y_span * 0.46)

    candles_js, vol_js, ma5_js, ma9_js, ma20_js, time_strs = [], [], [], [], [], []
    last_ts_sec = -1
    for _, row in df_win.iterrows():
        ts_ms = to_float(row["timestamp"])
        ts_sec = int(ts_ms // 1000) + tz_offset_sec
        if ts_sec <= last_ts_sec:
            ts_sec = last_ts_sec + step_sec
        last_ts_sec = ts_sec
        o_v, h_v, l_v, c_v, v_v = to_float(row["open"]), to_float(row["high"]), to_float(row["low"]), to_float(row["close"]), to_float(row["volume"])
        candles_js.append({"time": ts_sec, "open": o_v, "high": h_v, "low": l_v, "close": c_v})
        vol_js.append({"time": ts_sec, "value": v_v, "color": "rgba(38, 166, 154, 0.38)" if c_v >= o_v else "rgba(239, 83, 80, 0.38)"})
        ma5_js.append({"time": ts_sec, "value": to_float(row["ma5"])} if pd.notna(row.get("ma5")) else None)
        ma9_js.append({"time": ts_sec, "value": to_float(row["ma9"])} if pd.notna(row.get("ma9")) else None)
        ma20_js.append({"time": ts_sec, "value": to_float(row["ma20"])} if pd.notna(row.get("ma20")) else None)
        time_strs.append(datetime.fromtimestamp(ts_ms / 1000.0, tz=tz_use).strftime(f"%Y-%m-%d %H:%M {tz_suffix}"))

    tail_bars = max(5, min(42, int(round(len(candles_js) * 0.08))))
    ws_tail = [{"time": candles_js[-1]["time"] + (i * step_sec)} for i in range(1, tail_bars + 1)]
    decimals = 6 if entry_p < 0.01 else (5 if entry_p < 1 else (4 if entry_p < 10 else (3 if entry_p < 1000 else 2)))
    min_move = 10 ** (-decimals)
    lw_script_tags = _get_lw_charts_script_tags()

    pos_badge = "LONG-TERM REPLAY" if is_long else f"{track_type.replace('_', ' ')} REPLAY"
    price_diff_abs = abs(exit_p - entry_p)
    price_diff_prefix = "+" if exit_p >= entry_p else "-"
    exit_headline = f"+{total_realized_pct:.2f}% ({price_diff_prefix}{fmt_price(price_diff_abs)}) · {hold_str}"

    t_ko, _, t_en = build_x_hook_triplet(close_info)
    hook_card = (str(close_info.get("x_hook_en") or "").strip() or t_en) if is_en else (str(close_info.get("x_hook_ko") or "").strip() or t_ko)

    ui_badge = "🎙️ DANPYUNG PROFIT REPLAY · EN" if is_en else "🎙️ 단평 익절 리플레이 · KR / JP"
    ui_footer = VIDEO_FOOTER_NFA_EN if is_en else VIDEO_FOOTER_NFA_KR
    ui_clean_lbls = ("Entry Price", "Peak Gain", "Final Return", "Holding Time") if is_en else ("진입가격", "최고 상승폭", "최종 수익률", "보유 기간")
    ui_stages = {
        "prep": "🟢 ENTRY Spotted · Starting Replay" if is_en else "🟢 ENTRY 포착 완료 · 리플레이 시작",
        "tp3": "🚀 TP3 50% Secured · Trailing Active" if is_en else "🚀 TP3 50% 익절 · 트레일링 가동",
        "tp2": "🛡️ TP2 Confirmed · Stop Moved to TP1" if is_en else "🛡️ TP2 안착 · TP1 방어선 고정",
        "tp1": "✅ TP1 Hit · Holding 100%" if is_en else "✅ TP1 돌파 · 100% 홀딩",
    }

    in_fps, out_fps = 10, 30
    base_frames, thumb_end, play_start, play_end, outro_static_start = 305, 20, 30, 259, 270
    js_errors = []

    try:
        from playwright.sync_api import sync_playwright
        with tempfile.TemporaryDirectory() as tmpdir:
            segments, final_total_sec, tts_ok, tts_err = _build_profit_narration_plan(
                close_info, entry_p, total_realized_pct, created_at, tmpdir, base_total_sec=base_frames / float(in_fps), lang=lang
            )
            make_voiced = bool(tts_ok)
            make_clean = bool(include_clean) or (not make_voiced)
            voiced_frames = max(base_frames, int(round(final_total_sec * in_fps))) if make_voiced else base_frames
            sub_meta_js = [{"step": s["step"], "tag": s["tag"], "l1": s["l1"], "l2": s["l2"], "sub": s["sub"]} for s in segments]
            init_mode = "sub" if (make_voiced and not make_clean) else "clean"
            sub_jp_display = "none" if is_en else "block"

            html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">{lw_script_tags}
<style>
*{{box-sizing:border-box;margin:0;padding:0;-webkit-font-smoothing:antialiased}}
body{{background:#131722;font-family:'Noto Sans CJK KR','Noto Sans KR','Noto Sans CJK JP','Segoe UI',-apple-system,Roboto,sans-serif;color:#d1d4dc;width:960px;height:1280px;overflow:hidden}}
.tv-wrap{{width:960px;height:1280px;background:#131722;display:flex;flex-direction:column}}
.tv-topbar{{height:112px;background:#181c27;border-bottom:1px solid #2a2e39;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:10px 20px}}
.tb-row{{display:flex;align-items:center;justify-content:space-between}}
.tv-left,.tv-right{{display:flex;align-items:center;gap:10px}}
.tv-symbol{{font-size:26px;font-weight:900;color:#fff}}.tv-tf-badge{{background:#2a2e39;color:#e0e3eb;font-size:14px;font-weight:800;padding:4px 10px;border-radius:5px}}
.tv-replay-pill{{background:#2962ff;color:#fff;font-size:13px;font-weight:800;padding:5px 12px;border-radius:6px}}
.tv-handle-pill{{background:rgba(255,255,255,0.08);border:1px solid #3d4354;color:#f0f3fa;font-size:14px;font-weight:800;padding:5px 12px;border-radius:6px}}
.tv-ohlc{{display:flex;gap:12px;font-size:14px;font-weight:700;color:#9598a1}}.tv-val-up{{color:#26a69a}}.tv-val-dn{{color:#ef5350}}
#chart-box{{position:relative;width:960px;height:836px;overflow:hidden}}
.tv-ma-legend{{position:absolute;top:10px;left:14px;z-index:20;display:flex;gap:12px;font-size:12.5px;font-weight:700;background:rgba(19,23,34,0.88);padding:5px 12px;border-radius:6px;border:1px solid #2a2e39}}
.live-hud{{position:absolute;top:48px;left:50%;transform:translateX(-50%);z-index:25;display:flex;flex-direction:column;align-items:center;gap:8px;pointer-events:none}}
.pnl-pill{{background:rgba(15,20,32,0.90);border:2.5px solid #26a69a;border-radius:14px;padding:10px 30px;display:flex;flex-direction:column;align-items:center;min-width:238px;box-shadow:0 10px 28px rgba(0,0,0,0.55)}}
.pnl-label{{font-size:13px;font-weight:800;color:#9aa3b2;letter-spacing:1.2px}}.pnl-val{{font-size:44px;font-weight:900;color:#00e676;line-height:1.1;margin-top:2px}}
.stage-pill{{background:rgba(41,98,255,0.28);border:1.5px solid #4cc9f0;color:#e0f7fa;font-size:15px;font-weight:800;padding:7px 16px;border-radius:9px;display:none;white-space:nowrap}}
#result-overlay{{position:absolute;inset:0;pointer-events:none;z-index:30;opacity:0}}
#arc-svg{{position:absolute;inset:0;width:960px;height:836px;overflow:visible}}
.tv-callout{{position:absolute;background:linear-gradient(135deg,#2962ff 0%,#1d4ed8 100%);color:#fff;border-radius:12px;box-shadow:0 12px 28px rgba(0,0,0,0.62);white-space:nowrap}}
.tv-callout-exit{{padding:11px 18px}}.tv-callout-exit .co-main{{font-size:23px;font-weight:900}}.tv-callout-exit .co-sub{{font-size:16px;font-weight:800;color:#e8f0ff;margin-top:4px;display:flex;gap:12px}}
.tv-callout-entry{{padding:10px 16px}}.tv-callout-entry .co-main{{font-size:20px;font-weight:900}}.tv-callout-entry .co-sub{{font-size:15px;font-weight:700;color:#e0ecff;margin-top:3px}}
.co-caret-down{{position:absolute;bottom:-9px;width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-top:9px solid #1f52df}}
.co-caret-up{{position:absolute;top:-9px;width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:9px solid #2962ff}}
#bottom-panel{{height:300px;width:960px;background:linear-gradient(180deg,#121725 0%,#0c101a 100%);border-top:2px solid #272e40;padding:16px 24px;display:flex;flex-direction:column;justify-content:center}}
#clean-card{{display:flex;flex-direction:column;gap:14px;height:100%;justify-content:center}}
.cc-hook{{background:rgba(41,98,255,0.16);border:1px solid rgba(76,201,240,0.35);border-radius:12px;padding:12px 18px;font-size:19px;font-weight:800;color:#e8f1ff;text-align:center}}
.cc-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.cc-box{{background:#181e2e;border:1px solid #2a3144;border-radius:12px;padding:14px 12px;display:flex;flex-direction:column;align-items:center;gap:6px}}
.cc-k{{font-size:13px;font-weight:700;color:#9598a1}}.cc-v{{font-size:22px;font-weight:900;color:#fff}}.cc-v-profit{{color:#00e676;font-size:25px}}
#sub-card{{display:none;flex-direction:column;justify-content:center;gap:12px;height:100%}}
.sub-head{{display:flex;align-items:center;justify-content:space-between}}
.sub-badge{{background:#2962ff;color:#fff;font-size:13.5px;font-weight:800;padding:5px 14px;border-radius:999px;letter-spacing:0.3px}}
.sub-step{{background:rgba(255,255,255,0.08);border:1px solid #3a4258;color:#cbd5e1;font-size:13px;font-weight:800;padding:4px 12px;border-radius:999px}}
.sub-box{{width:100%;background:rgba(24,30,46,0.96);border:2px solid rgba(76,201,240,0.48);border-radius:16px;padding:18px 22px;text-align:center;box-shadow:0 10px 26px rgba(0,0,0,0.45);display:flex;flex-direction:column;gap:10px;justify-content:center;min-height:212px}}
.sub-l1{{font-size:27px;font-weight:900;color:#fff;line-height:1.34;word-break:keep-all;text-shadow:0 2px 6px rgba(0,0,0,0.55)}}
.sub-l2{{font-size:25px;font-weight:900;color:#ffd166;line-height:1.32;word-break:keep-all;text-shadow:0 2px 6px rgba(0,0,0,0.55)}}
.sub-jp{{display:{sub_jp_display};font-size:17px;font-weight:700;color:#93c5fd;line-height:1.32;border-top:1px solid rgba(255,255,255,0.10);padding-top:8px;margin-top:2px;word-break:keep-all}}
.tv-footer{{height:32px;background:#0a0d14;border-top:1px solid #1f2430;display:flex;align-items:center;justify-content:space-between;padding:0 18px;font-size:11.5px;color:#8e94a4;font-weight:600}}
.prog-wrap{{width:150px;height:5px;background:#222632;border-radius:3px;overflow:hidden}}.prog-fill{{height:100%;width:0%;background:linear-gradient(90deg,#2962ff,#00e676)}}
</style></head><body><div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tb-row"><div class="tv-left"><span class="tv-symbol">${base}USDT.P</span><span class="tv-tf-badge">{tf_label} · REPLAY</span></div><div class="tv-right"><span class="tv-replay-pill">{pos_badge}</span><span class="tv-handle-pill">{WATERMARK_HANDLE}</span></div></div>
    <div class="tb-row"><div class="tv-ohlc"><span>O <span id="oh-o" class="tv-val-up">{fmt_price(entry_p)}</span></span><span>H <span id="oh-h" class="tv-val-up">{fmt_price(entry_p)}</span></span><span>L <span id="oh-l" class="tv-val-up">{fmt_price(entry_p)}</span></span><span>C <span id="oh-c" class="tv-val-up">{fmt_price(entry_p)}</span></span></div><span id="top-time" style="font-size:13px;font-weight:700;color:#b2b5be">{time_strs[exit_idx]}</span></div>
  </div>
  <div id="chart-box">
    <div class="tv-ma-legend"><span style="color:#00e5ff">● MA(5) <span id="lg-m5">-</span></span><span style="color:#ffd166">● MA(9) <span id="lg-m9">-</span></span><span style="color:#2962ff">● MA(20) <span id="lg-m20">-</span></span></div>
    <div class="live-hud" id="live-hud"><div class="pnl-pill" id="pnl-box"><span class="pnl-label">LIVE RETURN</span><span class="pnl-val" id="pnl-val">+0.0%</span></div><div id="stage-pill" class="stage-pill"></div></div>
    <div id="result-overlay">
      <svg id="arc-svg" viewBox="0 0 960 836">
        <line id="exit-guide" x1="0" y1="0" x2="0" y2="0" stroke="#60a5fa" stroke-width="2.2" stroke-dasharray="5,4" opacity="0"/>
        <line id="entry-guide" x1="0" y1="0" x2="0" y2="0" stroke="#60a5fa" stroke-width="2.2" stroke-dasharray="5,4" opacity="0"/>
        <path id="arc-path" d="" fill="none" stroke="#2962ff" stroke-width="4.4" stroke-linecap="round"/>
        <circle id="dot-entry" cx="0" cy="0" r="6.5" fill="#2962ff" stroke="#fff" stroke-width="2.2"/>
        <circle id="dot-exit" cx="0" cy="0" r="6.5" fill="#2962ff" stroke="#fff" stroke-width="2.2"/>
        <polygon id="arc-head" points="0,0 -15,-7 -11,0 -15,7" fill="#3b82f6" stroke="#fff" stroke-width="0.8"/>
      </svg>
      <div id="callout-exit" class="tv-callout tv-callout-exit"><div class="co-main">{exit_headline}</div><div class="co-sub"><span>EXIT ${fmt_price(exit_p)}</span><span>🕒 {exit_dt_str}</span></div><div id="caret-exit" class="co-caret-down"></div></div>
      <div id="callout-entry" class="tv-callout tv-callout-entry"><div id="caret-entry" class="co-caret-up"></div><div class="co-main">ENTRY ${fmt_price(entry_p)}</div><div class="co-sub">🕒 {entry_dt_str}</div></div>
    </div>
  </div>
  <div id="bottom-panel">
    <div id="clean-card"><div class="cc-hook">📌 {html.escape(hook_card)}</div><div class="cc-grid"><div class="cc-box"><span class="cc-k">{ui_clean_lbls[0]}</span><span class="cc-v">${fmt_price(entry_p)}</span></div><div class="cc-box"><span class="cc-k">{ui_clean_lbls[1]}</span><span class="cc-v" style="color:#4cc9f0">+{max_fav_val:.1f}%</span></div><div class="cc-box"><span class="cc-k">{ui_clean_lbls[2]}</span><span class="cc-v cc-v-profit">+{total_realized_pct:.2f}%</span></div><div class="cc-box"><span class="cc-k">{ui_clean_lbls[3]}</span><span class="cc-v">{hold_str}</span></div></div></div>
    <div id="sub-card"><div class="sub-head"><span class="sub-badge">{ui_badge}</span><span class="sub-step" id="sub-step">1 / 5</span></div><div class="sub-box" id="sub-box"><div class="sub-l1" id="sub-l1"></div><div class="sub-l2" id="sub-l2"></div><div class="sub-jp" id="sub-jp"></div></div></div>
  </div>
  <div class="tv-footer"><span>{ui_footer}</span><div class="prog-wrap"><div class="prog-fill" id="prog-fill"></div></div></div>
</div><script>
const cAll={json.dumps(candles_js)},vAll={json.dumps(vol_js)},m5All={json.dumps(ma5_js)},m9All={json.dumps(ma9_js)},m20All={json.dumps(ma20_js)},tAll={json.dumps(time_strs)},wsTail={json.dumps(ws_tail)},subMeta={json.dumps(sub_meta_js, ensure_ascii=False)},uiS={json.dumps(ui_stages, ensure_ascii=False)};
const entryIdx={entry_idx},exitIdx={exit_idx},entryP={entry_p},exitP={exit_p},tp1P={tp1_p},tp2P={tp2_p},tp3P={tp3_p},slP={sl_p},decimals={decimals},yMinAnchor={y_min_anchor},yMaxAnchor={y_max_anchor};
function fmtP(v){{if(!v||v<=0)return '0.0000';return v>=1000?v.toLocaleString('en-US',{{minimumFractionDigits:2,maximumFractionDigits:2}}):v.toFixed(decimals);}}
const HEART_SVG='<svg style="display:inline-block;width:0.95em;height:0.95em;vertical-align:-0.12em;margin-left:4px" viewBox="0 0 24 24" fill="#ff3b5c"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>';
const COFFEE_SVG='<svg style="display:inline-block;width:1.0em;height:1.0em;vertical-align:-0.14em;margin-left:5px" viewBox="0 0 24 24" fill="none"><path d="M4 9h12v5a5 5 0 0 1-5 5H9a5 5 0 0 1-5-5V9z" fill="#d97706" stroke="#fde68a" stroke-width="1.6"/><path d="M16 10h2.2a2.3 2.3 0 0 1 0 4.6H16" stroke="#fde68a" stroke-width="1.8" stroke-linecap="round"/><path d="M8 6.2c0-1 .8-1.4.8-2.2M11.5 6.2c0-1 .8-1.4.8-2.2" stroke="#fde68a" stroke-width="1.6" stroke-linecap="round"/><path d="M3 21h14" stroke="#fde68a" stroke-width="1.8" stroke-linecap="round"/></svg>';
function setTextWithIcons(el,txt){{
  const s=String(txt||'');
  if(s.indexOf('❤️')===-1&&s.indexOf('☕')===-1){{el.textContent=s;return;}}
  const esc=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  el.innerHTML=esc.split('❤️').join(HEART_SVG).split('☕').join(COFFEE_SVG);
}}
const chart=LightweightCharts.createChart(document.getElementById('chart-box'),{{width:960,height:836,layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},watermark:{{visible:true,fontSize:66,horzAlign:'center',vertAlign:'center',color:'rgba(209,212,220,0.14)',text:'{WATERMARK_HANDLE}',fontStyle:'bold'}},rightPriceScale:{{borderColor:'#2a2e39',scaleMargins:{{top:0.06,bottom:0.09}},entireTextOnly:true}},timeScale:{{borderColor:'#2a2e39',timeVisible:true,rightOffset:2}},localization:{{priceFormatter:fmtP}}}});
chart.addLineSeries({{color:'rgba(0,0,0,0)',lineWidth:1,priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false}}).setData([{{time:cAll[0].time,value:yMinAnchor}},{{time:wsTail[wsTail.length-1].time,value:yMaxAnchor}}]);
const candleSeries=chart.addCandlestickSeries({{upColor:'#26a69a',downColor:'#ef5350',borderUpColor:'#26a69a',borderDownColor:'#ef5350',wickUpColor:'#26a69a',wickDownColor:'#ef5350',priceLineVisible:false,lastValueVisible:false,priceFormat:{{type:'custom',minMove:{min_move},formatter:fmtP}}}});
const ma5Series=chart.addLineSeries({{color:'#00e5ff',lineWidth:1.8,priceLineVisible:false,lastValueVisible:false}});
const ma9Series=chart.addLineSeries({{color:'#ffd166',lineWidth:1.8,priceLineVisible:false,lastValueVisible:false}});
const ma20Series=chart.addLineSeries({{color:'#2962ff',lineWidth:2.2,priceLineVisible:false,lastValueVisible:false}});
const volSeries=chart.addHistogramSeries({{priceFormat:{{type:'volume'}},priceScaleId:'vol_scale',priceLineVisible:false,lastValueVisible:false}});
chart.priceScale('vol_scale').applyOptions({{scaleMargins:{{top:0.90,bottom:0.0}}}});
[{{p:tp3P,c:'#ffd166',t:'TP3 (+{p3:g}%)'}},{{p:tp2P,c:'#06d6a0',t:'TP2 (+{p2:g}%)'}},{{p:tp1P,c:'#4cc9f0',t:'TP1 (+{p1:g}%)'}},{{p:slP,c:'#ef476f',t:'SL (-{sl_pct}%)'}}].forEach(x=>candleSeries.createPriceLine({{price:x.p,color:x.c,lineWidth:2,lineStyle:2,axisLabelVisible:true,title:x.t}}));
candleSeries.setData(cAll.concat(wsTail));chart.timeScale().fitContent();

const elO=document.getElementById('oh-o'),elH=document.getElementById('oh-h'),elL=document.getElementById('oh-l'),elC=document.getElementById('oh-c'),elTime=document.getElementById('top-time'),elM5=document.getElementById('lg-m5'),elM9=document.getElementById('lg-m9'),elM20=document.getElementById('lg-m20');
const elHud=document.getElementById('live-hud'),elPnl=document.getElementById('pnl-val'),elPnlBox=document.getElementById('pnl-box'),elStage=document.getElementById('stage-pill'),elProg=document.getElementById('prog-fill');
const elOverlay=document.getElementById('result-overlay'),elArcPath=document.getElementById('arc-path'),elArcHead=document.getElementById('arc-head'),elExitGuide=document.getElementById('exit-guide'),elEntryGuide=document.getElementById('entry-guide'),elDotEntry=document.getElementById('dot-entry'),elDotExit=document.getElementById('dot-exit'),elCoExit=document.getElementById('callout-exit'),elCoEntry=document.getElementById('callout-entry'),elCaretExit=document.getElementById('caret-exit'),elCaretEntry=document.getElementById('caret-entry');
const elCleanCard=document.getElementById('clean-card'),elSubCard=document.getElementById('sub-card'),elSubBox=document.getElementById('sub-box'),elSubStep=document.getElementById('sub-step'),elSubL1=document.getElementById('sub-l1'),elSubL2=document.getElementById('sub-l2'),elSubJp=document.getElementById('sub-jp');
let arcTotalLen=1,geometryReady=false,showExitGuide=false,showEntryGuide=false;

function setupResultOverlayGeometry(){{
  const psW=(chart.priceScale('right')&&chart.priceScale('right').width)?chart.priceScale('right').width():86;
  const plotRight=Math.max(360,960-(psW||86)-8),totalSlots=cAll.length+wsTail.length;
  let x1=chart.timeScale().timeToCoordinate(cAll[entryIdx].time),x2=chart.timeScale().timeToCoordinate(cAll[exitIdx].time);
  if(x1===null||!isFinite(x1))x1=32+(entryIdx/Math.max(1,totalSlots))*(plotRight-64);
  if(x2===null||!isFinite(x2))x2=32+(exitIdx/Math.max(1,totalSlots))*(plotRight-64);
  let y1=candleSeries.priceToCoordinate(entryP),y1Low=candleSeries.priceToCoordinate(cAll[entryIdx].low),y2=candleSeries.priceToCoordinate(exitP),y2High=candleSeries.priceToCoordinate(cAll[exitIdx].high);
  if(y1===null||!isFinite(y1))y1=560;if(y1Low===null||!isFinite(y1Low))y1Low=y1;if(y2===null||!isFinite(y2))y2=260;if(y2High===null||!isFinite(y2High))y2High=y2;
  y1=Math.max(110,Math.min(700,y1));y1Low=Math.max(y1,Math.min(710,y1Low));y2=Math.max(120,Math.min(640,y2));y2High=Math.max(95,Math.min(y2,y2High));
  let peakY=y2High;for(let i=entryIdx;i<=exitIdx;i++){{const hy=candleSeries.priceToCoordinate(cAll[i].high);if(hy!==null&&isFinite(hy)&&hy<peakY)peakY=hy;}}
  const dx=x2-x1,dy=y1-y2;
  const cx=(dx>=40)?Math.min(plotRight-14,x1+dx*0.90+14):Math.min(plotRight-14,Math.max(x1,x2)+64);
  const cy=(dx>=40)?Math.min(740,y1-dy*0.04+14):Math.min(740,y1-dy*0.25+16);
  elArcPath.setAttribute('d','M '+x1.toFixed(1)+' '+y1.toFixed(1)+' Q '+cx.toFixed(1)+' '+cy.toFixed(1)+' '+x2.toFixed(1)+' '+y2.toFixed(1));
  elDotEntry.setAttribute('cx',x1.toFixed(1));elDotEntry.setAttribute('cy',y1.toFixed(1));elDotExit.setAttribute('cx',x2.toFixed(1));elDotExit.setAttribute('cy',y2.toFixed(1));
  arcTotalLen=Math.max(1,elArcPath.getTotalLength());elArcPath.style.strokeDasharray=arcTotalLen.toFixed(1);
  const exitW=elCoExit.offsetWidth||360,exitH=elCoExit.offsetHeight||74;
  let exitLeft=Math.max(14,Math.min(plotRight-exitW-10,x2-exitW*0.52)),exitTop=Math.max(48,Math.min(y2,y2High,peakY+10)-exitH-46);
  elCoExit.style.left=exitLeft.toFixed(1)+'px';elCoExit.style.top=exitTop.toFixed(1)+'px';
  const exitCaretX=Math.max(16,Math.min(exitW-30,x2-exitLeft-9));elCaretExit.style.left=exitCaretX.toFixed(1)+'px';
  const exitTipX=exitLeft+exitCaretX+9,exitTipY=exitTop+exitH+9;
  if((y2-exitTipY)>14){{elExitGuide.setAttribute('x1',exitTipX.toFixed(1));elExitGuide.setAttribute('y1',exitTipY.toFixed(1));elExitGuide.setAttribute('x2',x2.toFixed(1));elExitGuide.setAttribute('y2',(y2-7).toFixed(1));showExitGuide=true;}}
  const entryW=elCoEntry.offsetWidth||240,entryH=elCoEntry.offsetHeight||66;
  let entryLeft=Math.max(14,Math.min(plotRight-entryW-10,x1-entryW*0.35)),entryTop=Math.min(798-entryH,Math.max(y1Low+46,exitTop+exitH+16));
  elCoEntry.style.left=entryLeft.toFixed(1)+'px';elCoEntry.style.top=entryTop.toFixed(1)+'px';
  const entryCaretX=Math.max(16,Math.min(entryW-30,x1-entryLeft-9));elCaretEntry.style.left=entryCaretX.toFixed(1)+'px';
  const entryTipX=entryLeft+entryCaretX+9,entryTipY=entryTop-9;
  if((entryTipY-y1)>16){{elEntryGuide.setAttribute('x1',x1.toFixed(1));elEntryGuide.setAttribute('y1',(y1+7).toFixed(1));elEntryGuide.setAttribute('x2',entryTipX.toFixed(1));elEntryGuide.setAttribute('y2',entryTipY.toFixed(1));showEntryGuide=true;}}
  geometryReady=true;
}}
function setResultOverlayVisual(op,ap,sc){{
  if(!geometryReady)setupResultOverlayGeometry();
  elOverlay.style.opacity=op.toFixed(2);if(op<=0.01)return;
  const p=Math.max(0.02,Math.min(1.0,ap)),curLen=arcTotalLen*p;
  elArcPath.style.strokeDashoffset=(arcTotalLen-curLen).toFixed(1);
  const pt=elArcPath.getPointAtLength(curLen),ptPrev=elArcPath.getPointAtLength(Math.max(0,curLen-3));
  elArcHead.setAttribute('transform','translate('+pt.x.toFixed(1)+','+pt.y.toFixed(1)+') rotate('+(Math.atan2(pt.y-ptPrev.y,pt.x-ptPrev.x)*180/Math.PI).toFixed(1)+')');
  elDotExit.style.opacity=(p>=0.95?'1':'0');
  const ba=sc?Math.min(1.0,op):0,er=(p>=0.85?ba:0);
  elCoEntry.style.opacity=ba.toFixed(2);elCoExit.style.opacity=er.toFixed(2);
  elEntryGuide.setAttribute('opacity',(showEntryGuide&&ba>0.5)?'0.80':'0');elExitGuide.setAttribute('opacity',(showExitGuide&&er>0.5)?'0.80':'0');
}}
window.__setBottomMode=function(mode,subIdx){{
  if(mode==='sub'&&subMeta.length>0){{
    elCleanCard.style.display='none';elSubCard.style.display='flex';
    const seg=subMeta[Math.max(0,Math.min(subMeta.length-1,subIdx||0))];
    elSubStep.textContent=seg.step;
    setTextWithIcons(elSubL1,seg.l1);setTextWithIcons(elSubL2,seg.l2);setTextWithIcons(elSubJp,seg.sub||'');
    if(seg.tag==='SUPPORT'){{
      elSubBox.style.padding='14px 20px';
      elSubBox.style.gap='7px';
      elSubL1.style.fontSize='23.5px';
      elSubL2.style.fontSize='22.5px';
      elSubJp.style.fontSize='15.5px';
      elSubL2.style.color='#fde68a';
    }} else {{
      elSubBox.style.padding='18px 22px';
      elSubBox.style.gap='10px';
      elSubL1.style.fontSize='27px';
      elSubL2.style.fontSize='25px';
      elSubJp.style.fontSize='17px';
      elSubL2.style.color=(seg.tag==='PROFIT')?'#00e676':'#ffd166';
    }}
  }} else {{elSubCard.style.display='none';elCleanCard.style.display='flex';}}
}};
const THUMB_END={thumb_end},PLAY_START={play_start},PLAY_END={play_end},postSteps=Math.max(1,exitIdx-entryIdx);
window.__renderFrame=function(f){{
  let showIdx=entryIdx,frac=1.0,isThumb=(f<THUMB_END),isPrep=(f>=THUMB_END&&f<PLAY_START),isPlay=(f>=PLAY_START&&f<=PLAY_END);
  if(isThumb||isPrep){{showIdx=isThumb?exitIdx:entryIdx;elProg.style.width='0.0%';}}
  else if(isPlay){{const p=(f-PLAY_START)/(PLAY_END-PLAY_START);elProg.style.width=(p*100).toFixed(1)+'%';const pos=entryIdx+p*postSteps;showIdx=Math.min(exitIdx,Math.floor(pos));frac=Math.min(1.0,Math.max(0.15,pos-Math.floor(pos)));if(f===PLAY_END){{showIdx=exitIdx;frac=1.0;}}}}
  else{{showIdx=exitIdx;elProg.style.width='100.0%';}}
  const dispC=[];
  for(let i=0;i<=showIdx;i++){{const src=cAll[i];if(i===showIdx&&i>entryIdx&&frac<0.999){{const cN=src.open+(src.close-src.open)*frac;dispC.push({{time:src.time,open:src.open,high:Math.max(src.open,cN,src.open+(src.high-src.open)*frac),low:Math.min(src.open,cN,src.open+(src.low-src.open)*frac),close:cN}});}}else dispC.push(src);}}
  for(let i=showIdx+1;i<cAll.length;i++)dispC.push({{time:cAll[i].time}});
  for(let i=0;i<wsTail.length;i++)dispC.push(wsTail[i]);
  candleSeries.setData(dispC);volSeries.setData(vAll.slice(0,showIdx+1));ma5Series.setData(m5All.slice(0,showIdx+1).filter(Boolean));ma9Series.setData(m9All.slice(0,showIdx+1).filter(Boolean));ma20Series.setData(m20All.slice(0,showIdx+1).filter(Boolean));
  const lb=dispC[showIdx],cls=(lb.close>=lb.open)?'tv-val-up':'tv-val-dn';
  elO.textContent=fmtP(lb.open);elO.className=cls;elH.textContent=fmtP(lb.high);elH.className=cls;elL.textContent=fmtP(lb.low);elL.className=cls;elC.textContent=fmtP(lb.close);elC.className=cls;elTime.textContent=tAll[showIdx];
  elM5.textContent=m5All[showIdx]?fmtP(m5All[showIdx].value):'-';elM9.textContent=m9All[showIdx]?fmtP(m9All[showIdx].value):'-';elM20.textContent=m20All[showIdx]?fmtP(m20All[showIdx].value):'-';
  const mk=[{{time:cAll[entryIdx].time,position:'belowBar',color:'#00e676',shape:'arrowUp',text:'ENTRY',size:2}}];
  if(isThumb||f>PLAY_END)mk.push({{time:cAll[exitIdx].time,position:'aboveBar',color:'#ffd166',shape:'arrowDown',text:'EXIT',size:2}});
  candleSeries.setMarkers(mk);
  if(isThumb){{elHud.style.opacity='0';setResultOverlayVisual(1.0,1.0,true);}}
  else if(isPrep||isPlay){{
    elHud.style.opacity='1';setResultOverlayVisual(0.0,0.0,false);
    let h1=false,h2=false,h3=false;
    for(let i=entryIdx;i<=showIdx;i++){{const b=dispC[i];if(b.high>=tp1P)h1=true;if(b.close>=tp2P||b.high>=tp3P)h2=true;if(b.high>=tp3P)h3=true;}}
    const lp=isPrep?0.0:((lb.close-entryP)/entryP*100.0);
    elPnl.textContent=(lp>=0?'+':'')+lp.toFixed(1)+'%';elPnl.style.color=lp>=0?'#00e676':'#ef5350';elPnlBox.style.borderColor=lp>=0?'#26a69a':'#ef5350';
    if(isPrep){{elStage.style.display='block';elStage.style.borderColor='#00e676';elStage.textContent=uiS.prep;}}
    else if(h3){{elStage.style.display='block';elStage.style.borderColor='#ffd166';elStage.textContent=uiS.tp3;}}
    else if(h2){{elStage.style.display='block';elStage.style.borderColor='#06d6a0';elStage.textContent=uiS.tp2;}}
    else if(h1){{elStage.style.display='block';elStage.style.borderColor='#4cc9f0';elStage.textContent=uiS.tp1;}}
    else elStage.style.display='none';
  }} else {{elHud.style.opacity='0';setResultOverlayVisual(1.0,Math.min(1.0,(f-PLAY_END)/10.0),true);}}
}};
requestAnimationFrame(()=>requestAnimationFrame(()=>{{setupResultOverlayGeometry();window.__setBottomMode('{init_mode}',0);window.__renderFrame(0);window.__REPLAY_READY=true;}}));
</script></body></html>"""

            html_path = os.path.join(tmpdir, f"replay_vertical_{lang}.html")
            out_clean_mp4 = os.path.join(tmpdir, f"replay_clean_{lang}.mp4")
            out_voiced_mp4 = os.path.join(tmpdir, f"replay_voiced_{lang}.mp4")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            def _sub_idx_for_frame(fi_num):
                if not segments:
                    return 0
                t_sec = fi_num / float(in_fps)
                for si, seg in enumerate(segments):
                    if seg["sub_start"] <= t_sec < seg["sub_end"]:
                        return si
                return len(segments) - 1

            total_max_frames = max(base_frames if make_clean else 0, voiced_frames if make_voiced else 0)
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"])
                try:
                    page = browser.new_page(viewport={"width": CHART_CSS_W, "height": CHART_CSS_H}, device_scale_factor=1.5)
                    page.on("pageerror", lambda err: js_errors.append(str(err)))
                    page.goto(Path(html_path).resolve().as_uri(), wait_until="load", timeout=25000)
                    page.wait_for_function("window.__REPLAY_READY === true", timeout=15000)
                    page.wait_for_timeout(120)

                    prev_sub_idx = -1
                    for fi in range(total_max_frames):
                        chart_is_static = (1 <= fi < thumb_end) or ((thumb_end + 1) <= fi < play_start) or (fi > outro_static_start)
                        cur_sub_idx = _sub_idx_for_frame(fi) if make_voiced else 0
                        clean_png = os.path.join(tmpdir, f"c_{fi:04d}.png")
                        sub_png = os.path.join(tmpdir, f"s_{fi:04d}.png")

                        need_chart = not chart_is_static
                        need_clean = make_clean and (fi < base_frames) and need_chart
                        need_sub = make_voiced and (fi < voiced_frames) and (need_chart or cur_sub_idx != prev_sub_idx)

                        if need_chart:
                            page.evaluate(f"window.__renderFrame({fi})")
                        if make_clean and fi < base_frames:
                            if need_clean or fi == 0:
                                page.evaluate("window.__setBottomMode('clean', 0)")
                                page.screenshot(path=clean_png, type="png")
                            else:
                                ref_fi = 0 if fi < thumb_end else (thumb_end if fi < play_start else outro_static_start)
                                shutil.copyfile(os.path.join(tmpdir, f"c_{ref_fi:04d}.png"), clean_png)
                        if make_voiced and fi < voiced_frames:
                            if need_sub or fi == 0:
                                page.evaluate(f"window.__setBottomMode('sub', {cur_sub_idx})")
                                page.screenshot(path=sub_png, type="png")
                            else:
                                shutil.copyfile(os.path.join(tmpdir, f"s_{fi - 1:04d}.png"), sub_png)
                            prev_sub_idx = cur_sub_idx
                finally:
                    browser.close()

            clean_bytes = None
            if make_clean:
                clean_bytes, c_err = _encode_hd_mp4(os.path.join(tmpdir, "c_%04d.png"), out_clean_mp4, in_fps, out_fps)
                if not clean_bytes:
                    LAST_VIDEO_ERROR = f"클린 영상({lang}) 오류: {c_err}"

            voiced_bytes, voiced_error = None, (tts_err if not make_voiced else "")
            if make_voiced and segments:
                voiced_bytes, v_err = _encode_hd_mp4(
                    os.path.join(tmpdir, "s_%04d.png"), out_voiced_mp4, in_fps, out_fps,
                    segments=segments, total_sec=final_total_sec,
                )
                if not voiced_bytes:
                    voiced_error = f"음성 합성({lang}) 오류: {v_err}"

            print(f"[익절 리플레이 완료({lang})] {symbol} ({tf_label}, {len(df_win)}봉 · {VIDEO_W}x{VIDEO_H}) | 5단계 · TTS={'OK' if voiced_bytes else 'FAIL'} ({voiced_frames / in_fps:.1f}초)")
            return {
                "clean": clean_bytes, "voiced": voiced_bytes,
                "width": VIDEO_W, "height": VIDEO_H,
                "clean_duration": round(base_frames / float(in_fps), 1),
                "voiced_duration": round(voiced_frames / float(in_fps), 1),
                "voiced_error": voiced_error,
            }
    except Exception as e:
        err_extra = f" | JS: {js_errors[0][:120]}" if js_errors else ""
        LAST_VIDEO_ERROR = f"리플레이({lang}) 예외: {str(e)[:150]}{err_extra}"
        print(f"[세로형 리플레이 예외] {symbol}: {LAST_VIDEO_ERROR}")
        return None


def generate_profit_replay_video_bytes(close_info, lang="kr_jp"):
    """단일 영상 호환 래퍼"""
    bundle = generate_profit_replay_videos_bundle(close_info, include_clean=False, lang=lang)
    if isinstance(bundle, dict):
        return bundle.get("voiced") or bundle.get("clean")
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
    for attempt in range(3):
        try:
            resp = requests.post(url, json=payload, timeout=12)
            if resp.status_code == 200:
                return True
            if resp.status_code == 429:
                wait_s = int(to_float((resp.json().get("parameters") or {}).get("retry_after"), 3))
                time.sleep(wait_s + 0.5)
                continue
            print(f"[sendMessage 오류 {attempt + 1}/3] HTTP {resp.status_code}: {resp.text[:160]}")
        except Exception as e:
            print(f"[sendMessage 예외 {attempt + 1}/3] {e}")
        time.sleep(1.0)
    return False


def send_telegram_photo_with_caption(photo_bytes, caption_text="", reply_markup=None):
    """초고화질 무손실 PNG 원본 전송 (sendDocument)"""
    token = (TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    if not photo_bytes or not token or not chat_id:
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendDocument")
    for attempt in range(3):
        try:
            files = {"document": ("danpyung_chart_hd.png", photo_bytes, "image/png")}
            data = {"chat_id": chat_id}
            if caption_text:
                data["caption"] = caption_text
            if reply_markup:
                data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False) if isinstance(reply_markup, dict) else str(reply_markup)
            resp = requests.post(url, data=data, files=files, timeout=25)
            if resp.status_code == 200:
                time.sleep(0.7)
                return True
            if resp.status_code == 429:
                wait_s = int(to_float((resp.json().get("parameters") or {}).get("retry_after"), 3))
                time.sleep(wait_s + 0.5)
                continue
            print(f"[sendDocument(HD차트) 오류 {attempt + 1}/3] HTTP {resp.status_code}: {resp.text[:160]}")
        except Exception as e:
            print(f"[sendDocument(HD차트) 예외 {attempt + 1}/3] {e}")
        time.sleep(1.2)
    return False


def send_telegram_video_with_caption(
    video_bytes,
    caption_text="",
    reply_markup=None,
    filename="profit_replay.mp4",
    width=VIDEO_W,
    height=VIDEO_H,
    duration=30,
):
    """세로형(1080x1440) 풀HD MP4 영상을 텔레그램 sendVideo로 전송"""
    global LAST_VIDEO_ERROR
    token = (TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    if not video_bytes or not token or not chat_id:
        LAST_VIDEO_ERROR = "영상 데이터 또는 텔레그램 토큰/챗ID 누락"
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendVideo")
    last_resp_err = ""
    for attempt in range(3):
        try:
            files = {"video": (filename, video_bytes, "video/mp4")}
            data = {
                "chat_id": chat_id,
                "width": str(int(to_float(width, VIDEO_W))),
                "height": str(int(to_float(height, VIDEO_H))),
                "duration": str(max(1, int(round(to_float(duration, 30))))),
                "supports_streaming": "true",
            }
            if caption_text:
                data["caption"] = caption_text
            if reply_markup:
                data["reply_markup"] = json.dumps(reply_markup, ensure_ascii=False) if isinstance(reply_markup, dict) else str(reply_markup)
            resp = requests.post(url, data=data, files=files, timeout=90)
            if resp.status_code == 200:
                time.sleep(0.8)
                return True
            if resp.status_code == 429:
                wait_s = int(to_float((resp.json().get("parameters") or {}).get("retry_after"), 3))
                time.sleep(wait_s + 0.5)
                continue
            last_resp_err = f"Telegram HTTP {resp.status_code}: {resp.text[:160]}"
            print(f"[sendVideo 오류 {attempt + 1}/3] {last_resp_err}")
        except Exception as e:
            last_resp_err = f"Telegram 전송 예외: {str(e)[:150]}"
            print(f"[sendVideo 예외 {attempt + 1}/3] {last_resp_err}")
        time.sleep(1.5)
    LAST_VIDEO_ERROR = last_resp_err or "Telegram sendVideo 3회 실패"
    print(f"[sendVideo 실패] {LAST_VIDEO_ERROR}")
    return False
# === END OF FILE ===
