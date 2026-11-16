import os
import io
import sys
import time
import json
import tempfile
import subprocess
import requests
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
    TELEGRAM_BOT_TOKEN = (
        os.getenv("TELEGRAM_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or ""
    ).strip()

if not TELEGRAM_CHAT_ID:
    TELEGRAM_CHAT_ID = (
        os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("CHAT_ID")
        or ""
    ).strip()

_PLAYWRIGHT_READY = False


def to_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except (TypeError, ValueError):
        return default


def fmt_price(price):
    p = to_float(price, 0.0)
    if p <= 0:
        return "0.0000"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 10:
        return f"{p:.3f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    return f"{p:.6f}"


def fmt_vol(v):
    val = to_float(v, 0.0)
    if val >= 1e9:
        return f"{val / 1e9:.2f}B"
    if val >= 1e6:
        return f"{val / 1e6:.2f}M"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K"
    return f"{val:.0f}"


def _ensure_playwright():
    """런타임에 playwright 및 chromium이 없으면 1회 자동 설치"""
    global _PLAYWRIGHT_READY
    if _PLAYWRIGHT_READY:
        return True
    try:
        import playwright
        _PLAYWRIGHT_READY = True
        return True
    except ImportError:
        try:
            print("[알림] TradingView 엔진 렌더링용 Playwright 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "playwright"])
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
            _PLAYWRIGHT_READY = True
            return True
        except Exception as e:
            print(f"[경고] Playwright 설치 실패: {e}")
            return False


def fetch_exact_bybit_klines(symbol, interval="15", limit=130):
    """
    Bybit V5 공식 미러 서버에서 실제 15분봉(OHLC)과 거래량(Volume)을 조회해
    과거 -> 현재 시간 오름차순으로 정렬합니다. (24시간 96캔들 + 이평선 계산용 여유분 포함 130캔들)
    """
    bybit_params = {
        "category": "linear",
        "symbol": symbol,
        "interval": str(interval),
        "limit": int(limit),
    }
    bybit_hosts = (
        "https://api.bybit-tr.com",
        "https://api.bybit.kz",
        "https://api.bytick.com",
        "https://api.bybit.com",
    )

    for host in bybit_hosts:
        try:
            resp = requests.get(
                f"{host}/v5/market/kline",
                params=bybit_params,
                headers=BROWSER_HEADERS,
                timeout=4,
            )
            if resp.status_code == 200:
                rows = resp.json().get("result", {}).get("list", [])
                if isinstance(rows, list) and len(rows) >= 30:
                    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(rows[0])]
                    df = pd.DataFrame(rows, columns=cols)
                    for c in ("timestamp", "open", "high", "low", "close", "volume"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    if df["timestamp"].max() < 1e11:
                        df["timestamp"] = df["timestamp"] * 1000.0
                    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp", ascending=True).reset_index(drop=True)
                    if len(df) >= 30:
                        return df
        except Exception:
            continue

    try:
        r = requests.get(
            "https://api.bitget.com/api/v2/mix/market/candles",
            params={"symbol": symbol, "productType": "USDT-FUTURES", "granularity": "15m", "limit": str(limit)},
            headers=BROWSER_HEADERS,
            timeout=4,
        )
        if r.status_code == 200:
            rows = r.json().get("data", [])
            if isinstance(rows, list) and len(rows) >= 30:
                cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(rows[0])]
                df = pd.DataFrame(rows, columns=cols)
                for c in ("timestamp", "open", "high", "low", "close", "volume"):
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                if df["timestamp"].max() < 1e11:
                    df["timestamp"] = df["timestamp"] * 1000.0
                df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp", ascending=True).reset_index(drop=True)
                return df
    except Exception:
        pass

    return pd.DataFrame()


def build_briefing_message(res, now_str):
    base = res.get("base_coin") or res.get("base") or ""
    symbol = res.get("symbol", "")
    title = res.get("track_title", "🚀 [타점 포착]")
    score = res.get("score", 85)
    price = to_float(res.get("price"))
    chg = to_float(res.get("change_24h"))
    turn_m = to_float(res.get("turnover_m"))
    oi_m = to_float(res.get("oi_m"))
    oi_ratio = to_float(res.get("oi_ratio"))
    fund = to_float(res.get("funding_pct"))
    htf = res.get("htf_status", "1H 중심선지지")
    rsi15 = to_float(res.get("rsi6_15m"))
    kdj15 = to_float(res.get("kdj_j_15m"))

    e_low = to_float(res.get("entry_low"), price * 0.9975)
    e_high = to_float(res.get("entry_high"), price * 1.0010)
    sl_p = to_float(res.get("stop_price"))
    sl_pct = int(round(to_float(res.get("sl_pct"), FIXED_SL_PCT)))
    sl_note = res.get("sl_note", f"고정손절(-{sl_pct}%)")

    tps = res.get("tps") or [price * 1.08, price * 1.16, price * 1.24]
    tp_pcts = res.get("tp_pcts") or [8.0, 16.0, 24.0]
    tp_labels = res.get("tp_labels") or ["1차목표", "2차목표", "3차목표"]
    reasons = res.get("reasons") or []

    lines = [
        f"{title} ${base} ({symbol})",
        f"🕒 포착 시각: {now_str} | 종합점수: {score}점",
        f"• 현재가: {fmt_price(price)} USDT (24h {chg:+.2f}%)",
        f"• 수급: 거래대금 {turn_m:.1f}M | OI {oi_m:.1f}M (비중 {oi_ratio:.0f}%) | 펀딩 {fund:+.4f}%",
        f"• 멀티프레임: {htf} | 15m RSI6 {rsi15:.1f} · J {kdj15:.1f}",
        "",
        "🎯 [매매 전략 가이드]",
        f"• 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT",
        f"• 손절가(SL): {fmt_price(sl_p)} USDT (-{sl_pct}% | {sl_note})",
        f"• 1차 익절(40%): {fmt_price(tps[0])} USDT (+{int(round(to_float(tp_pcts[0])))}% | {tp_labels[0]})",
        f"• 2차 익절(30%): {fmt_price(tps[1])} USDT (+{int(round(to_float(tp_pcts[1])))}% | {tp_labels[1]})",
        f"• 3차 익절(30%): {fmt_price(tps[2])} USDT (+{int(round(to_float(tp_pcts[2])))}% | {tp_labels[2]})",
    ]

    if reasons:
        lines.append("\n🔍 [포착 근거]")
        for r in reasons:
            lines.append(f"• {r}")

    return "\n".join(lines)


def build_x_post_message(res):
    """요청하신 최신 X(트위터) 전용 양식 반영"""
    base = res.get("base_coin") or res.get("base") or ""
    symbol = res.get("symbol", "")
    price = to_float(res.get("price"))
    e_low = to_float(res.get("entry_low"), price * 0.9975)
    e_high = to_float(res.get("entry_high"), price * 1.0010)
    sl_p = to_float(res.get("stop_price"))
    sl_pct = int(round(to_float(res.get("sl_pct"), FIXED_SL_PCT)))

    tps = res.get("tps") or [price * 1.08, price * 1.16, price * 1.24]
    tp_pcts = res.get("tp_pcts") or [8.0, 16.0, 24.0]

    p1 = int(round(to_float(tp_pcts[0], 8.0)))
    p2 = int(round(to_float(tp_pcts[1], 16.0)))
    p3 = int(round(to_float(tp_pcts[2], 24.0)))

    return (
        "혹시 이 종목을 보유하고 있는 분 계신가요?\n"
        "😎 개인 검색기에 포착된 종목😎\n\n"
        f"${base} ({symbol})\n\n"
        f"- 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT\n"
        f"- SL: {fmt_price(sl_p)} (-{sl_pct}%)\n"
        f"- tp1: {fmt_price(tps[0])}(+{p1}%)\n"
        f"- tp2: {fmt_price(tps[1])}(+{p2}%)\n"
        f"- tp3: {fmt_price(tps[2])}(+{p3}%)"
    )


def _render_tradingview_lightweight_chart(df, res):
    """
    [트레이딩뷰 Lightweight Charts 15분봉 24시간(96캔들) 렌더링 엔진]
    1) 15분봉 96개(24시간 흐름) 표시 + 이평선 MA(5), MA(9), MA(20) 적용
    2) 양방향 비선형 스케일(Two-Sided Piecewise Scale)로 SL 하단 빈 여백과 TP 상단 여백을 모두 압축해
       캔들 몸통이 화면 중앙을 큼직하게 채우도록 확대
    3) 거래량 바 높이를 하단 10%로 축소 & 진입한 마지막 캔들 봉 하단에 '▲ ENTRY' 화살표 마커 표시
    """
    if not _ensure_playwright():
        return None

    symbol = res.get("symbol", "")
    base = res.get("base_coin") or res.get("base") or symbol.replace("USDT", "")
    score = res.get("score", 85)
    track_type = res.get("track_type", "TRACK_A")

    latest_row = df.iloc[-1]
    price = to_float(latest_row["close"], to_float(res.get("price")))
    open_last = to_float(latest_row["open"], price)
    high_last = to_float(latest_row["high"], price)
    low_last = to_float(latest_row["low"], price)
    vol_last = to_float(latest_row["volume"], 0.0)

    orig_entry = to_float(res.get("price"), price)
    ratio_adj = price / orig_entry if orig_entry > 0 else 1.0

    sl_pct = int(round(to_float(res.get("sl_pct"), FIXED_SL_PCT)))
    sl_p = to_float(res.get("stop_price"), price * (1.0 - sl_pct / 100.0)) * ratio_adj
    tps = res.get("tps") or [price * 1.08, price * 1.16, price * 1.24]
    tp_pcts = res.get("tp_pcts") or [8.0, 16.0, 24.0]

    tp1_p = to_float(tps[0]) * ratio_adj
    tp2_p = to_float(tps[1]) * ratio_adj
    tp3_p = to_float(tps[2]) * ratio_adj
    p1 = int(round(to_float(tp_pcts[0], 8.0)))
    p2 = int(round(to_float(tp_pcts[1], 16.0)))
    p3 = int(round(to_float(tp_pcts[2], 24.0)))

    # [핵심] 96개 캔들의 실제 고저점 범위를 기준으로 상단(TP)과 하단(SL) 빈 여백을 동시에 압축
    candle_min_low = to_float(df["low"].min(), price * 0.97)
    candle_max_high = to_float(df["high"].max(), price * 1.02)

    low_piv = max(sl_p * 1.005, min(candle_min_low, price * 0.988))
    high_piv = min(tp1_p * 0.990, max(candle_max_high, price * 1.010))
    if high_piv <= low_piv:
        low_piv = price * 0.985
        high_piv = price * 1.015

    core_candle_span = max(high_piv - low_piv, price * 0.018)
    tp_span = max(tp3_p - high_piv, price * 0.04)
    sl_span = max(low_piv - sl_p, price * 0.01)

    # 상단 TP 구간은 캔들 높이의 48% 수준으로 압축, 하단 SL 여백은 캔들 높이의 16% 수준으로 대폭 축소!
    up_comp = float(np.clip((core_candle_span * 0.48) / tp_span, 0.08, 0.65))
    down_comp = float(np.clip((core_candle_span * 0.16) / sl_span, 0.08, 0.55)) if sl_p < low_piv else 1.0

    def _fwd(val):
        v = to_float(val, price)
        if v > high_piv:
            return high_piv + (v - high_piv) * up_comp
        if v < low_piv:
            return low_piv - (low_piv - v) * down_comp
        return v

    candles_js = []
    vol_js = []
    ma5_js = []
    ma9_js = []
    ma20_js = []

    for _, row in df.iterrows():
        ts_sec = int(to_float(row["timestamp"]) // 1000) + (9 * 3600)
        o_v = to_float(row["open"])
        h_v = to_float(row["high"])
        l_v = to_float(row["low"])
        c_v = to_float(row["close"])
        v_v = to_float(row["volume"])

        candles_js.append({
            "time": ts_sec,
            "open": _fwd(o_v),
            "high": _fwd(h_v),
            "low": _fwd(l_v),
            "close": _fwd(c_v),
        })
        vol_js.append({
            "time": ts_sec,
            "value": v_v,
            "color": "rgba(38, 166, 154, 0.38)" if c_v >= o_v else "rgba(239, 83, 80, 0.38)",
        })
        if pd.notna(row.get("ma5")):
            ma5_js.append({"time": ts_sec, "value": _fwd(row["ma5"])})
        if pd.notna(row.get("ma9")):
            ma9_js.append({"time": ts_sec, "value": _fwd(row["ma9"])})
        if pd.notna(row.get("ma20")):
            ma20_js.append({"time": ts_sec, "value": _fwd(row["ma20"])})

    last_ts_sec = candles_js[-1]["time"]
    whitespace_js = [{"time": last_ts_sec + (i * 900)} for i in range(1, 6)]

    last_kst_dt = datetime.fromtimestamp(to_float(latest_row["timestamp"]) / 1000.0, tz=KST)
    kst_str = last_kst_dt.strftime("%Y-%m-%d %H:%M KST")

    track_badge = {
        "TRACK_A": "TRACK A · Pullback",
        "TRACK_B": "TRACK B · Breakout",
        "TRACK_C": "TRACK C · Leader Fib 0.5",
    }.get(track_type, track_type)

    ma5_curr = fmt_price(df["ma5"].iloc[-1])
    ma9_curr = fmt_price(df["ma9"].iloc[-1])
    ma20_curr = fmt_price(df["ma20"].iloc[-1])

    decimals = 6 if price < 0.01 else (5 if price < 1 else (4 if price < 10 else (3 if price < 1000 else 2)))
    min_move = 10 ** (-decimals)

    html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/lightweight-charts@4.2.0/dist/lightweight-charts.standalone.production.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: #131722;
    font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif;
    color: #d1d4dc;
    width: 1160px;
    height: 680px;
    overflow: hidden;
  }}
  .tv-wrap {{
    position: relative;
    width: 1160px;
    height: 680px;
    background: #131722;
    border: 1px solid #2a2e39;
    display: flex;
    flex-direction: column;
  }}
  .tv-topbar {{
    height: 42px;
    background: #181c27;
    border-bottom: 1px solid #2a2e39;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 16px;
    z-index: 20;
  }}
  .tv-left {{ display: flex; align-items: center; gap: 11px; }}
  .tv-symbol {{ font-size: 16px; font-weight: 800; color: #ffffff; letter-spacing: 0.3px; }}
  .tv-tf-badge {{
    background: #2a2e39; color: #d1d4dc; font-size: 11.5px; font-weight: 700;
    padding: 3px 8px; border-radius: 4px;
  }}
  .tv-exchange {{ color: #787b86; font-size: 12px; font-weight: 600; }}
  .tv-ohlc {{ display: flex; gap: 10px; font-size: 12px; font-weight: 600; margin-left: 6px; }}
  .tv-ohlc span b {{ color: #787b86; font-weight: 500; margin-right: 2px; }}
  .tv-val-up {{ color: #26a69a; }}
  .tv-right {{ display: flex; align-items: center; gap: 10px; }}
  .tv-score-pill {{
    background: linear-gradient(90deg, #2962ff, #1e4bd8);
    color: #ffffff; font-size: 12px; font-weight: 700;
    padding: 4px 11px; border-radius: 5px;
  }}
  .tv-time {{ color: #9598a1; font-size: 12px; font-weight: 600; }}
  #chart-box {{ position: relative; flex: 1; width: 100%; height: 638px; }}
  .tv-ma-legend {{
    position: absolute; top: 10px; left: 14px; z-index: 15;
    display: flex; gap: 14px; font-size: 12px; font-weight: 700;
    background: rgba(19, 23, 34, 0.84); padding: 5px 11px;
    border-radius: 4px; border: 1px solid #2a2e39;
  }}
  .ma5 {{ color: #00e5ff; }}
  .ma9 {{ color: #ffd166; }}
  .ma20 {{ color: #2962ff; }}
  .vol-lbl {{ color: #9598a1; }}
  .tv-watermark {{
    position: absolute; top: 44%; left: 46%; transform: translate(-50%, -50%);
    font-size: 46px; font-weight: 900; color: rgba(255, 255, 255, 0.03);
    text-align: center; pointer-events: none; z-index: 1; line-height: 1.15;
  }}
</style>
</head>
<body>
<div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tv-left">
      <span class="tv-symbol">${base}USDT.P</span>
      <span class="tv-tf-badge">15m · 24H (96 Bars)</span>
      <span class="tv-exchange">BYBIT PERPETUAL</span>
      <div class="tv-ohlc">
        <span><b>O</b><span class="tv-val-up">{fmt_price(open_last)}</span></span>
        <span><b>H</b><span class="tv-val-up">{fmt_price(high_last)}</span></span>
        <span><b>L</b><span class="tv-val-up">{fmt_price(low_last)}</span></span>
        <span><b>C</b><span class="tv-val-up">{fmt_price(price)}</span></span>
        <span><b>Vol(15m)</b><span class="tv-val-up">{fmt_vol(vol_last)}</span></span>
      </div>
    </div>
    <div class="tv-right">
      <span class="tv-time">{kst_str}</span>
      <span class="tv-score-pill">{track_badge} · {score} pts</span>
    </div>
  </div>
  <div id="chart-box">
    <div class="tv-ma-legend">
      <span class="ma5">● MA(5) {ma5_curr}</span>
      <span class="ma9">● MA(9) {ma9_curr}</span>
      <span class="ma20">● MA(20) {ma20_curr}</span>
      <span class="vol-lbl">Vol {fmt_vol(vol_last)}</span>
    </div>
    <div class="tv-watermark">{base}USDT<br>BYBIT · 15M</div>
  </div>
</div>
<script>
  const lowPiv = {low_piv};
  const highPiv = {high_piv};
  const upComp = {up_comp};
  const downComp = {down_comp};
  const decimals = {decimals};

  function invScale(y) {{
    if (y > highPiv) return highPiv + (y - highPiv) / Math.max(upComp, 1e-6);
    if (y < lowPiv) return lowPiv - (lowPiv - y) / Math.max(downComp, 1e-6);
    return y;
  }}

  function fmtReal(val) {{
    const p = invScale(val);
    if (p <= 0) return '0.0000';
    if (p >= 1000) return p.toLocaleString('en-US', {{minimumFractionDigits: 2, maximumFractionDigits: 2}});
    return p.toFixed(decimals);
  }}

  const container = document.getElementById('chart-box');
  const chart = LightweightCharts.createChart(container, {{
    width: 1160,
    height: 638,
    layout: {{
      background: {{ type: 'solid', color: '#131722' }},
      textColor: '#d1d4dc',
      fontSize: 12,
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, sans-serif",
    }},
    grid: {{
      vertLines: {{ color: '#1e222d', style: LightweightCharts.LineStyle.Solid }},
      horzLines: {{ color: '#1e222d', style: LightweightCharts.LineStyle.Solid }},
    }},
    rightPriceScale: {{
      borderColor: '#2a2e39',
      scaleMargins: {{ top: 0.05, bottom: 0.12 }},
      entireTextOnly: true,
    }},
    timeScale: {{
      borderColor: '#2a2e39',
      timeVisible: true,
      secondsVisible: false,
      rightOffset: 5,
      barSpacing: 9.8,
    }},
    localization: {{
      priceFormatter: fmtReal,
    }},
  }});

  const candleSeries = chart.addCandlestickSeries({{
    upColor: '#26a69a',
    downColor: '#ef5350',
    borderUpColor: '#26a69a',
    borderDownColor: '#ef5350',
    wickUpColor: '#26a69a',
    wickDownColor: '#ef5350',
    priceFormat: {{ type: 'custom', minMove: {min_move}, formatter: fmtReal }},
  }});

  const candlesData = {json.dumps(candles_js)};
  const whitespaceData = {json.dumps(whitespace_js)};
  candleSeries.setData([...candlesData, ...whitespaceData]);

  // 진입한 마지막 캔들 하단에 진입 화살표 아이콘(▲ ENTRY) 표시
  candleSeries.setMarkers([
    {{
      time: candlesData[candlesData.length - 1].time,
      position: 'belowBar',
      color: '#00e676',
      shape: 'arrowUp',
      text: 'ENTRY',
      size: 2,
    }}
  ]);

  // 이평선 5, 9, 20 렌더링
  const ma5Series = chart.addLineSeries({{ color: '#00e5ff', lineWidth: 1.6, priceLineVisible: false, lastValueVisible: false }});
  ma5Series.setData({json.dumps(ma5_js)});

  const ma9Series = chart.addLineSeries({{ color: '#ffd166', lineWidth: 1.6, priceLineVisible: false, lastValueVisible: false }});
  ma9Series.setData({json.dumps(ma9_js)});

  const ma20Series = chart.addLineSeries({{ color: '#2962ff', lineWidth: 2.0, priceLineVisible: false, lastValueVisible: false }});
  ma20Series.setData({json.dumps(ma20_js)});

  // 거래량 영역을 하단 10%로 축소해 캔들 영역 극대화
  const volSeries = chart.addHistogramSeries({{
    priceFormat: {{ type: 'volume' }},
    priceScaleId: 'vol_scale',
    priceLineVisible: false,
    lastValueVisible: false,
  }});
  chart.priceScale('vol_scale').applyOptions({{
    scaleMargins: {{ top: 0.90, bottom: 0.00 }},
  }});
  volSeries.setData({json.dumps(vol_js)});

  const targets = [
    {{ price: {_fwd(tp3_p)}, color: '#ffd166', title: 'TP3 (+{p3}%)', style: LightweightCharts.LineStyle.Dashed, width: 2 }},
    {{ price: {_fwd(tp2_p)}, color: '#06d6a0', title: 'TP2 (+{p2}%)', style: LightweightCharts.LineStyle.Dashed, width: 2 }},
    {{ price: {_fwd(tp1_p)}, color: '#4cc9f0', title: 'TP1 (+{p1}%)', style: LightweightCharts.LineStyle.Dashed, width: 2 }},
    {{ price: {_fwd(price)}, color: '#ffffff', title: 'ENTRY', style: LightweightCharts.LineStyle.Solid, width: 2 }},
    {{ price: {_fwd(sl_p)}, color: '#ef476f', title: 'SL (-{sl_pct}%)', style: LightweightCharts.LineStyle.Dashed, width: 2 }},
  ];

  targets.forEach(t => {{
    candleSeries.createPriceLine({{
      price: t.price,
      color: t.color,
      lineWidth: t.width,
      lineStyle: t.style,
      axisLabelVisible: true,
      title: t.title,
    }});
  }});

  // 상단 TP3와 하단 SL까지 여백 없이 딱 맞게 들어오도록 바운딩
  const boundsSeries = chart.addLineSeries({{
    color: 'rgba(0,0,0,0)',
    lineWidth: 1,
    priceLineVisible: false,
    lastValueVisible: false,
  }});
  boundsSeries.setData([
    {{ time: candlesData[0].time, value: {_fwd(sl_p) - core_candle_span * 0.03} }},
    {{ time: candlesData[candlesData.length - 1].time, value: {_fwd(tp3_p) + core_candle_span * 0.04} }},
  ]);

  chart.timeScale().fitContent();
  window.__CHART_DONE = true;
</script>
</body>
</html>"""

    try:
        from playwright.sync_api import sync_playwright
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            tmp_html_path = f.name

        png_bytes = None
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page(viewport={"width": 1160, "height": 680}, device_scale_factor=1.5)
            page.goto(f"file://{tmp_html_path}", wait_until="networkidle", timeout=20000)
            page.wait_for_function("window.__CHART_DONE === true", timeout=8000)
            page.wait_for_timeout(250)
            png_bytes = page.screenshot(type="png")
            browser.close()

        try:
            os.remove(tmp_html_path)
        except Exception:
            pass
        return png_bytes
    except Exception as e:
        print(f"[TradingView Lightweight Charts 캡처 오류] {e}")
        return None


def generate_signal_chart_bytes(res):
    """
    Bybit V5 실제 15분봉 130캔들을 조회하여 MA(5, 9, 20)을 계산한 뒤,
    최근 24시간(96캔들)을 슬라이싱해 TradingView Lightweight Charts 이미지를 생성합니다.
    """
    symbol = res.get("symbol", "")
    price = to_float(res.get("price"))
    if not symbol or price <= 0:
        return None

    df = fetch_exact_bybit_klines(symbol, interval="15", limit=130)
    if df is None or df.empty or len(df) < 30:
        return None

    df = df.copy()
    df["ma5"] = df["close"].rolling(5, min_periods=2).mean()
    df["ma9"] = df["close"].rolling(9, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()

    # 정확히 24시간 분량인 최근 96개 15분봉 슬라이싱
    df = df.iloc[-96:].reset_index(drop=True)

    return _render_tradingview_lightweight_chart(df, res)


def send_telegram_single(text):
    if not text or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True},
            timeout=12,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[텔레그램 전송 오류] {e}")
        return False


def send_telegram_photo_with_caption(photo_bytes, caption_text):
    token = (
        TELEGRAM_BOT_TOKEN
        or os.getenv("TELEGRAM_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or ""
    ).strip()
    chat_id = (
        TELEGRAM_CHAT_ID
        or os.getenv("TELEGRAM_CHAT_ID")
        or os.getenv("CHAT_ID")
        or ""
    ).strip()

    if not photo_bytes or not token or not chat_id:
        return False

    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    for attempt in range(3):
        try:
            files = {"photo": ("tv_signal_chart.png", photo_bytes, "image/png")}
            data = {"chat_id": chat_id, "caption": caption_text}
            resp = requests.post(url, data=data, files=files, timeout=18)
            if resp.status_code == 200:
                time.sleep(0.7)
                return True
            if resp.status_code == 429:
                retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                time.sleep(retry_after + 0.5)
                continue
        except Exception as e:
            print(f"[텔레그램 차트 사진 전송 재시도 {attempt + 1}/3] {e}")
        time.sleep(1.2)
    return False


def send_uai_sample_test():
    """직전 포착된 $BNC 종목으로 24시간(96캔들) + MA(5,9,20) + SL여백 축소 + 진입마커 비교 테스트 발송"""
    sample_res = {
        "symbol": "BNCUSDT",
        "base_coin": "BNC",
        "base": "BNC",
        "track_type": "TRACK_B",
        "track_title": "🚀 [2차 돌파/리테스트 타점 포착]",
        "score": 97,
        "price": 3.8219,
        "entry_low": 3.8123,
        "entry_high": 3.8257,
        "stop_price": 3.5926,
        "sl_pct": 6,
        "tps": [4.1659, 4.5481, 4.9685],
        "tp_pcts": [9.0, 19.0, 30.0],
    }
    x_caption = build_x_post_message(sample_res)
    chart_bytes = generate_signal_chart_bytes(sample_res)
    if chart_bytes:
        ok = send_telegram_photo_with_caption(chart_bytes, x_caption)
        print(f"[TradingView 24H(96캔들) 차트 발송 결과]: {'성공' if ok else '실패'}")
        return ok
    return send_telegram_single(x_caption)


if __name__ == "__main__":
    send_uai_sample_test()
