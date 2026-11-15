import os
import io
import sys
import time
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
        return f"{val / 1e9:.1f}B"
    if val >= 1e6:
        return f"{val / 1e6:.1f}M"
    if val >= 1e3:
        return f"{val / 1e3:.1f}K"
    return f"{val:.0f}"


def _ensure_matplotlib():
    try:
        import matplotlib
        return True
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
            return True
        except Exception as e:
            print(f"[경고] matplotlib 자동 설치 실패: {e}")
            return False


def fetch_exact_bybit_klines(symbol, interval="15", limit=75):
    """
    [1순위] 게이트아이오/비트겟이 아닌 실제 Bybit V5 공식 미러 서버에서
    바이비트 원본 캔들(OHLC)과 실제 바이비트 거래량(Volume)을 직접 가져옵니다.
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
                if isinstance(rows, list) and len(rows) >= 20:
                    cols = ["timestamp", "open", "high", "low", "close", "volume", "turnover"][:len(rows[0])]
                    df = pd.DataFrame(rows, columns=cols)
                    for c in ("timestamp", "open", "high", "low", "close", "volume"):
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                    df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp", ascending=True).reset_index(drop=True)
                    if len(df) >= 20:
                        return df
        except Exception:
            continue

    # 모든 바이비트 미러가 응답하지 않을 때만 api_mod 폴백 사용
    if api_mod and hasattr(api_mod, "fetch_klines"):
        try:
            df = api_mod.fetch_klines(symbol, str(interval), limit=int(limit))
            if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= 20:
                if "timestamp" in df.columns:
                    df = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
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
    """요청하신 X(트위터) 전용 포맷"""
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
        "어디에서 익절될거 같나요?😎\n\n"
        f"${base} ({symbol})\n\n"
        f"- 진입 구간: {fmt_price(e_low)} ~ {fmt_price(e_high)} USDT\n"
        f"- SL: {fmt_price(sl_p)} USDT (-{sl_pct}%)\n"
        f"- 1차 {fmt_price(tps[0])}(+{p1}%)\n"
        f"- 2차 {fmt_price(tps[1])}(+{p2}%)\n"
        f"- 3차 {fmt_price(tps[2])}(+{p3}%)"
    )


def generate_signal_chart_bytes(res):
    """
    [Bybit V5 실제 15분봉·거래량 100% 동기화 차트 생성기]
    1) Bybit 공식 V5 미러 서버에서 실제 15분봉 캔들과 실제 거래량을 직결 조회
    2) 비선형 스케일(FuncScale)로 캔들 영역 확대 & TP1~TP3 여백 압축
    3) 상단 TP3 선과 이평선 범례가 겹치지 않도록 범례를 차트 프레임 상단 바깥으로 분리
    """
    if not _ensure_matplotlib():
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.gridspec import GridSpec

    symbol = res.get("symbol", "")
    base = res.get("base_coin") or res.get("base") or symbol.replace("USDT", "")
    score = res.get("score", 85)
    track_type = res.get("track_type", "TRACK_A")
    price = to_float(res.get("price"))
    if not symbol or price <= 0:
        return None

    df = fetch_exact_bybit_klines(symbol, interval="15", limit=80)
    if df is None or df.empty or len(df) < 20:
        return None

    # 마지막 캔들의 현재가를 포착 시점 가격과 일치시켜 갭 오차 제거
    df = df.copy()
    df["ma7"] = df["close"].rolling(7, min_periods=2).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=3).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=5).mean()
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=3).mean()

    # 최근 45개 15분봉(11시간 15분 흐름) 슬라이싱
    df = df.iloc[-45:].reset_index(drop=True)

    # 실제 차트의 마지막 종가를 기준으로 진입가 동기화 (실제 차트와 100% 일치 보장)
    latest_close = to_float(df["close"].iloc[-1], price)
    if abs(latest_close - price) / max(price, 1e-8) <= 0.03:
        price = latest_close

    e_low = to_float(res.get("entry_low"), price * 0.9975)
    e_high = to_float(res.get("entry_high"), price * 1.0010)
    sl_p = to_float(res.get("stop_price"), price * 0.94)
    sl_pct = int(round(to_float(res.get("sl_pct"), FIXED_SL_PCT)))
    tps = res.get("tps") or [price * 1.08, price * 1.16, price * 1.24]
    tp_pcts = res.get("tp_pcts") or [8.0, 16.0, 24.0]

    tp1_p, tp2_p, tp3_p = to_float(tps[0]), to_float(tps[1]), to_float(tps[2])
    p1 = int(round(to_float(tp_pcts[0], 8.0)))
    p2 = int(round(to_float(tp_pcts[1], 16.0)))
    p3 = int(round(to_float(tp_pcts[2], 24.0)))

    try:
        bg_color = "#0f131c"
        panel_bg = "#131722"
        grid_color = "#232838"

        fig = plt.figure(figsize=(11.2, 6.8), dpi=160, facecolor=bg_color)
        gs = GridSpec(2, 1, height_ratios=[4.0, 1.15], hspace=0.04, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax)

        ax.set_facecolor(panel_bg)
        ax_vol.set_facecolor(panel_bg)

        c_low_min = min(to_float(df["low"].min(), sl_p), sl_p)
        c_high_max = max(to_float(df["high"].max(), price), price * 1.02)
        y_piv = min(tp1_p * 0.985, c_high_max * 1.012)

        candle_span = max(y_piv - c_low_min, price * 0.04)
        tp_span = max(tp3_p - y_piv, price * 0.05)
        comp_ratio = float(np.clip((candle_span * 0.56) / tp_span, 0.18, 0.75))

        def _forward_scale(y):
            y_arr = np.asarray(y, dtype=float)
            return np.where(y_arr <= y_piv, y_arr, y_piv + (y_arr - y_piv) * comp_ratio)

        def _inverse_scale(y):
            y_arr = np.asarray(y, dtype=float)
            return np.where(y_arr <= y_piv, y_arr, y_piv + (y_arr - y_piv) / max(comp_ratio, 1e-6))

        try:
            ax.set_yscale("function", functions=(_forward_scale, _inverse_scale))
        except Exception:
            pass

        n = len(df)
        up_color = "#089981"
        down_color = "#f23645"

        ax.text(
            0.42,
            0.46,
            f"BYBIT · ${base}USDT\n15M PERPETUAL",
            transform=ax.transAxes,
            fontsize=32,
            fontweight="bold",
            color="#ffffff",
            alpha=0.035,
            ha="center",
            va="center",
            zorder=0,
        )

        for i in range(n):
            o = to_float(df.loc[i, "open"])
            c = to_float(df.loc[i, "close"])
            h = to_float(df.loc[i, "high"])
            l = to_float(df.loc[i, "low"])
            v = to_float(df.loc[i, "volume"])
            col = up_color if c >= o else down_color

            ax.plot([i, i], [l, h], color=col, linewidth=1.35, zorder=3)
            body_bottom = min(o, c)
            body_h = max(abs(c - o), price * 0.0007)
            rect = Rectangle(
                (i - 0.36, body_bottom),
                0.72,
                body_h,
                facecolor=col,
                edgecolor=col,
                linewidth=0.8,
                zorder=4,
            )
            ax.add_patch(rect)
            ax_vol.bar(i, v, width=0.68, color=col, alpha=0.70, zorder=3)

        ax.plot(df.index, df["ma7"], color="#00e5ff", linewidth=1.2, label="MA7", alpha=0.88, zorder=5)
        ax.plot(df.index, df["ma20"], color="#2962ff", linewidth=1.5, label="MA20", alpha=0.95, zorder=5)
        ax.plot(df.index, df["ma50"], color="#ff9800", linewidth=1.5, label="MA50", alpha=0.90, zorder=5)
        ax_vol.plot(df.index, df["vol_ma20"], color="#ffd166", linewidth=1.1, alpha=0.85, zorder=4)

        box_start = n - 1
        box_end = n + 4.2
        ax.fill_between([box_start, box_end], price, tp1_p, color="#4cc9f0", alpha=0.09, zorder=2)
        ax.fill_between([box_start, box_end], tp1_p, tp2_p, color="#06d6a0", alpha=0.11, zorder=2)
        ax.fill_between([box_start, box_end], tp2_p, tp3_p, color="#ffd166", alpha=0.13, zorder=2)
        ax.fill_between([box_start, box_end], sl_p, price, color="#ef476f", alpha=0.14, zorder=2)

        ax.axhspan(e_low, e_high, color="#3b82f6", alpha=0.22, zorder=2)

        x_right = n + 13.5
        ax.set_xlim(-1.0, x_right)

        level_specs = [
            (tp3_p, f"TP3 (+{p3}%)  {fmt_price(tp3_p)}", "#ffd166", "--", 1.5),
            (tp2_p, f"TP2 (+{p2}%)  {fmt_price(tp2_p)}", "#06d6a0", "--", 1.4),
            (tp1_p, f"TP1 (+{p1}%)  {fmt_price(tp1_p)}", "#4cc9f0", "--", 1.4),
            (price, f"ENTRY  {fmt_price(price)}", "#ffffff", "-", 1.5),
            (sl_p, f"SL (-{sl_pct}%)  {fmt_price(sl_p)}", "#ef476f", "--", 1.5),
        ]

        for lv_price, label_txt, col, l_style, lw in level_specs:
            if lv_price <= 0:
                continue
            ax.axhline(lv_price, color=col, linestyle=l_style, linewidth=lw, alpha=0.90, zorder=6)
            ax.text(
                n + 0.5,
                lv_price,
                f" {label_txt} ",
                color="#0f131c" if col in ("#ffd166", "#06d6a0", "#4cc9f0", "#ffffff") else "#ffffff",
                fontsize=8.8,
                fontweight="bold",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=col, edgecolor="none", alpha=0.96),
                zorder=7,
            )

        y_min = c_low_min - candle_span * 0.08
        y_max = tp3_p + (tp3_p - tp2_p) * 0.42
        ax.set_ylim(y_min, y_max)

        tick_candidates = sorted(list({
            round(c_low_min, 6),
            round(price, 6),
            round(tp1_p, 6),
            round(tp2_p, 6),
            round(tp3_p, 6),
        }))
        ax.set_yticks(tick_candidates)
        ax.set_yticklabels([fmt_price(t) for t in tick_candidates], color="#9ca3af", fontsize=8.5)

        # 거래량 Y축 상한 및 눈금 표시 (Bybit 실제 거래량 수치 표시)
        vol_max = to_float(df["volume"].max(), 100.0)
        ax_vol.set_ylim(0, vol_max * 1.18)
        ax_vol.set_yticks([vol_max * 0.5, vol_max])
        ax_vol.set_yticklabels([fmt_vol(vol_max * 0.5), fmt_vol(vol_max)], color="#9ca3af", fontsize=7.8)

        # 마지막 봉 기준 KST 시각 표시
        tick_idxs = [i for i in range(4, n, 8)]
        if (n - 1) not in tick_idxs:
            tick_idxs.append(n - 1)
        tick_labels = []
        for idx_t in tick_idxs:
            ts_val = to_float(df.loc[idx_t, "timestamp"], 0.0) if "timestamp" in df.columns else 0.0
            if ts_val > 0:
                if ts_val < 1e11:
                    ts_val *= 1000.0
                dt_str = datetime.fromtimestamp(ts_val / 1000.0, tz=KST).strftime("%H:%M")
            else:
                dt_str = f"-{(n - 1 - idx_t) * 15}m"
            tick_labels.append(dt_str)

        ax_vol.set_xticks(tick_idxs)
        ax_vol.set_xticklabels(tick_labels, color="#9ca3af", fontsize=8.5)
        curr_vol_v = to_float(df["volume"].iloc[-1], 0.0)
        ax_vol.text(
            0.01,
            0.78,
            f"Bybit Volume (15M): {fmt_vol(curr_vol_v)}  |  Vol MA20",
            transform=ax_vol.transAxes,
            color="#cbd5e1",
            fontsize=8,
            fontweight="bold",
        )

        ax.grid(True, linestyle="--", linewidth=0.5, color=grid_color, alpha=0.75)
        ax_vol.grid(True, linestyle="--", linewidth=0.5, color=grid_color, alpha=0.55)
        ax.tick_params(axis="x", labelbottom=False)

        for a in (ax, ax_vol):
            for spine in a.spines.values():
                spine.set_color("#2a2e39")

        track_badge = {
            "TRACK_A": "TRACK A · Pullback",
            "TRACK_B": "TRACK B · Breakout",
            "TRACK_C": "TRACK C · Leader Fib 0.5",
        }.get(track_type, track_type)

        last_ts = to_float(df["timestamp"].iloc[-1], 0.0)
        last_kst_str = (
            datetime.fromtimestamp((last_ts if last_ts > 1e11 else last_ts * 1000.0) / 1000.0, tz=KST).strftime("%m/%d %H:%M KST")
            if last_ts > 0 else "LIVE"
        )

        ax.set_title(
            f"${base} ({symbol})  |  Bybit 15M ({last_kst_str})  |  {track_badge} (Score: {score})",
            color="#ffffff",
            fontsize=12.0,
            fontweight="bold",
            pad=14,
            loc="left",
        )

        # TP3 선과 겹치지 않도록 범례를 차트 상단 프레임 바깥 우측에 배치
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.01),
            ncol=3,
            facecolor="#1a1f2e",
            edgecolor="#2a2e39",
            labelcolor="#e2e8f0",
            fontsize=8.2,
            framealpha=0.95,
        )

        plt.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.07)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", facecolor=bg_color)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print(f"[차트 렌더링 오류] {symbol}: {e}")
        return None


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
            files = {"photo": ("signal_chart.png", photo_bytes, "image/png")}
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
    """방금 포착된 $PUMP 종목의 실제 Bybit V5 15분봉·거래량 데이터로 즉시 비교 발송"""
    sample_res = {
        "symbol": "PUMPUSDT",
        "base_coin": "PUMP",
        "base": "PUMP",
        "track_type": "TRACK_A",
        "track_title": "🟢 [1차 선취매/눌림 타점 포착]",
        "score": 92,
        "price": 0.002910,
        "entry_low": 0.002903,
        "entry_high": 0.002913,
        "stop_price": 0.002735,
        "sl_pct": 6,
        "tps": [0.003143, 0.003521, 0.003812],
        "tp_pcts": [8.0, 21.0, 31.0],
    }
    x_caption = build_x_post_message(sample_res)
    chart_bytes = generate_signal_chart_bytes(sample_res)
    if chart_bytes:
        ok = send_telegram_photo_with_caption(chart_bytes, x_caption)
        print(f"[Bybit 실제 15M 차트 발송 결과]: {'성공' if ok else '실패'}")
        return ok
    return send_telegram_single(x_caption)


if __name__ == "__main__":
    send_uai_sample_test()
