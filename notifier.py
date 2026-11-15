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
except Exception:
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    FIXED_SL_PCT = 6
    KST = timezone(timedelta(hours=9))

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


def _ensure_matplotlib():
    """런타임에 matplotlib가 없으면 자동 설치"""
    try:
        import matplotlib
        return True
    except ImportError:
        try:
            print("[알림] matplotlib 자동 설치 중...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
            return True
        except Exception as e:
            print(f"[경고] matplotlib 자동 설치 실패: {e}")
            return False


def _build_sample_breakout_df(entry_price=0.09520):
    """테스트 발송용 15분봉 45캔들(박스권 응축 -> 거래량 폭발 돌파 -> 눌림 지지) 생성"""
    np.random.seed(42)
    rows = []
    now_ms = int(time.time() * 1000)
    base_p = entry_price * 0.965
    curr = base_p

    for i in range(34):
        ts = now_ms - (45 - i) * 15 * 60 * 1000
        wave = np.sin(i * 0.4) * (entry_price * 0.006)
        noise = np.random.uniform(-0.0025, 0.0028) * entry_price
        o = curr
        c = base_p + wave + noise
        h = max(o, c) + np.random.uniform(0.0010, 0.0035) * entry_price
        l = min(o, c) - np.random.uniform(0.0010, 0.0035) * entry_price
        v = np.random.uniform(800, 1800)
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
        curr = c

    break_steps = [
        (0.976, 3200), (0.988, 5400), (1.004, 6800),
        (1.015, 4900), (1.008, 2100), (1.001, 1400),
        (0.996, 950), (0.998, 1100), (1.002, 2600),
        (0.999, 1300), (1.000, 3100)
    ]
    for idx_b, (ratio, vol_v) in enumerate(break_steps):
        ts = now_ms - (11 - idx_b) * 15 * 60 * 1000
        o = curr
        c = entry_price * ratio
        h = max(o, c) + entry_price * 0.0040
        l = min(o, c) - entry_price * 0.0030
        rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": vol_v})
        curr = c

    return pd.DataFrame(rows)


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
    [고급 트레이딩뷰 스타일 15분봉 차트 생성기]
    1) 15분봉(최근 45캔들)을 사용하여 캔들 크기와 추세 가독성을 대폭 확대
    2) 비선형 압축 스케일(FuncScale)을 적용하여 하단 캔들 영역이 화면의 62%를 시원하게 차지하고,
       상단 TP1~TP3 사이의 과도한 빈 여백을 자동으로 압축
    3) 롱 포지션 수익/손절 음영 박스 + 하단 거래량(Volume) 서브 패널 + 배경 워터마크 추가
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

    df = pd.DataFrame()
    if not res.get("_is_test_sample"):
        # 15분봉 65캔들 조회 (이동평균선 계산 후 최근 45캔들 슬라이싱)
        if api_mod and hasattr(api_mod, "fetch_klines"):
            try:
                df = api_mod.fetch_klines(symbol, "15", limit=65)
            except Exception:
                df = pd.DataFrame()

        if df is None or df.empty or len(df) < 20:
            try:
                r = requests.get(
                    "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                    params={"contract": f"{base}_USDT", "interval": "15m", "limit": 65},
                    timeout=5,
                )
                if r.status_code == 200 and isinstance(r.json(), list) and r.json():
                    raw_df = pd.DataFrame(r.json()).rename(
                        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                    )
                    raw_df["timestamp"] = pd.to_numeric(raw_df["timestamp"], errors="coerce") * 1000.0
                    for col in ("open", "high", "low", "close", "volume"):
                        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
                    df = raw_df.dropna(subset=["close"]).sort_values("timestamp").reset_index(drop=True)
            except Exception:
                pass

    if df is None or df.empty or len(df) < 20:
        df = _build_sample_breakout_df(price)

    if "volume" not in df.columns:
        df["volume"] = 1000.0

    df["ma7"] = df["close"].rolling(7, min_periods=2).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=3).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=5).mean()
    df["vol_ma20"] = df["volume"].rolling(20, min_periods=3).mean()

    # 최근 45개 15분봉만 남겨서 캔들 하나하나를 큼직하게 표시
    df = df.iloc[-45:].reset_index(drop=True)

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
        gs = GridSpec(2, 1, height_ratios=[4.2, 1.0], hspace=0.04, figure=fig)
        ax = fig.add_subplot(gs[0])
        ax_vol = fig.add_subplot(gs[1], sharex=ax)

        ax.set_facecolor(panel_bg)
        ax_vol.set_facecolor(panel_bg)

        # [핵심] 캔들 영역은 크게 키우고 TP1~TP3 사이의 빈 공간은 압축하는 이중 스케일(Piecewise Scale)
        c_low_min = min(to_float(df["low"].min(), sl_p), sl_p)
        c_high_max = max(to_float(df["high"].max(), price), price * 1.025)
        y_piv = min(tp1_p * 0.985, c_high_max * 1.012)

        candle_span = max(y_piv - c_low_min, price * 0.04)
        tp_span = max(tp3_p - y_piv, price * 0.05)
        # TP 영역이 차트 상단 38% 안에 오밀조밀하게 들어오도록 압축 비율 자동 계산
        comp_ratio = float(np.clip((candle_span * 0.58) / tp_span, 0.18, 0.75))

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

        # 배경 중앙 워터마크 (트레이딩뷰 스타일)
        ax.text(
            0.42,
            0.48,
            f"${base}\n15M CHART",
            transform=ax.transAxes,
            fontsize=38,
            fontweight="bold",
            color="#ffffff",
            alpha=0.035,
            ha="center",
            va="center",
            zorder=0,
        )

        # 캔들스틱 & 하단 거래량 바 렌더링 (캔들 폭을 0.72로 넓혀 시원하게 표시)
        for i in range(n):
            o = to_float(df.loc[i, "open"])
            c = to_float(df.loc[i, "close"])
            h = to_float(df.loc[i, "high"])
            l = to_float(df.loc[i, "low"])
            v = to_float(df.loc[i, "volume"])
            col = up_color if c >= o else down_color

            ax.plot([i, i], [l, h], color=col, linewidth=1.35, zorder=3)
            body_bottom = min(o, c)
            body_h = max(abs(c - o), price * 0.0008)
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

            ax_vol.bar(i, v, width=0.68, color=col, alpha=0.65, zorder=3)

        # 이평선 렌더링 (MA7 민트, MA20 블루, MA50 오렌지)
        ax.plot(df.index, df["ma7"], color="#00e5ff", linewidth=1.2, label="MA7", alpha=0.85, zorder=5)
        ax.plot(df.index, df["ma20"], color="#2962ff", linewidth=1.5, label="MA20", alpha=0.95, zorder=5)
        ax.plot(df.index, df["ma50"], color="#ff9800", linewidth=1.5, label="MA50", alpha=0.90, zorder=5)
        ax_vol.plot(df.index, df["vol_ma20"], color="#ffd166", linewidth=1.1, alpha=0.85, zorder=4)

        # 우측 타점 프로젝션 영역 (트레이딩뷰 Long Position Tool 스타일 수익/손절 박스)
        box_start = n - 1
        box_end = n + 4.2
        ax.fill_between([box_start, box_end], price, tp1_p, color="#4cc9f0", alpha=0.09, zorder=2)
        ax.fill_between([box_start, box_end], tp1_p, tp2_p, color="#06d6a0", alpha=0.11, zorder=2)
        ax.fill_between([box_start, box_end], tp2_p, tp3_p, color="#ffd166", alpha=0.13, zorder=2)
        ax.fill_between([box_start, box_end], sl_p, price, color="#ef476f", alpha=0.14, zorder=2)

        # 진입 구간 밴드 강조
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
        y_max = tp3_p + (tp3_p - tp2_p) * 0.35
        ax.set_ylim(y_min, y_max)

        # Y축 눈금이 겹치지 않도록 주요 가격대만 깔끔하게 표시
        tick_candidates = sorted(list({
            round(c_low_min, 6),
            round(price, 6),
            round(tp1_p, 6),
            round(tp2_p, 6),
            round(tp3_p, 6),
        }))
        ax.set_yticks(tick_candidates)
        ax.set_yticklabels([fmt_price(t) for t in tick_candidates], color="#9ca3af", fontsize=8.5)

        # X축 하단 시간 라벨 (15분봉 KST 기준)
        tick_idxs = [i for i in range(0, n, 8)]
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
        ax_vol.set_yticks([])
        ax_vol.text(
            0.01, 0.78, "Volume (15M) & MA20",
            transform=ax_vol.transAxes, color="#9ca3af", fontsize=8, fontweight="bold"
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

        ax.set_title(
            f"${base} ({symbol})  |  15M Chart & Targets  |  {track_badge} (Score: {score})",
            color="#ffffff",
            fontsize=12.5,
            fontweight="bold",
            pad=12,
            loc="left",
        )
        ax.legend(
            loc="upper left",
            ncol=3,
            facecolor="#1a1f2e",
            edgecolor="#2a2e39",
            labelcolor="#e2e8f0",
            fontsize=8.5,
        )

        plt.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.07)
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
    """생성된 차트 이미지와 X 양식 캡션을 하나의 사진 메시지로 전송"""
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
        print("[경고] 사진 바이트 또는 텔레그램 토큰/챗ID 누락")
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
            print(f"[텔레그램 사진 전송 응답 오류] HTTP {resp.status_code}: {resp.text}")
            if resp.status_code == 429:
                retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                time.sleep(retry_after + 0.5)
                continue
        except Exception as e:
            print(f"[텔레그램 차트 사진 전송 재시도 {attempt + 1}/3] {e}")
        time.sleep(1.2)
    return False


def send_uai_sample_test():
    """방금 포착된 $BERA 수치 기준으로 업그레이드된 15M 고화질 차트 테스트 발송"""
    sample_res = {
        "symbol": "BERAUSDT",
        "base_coin": "BERA",
        "base": "BERA",
        "track_type": "TRACK_A",
        "track_title": "🧪 [15M 차트 업그레이드 프리뷰]",
        "score": 85,
        "price": 0.09520,
        "entry_low": 0.09496,
        "entry_high": 0.09530,
        "stop_price": 0.08949,
        "sl_pct": 6,
        "tps": [0.10282, 0.11519, 0.12471],
        "tp_pcts": [8.0, 21.0, 31.0],
    }
    x_caption = build_x_post_message(sample_res)
    chart_bytes = generate_signal_chart_bytes(sample_res)
    if chart_bytes:
        ok = send_telegram_photo_with_caption(chart_bytes, x_caption)
        print(f"[15M 고화질 차트 테스트 발송 결과]: {'성공' if ok else '실패'}")
        return ok
    else:
        ok = send_telegram_single(x_caption)
        return ok


if __name__ == "__main__":
    send_uai_sample_test()
