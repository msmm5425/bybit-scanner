import os
import io
import sys
import time
import subprocess
import requests
import numpy as np
import pandas as pd

try:
    import config as cfg
    TELEGRAM_BOT_TOKEN = getattr(cfg, "TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = getattr(cfg, "TELEGRAM_CHAT_ID", "")
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception:
    cfg = None
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""
    FIXED_SL_PCT = 6

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
    """워크플로우에 matplotlib가 없더라도 런타임에 자동 설치하여 차트 생성 보장"""
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


def _build_sample_breakout_df(entry_price=0.03188):
    """테스트 발송 또는 거래소 캔들 미조회 시 박스권 응축 -> 돌파 -> 눌림목 65캔들 샘플 생성"""
    np.random.seed(42)
    rows = []
    base_p = entry_price * 0.968
    curr = base_p

    # 1~52번 캔들: 2시간 이상 타이트한 박스권 응축
    for i in range(52):
        wave = np.sin(i * 0.35) * (entry_price * 0.004)
        noise = np.random.uniform(-0.0018, 0.0020) * entry_price
        o = curr
        c = base_p + wave + noise
        h = max(o, c) + np.random.uniform(0.0005, 0.0022) * entry_price
        l = min(o, c) - np.random.uniform(0.0005, 0.0022) * entry_price
        rows.append({"open": o, "high": h, "low": l, "close": c})
        curr = c

    # 53~59번 캔들: 박스권 상단 초동 돌파 양봉 분출
     break_steps = [0.975, 0.982, 0.991, 1.002, 1.012, 1.018, 1.010]
    for ratio in break_steps:
        o = curr
        c = entry_price * ratio
        h = max(o, c) + entry_price * 0.0035
        l = min(o, c) - entry_price * 0.0020
        rows.append({"open": o, "high": h, "low": l, "close": c})
        curr = c

    # 60~65번 캔들: 전고점 S/R Flip 및 MA20 지지 리테스트 후 재반등 타점(entry_price)
    retest_steps = [1.004, 0.998, 0.995, 0.997, 0.999, 1.000]
    for ratio in retest_steps:
        o = curr
        c = entry_price * ratio
        h = max(o, c) + entry_price * 0.0022
        l = min(o, c) - entry_price * 0.0025
        rows.append({"open": o, "high": h, "low": l, "close": c})
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
    포착 종목의 최근 5분봉 캔들과 MA20/MA50, 진입가·SL·1~3차 TP 수평선을 그린
    트레이딩뷰 다크모드 스타일 PNG 이미지를 메모리(BytesIO)로 생성합니다.
    """
    if not _ensure_matplotlib():
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    symbol = res.get("symbol", "")
    base = res.get("base_coin") or res.get("base") or symbol.replace("USDT", "")
    score = res.get("score", 85)
    track_type = res.get("track_type", "TRACK_A")
    price = to_float(res.get("price"))
    if not symbol or price <= 0:
        return None

    df = pd.DataFrame()
    if not res.get("_is_test_sample"):
        if api_mod and hasattr(api_mod, "fetch_klines"):
            try:
                df = api_mod.fetch_klines(symbol, "5", limit=75)
            except Exception:
                df = pd.DataFrame()

        if df is None or df.empty or len(df) < 20:
            try:
                r = requests.get(
                    "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                    params={"contract": f"{base}_USDT", "interval": "5m", "limit": 75},
                    timeout=5,
                )
                if r.status_code == 200 and isinstance(r.json(), list) and r.json():
                    raw_df = pd.DataFrame(r.json()).rename(
                        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
                    )
                    for col in ("open", "high", "low", "close"):
                        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")
                    df = raw_df.dropna(subset=["close"]).reset_index(drop=True)
            except Exception:
                pass

    if df is None or df.empty or len(df) < 20:
        if res.get("_is_test_sample"):
            df = _build_sample_breakout_df(price)
        else:
            return None

    df = df.iloc[-65:].reset_index(drop=True)
    df["ma20"] = df["close"].rolling(20, min_periods=3).mean()
    df["ma50"] = df["close"].rolling(50, min_periods=5).mean()

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
        fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=130)
        bg_color = "#131722"
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)

        n = len(df)
        for i in range(n):
            o = to_float(df.loc[i, "open"])
            c = to_float(df.loc[i, "close"])
            h = to_float(df.loc[i, "high"])
            l = to_float(df.loc[i, "low"])
            color = "#089981" if c >= o else "#f23645"

            ax.plot([i, i], [l, h], color=color, linewidth=1.1, zorder=2)
            body_bottom = min(o, c)
            body_h = max(abs(c - o), price * 0.0006)
            rect = Rectangle(
                (i - 0.32, body_bottom),
                0.64,
                body_h,
                facecolor=color,
                edgecolor=color,
                zorder=3,
            )
            ax.add_patch(rect)

        ax.plot(df.index, df["ma20"], color="#2962ff", linewidth=1.3, label="MA20", alpha=0.9, zorder=4)
        ax.plot(df.index, df["ma50"], color="#ff9800", linewidth=1.3, label="MA50", alpha=0.85, zorder=4)

        x_right = n + 14
        ax.set_xlim(-1.5, x_right)

        ax.axhspan(e_low, e_high, color="#3b82f6", alpha=0.18, zorder=1)

        level_specs = [
            (tp3_p, f"TP3 (+{p3}%)  {fmt_price(tp3_p)}", "#ffd166", "--", 1.5),
            (tp2_p, f"TP2 (+{p2}%)  {fmt_price(tp2_p)}", "#06d6a0", "--", 1.4),
            (tp1_p, f"TP1 (+{p1}%)  {fmt_price(tp1_p)}", "#4cc9f0", "--", 1.4),
            (price, f"ENTRY  {fmt_price(price)}", "#ffffff", "-", 1.3),
            (sl_p, f"SL (-{sl_pct}%)  {fmt_price(sl_p)}", "#ef476f", "--", 1.5),
        ]

        for lv_price, label_txt, col, l_style, lw in level_specs:
            if lv_price <= 0:
                continue
            ax.axhline(lv_price, color=col, linestyle=l_style, linewidth=lw, alpha=0.88, zorder=5)
            ax.text(
                n + 0.6,
                lv_price,
                f" {label_txt} ",
                color="#131722" if col in ("#ffd166", "#06d6a0", "#4cc9f0", "#ffffff") else "#ffffff",
                fontsize=8.5,
                fontweight="bold",
                va="center",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=col, edgecolor="none", alpha=0.95),
                zorder=6,
            )

        y_min = min(to_float(df["low"].min(), sl_p), sl_p) * 0.975
        y_max = max(to_float(df["high"].max(), tp3_p), tp3_p) * 1.030
        ax.set_ylim(y_min, y_max)

        ax.grid(True, linestyle=":", linewidth=0.6, color="#2a2e39", alpha=0.8)
        ax.tick_params(axis="y", colors="#b2b5be", labelsize=9)
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a2e39")

        ax.set_title(
            f"${base} ({symbol})  |  5M Chart & Targets  |  {track_type} (Score: {score})",
            color="#ffffff",
            fontsize=12,
            fontweight="bold",
            pad=12,
            loc="left",
        )
        ax.legend(loc="upper left", facecolor="#1e222d", edgecolor="#2a2e39", labelcolor="#d1d4dc", fontsize=8.5)

        plt.tight_layout()
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
    if not photo_bytes or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    for attempt in range(3):
        try:
            files = {"photo": ("signal_chart.png", photo_bytes, "image/png")}
            data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption_text}
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
    """$UAI 예시 데이터로 차트 이미지 + X 양식 캡션을 텔레그램 방에 즉시 테스트 발송"""
    sample_res = {
        "_is_test_sample": True,
        "symbol": "UAIUSDT",
        "base_coin": "UAI",
        "base": "UAI",
        "track_type": "TRACK_B",
        "track_title": "🧪 [차트+X양식 테스트 발송]",
        "score": 92,
        "price": 0.03188,
        "entry_low": 0.03181,
        "entry_high": 0.03192,
        "stop_price": 0.02998,
        "sl_pct": 6,
        "tps": [0.03444, 0.03859, 0.04178],
        "tp_pcts": [8.0, 21.0, 31.0],
    }
    x_caption = build_x_post_message(sample_res)
    chart_bytes = generate_signal_chart_bytes(sample_res)
    if chart_bytes:
        ok = send_telegram_photo_with_caption(chart_bytes, x_caption)
        print(f"[$UAI 차트+X양식 테스트 발송 결과]: {'성공' if ok else '실패'}")
        return ok
    else:
        ok = send_telegram_single(x_caption)
        print(f"[$UAI 텍스트 폴백 발송 결과]: {'성공' if ok else '실패'}")
        return ok


if __name__ == "__main__":
    send_uai_sample_test()
