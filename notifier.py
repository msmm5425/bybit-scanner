import time
import requests

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    to_float,
)


def fmt_price(price):
    p = to_float(price)
    if p <= 0:
        return "0"
    if p >= 1000:
        return f"{p:,.2f}"
    if p >= 10:
        return f"{p:.3f}"
    if p >= 1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.5f}"
    if p >= 0.0001:
        return f"{p:.6f}"
    return f"{p:.8f}"


def fmt_pct(pct_val):
    v = round(to_float(pct_val), 1)
    return f"{int(v)}" if v == int(v) else f"{v:.1f}"


def build_briefing_message(res, now_str):
    short_sym = res["symbol"].replace("USDT", "")
    return (
        f"{res['track_title']} ${short_sym} ({res['symbol']} | {res['score']}점)\n"
        f"🕒 포착 시각: {now_str}\n"
        f"• 현재가: {fmt_price(res['last_price'])} USDT (24h {res['change_24h']:+.2f}%)\n"
        f"• 거래대금: {res['turnover_m']:.1f}M | OI: {res['oi_m']:.1f}M | 펀딩비: {res['funding_rate']:+.4f}%\n"
        f"• [멀티 프레임 진단]: {res['htf_status']} | 15m RSI6 {res['rsi6_15m']:.1f} (<75) | J값 {res['kdj_j_15m']:.1f} (<88)\n"
        f"• [포착 근거]: {' / '.join(res['matched_tags'])}\n"
        f"• [핵심 기준선 & 전략적 3단계 타점 가이드]:\n"
        f"  - 1차 지지선(매집 베이스 하단): {fmt_price(res['support_line'])} USDT\n"
        f"  - 2차 돌파선(스윙 고점): {fmt_price(res['swing_high_line'])} USDT\n"
        f"  - 권장 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n"
        f"  - 전략적 손절가(SL): {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%·{res['sl_reason']})\n"
        f"  - 3단계 익절(TP): "
        f"1차 {fmt_price(res['tps'][0])}(+{fmt_pct(res['tp_pcts'][0])}%·{res['tp_labels'][0]}) / "
        f"2차 {fmt_price(res['tps'][1])}(+{fmt_pct(res['tp_pcts'][1])}%·{res['tp_labels'][1]}) / "
        f"3차 {fmt_price(res['tps'][2])}(+{fmt_pct(res['tp_pcts'][2])}%·{res['tp_labels'][2]})"
    )


def build_x_post_message(res):
    short_sym = res["symbol"].replace("USDT", "")
    return (
        f"${short_sym} ({res['symbol']})\n\n"
        f"- 진입 구간: {fmt_price(res['entry_low'])} ~ {fmt_price(res['entry_high'])} USDT\n\n"
        f"- SL: {fmt_price(res['stop_price'])} USDT (-{res['sl_pct']}%)\n\n"
        f"- 1차 {fmt_price(res['tps'][0])}(+{fmt_pct(res['tp_pcts'][0])}%)\n"
        f"- 2차 {fmt_price(res['tps'][1])}(+{fmt_pct(res['tp_pcts'][1])}%)\n"
        f"- 3차 {fmt_price(res['tps'][2])}(+{fmt_pct(res['tp_pcts'][2])}%)"
    )


def send_telegram_single(text):
    print(text)
    print("--------------------------------------------------")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        time.sleep(0.5)
    except Exception as e:
        print(f"[Telegram 전송 오류] {e}")
