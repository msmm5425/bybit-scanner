import os, re, io, sys, time, json, shutil, asyncio, tempfile, threading, subprocess, urllib.parse, requests
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

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
VIDEO_W = 960
VIDEO_H = 1280
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


def _fetch_replay_klines_window(symbol, interval, start_ms, end_ms, max_bars=850):
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


def _build_supply_overlay_line(res):
    """시총 · Bybit OI/시총 · 회전율 · 5대 거래소(BN·GATE·BG·BYBIT·OKX) L/S 요약 라인 생성"""
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
        if t_val <= 0.75:
            ls_badge = "🟢숏쏠림"
        elif t_val >= 2.50:
            ls_badge = "🔴롱과밀"
        else:
            ls_badge = "⚪중립"
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
        lines.append(
            "\n⚠️ [장기 진입 전 10초 체크]: 30일 내 대형 클리프 언락(Tokenomist) · 바이낸스 유의종목(Monitoring Tag) 여부 확인 권장"
        )

    return "\n".join(lines)


def _pick_pair(pairs, symbol):
    if not pairs:
        return "", ""
    seed = sum(ord(ch) for ch in str(symbol or "BTC"))
    return pairs[seed % len(pairs)]


def build_x_hook_pair(res):
    """
    [36종 전략별 한/일 네이티브 훅(Hook) 1:1 매핑 엔진]
    - 반환값: (한글 간략 포착근거, 자연스러운 일본어 아나운서 번역)
    """
    symbol = str(res.get("symbol", ""))
    track_type = str(res.get("track_type", "TRACK_A")).upper()
    r_text = " ".join(str(x) for x in (res.get("reasons") or []))
    title_str = str(res.get("track_title", ""))

    # 1) 장기 포지션 (Track D1 ~ D6)
    if track_type == "TRACK_D" or res.get("is_long_term"):
        sub_s = str(res.get("sub_strategy", "")).upper()
        if "D6" in sub_s or "스프링" in sub_s:
            return _pick_pair([
                ("신저가 개미털기(스프링) 후 다개월 바닥선 거래량 동반 재탈환 포착",
                 "安値更新のふるい落とし後、出来高を伴う底値ライン奪還シグナル"),
                ("공포 투매 물량 흡수 완료! 바닥선 강하게 재탈환하는 역발상 대시세 초입",
                 "パニック売りを吸収し、底値ラインを力強く奪還する逆張り初動シグナル"),
                ("개미 손절 털어낸 뒤 거래량 터지며 바닥 위로 올라탄 와이코프 스프링 포착",
                 "損切りを巻き込んだ直後に出来高急増で反発するワイコフスプリング"),
            ], symbol)
        if "D1" in sub_s:
            return _pick_pair([
                ("가격 횡보 속 세력 OI·OBV 선행 매집 포착! 조용히 물량 잠기는 대시세 초입",
                 "レンジ推移の中で大口の建玉とOBVが先行して積み上がる静かな仕込み初動"),
                ("티 안 나게 세력 수급이 먼저 뚫린 스텔스 바닥 매집 구간 포착",
                 "目立たない形で大口需給が先行突破したステルス底値仕込みシグナル"),
            ], symbol)
        if "D2" in sub_s:
            return _pick_pair([
                ("세력 매집봉 출현 후 거래량 건조 눌림 마무리! 2차 대시세 출발 직전",
                 "大口の買い集め後に出来高が細る押し目を完了した上昇出発直前シグナル"),
                ("매집봉 지지선 방어 후 거래량 말리며 반등 턴 나오는 세력 눌림목 포착",
                 "大口サポートを維持し、出来高減少から反発に転じた押し目シグナル"),
            ], symbol)
        if "D3" in sub_s:
            return _pick_pair([
                ("30일 장기 박스권 상단 거래량 폭발 첫 돌파! 큰 추세 파동 개시 포착",
                 "30日間の長期レンジ上限を出来高急増で初突破したトレンド開始シグナル"),
                ("한 달간 응축된 장기 박스권 저항을 거래량 싣고 뚫어낸 대시세 초입",
                 "1か月間凝縮された長期抵抗帯を出来高を伴って突破した大相場初動"),
            ], symbol)
        if "D4" in sub_s:
            return _pick_pair([
                ("장기 하락추세 탈출 및 4H 50MA 거래량 돌파 안착! 추세 턴어라운드 초입",
                 "長期下落トレンドからの脱出と4時間足50MA突破定着の転換初動"),
                ("이중바닥 완성 후 주요 추세선 돌파! 중장기 상승 1파동 시작점 포착",
                 "ダブルボトム完成後に主要トレンドラインを突破した上昇第1波の起点"),
            ], symbol)
        if "D5" in sub_s:
            return _pick_pair([
                ("바닥권 4H PPO·RSI·OBV 3중 강세 다이버전스 완성! 세력 바닥 흡수 포착",
                 "底値圏でのPPO・RSI・OBV三重強気ダイバージェンス完成シグナル"),
                ("하락 에너지는 멈추고 세력 매수세는 유입되는 매크로 바닥 반전 초입",
                 "下落圧力が止まり大口の買いが流入するマクロ底値反転初動"),
            ], symbol)
        return _pick_pair([
            ("세력 4H·1D 매집 후 대시세 1파동 초입 포착",
             "4時間足と日足での大口仕込み完了後の上昇第1波初動シグナル"),
            ("상위 프레임 바닥 숙성 완료! 세력 수급 결집되는 중장기 대시세 초입",
             "上位足の底固めが完了し大口需給が集中する中長期上昇初動"),
        ], symbol)

    # 2) 스탑헌팅 후 재진입
    if "재진입" in title_str or "재탈환" in r_text:
        return _pick_pair([
            ("개미털기 바닥 확인 후 거래량 실리며 V자 반등 시도중",
             "ふるい落としの底打ち確認後、出来高を伴うV字反発シグナル"),
            ("스탑헌팅 악성 매물 소화 완료! 비과열 구간에서 V자 재탈환 시도중",
             "ストップ狩りの売りを消化し、非過熱圏からV字奪還を狙うシグナル"),
            ("밑꼬리 개미털기 후 매수세 재유입되며 단기 추세 재가동중",
             "下ヒゲのふるい落とし後に買いが再流入する短期トレンド再始動シグナル"),
        ], symbol)

    # 3) 수급심층 특화 훅
    oi_mc_pct = to_float(res.get("oi_mc_pct"), 0.0)
    top_ls = to_float(res.get("top_ls"), 0.0)
    retail_ls = to_float(res.get("retail_ls"), 0.0)
    ls_div = to_float(res.get("ls_divergence"), 0.0)
    if top_ls >= 1.50 and 0 < retail_ls <= 1.05 and ls_div >= 1.60:
        return _pick_pair([
            ("고래는 롱 매집, 개미는 숏 쏠림! 스마트머니 포지션 괴리 폭발 초입",
             "クジラはロング、個人はショート偏重のスマートマネー乖離シグナル"),
            ("탑트레이더 롱 베팅 vs 일반투자자 숏 쏠림! 수급 비대칭 급등 후보 포착",
             "トップトレーダーのロング優位と個人ショート偏重の需給非対称シグナル"),
        ], symbol)
    if oi_mc_pct >= 15.0:
        return _pick_pair([
            ("시총 대비 선물 OI 압축 임계치 도달! 레버리지 화약고 점화 초입 포착",
             "時価総額に対する先物建玉の圧縮が臨界点に達したレバレッジ点火初動"),
            ("과적재된 선물 OI에 매수 거래량 점화! 강한 변동성 분출 직전 타점",
             "積み上がった先物建玉に買い出来高が点火したボラティリティ噴出直前"),
        ], symbol)

    # 4) 단기·스윙 트랙별 맞춤 훅
    if track_type == "TRACK_H":
        return _pick_pair([
            ("최근 급등 대장주 Top 5 발사 직전 DNA 패턴과 일치하는 초입 포착",
             "直近急騰トップ5銘柄の発射直前チャートDNAに一致した初動シグナル"),
            ("최근 시장을 주도한 급등주들의 발사 직전 차트 공식을 그대로 닮은 초입",
             "市場を牽引した急騰銘柄の発射直前パターンを再現した初動シグナル"),
            ("급등주 역추적 알고리즘이 포착한 에너지 응축·거래량 점화 초입 타점",
             "急騰銘柄アルゴリズムが捉えたエネルギー凝縮と出来高点火の初動"),
        ], symbol)
    if track_type == "TRACK_E":
        return _pick_pair([
            ("24H 저점 유동성 스윕(개미털기) 직후 강한 반전 매수세 유입중",
             "24時間安値の流動性スイープ直後に強い反転買いが流入したシグナル"),
            ("전저점 스탑헌팅 아래꼬리 흡수 후 강세 다이버전스와 함께 급반전 시도중",
             "直近安値のストップ狩りを吸収し強気ダイバージェンスで反転するシグナル"),
            ("저점 이탈 공포 물량을 세력이 받아먹고 지지선 탈환하는 SFP 반전 타점",
             "安値割れの投げ売りを大口が吸収しサポートを奪還するSFP反転シグナル"),
        ], symbol)
    if track_type == "TRACK_F":
        return _pick_pair([
            ("비트코인 흔들림 속에서도 역행 강세 보이는 상대강도 대장주 포착",
             "ビットコイン調整局面でも逆行高を見せる相対強度リーダー銘柄シグナル"),
            ("시장 조정장에서도 고점권 버티며 독주하는 상대강도(RS) 1티어 주도주",
             "相場全体の調整をものともせず高値圏を維持する相対強度トップ銘柄"),
            ("비트코인 약세를 무시하고 OBV 신고가와 함께 치고 나가는 역행 대장주",
             "ビットコイン弱含みを無視しOBV高値更新で上昇する逆行リーダー銘柄"),
        ], symbol)
    if track_type == "TRACK_G":
        return _pick_pair([
            ("깊은 음수 펀딩비와 OI 집중 속 숏스퀴즈 강제청산 초입 포착",
             "深いマイナス金利と建玉集中によるショートスクイーズ初動シグナル"),
            ("과도하게 쌓인 숏 포지션 위에 매수 폭발! 연쇄 숏스퀴즈 발사 직전",
             "積み上がったショートに買いが殺到する連鎖ショートスクイーズ直前"),
            ("음수 펀딩비 연료 가득 찬 상태에서 저항선 돌파하는 숏스퀴즈 초입 타점",
             "マイナス金利を燃料に抵抗線を突破するショートスクイーズ初動シグナル"),
        ], symbol)
    if track_type == "TRACK_C":
        return _pick_pair([
            ("급등 대장주 핵심 눌림목 지지 후 2차 파동 시도중",
             "急騰リーダー銘柄の重要押し目サポートからの第2波上昇シグナル"),
            ("1파 급등 후 거래량 건조 조정 마무리! 피보나치 핵심 지지선 반등 타점",
             "第1波急騰後の出来高減少調整を終えフィボナッチ支持線で反発するシグナル"),
            ("대장주 건강한 눌림목 소화 완료! 5분봉 추세 전환과 함께 2차 상승 시동",
             "健全な押し目を消化し5分足トレンド転換とともに第2波へ移行するシグナル"),
        ], symbol)
    if track_type == "TRACK_B":
        return _pick_pair([
            ("거래량이 살아나면서 24H 전고점 돌파 시도중",
             "出来高の急増を伴う24時間高値ブレイクアウトシグナル"),
            ("24시간 주요 저항선을 건전한 양봉과 거래량으로 뚫어낸 구조 돌파 초입",
             "24時間主要抵抗線を力強い陽線と出来高で突破した構造ブレイク初動"),
            ("돌파 후 지지 전환(S-R Flip) 확인! 매물벽 소화하고 상방 확장 시도중",
             "ブレイク後のロールリバーサルを確認し上値拡大を狙うシグナル"),
        ], symbol)
    if track_type == "TRACK_A" or "응축" in r_text or "스퀴즈" in r_text:
        return _pick_pair([
            ("박스권 에너지 응축 후 거래량 살아나며 추세 전환중",
             "レンジでのエネルギー凝縮後、出来高回復とともにトレンド転換するシグナル"),
            ("볼린저·켈트너 초응축(TTM 스퀴즈) 상방 발사! 매수 델타 유입 초입",
             "TTMスクイーズの極限収縮から上放れし買いデルタが流入した初動"),
            ("좁아진 변동성 박스를 거래량 싣고 뚫어내는 에너지 폭발 초입 포착",
             "狭まったボラティリティレンジを出来高を伴って上抜ける爆発初動"),
        ], symbol)
    if "음수 펀딩" in r_text:
        return (
            "숏 스퀴즈 수급 유입되며 단기 추세 반등 시도중",
            "ショートスクイーズ需給の流入による短期トレンド反発シグナル",
        )
    return (
        "눌림목 거래량 건조 후 수급 살아나며 반등 시도중",
        "押し目での出来高減少後、需給回復とともに反発を狙うシグナル",
    )


def build_x_hook_line(res):
    return build_x_hook_pair(res)[0]


def build_x_post_message(res):
    """기존 확정된 일본어 최상단 헤더 + 한국어 본문 X 포스팅 양식 100% 유지"""
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

    p1 = round(to_float(tp_pcts[0], 8.0), 1)
    p2 = round(to_float(tp_pcts[1], 15.0), 1)
    p3 = round(to_float(tp_pcts[2], 22.0), 1)
    hook_line = build_x_hook_line(res)
    jp_header = "上昇可能性の高いコインをスキャンしました\n#仮想通貨 #コインシグナル\n\n"
    footer_tags = "#코인시황 #코인신호 #코인 #재테크"

    if is_long:
        return (
            f"{jp_header}"
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
        f"{jp_header}"
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
    """OS 공용 Path.as_uri() · 임시 HTML finally 정리 · 비선형 압축 표식 · 예외 진단 로깅"""
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
    open_last, high_last, low_last = (
        to_float(latest_row["open"], price),
        to_float(latest_row["high"], price),
        to_float(latest_row["low"], price),
    )

    orig_entry = to_float(res.get("price"), price)
    ratio_adj = price / orig_entry if orig_entry > 0 else 1.0
    sl_pct = int(round(to_float(res.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
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
    comp_badge_html = (
        '<span style="color:#9598a1">⚡Y축 TP구간 압축</span>'
        if (up_comp < 0.98 or down_comp < 0.98)
        else ""
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
    lw_script_tags = _get_lw_charts_script_tags()

    html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
{lw_script_tags}
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#131722;font-family:-apple-system,BlinkMacSystemFont,Roboto,'Noto Sans CJK KR','Noto Sans KR',sans-serif;color:#d1d4dc;width:1160px;height:680px;overflow:hidden}}
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
      {comp_badge_html}
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
    color:fl.color,lineWidth:fl.lineWidth,lineStyle:fl.lineStyle,
    priceLineVisible:false,lastValueVisible:false,crosshairMarkerVisible:false
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

    tmp_html_path = None
    try:
        from playwright.sync_api import sync_playwright
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as f:
            f.write(html_content)
            tmp_html_path = f.name
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"])
            try:
                page = browser.new_page(viewport={"width": 1160, "height": 680}, device_scale_factor=1.5)
                page.goto(Path(tmp_html_path).resolve().as_uri(), wait_until="load", timeout=25000)
                page.wait_for_function("window.__CHART_DONE === true", timeout=10000)
                page.wait_for_timeout(200)
                png_bytes = page.screenshot(type="png")
            finally:
                browser.close()
        return png_bytes
    except Exception as e:
        print(f"[차트 렌더링 예외] {symbol}: {e}")
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


def _probe_audio_duration_sec(audio_path, default_sec=3.0):
    """ffprobe로 음성 클립 실제 재생 길이(초) 정밀 측정"""
    if not audio_path or not os.path.exists(audio_path):
        return default_sec
    if shutil.which("ffprobe"):
        try:
            r = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                d = to_float(r.stdout.strip(), 0.0)
                if d > 0.1:
                    return d
        except Exception:
            pass
    return default_sec


def _synth_jp_voice_segment(text, out_mp3):
    """
    일본어 여성 아나운서(ja-JP-NanamiNeural · 1.1배속) 음성 파일 생성
    - Playwright 이벤트 루프 충돌 방지를 위해 전용 스레드 + 새 이벤트 루프에서 실행
    - edge-tts 일시 오류 시 gTTS + ffmpeg atempo=1.1 자동 폴백
    """
    err_box = []

    def _run_edge():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            import edge_tts

            async def _go():
                comm = edge_tts.Communicate(text=text, voice=JP_TTS_VOICE, rate=JP_TTS_RATE)
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

    # 비상 백업: gTTS + ffmpeg 1.1배속
    try:
        from gtts import gTTS
        raw_mp3 = out_mp3 + ".raw.mp3"
        gTTS(text=text, lang="ja").save(raw_mp3)
        if os.path.exists(raw_mp3) and os.path.getsize(raw_mp3) > 300:
            r = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", raw_mp3,
                    "-filter:a", "atempo=1.1",
                    "-b:a", "128k",
                    out_mp3,
                ],
                capture_output=True,
                text=True,
                timeout=20,
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


def _build_narration_plan(close_info, entry_p, total_realized_pct, created_at, tmpdir, base_total_sec=30.5):
    """
    4단계 한글 자막 + 일본어 여성 아나운서(1.1배속) 음성 클립 생성 및 타임라인 정밀 동기화
    - 반환: (segments_meta_list, final_total_sec, tts_ok, tts_err)
    """
    base = str(close_info.get("base_coin") or str(close_info.get("symbol", "")).replace("USDT", "")).upper()
    dt_cap = datetime.fromtimestamp(created_at, tz=KST)
    date_ko = f"{dt_cap.month}월 {dt_cap.day}일"
    date_jp = f"{dt_cap.month}月{dt_cap.day}日"

    hook_ko = str(close_info.get("x_hook_ko") or "").strip()
    hook_jp = str(close_info.get("x_hook_jp") or "").strip()
    if not hook_ko or not hook_jp:
        p_ko, p_jp = build_x_hook_pair(close_info)
        hook_ko = hook_ko or p_ko
        hook_jp = hook_jp or p_jp

    entry_str = fmt_price(entry_p)
    entry_jp_num = entry_str.replace(",", "")
    pct_str = f"{total_realized_pct:.2f}"
    pct_jp_num = f"{total_realized_pct:.1f}"

    raw_specs = [
        {
            "step": "1 / 4",
            "tag": "INTRO",
            "ko_line1": "안녕하세요, 단평입니다 👋",
            "ko_line2": f"${base} 실전 검색기 익절 리플레이 브리핑",
            "jp_text": "こんにちは、ダンピョンです。",
        },
        {
            "step": "2 / 4",
            "tag": "SIGNAL",
            "ko_line1": f'{date_ko}, "{hook_ko}" 근거로',
            "ko_line2": f"${base} 종목이 검색기에 포착되었습니다",
            "jp_text": f"{date_jp}、「{hook_jp}」の根拠で、{base}がスキャナーに捕捉されました。",
        },
        {
            "step": "3 / 4",
            "tag": "PROFIT",
            "ko_line1": f"진입가격 {entry_str} 달러 · 최종 수익률 +{pct_str}%",
            "ko_line2": "달성하였습니다! 축하드립니다!! 🎉",
            "jp_text": f"エントリー価格、{entry_jp_num}ドル。最終収益率、プラス{pct_jp_num}パーセントを達成しました！おめでとうございます！",
        },
        {
            "step": "4 / 4",
            "tag": "FOLLOW",
            "ko_line1": "팔로우 + 알림설정 부탁드립니다 🔔",
            "ko_line2": f"X · {WATERMARK_HANDLE} 실시간 상승 타점 알림",
            "jp_text": "フォローと通知設定、よろしくお願いします！",
        },
    ]

    durs = []
    paths = []
    tts_ok = True
    tts_err = ""
    for idx, sp in enumerate(raw_specs):
        mp3_p = os.path.join(tmpdir, f"tts_seg_{idx}.mp3")
        ok, err = _synth_jp_voice_segment(sp["jp_text"], mp3_p)
        if not ok:
            tts_ok = False
            tts_err = err
            break
        dur = _probe_audio_duration_sec(mp3_p, default_sec=3.5)
        durs.append(dur)
        paths.append(mp3_p)

    if not tts_ok or len(durs) != 4:
        return [], base_total_sec, False, tts_err

    # 30.5초 리플레이 타임라인에 4단계 음성이 자연스럽게 겹치지 않고 배치되도록 시작/종료 초 계산
    s0 = 0.25
    e0 = s0 + durs[0]

    s1 = max(e0 + 0.55, 2.30)
    e1 = s1 + durs[1]

    s2 = max(e1 + 1.20, 16.20)
    e2 = s2 + durs[2]

    s3 = max(e2 + 0.85, 25.40)
    e3 = s3 + durs[3]

    final_total_sec = max(base_total_sec, e3 + 0.90)

    segments = []
    starts = [s0, s1, s2, s3]
    ends = [e0, e1, e2, e3]
    for i, sp in enumerate(raw_specs):
        show_start = 0.0 if i == 0 else starts[i] - 0.15
        show_end = final_total_sec if i == 3 else (starts[i + 1] - 0.15)
        segments.append({
            "step": sp["step"],
            "tag": sp["tag"],
            "ko_line1": sp["ko_line1"],
            "ko_line2": sp["ko_line2"],
            "audio_path": paths[i],
            "audio_start": round(starts[i], 3),
            "audio_dur": round(durs[i], 3),
            "audio_end": round(ends[i], 3),
            "sub_start": round(show_start, 3),
            "sub_end": round(show_end, 3),
        })

    return segments, final_total_sec, True, ""


def generate_profit_replay_videos_bundle(close_info):
    """
    [세로형(3:4 · 960x1280) 2종 리플레이 영상 동시 제작 엔진 v8]
    - 영상 1 (clean): 무음 클린 세로형 (하단 진입·최고·마감·수익 요약 카드)
    - 영상 2 (voiced): 한글 대형 자막 + 일본어 여성 아나운서(1.1배속) 음성 싱크 세로형
    - 단일 Playwright 세션에서 차트 1회 렌더링 후 하단 패널만 즉시 스왑 캡처하여 속도 최적화
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

    tf_ladder = [
        ("1", "1m", 60, 820),
        ("5", "5m", 300, 420),
        ("15", "15m", 900, 420),
        ("60", "1H", 3600, 260),
        ("240", "4H", 14400, 260),
    ]
    if hold_hours < 10.0:
        start_tier = 0
    elif hold_hours < 20.0:
        start_tier = 1
    elif hold_hours < 70.0:
        start_tier = 2
    elif hold_hours < 90.0:
        start_tier = 3
    else:
        start_tier = 4

    created_ms = created_at * 1000
    closed_ms = closed_at * 1000

    df_raw = pd.DataFrame()
    tf_code, tf_label, step_sec = tf_ladder[start_tier][:3]
    best_fallback = None

    for idx_t in range(start_tier, len(tf_ladder)):
        cand_code, cand_lbl, cand_step, cand_max_bars = tf_ladder[idx_t]
        req_start_ms = created_ms - (55 * cand_step * 1000)
        req_end_ms = closed_ms + (2 * cand_step * 1000)
        cand_df = _fetch_replay_klines_window(
            symbol,
            interval=cand_code,
            start_ms=req_start_ms,
            end_ms=req_end_ms,
            max_bars=cand_max_bars,
        )
        if cand_df.empty or len(cand_df) < 10:
            continue
        if best_fallback is None:
            best_fallback = (cand_df, cand_code, cand_lbl, cand_step)
        earliest_ts = int(to_float(cand_df["timestamp"].iloc[0], 0))
        if earliest_ts <= (created_ms + cand_step * 1000 * 3):
            df_raw, tf_code, tf_label, step_sec = cand_df, cand_code, cand_lbl, cand_step
            break

    if df_raw.empty and best_fallback is not None:
        df_raw, tf_code, tf_label, step_sec = best_fallback

    if df_raw.empty:
        LAST_VIDEO_ERROR = f"{symbol} {tf_label} 캔들 데이터 API 조회 실패"
        print(f"[리플레이 영상 실패] {LAST_VIDEO_ERROR}")
        return None

    df = df_raw.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp", ascending=True).reset_index(drop=True)
    df["ma5"] = df["close"].rolling(5, min_periods=2).mean()
    df["ma9"] = df["close"].rolling(9, min_periods=3).mean()
    df["ma20"] = df["close"].rolling(20, min_periods=5).mean()

    ts_arr = df["timestamp"].astype("int64").values
    entry_matches = np.where(ts_arr <= created_ms)[0]
    entry_abs = int(entry_matches[-1]) if len(entry_matches) > 0 else 0

    exit_matches = np.where(ts_arr <= closed_ms)[0]
    exit_abs = int(exit_matches[-1]) if len(exit_matches) > 0 else (len(df) - 1)
    if exit_abs < entry_abs:
        exit_abs = min(len(df) - 1, entry_abs)

    post_count = (exit_abs - entry_abs) + 1
    pre_bars = 24 if post_count <= 90 else (18 if post_count <= 220 else 14)
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
    sl_pct = int(round(to_float(close_info.get("sl_pct"), LONGTERM_SL_PCT if is_long else FIXED_SL_PCT)))
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
    max_fav_val = round(max(to_float(close_info.get("max_favorable_pct"), total_realized_pct), (max_high_val - entry_p) / entry_p * 100.0), 1)
    hold_str = _fmt_hold_duration(hold_minutes)

    entry_dt_str = datetime.fromtimestamp(created_at, tz=KST).strftime("%Y-%m-%d %H:%M")
    exit_dt_str = datetime.fromtimestamp(closed_at, tz=KST).strftime("%Y-%m-%d %H:%M")

    win_low = min(to_float(df_win["low"].min(), entry_p * 0.94), sl_p, entry_p)
    win_high = max(to_float(df_win["high"].max(), entry_p * 1.15), tp2_p, exit_p)
    if close_type == "TP3_TRAILING" or max_high_val >= tp3_p * 0.92:
        win_high = max(win_high, tp3_p)
    y_span = max(win_high - win_low, entry_p * 0.05)
    y_min_anchor = win_low - (y_span * 0.28)
    y_max_anchor = win_high + (y_span * 0.34)

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

    tail_bars = max(5, min(42, int(round(len(candles_js) * 0.08))))
    ws_tail = [{"time": candles_js[-1]["time"] + (i * step_sec)} for i in range(1, tail_bars + 1)]
    decimals = 6 if entry_p < 0.01 else (5 if entry_p < 1 else (4 if entry_p < 10 else (3 if entry_p < 1000 else 2)))
    min_move = 10 ** (-decimals)
    lw_script_tags = _get_lw_charts_script_tags()

    pos_badge = "LONG-TERM REPLAY" if is_long else f"{track_type.replace('_', ' ')} REPLAY"
    price_diff_abs = abs(exit_p - entry_p)
    price_diff_prefix = "+" if exit_p >= entry_p else "-"
    exit_headline = f"+{total_realized_pct:.2f}% ({price_diff_prefix}{fmt_price(price_diff_abs)}) · {hold_str}"
    hook_ko_card = str(close_info.get("x_hook_ko") or "").strip() or build_x_hook_line(close_info)

    in_fps = 10
    out_fps = 30
    base_frames = 305
    thumb_end = 20
    play_start = 30
    play_end = 259
    outro_static_start = 270

    js_errors = []
    try:
        from playwright.sync_api import sync_playwright
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) Playwright 실행 전 일본어 여성 아나운서(1.1배속) 4단계 음성 생성
            segments, final_total_sec, tts_ok, tts_err = _build_narration_plan(
                close_info, entry_p, total_realized_pct, created_at, tmpdir, base_total_sec=base_frames / float(in_fps)
            )
            voiced_frames = max(base_frames, int(round(final_total_sec * in_fps))) if tts_ok else base_frames
            sub_meta_js = [
                {
                    "step": seg["step"],
                    "tag": seg["tag"],
                    "l1": seg["ko_line1"],
                    "l2": seg["ko_line2"],
                    "s": seg["sub_start"],
                    "e": seg["sub_end"],
                }
                for seg in segments
            ]

            html_content = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
{lw_script_tags}
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#131722;font-family:'Noto Sans CJK KR','Noto Sans KR','Malgun Gothic',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#d1d4dc;width:960px;height:1280px;overflow:hidden}}
.tv-wrap{{width:960px;height:1280px;background:#131722;display:flex;flex-direction:column}}
.tv-topbar{{height:112px;background:#181c27;border-bottom:1px solid #2a2e39;display:flex;flex-direction:column;justify-content:center;gap:8px;padding:10px 20px}}
.tb-row{{display:flex;align-items:center;justify-content:space-between}}
.tv-left,.tv-right{{display:flex;align-items:center;gap:10px}}
.tv-symbol{{font-size:26px;font-weight:900;color:#fff;letter-spacing:0.2px}}
.tv-tf-badge{{background:#2a2e39;color:#e0e3eb;font-size:14px;font-weight:800;padding:4px 10px;border-radius:5px}}
.tv-replay-pill{{background:#2962ff;color:#fff;font-size:13px;font-weight:800;padding:5px 12px;border-radius:6px}}
.tv-handle-pill{{background:rgba(255,255,255,0.08);border:1px solid #3d4354;color:#f0f3fa;font-size:14px;font-weight:800;padding:5px 12px;border-radius:6px}}
.tv-ohlc{{display:flex;gap:12px;font-size:14px;font-weight:700;color:#9598a1}}
.tv-val-up{{color:#26a69a}}
.tv-val-dn{{color:#ef5350}}
#chart-box{{position:relative;width:960px;height:836px;overflow:hidden}}
.tv-ma-legend{{position:absolute;top:12px;left:14px;z-index:20;display:flex;gap:12px;font-size:13px;font-weight:700;background:rgba(19,23,34,0.88);padding:6px 12px;border-radius:6px;border:1px solid #2a2e39}}
.live-hud{{position:absolute;top:52px;right:94px;z-index:25;display:flex;flex-direction:column;align-items:flex-end;gap:8px}}
.pnl-pill{{background:rgba(19,23,34,0.92);border:2px solid #26a69a;border-radius:10px;padding:8px 16px;display:flex;flex-direction:column;align-items:flex-end;min-width:152px;box-shadow:0 8px 22px rgba(0,0,0,0.48)}}
.pnl-label{{font-size:11px;font-weight:800;color:#9598a1;letter-spacing:0.6px}}
.pnl-val{{font-size:28px;font-weight:900;color:#00e676;line-height:1.15}}
.stage-pill{{background:rgba(41,98,255,0.28);border:1.5px solid #4cc9f0;color:#e0f7fa;font-size:14px;font-weight:800;padding:8px 14px;border-radius:9px;display:none}}

#result-overlay{{position:absolute;inset:0;pointer-events:none;z-index:30;opacity:0}}
#arc-svg{{position:absolute;inset:0;width:960px;height:836px;overflow:visible}}
.tv-callout{{
  position:absolute;
  background:linear-gradient(135deg,#2962ff 0%,#1d4ed8 100%);
  color:#ffffff;
  border-radius:12px;
  box-shadow:0 12px 28px rgba(0,0,0,0.62), 0 0 0 1.5px rgba(255,255,255,0.20) inset;
  white-space:nowrap;
}}
.tv-callout-exit{{padding:11px 18px}}
.tv-callout-exit .co-main{{font-size:23px;font-weight:900;letter-spacing:-0.3px;line-height:1.2;color:#ffffff}}
.tv-callout-exit .co-sub{{font-size:16px;font-weight:800;color:#e8f0ff;margin-top:4px;display:flex;align-items:center;gap:12px}}
.tv-callout-entry{{padding:10px 16px}}
.tv-callout-entry .co-main{{font-size:20px;font-weight:900;line-height:1.2;color:#ffffff}}
.tv-callout-entry .co-sub{{font-size:15px;font-weight:700;color:#e0ecff;margin-top:3px}}
.co-caret-down{{position:absolute;bottom:-9px;width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-top:9px solid #1f52df}}
.co-caret-up{{position:absolute;top:-9px;width:0;height:0;border-left:9px solid transparent;border-right:9px solid transparent;border-bottom:9px solid #2962ff}}

#bottom-panel{{height:300px;width:960px;background:linear-gradient(180deg,#121725 0%,#0c101a 100%);border-top:2px solid #272e40;padding:18px 24px;display:flex;flex-direction:column;justify-content:center}}
#clean-card{{display:flex;flex-direction:column;gap:14px;height:100%;justify-content:center}}
.cc-hook{{background:rgba(41,98,255,0.16);border:1px solid rgba(76,201,240,0.35);border-radius:12px;padding:12px 18px;font-size:19px;font-weight:800;color:#e8f1ff;line-height:1.35;text-align:center}}
.cc-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.cc-box{{background:#181e2e;border:1px solid #2a3144;border-radius:12px;padding:14px 12px;display:flex;flex-direction:column;align-items:center;gap:6px}}
.cc-k{{font-size:13px;font-weight:700;color:#9598a1}}
.cc-v{{font-size:22px;font-weight:900;color:#ffffff}}
.cc-v-profit{{color:#00e676;font-size:25px}}

#sub-card{{display:none;flex-direction:column;justify-content:center;align-items:center;gap:16px;height:100%}}
.sub-head{{display:flex;align-items:center;gap:10px}}
.sub-badge{{background:#2962ff;color:#fff;font-size:14px;font-weight:800;padding:5px 14px;border-radius:999px;letter-spacing:0.3px}}
.sub-step{{background:rgba(255,255,255,0.08);border:1px solid #3a4258;color:#cbd5e1;font-size:13px;font-weight:800;padding:4px 11px;border-radius:999px}}
.sub-box{{width:100%;background:rgba(24,30,46,0.95);border:2px solid rgba(76,201,240,0.45);border-radius:16px;padding:22px 24px;text-align:center;box-shadow:0 10px 26px rgba(0,0,0,0.45);display:flex;flex-direction:column;gap:10px;justify-content:center;min-height:180px}}
.sub-l1{{font-size:30px;font-weight:900;color:#ffffff;line-height:1.38;word-break:keep-all;text-shadow:0 2px 6px rgba(0,0,0,0.55)}}
.sub-l2{{font-size:27px;font-weight:900;color:#ffd166;line-height:1.35;word-break:keep-all;text-shadow:0 2px 6px rgba(0,0,0,0.55)}}

.tv-footer{{height:32px;background:#0a0d14;border-top:1px solid #1f2430;display:flex;align-items:center;justify-content:space-between;padding:0 18px;font-size:11px;color:#787b86;font-weight:600}}
.prog-wrap{{width:150px;height:5px;background:#222632;border-radius:3px;overflow:hidden}}
.prog-fill{{height:100%;width:0%;background:linear-gradient(90deg,#2962ff,#00e676)}}
</style></head>
<body>
<div class="tv-wrap">
  <div class="tv-topbar">
    <div class="tb-row">
      <div class="tv-left">
        <span class="tv-symbol">${base}USDT.P</span>
        <span class="tv-tf-badge">{tf_label} · REPLAY</span>
      </div>
      <div class="tv-right">
        <span class="tv-replay-pill">{pos_badge}</span>
        <span class="tv-handle-pill">{WATERMARK_HANDLE}</span>
      </div>
    </div>
    <div class="tb-row">
      <div class="tv-ohlc">
        <span>O <span id="oh-o" class="tv-val-up">{fmt_price(entry_p)}</span></span>
        <span>H <span id="oh-h" class="tv-val-up">{fmt_price(entry_p)}</span></span>
        <span>L <span id="oh-l" class="tv-val-up">{fmt_price(entry_p)}</span></span>
        <span>C <span id="oh-c" class="tv-val-up">{fmt_price(entry_p)}</span></span>
      </div>
      <span id="top-time" style="font-size:13px;font-weight:700;color:#b2b5be">{kst_strs[exit_idx]}</span>
    </div>
  </div>

  <div id="chart-box">
    <div class="tv-ma-legend">
      <span style="color:#00e5ff">● MA(5) <span id="lg-m5">-</span></span>
      <span style="color:#ffd166">● MA(9) <span id="lg-m9">-</span></span>
      <span style="color:#2962ff">● MA(20) <span id="lg-m20">-</span></span>
    </div>
    <div class="live-hud" id="live-hud">
      <div class="pnl-pill" id="pnl-box">
        <span class="pnl-label">LIVE RETURN</span>
        <span class="pnl-val" id="pnl-val">+0.0%</span>
      </div>
      <div id="stage-pill" class="stage-pill"></div>
    </div>

    <div id="result-overlay">
      <svg id="arc-svg" viewBox="0 0 960 836">
        <defs>
          <filter id="arc-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="0" dy="2" stdDeviation="3.5" flood-color="#2962ff" flood-opacity="0.6"/>
          </filter>
        </defs>
        <line id="exit-guide" x1="0" y1="0" x2="0" y2="0" stroke="#60a5fa" stroke-width="2.2" stroke-dasharray="5,4" opacity="0"/>
        <line id="entry-guide" x1="0" y1="0" x2="0" y2="0" stroke="#60a5fa" stroke-width="2.2" stroke-dasharray="5,4" opacity="0"/>
        <path id="arc-path" d="" fill="none" stroke="#2962ff" stroke-width="4.4" stroke-linecap="round" filter="url(#arc-glow)"/>
        <circle id="dot-entry" cx="0" cy="0" r="6.5" fill="#2962ff" stroke="#ffffff" stroke-width="2.2"/>
        <circle id="dot-exit" cx="0" cy="0" r="6.5" fill="#2962ff" stroke="#ffffff" stroke-width="2.2"/>
        <polygon id="arc-head" points="0,0 -15,-7 -11,0 -15,7" fill="#3b82f6" stroke="#ffffff" stroke-width="0.8"/>
      </svg>

      <div id="callout-exit" class="tv-callout tv-callout-exit">
        <div class="co-main">{exit_headline}</div>
        <div class="co-sub"><span>EXIT ${fmt_price(exit_p)}</span><span>🕒 {exit_dt_str}</span></div>
        <div id="caret-exit" class="co-caret-down"></div>
      </div>

      <div id="callout-entry" class="tv-callout tv-callout-entry">
        <div id="caret-entry" class="co-caret-up"></div>
        <div class="co-main">ENTRY ${fmt_price(entry_p)}</div>
        <div class="co-sub">🕒 {entry_dt_str}</div>
      </div>
    </div>
  </div>

  <div id="bottom-panel">
    <div id="clean-card">
      <div class="cc-hook">📌 {hook_ko_card}</div>
      <div class="cc-grid">
        <div class="cc-box"><span class="cc-k">진입가격</span><span class="cc-v">${fmt_price(entry_p)}</span></div>
        <div class="cc-box"><span class="cc-k">최고 상승폭</span><span class="cc-v" style="color:#4cc9f0">+{max_fav_val:.1f}%</span></div>
        <div class="cc-box"><span class="cc-k">최종 수익률</span><span class="cc-v cc-v-profit">+{total_realized_pct:.2f}%</span></div>
        <div class="cc-box"><span class="cc-k">보유 기간</span><span class="cc-v">{hold_str}</span></div>
      </div>
    </div>

    <div id="sub-card">
      <div class="sub-head">
        <span class="sub-badge">🎙️ 단평 리플레이 브리핑 · JP VOICE</span>
        <span class="sub-step" id="sub-step">1 / 4</span>
      </div>
      <div class="sub-box">
        <div class="sub-l1" id="sub-l1"></div>
        <div class="sub-l2" id="sub-l2"></div>
      </div>
    </div>
  </div>

  <div class="tv-footer">
    <span>※ 본 영상은 스캐너 기준 시뮬레이션 리플레이이며 실제 체결 결과와 다를 수 있습니다.</span>
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
const subMeta={json.dumps(sub_meta_js, ensure_ascii=False)};
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
  width:960, height:836,
  layout:{{background:{{type:'solid',color:'#131722'}},textColor:'#d1d4dc',fontSize:12}},
  grid:{{vertLines:{{color:'#1e222d'}},horzLines:{{color:'#1e222d'}}}},
  watermark:{{
    visible:true,
    fontSize:66,
    horzAlign:'center',
    vertAlign:'center',
    color:'rgba(209, 212, 220, 0.14)',
    text:'{WATERMARK_HANDLE}',
    fontFamily:'-apple-system,BlinkMacSystemFont,Roboto,sans-serif',
    fontStyle:'bold'
  }},
  rightPriceScale:{{borderColor:'#2a2e39',scaleMargins:{{top:0.04,bottom:0.09}},entireTextOnly:true}},
  timeScale:{{borderColor:'#2a2e39',timeVisible:true,rightOffset:2}},
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
chart.priceScale('vol_scale').applyOptions({{scaleMargins:{{top:0.90,bottom:0.0}}}});

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
const elExitGuide=document.getElementById('exit-guide'), elEntryGuide=document.getElementById('entry-guide');
const elDotEntry=document.getElementById('dot-entry'), elDotExit=document.getElementById('dot-exit');
const elCoExit=document.getElementById('callout-exit'), elCoEntry=document.getElementById('callout-entry');
const elCaretExit=document.getElementById('caret-exit'), elCaretEntry=document.getElementById('caret-entry');
const elCleanCard=document.getElementById('clean-card'), elSubCard=document.getElementById('sub-card');
const elSubStep=document.getElementById('sub-step'), elSubL1=document.getElementById('sub-l1'), elSubL2=document.getElementById('sub-l2');

let arcTotalLen=1;
let geometryReady=false;
let showExitGuide=false;
let showEntryGuide=false;

function setupResultOverlayGeometry(){{
  const psW=(chart.priceScale('right') && chart.priceScale('right').width) ? chart.priceScale('right').width() : 86;
  const plotRight=Math.max(360, 960 - (psW || 86) - 8);
  const totalSlots=cAll.length + wsTail.length;

  let x1=chart.timeScale().timeToCoordinate(cAll[entryIdx].time);
  let x2=chart.timeScale().timeToCoordinate(cAll[exitIdx].time);
  if(x1===null || !isFinite(x1)) x1=32 + (entryIdx / Math.max(1, totalSlots)) * (plotRight - 64);
  if(x2===null || !isFinite(x2)) x2=32 + (exitIdx / Math.max(1, totalSlots)) * (plotRight - 64);

  let y1=candleSeries.priceToCoordinate(entryP);
  let y1Low=candleSeries.priceToCoordinate(cAll[entryIdx].low);
  let y2=candleSeries.priceToCoordinate(exitP);
  let y2High=candleSeries.priceToCoordinate(cAll[exitIdx].high);
  if(y1===null || !isFinite(y1)) y1=560;
  if(y1Low===null || !isFinite(y1Low)) y1Low=y1;
  if(y2===null || !isFinite(y2)) y2=260;
  if(y2High===null || !isFinite(y2High)) y2High=y2;

  y1=Math.max(110, Math.min(700, y1));
  y1Low=Math.max(y1, Math.min(710, y1Low));
  y2=Math.max(120, Math.min(640, y2));
  y2High=Math.max(95, Math.min(y2, y2High));

  let peakY=y2High;
  for(let i=entryIdx;i<=exitIdx;i++){{
    const hy=candleSeries.priceToCoordinate(cAll[i].high);
    if(hy!==null && isFinite(hy) && hy < peakY) peakY=hy;
  }}

  const dx=x2 - x1;
  const dy=y1 - y2;
  let cx, cy;
  if(dx >= 40){{
    cx = Math.min(plotRight - 14, x1 + dx * 0.90 + 14);
    cy = Math.min(740, y1 - dy * 0.04 + 14);
  }} else {{
    cx = Math.min(plotRight - 14, Math.max(x1, x2) + 64);
    cy = Math.min(740, y1 - dy * 0.25 + 16);
  }}

  const dStr='M ' + x1.toFixed(1) + ' ' + y1.toFixed(1) + ' Q ' + cx.toFixed(1) + ' ' + cy.toFixed(1) + ' ' + x2.toFixed(1) + ' ' + y2.toFixed(1);
  elArcPath.setAttribute('d', dStr);
  elDotEntry.setAttribute('cx', x1.toFixed(1));
  elDotEntry.setAttribute('cy', y1.toFixed(1));
  elDotExit.setAttribute('cx', x2.toFixed(1));
  elDotExit.setAttribute('cy', y2.toFixed(1));

  arcTotalLen=Math.max(1, elArcPath.getTotalLength());
  elArcPath.style.strokeDasharray=arcTotalLen.toFixed(1);

  const exitW=elCoExit.offsetWidth || 360;
  const exitH=elCoExit.offsetHeight || 74;
  let exitLeft=Math.max(14, Math.min(plotRight - exitW - 10, x2 - exitW * 0.52));
  let exitTop=Math.min(y2, y2High, peakY + 10) - exitH - 46;
  if(exitTop < 48) exitTop = 48;

  elCoExit.style.left=exitLeft.toFixed(1) + 'px';
  elCoExit.style.top=exitTop.toFixed(1) + 'px';
  const exitCaretX=Math.max(16, Math.min(exitW - 30, x2 - exitLeft - 9));
  elCaretExit.style.left=exitCaretX.toFixed(1) + 'px';

  const exitTipX = exitLeft + exitCaretX + 9;
  const exitTipY = exitTop + exitH + 9;
  if((y2 - exitTipY) > 14){{
    elExitGuide.setAttribute('x1', exitTipX.toFixed(1));
    elExitGuide.setAttribute('y1', exitTipY.toFixed(1));
    elExitGuide.setAttribute('x2', x2.toFixed(1));
    elExitGuide.setAttribute('y2', (y2 - 7).toFixed(1));
    showExitGuide = true;
  }} else {{
    showExitGuide = false;
  }}

  const entryW=elCoEntry.offsetWidth || 240;
  const entryH=elCoEntry.offsetHeight || 66;
  let entryLeft=Math.max(14, Math.min(plotRight - entryW - 10, x1 - entryW * 0.35));
  let entryTop = y1Low + 46;
  if(entryTop + entryH > 798) entryTop = 798 - entryH;
  if(entryTop < exitTop + exitH + 16){{
    entryTop = Math.min(798 - entryH, exitTop + exitH + 16);
  }}
  elCoEntry.style.left=entryLeft.toFixed(1) + 'px';
  elCoEntry.style.top=entryTop.toFixed(1) + 'px';
  const entryCaretX=Math.max(16, Math.min(entryW - 30, x1 - entryLeft - 9));
  elCaretEntry.style.left=entryCaretX.toFixed(1) + 'px';

  const entryTipX = entryLeft + entryCaretX + 9;
  const entryTipY = entryTop - 9;
  if((entryTipY - y1) > 16){{
    elEntryGuide.setAttribute('x1', x1.toFixed(1));
    elEntryGuide.setAttribute('y1', (y1 + 7).toFixed(1));
    elEntryGuide.setAttribute('x2', entryTipX.toFixed(1));
    elEntryGuide.setAttribute('y2', entryTipY.toFixed(1));
    showEntryGuide = true;
  }} else {{
    showEntryGuide = false;
  }}

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
  elEntryGuide.setAttribute('opacity', (showEntryGuide && boxAlpha > 0.5) ? '0.80' : '0');
  elExitGuide.setAttribute('opacity', (showExitGuide && exitReady > 0.5) ? '0.80' : '0');
}}

window.__setBottomMode=function(mode, subIdx){{
  if(mode === 'sub' && subMeta.length > 0){{
    elCleanCard.style.display='none';
    elSubCard.style.display='flex';
    const seg=subMeta[Math.max(0, Math.min(subMeta.length - 1, subIdx || 0))];
    elSubStep.textContent=seg.step;
    elSubL1.textContent=seg.l1;
    elSubL2.textContent=seg.l2;
    elSubL2.style.color=(seg.tag === 'PROFIT') ? '#00e676' : '#ffd166';
  }} else {{
    elSubCard.style.display='none';
    elCleanCard.style.display='flex';
  }}
}};

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
      elStage.style.background='rgba(0,230,118,0.18)';
      elStage.textContent='🟢 ENTRY 포착 완료 · 리플레이 시작';
    }} else if(hitTp3){{
      elStage.style.display='block';
      elStage.style.borderColor='#ffd166';
      elStage.style.background='rgba(255,209,102,0.20)';
      elStage.textContent='🚀 TP3 50% 익절 · 트레일링 가동';
    }} else if(hitTp2){{
      elStage.style.display='block';
      elStage.style.borderColor='#06d6a0';
      elStage.style.background='rgba(6,214,160,0.20)';
      elStage.textContent='🛡️ TP2 안착 · TP1 방어선 고정';
    }} else if(hitTp1){{
      elStage.style.display='block';
      elStage.style.borderColor='#4cc9f0';
      elStage.style.background='rgba(76,201,240,0.20)';
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
  window.__setBottomMode('clean', 0);
  window.__renderFrame(0);
  window.__REPLAY_READY=true;
}}));
</script></body></html>"""

            html_path = os.path.join(tmpdir, "replay_vertical.html")
            out_clean_mp4 = os.path.join(tmpdir, "replay_clean.mp4")
            out_voiced_mp4 = os.path.join(tmpdir, "replay_voiced.mp4")
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

            total_max_frames = max(base_frames, voiced_frames)
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
                )
                try:
                    page = browser.new_page(viewport={"width": VIDEO_W, "height": VIDEO_H}, device_scale_factor=1)
                    page.on("pageerror", lambda err: js_errors.append(str(err)))
                    page.goto(Path(html_path).resolve().as_uri(), wait_until="load", timeout=25000)
                    page.wait_for_function("window.__REPLAY_READY === true", timeout=15000)
                    page.wait_for_timeout(120)

                    prev_sub_idx = -1
                    for fi in range(total_max_frames):
                        chart_is_static = (
                            (1 <= fi < thumb_end)
                            or ((thumb_end + 1) <= fi < play_start)
                            or (fi > outro_static_start)
                        )
                        cur_sub_idx = _sub_idx_for_frame(fi) if tts_ok else 0

                        clean_jpg = os.path.join(tmpdir, f"c_{fi:04d}.jpg")
                        sub_jpg = os.path.join(tmpdir, f"s_{fi:04d}.jpg")

                        need_chart_render = not chart_is_static
                        need_clean_shot = (fi < base_frames) and need_chart_render
                        need_sub_shot = tts_ok and (fi < voiced_frames) and (need_chart_render or cur_sub_idx != prev_sub_idx)

                        if need_chart_render:
                            page.evaluate(f"window.__renderFrame({fi})")

                        if fi < base_frames:
                            if need_clean_shot or fi == 0:
                                page.evaluate("window.__setBottomMode('clean', 0)")
                                page.screenshot(path=clean_jpg, type="jpeg", quality=88)
                            else:
                                ref_fi = 0 if fi < thumb_end else (thumb_end if fi < play_start else outro_static_start)
                                shutil.copyfile(os.path.join(tmpdir, f"c_{ref_fi:04d}.jpg"), clean_jpg)

                        if tts_ok and fi < voiced_frames:
                            if need_sub_shot or fi == 0:
                                page.evaluate(f"window.__setBottomMode('sub', {cur_sub_idx})")
                                page.screenshot(path=sub_jpg, type="jpeg", quality=88)
                            else:
                                shutil.copyfile(os.path.join(tmpdir, f"s_{fi - 1:04d}.jpg"), sub_jpg)
                            prev_sub_idx = cur_sub_idx
                finally:
                    browser.close()

            # 2) 영상 1 인코딩 (클린 세로형 · 무음)
            cmd_clean = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(in_fps),
                "-i", os.path.join(tmpdir, "c_%04d.jpg"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "21",
                "-r", str(out_fps),
                "-movflags", "+faststart",
                out_clean_mp4,
            ]
            res_c = subprocess.run(cmd_clean, capture_output=True, text=True, timeout=90)
            clean_bytes = None
            if res_c.returncode == 0 and os.path.exists(out_clean_mp4):
                with open(out_clean_mp4, "rb") as vf:
                    clean_bytes = vf.read()
            else:
                LAST_VIDEO_ERROR = f"클린 영상 ffmpeg 오류: {(res_c.stderr or '')[:180]}"
                print(f"[리플레이 클린 영상 실패] {LAST_VIDEO_ERROR}")

            # 3) 영상 2 인코딩 (한글 자막 + 일본어 여성 아나운서 1.1배속 음성)
            voiced_bytes = None
            voiced_error = tts_err if not tts_ok else ""
            if tts_ok and segments:
                audio_inputs = []
                fc_parts = []
                concat_labels = []
                cursor = 0.0
                for idx, seg in enumerate(segments):
                    audio_inputs.extend(["-i", seg["audio_path"]])
                    in_idx = idx + 1  # 0번 입력은 이미지 시퀀스
                    gap = max(0.05, seg["audio_start"] - cursor)
                    fc_parts.append(f"aevalsrc=0:s=44100:c=stereo:d={gap:.3f}[g{idx}]")
                    fc_parts.append(f"[{in_idx}:a]aresample=44100,aformat=channel_layouts=stereo,volume=1.18[s{idx}]")
                    concat_labels.extend([f"[g{idx}]", f"[s{idx}]"])
                    cursor = seg["audio_start"] + seg["audio_dur"]

                trail_gap = max(0.20, final_total_sec - cursor)
                fc_parts.append(f"aevalsrc=0:s=44100:c=stereo:d={trail_gap:.3f}[gend]")
                concat_labels.append("[gend]")
                fc_parts.append(f"{''.join(concat_labels)}concat=n={len(concat_labels)}:v=0:a=1[aout]")
                filter_complex = ";".join(fc_parts)

                cmd_voiced = [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", str(in_fps),
                    "-i", os.path.join(tmpdir, "s_%04d.jpg"),
                    *audio_inputs,
                    "-filter_complex", filter_complex,
                    "-map", "0:v:0",
                    "-map", "[aout]",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "fast",
                    "-crf", "21",
                    "-r", str(out_fps),
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-ar", "44100",
                    "-shortest",
                    "-movflags", "+faststart",
                    out_voiced_mp4,
                ]
                res_v = subprocess.run(cmd_voiced, capture_output=True, text=True, timeout=110)
                if res_v.returncode == 0 and os.path.exists(out_voiced_mp4):
                    with open(out_voiced_mp4, "rb") as vf:
                        voiced_bytes = vf.read()
                else:
                    voiced_error = f"음성 합성 ffmpeg 오류: {(res_v.stderr or '')[:180]}"
                    print(f"[리플레이 음성 영상 실패] {voiced_error}")

            print(
                f"[세로형 리플레이 완료] {symbol} ({tf_label}, {len(df_win)}봉 · 960x1280) | "
                f"클린={'OK' if clean_bytes else 'FAIL'} ({base_frames / in_fps:.1f}초) · "
                f"자막/JP음성={'OK' if voiced_bytes else 'FAIL'} ({voiced_frames / in_fps:.1f}초)"
            )
            return {
                "clean": clean_bytes,
                "voiced": voiced_bytes,
                "width": VIDEO_W,
                "height": VIDEO_H,
                "clean_duration": round(base_frames / float(in_fps), 1),
                "voiced_duration": round(voiced_frames / float(in_fps), 1),
                "voiced_error": voiced_error,
            }
    except Exception as e:
        err_extra = f" | JS: {js_errors[0][:120]}" if js_errors else ""
        LAST_VIDEO_ERROR = f"렌더링 예외: {str(e)[:150]}{err_extra}"
        print(f"[세로형 리플레이 예외] {symbol}: {LAST_VIDEO_ERROR}")
        return None


def generate_profit_replay_video_bytes(close_info):
    """단일 영상 호환 래퍼 (구버전 호출 대비)"""
    bundle = generate_profit_replay_videos_bundle(close_info)
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
    token = (TELEGRAM_BOT_TOKEN or os.getenv("TELEGRAM_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (TELEGRAM_CHAT_ID or os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()
    if not photo_bytes or not token or not chat_id:
        return False
    url = clean_url(f"{SCHEME}api.telegram.org/bot{token}/sendPhoto")
    for attempt in range(3):
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
            if resp.status_code == 429:
                wait_s = int(to_float((resp.json().get("parameters") or {}).get("retry_after"), 3))
                time.sleep(wait_s + 0.5)
                continue
            print(f"[sendPhoto 오류 {attempt + 1}/3] HTTP {resp.status_code}: {resp.text[:160]}")
        except Exception as e:
            print(f"[sendPhoto 예외 {attempt + 1}/3] {e}")
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
    """세로형(960x1280) HD MP4 리플레이 영상을 텔레그램 sendVideo로 전송"""
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
            resp = requests.post(url, data=data, files=files, timeout=60)
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
