import sys
import time
import numpy as np
from datetime import datetime, timedelta, timezone

try:
    import config as cfg
    HISTORY_FILE = getattr(cfg, "HISTORY_FILE", "analyzer_history.json")
    KST = getattr(cfg, "KST", timezone(timedelta(hours=9)))
    FIXED_SL_PCT = int(getattr(cfg, "FIXED_SL_PCT", 6))
except Exception:
    cfg = None
    HISTORY_FILE = "analyzer_history.json"
    KST = timezone(timedelta(hours=9))
    FIXED_SL_PCT = 6

from wave_engine import (
    to_float,
    fmt_price,
    send_telegram_chunked,
)
from postmortem import (
    load_json_file,
    save_json_file,
    fetch_top_gainer_candidates,
    reverse_engineer_top_coin,
    analyze_scanner_coins_and_postmortem,
    calibrate_3wave_tp_system,
    normalize_history_data,
    evaluate_48h_history,
)


def main():
    now_ts = int(time.time())
    now_dt = datetime.now(KST)
    now_str = now_dt.strftime("%Y-%m-%d %H:%M KST")

    candidates = fetch_top_gainer_candidates()
    if not candidates:
        print("[치명적 오류] Top 5 후보군 조회 실패")
        sys.exit(1)

    candidates.sort(
        key=lambda x: to_float(x.get("change_24h"), to_float(x.get("price24hPcnt")) * 100.0),
        reverse=True,
    )

    analyzed_top5 = []
    for c in candidates[:25]:
        res = reverse_engineer_top_coin(c, now_ts)
        if res:
            analyzed_top5.append(res)
        if len(analyzed_top5) >= 5:
            break

    if not analyzed_top5:
        print("[치명적 오류] 상위 종목 캔들 분석 결과 없음")
        sys.exit(1)

    num_icons = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    msg_lines = [
        "📊 [Bybit Futures 급등 Top 5 초동 역추적 & 1·2·3차 파동 로데이터 학습]",
        f"🕒 기준 시각: {now_str} (순수 코인 상위 5개)\n",
    ]

    for i, item in enumerate(analyzed_top5):
        icon = num_icons[i] if i < len(num_icons) else f"{i+1}."
        alpha_str = " / ".join(item["alphas"]) if item["alphas"] else "표준 수급 돌파"
        w1 = item["waves"]["w1"]
        w2 = item["waves"]["w2"]
        w3 = item["waves"]["w3"]

        msg_lines.extend(
            [
                f"{icon} ${item['base']} ({item['symbol']} | +{item['change_24h']:.2f}% | 현재가 {fmt_price(item['price'])} USDT)",
                f"• 거래대금: {item['turnover_m']:.1f}M | 미결제약정(OI): {item['oi_m']:.1f}M (비중 {item['oi_ratio']:.0f}%) | 펀딩비: {item['funding_pct']:+.4f}%",
                f"• [초동 발화점]: {item['ign_time_str']} (발화가 {fmt_price(item['ign_price'])} → 최종고점까지 +{item['pump_from_ign_pct']:.1f}% 폭등 | {item['hours_to_peak']}h 소요)",
                "• [🌊 1·2·3차 파동별 조정·고점 형성 로데이터(Raw Data) 분석]:",
                f"  - 1차(첫 조정 고점): +{w1['gain_pct']:.1f}% ({fmt_price(w1['price'])}) | Fib {w1['fib_mult']:.2f}배 | MA20이격 +{w1['ma20_ext_pct']:.1f}% | RSI6 {w1['rsi6']:.0f}·J {w1['kdj_j']:.0f} → -{w1['pullback_pct']:.1f}% 눌림({w1['hold_tag']} 지지)",
                f"  - 2차(중간 조정 고점): +{w2['gain_pct']:.1f}% ({fmt_price(w2['price'])}) | Fib {w2['fib_mult']:.2f}배 | MA20이격 +{w2['ma20_ext_pct']:.1f}% | RSI6 {w2['rsi6']:.0f}·J {w2['kdj_j']:.0f} → -{w2['pullback_pct']:.1f}% 숨고르기({w2['hold_tag']} 지지)",
                f"  - 3차(최종 클라이맥스): +{w3['gain_pct']:.1f}% ({fmt_price(w3['price'])}) | Fib {w3['fib_mult']:.2f}배 | MA20이격 +{w3['ma20_ext_pct']:.1f}% | RSI6 {w3['rsi6']:.0f}·J {w3['kdj_j']:.0f} | 거래량 {w3['vol_mult']:.1f}배",
                "• [발화 직전 컨펌된 초동 전략]:",
                f"  - 5분봉 이평선: {item['ma_str']} | 직전 2h 박스 {item['pre_box_pct']:.1f}% 응축 | OBV/CVD {item['obv_str']}",
                f"  - 초동 수급: 평소 대비 [{item['ign_vol_mult']:.1f}배 폭발] (발화전 포켓 {int(round(item['pre5_min_vol_ratio']*100))}%) | 돌파 후: {item['post_pattern_str']}",
                "• [🔍 포착된 전문가급 신규 알파 전략]:",
                f"  - {alpha_str}\n",
            ]
        )

    sec_postmortem, prescriptions, scanner_wave_stats = analyze_scanner_coins_and_postmortem(now_ts)
    tp_tuning = calibrate_3wave_tp_system(analyzed_top5, scanner_wave_stats)

    raw_history = load_json_file(HISTORY_FILE, {"cohorts": []})
    history_list = normalize_history_data(raw_history)
    sec_48h = evaluate_48h_history(history_list, now_ts, tp_tuning)
    msg_lines.append(sec_48h)
    msg_lines.append(sec_postmortem)

    today_key = now_dt.strftime("%Y-%m-%d")
    history_list = [h for h in history_list if not str(h.get("date_str", "")).startswith(today_key)]
    history_list.append(
        {
            "date_str": today_key,
            "timestamp": now_ts,
            "coins": [
                {
                    "symbol": x["symbol"],
                    "ign_price": x["ign_price"],
                    "support_line": x["support_line"],
                    "swing_high_line": x["swing_high_line"],
                }
                for x in analyzed_top5
            ],
        }
    )
    history_list = history_list[-10:]
    save_json_file(HISTORY_FILE, {
        "cohorts": history_list,
        "dynamic_tp_tuning": tp_tuning,
        "updated_at": now_str,
    })

    n = len(analyzed_top5)
    c_short = sum(1 for x in analyzed_top5 if x["short_aligned"])
    c_full = sum(1 for x in analyzed_top5 if x["full_aligned"])
    c_obv = sum(1 for x in analyzed_top5 if x["obv_up"])
    c_cvd = sum(1 for x in analyzed_top5 if x["cvd_up"])
    c_spring = sum(1 for x in analyzed_top5 if x["cond_spring"])
    c_htf = sum(1 for x in analyzed_top5 if x["htf_aligned"])
    c_pocket = sum(1 for x in analyzed_top5 if x["is_pocket_pivot"])
    c_ribbon = sum(1 for x in analyzed_top5 if x["pre5_min_ribbon"] <= 2.2)
    c_bull_flag = sum(1 for x in analyzed_top5 if x["is_bull_flag"])
    c_fvg50 = sum(1 for x in analyzed_top5 if x["is_fvg_50_hold"])
    c_sr = sum(1 for x in analyzed_top5 if x["is_sr_flip"])
    c_sqz = sum(1 for x in analyzed_top5 if x["ttm_sqz"])
    c_fund = sum(1 for x in analyzed_top5 if x["funding_pct"] <= 0.015)

    rec_vol_min = round(max(2.6, min(4.0, float(np.percentile([x["ign_vol_mult"] for x in analyzed_top5], 15)) * 0.85)), 1)
    rec_box_max = round(max(5.5, min(9.5, float(np.median([x["pre_box_pct"] for x in analyzed_top5])) * 1.6)), 1)
    rec_dryup = round(max(0.18, min(0.35, float(np.median([x["dryup_ratio"] for x in analyzed_top5])) + 0.05)), 2)
    avg_hours = round(float(np.mean([x["hours_to_peak"] for x in analyzed_top5])), 1)

    tp1_rec = int(tp_tuning["tp1_min_pct"])
    tp2_rec = int(tp_tuning["tp2_min_pct"])
    tp3_rec = int(tp_tuning["tp3_min_pct"])
    fib1_rec = tp_tuning["tp1_fib_mult"]
    fib2_rec = tp_tuning["tp2_fib_mult"]
    fib3_rec = tp_tuning["tp3_fib_mult"]
    ext1_rec = tp_tuning["tp1_ma20_ext_pct"]
    ext2_rec = tp_tuning["tp2_ma20_ext_pct"]
    ext3_rec = tp_tuning["tp3_ma20_ext_pct"]

    msg_lines.extend(
        [
            "━━━━━━━━━━━━━━━━━━",
            "💡 [내 전략(scanner.py / analyzer.py) 실시간 업데이트 피드백 (제미나이 복붙용)]\n",
            "[1. 기존 vs 전문가 신규 알파 전략 적중률 비교]",
            f"• 기존 5분봉 단기정배열(5>20>50): {n}종목 중 {c_short}종목 (완전정배열 {c_full}종목)",
            f"• 기존 박스권 OBV 우상향 매집: {n}종목 중 {c_obv}종목 | CVD 흡수 다이버전스: {c_cvd}종목",
            f"• [전문가] High-Tight Bull Flag 공중눌림(상단 38.2% 사수): {n}종목 중 {c_bull_flag}종목",
            f"• [전문가] 포켓 피벗(Pocket Pivot 초동 양봉 직격): {n}종목 중 {c_pocket}종목",
            f"• [전문가] 장대양봉 50% 중심선(FVG/Equilibrium) 칼지지: {n}종목 중 {c_fvg50}종목",
            f"• [전문가] 전고점 S/R Flip 칼지지: {n}종목 중 {c_sr}종목 | 유동성 스윕(Spring): {c_spring}종목",
            f"• [전문가] 이평선 리본 초밀집(<=2.2%): {n}종목 중 {c_ribbon}종목 | TTM 켈트너 스퀴즈: {c_sqz}종목",
            f"• [전문가] 상위프레임(1H 중심선) 구조적 지지: {n}종목 중 {c_htf}종목 | 청정/음수 펀딩: {c_fund}종목\n",
            "[2. analyzer.py & config.py 파라미터 튜닝 권장값 (자동 동기화 완료)]",
            f"• EARLY_VOL_MULT_MIN = {rec_vol_min} (초동 최소 거래량 배수)",
            f"• PRE_BOX_RANGE_MAX_PCT = {rec_box_max}% (직전 2시간 변동폭 상한)",
            f"• PULLBACK_DRYUP_MAX = {rec_dryup} (발화봉 대비 눌림 거래량 상한)",
            f"• TP1_MIN_PCT = {tp1_rec}% (Fib {fib1_rec}배 / MA20이격 +{ext1_rec}%)",
            f"• TP2_MIN_PCT = {tp2_rec}% (Fib {fib2_rec}배 / MA20이격 +{ext2_rec}%)",
            f"• TP3_MIN_PCT = {tp3_rec}% (Fib {fib3_rec}배 / MA20이격 +{ext3_rec}%)\n",
            "[3. analyzer.py 추가/보완할 전문가 파이썬 조건식]",
            "• cond_bull_flag = (close > prev_box_high) and (pullback_low >= ignition_high - (ignition_high - prev_box_high) * 0.382) and (close >= ma5 * 0.998)",
            "• cond_pocket_pivot = is_current_ignition and (close > open_p) and (pre_vol_pocket_ratio <= 0.40) and (curr_vol_mult >= EARLY_VOL_MULT_MIN)",
            "• cond_fvg_mid_hold = (pullback_low >= (ignition_open + ignition_high) * 0.5 * 0.998) and (pullback_vol_ratio <= 0.35)\n",
            "[4. 오답 노트(손절 역추적) 기반 방어 필터 업데이트 요약]",
        ]
    )

    if prescriptions:
        msg_lines.extend(prescriptions)
    else:
        msg_lines.append("• 최근 48h 손절 특이사항 없음 (현행 방어 필터 유지 권장)")

    msg_lines.extend(
        [
            "\n[5. Top 5 대장주 + 내 스캐너 코인(오답노트) 로데이터 결합 1·2·3차 익절(TP) 학습 결과]",
            f"• 실전 고정 손절폭(SL): -{FIXED_SL_PCT}% (급등 전 휩소 방어 & 1R +{FIXED_SL_PCT}% 본전방어 연동)",
            f"• 파동별 고점 형성 위치 심층 분석 (Top5 5건 + 스캐너 포착 {tp_tuning['pm_sample_cnt']}건 결합 | 최종고점 평균 {avg_hours}h 소요):",
            f"  - 1차 TP (첫 조정 형성 고점): +{tp1_rec}% (Top5 첫조정 +{tp_tuning['top5_w1_med']}%·스캐너코인 첫조정 +{tp_tuning['pm_w1_med']}% 결합 | Fib {fib1_rec}배 | MA20이격 +{ext1_rec}% | RSI6 {tp_tuning['tp1_rsi6']:.0f})",
            f"  - 2차 TP (최종고점 전 중간조정 고점): +{tp2_rec}% (Top5 2차고점 +{tp_tuning['top5_w2_med']}%·스캐너코인 2차고점 +{tp_tuning['pm_w2_med']}% 결합 | Fib {fib2_rec}배 | MA20이격 +{ext2_rec}% | RSI6 {tp_tuning['tp2_rsi6']:.0f})",
            f"  - 3차 TP (최종 클라이맥스 고점): +{tp3_rec}% (Top5 최종고점 +{tp_tuning['top5_w3_med']}%·스캐너코인 최종고점 +{tp_tuning['pm_w3_med']}% 결합 | Fib {fib3_rec}배 | MA20이격 +{ext3_rec}% | RSI6 {tp_tuning['tp3_rsi6']:.0f})",
        ]
    )

    full_report = "\n".join(msg_lines)
    send_telegram_chunked(full_report)
    print(f"[{now_str}] 전문가급 Top 5 + 스캐너 코인 3파동 로데이터 심층 분석 브리핑 전송 완료")


if __name__ == "__main__":
    main()
