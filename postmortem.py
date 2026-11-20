import os, json, time, threading
from collections import Counter

POSTMORTEM_DB_FILE = "postmortem_db.json"
RETENTION_SEC = 14 * 86400
_DB_LOCK = threading.Lock()


def _to_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v
    except Exception:
        return default


def _load_db_unlocked():
    if os.path.exists(POSTMORTEM_DB_FILE):
        try:
            with open(POSTMORTEM_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("records"), list):
                return data
        except Exception:
            pass
    return {"records": [], "updated_at": 0}


def _save_db_unlocked(data):
    tmp = f"{POSTMORTEM_DB_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, POSTMORTEM_DB_FILE)
    except Exception:
        pass


def classify_sl_cause(sl_item):
    """손절 시점의 지표와 반등폭으로 주요 손절 원인 진단"""
    rsi_v = _to_float(sl_item.get("rsi6_15m"), 50.0)
    vol_m = _to_float(sl_item.get("vol_mult"), 1.0)
    max_f = _to_float(sl_item.get("max_favorable_pct"), 0.0)
    hold_m = int(_to_float(sl_item.get("hold_minutes"), 0))

    if rsi_v >= 70.0:
        return "단기 고점 RSI 과열 추격"
    if vol_m < 1.8 and max_f < 2.0:
        return "돌파 거래량 부족(가짜 돌파)"
    if hold_m <= 45 and max_f < 1.5:
        return "진입 직후 급락(스탑헌팅/비트 연동)"
    if max_f >= 4.0:
        return f"+{max_f:.1f}% 반등 후 차익 매물 출회"
    return "비트 연동 급락 및 지지선 이탈"


def record_sl_postmortem(sl_payload):
    """손절 기록을 로컬 오답노트 DB(postmortem_db.json)에 스레드 안전하게 누적 저장"""
    if not isinstance(sl_payload, dict):
        return False

    now_ts = int(time.time())
    sym = str(sl_payload.get("symbol", "")).upper().strip()
    c_at = int(_to_float(sl_payload.get("created_at"), now_ts))
    sl_at = int(_to_float(sl_payload.get("sl_at"), now_ts))
    if not sym:
        return False

    rec_id = f"{sym}_{c_at}"
    record = {
        "id": rec_id,
        "symbol": sym,
        "track_type": str(sl_payload.get("track_type", "TRACK_A")),
        "score": int(_to_float(sl_payload.get("score"), 90)),
        "entry_price": _to_float(sl_payload.get("entry_price")),
        "stop_price": _to_float(sl_payload.get("stop_price")),
        "sl_trigger_price": _to_float(sl_payload.get("sl_trigger_price")),
        "sl_pct": int(round(_to_float(sl_payload.get("sl_pct"), 6))),
        "created_at": c_at,
        "sl_at": sl_at,
        "hold_minutes": int(_to_float(sl_payload.get("hold_minutes"), 0)),
        "max_high_price": _to_float(sl_payload.get("max_high_price")),
        "max_favorable_pct": _to_float(sl_payload.get("max_favorable_pct")),
        "vol_mult": _to_float(sl_payload.get("vol_mult"), 1.0),
        "rsi6_15m": _to_float(sl_payload.get("rsi6_15m"), 50.0),
        "oi_m": _to_float(sl_payload.get("oi_m"), 0.0),
        "funding_pct": _to_float(sl_payload.get("funding_pct"), 0.0),
        "cause": classify_sl_cause(sl_payload),
    }

    with _DB_LOCK:
        db = _load_db_unlocked()
        recs = [
            r for r in db.get("records", [])
            if isinstance(r, dict)
            and r.get("id") != rec_id
            and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= RETENTION_SEC
        ]
        recs.insert(0, record)
        db["records"] = recs[:200]
        db["updated_at"] = now_ts
        _save_db_unlocked(db)
    return True


def get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult):
    """
    단기 반복 손절 종목 또는 최근 손절 빈발 패턴(과열 추격/저거래량)에 0~15점 자동 감점 반환
    """
    sym = str(symbol or "").upper().strip()
    tk = str(track_type or "TRACK_A").upper().strip()
    rsi_v = _to_float(rsi6_15m, 50.0)
    vol_v = _to_float(vol_mult, 1.0)
    now_ts = int(time.time())

    with _DB_LOCK:
        db = _load_db_unlocked()
    recs = db.get("records", [])
    if not recs:
        return 0

    penalty = 0

    # 1) 최근 48시간 내 동일 종목 손절 이력 감점 (1회당 7점, 최대 12점)
    recent_sym_sls = [
        r for r in recs
        if isinstance(r, dict)
        and str(r.get("symbol", "")).upper() == sym
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 172800
    ]
    if recent_sym_sls:
        penalty += min(12, len(recent_sym_sls) * 7)

    # 2) 최근 7일 내 동일 트랙의 과열/저거래량 손절 패턴 학습 감점
    recent_tk_sls = [
        r for r in recs
        if isinstance(r, dict)
        and str(r.get("track_type", "")).upper() == tk
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 604800
    ]
    if len(recent_tk_sls) >= 2:
        overheat_cnt = sum(1 for r in recent_tk_sls if _to_float(r.get("rsi6_15m"), 50.0) >= 68.0)
        lowvol_cnt = sum(1 for r in recent_tk_sls if _to_float(r.get("vol_mult"), 1.0) < 1.8)
        if rsi_v >= 68.0 and overheat_cnt >= 2:
            penalty += 5
        if vol_v < 1.8 and lowvol_cnt >= 2:
            penalty += 4

    return int(min(15, max(0, penalty)))


def build_daily_postmortem_summary(sls_recent=None):
    """일간 리포트용 최근 손절 오답노트 원인 요약 텍스트 리스트 반환"""
    now_ts = int(time.time())
    merged = {}

    with _DB_LOCK:
        db = _load_db_unlocked()
    for r in db.get("records", []):
        if isinstance(r, dict) and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 604800:
            k = r.get("id") or f"{r.get('symbol')}_{r.get('created_at')}"
            merged[k] = r

    if isinstance(sls_recent, list):
        for s in sls_recent:
            if isinstance(s, dict) and (now_ts - int(_to_float(s.get("sl_at"), 0))) <= 604800:
                k = f"{s.get('symbol')}_{s.get('created_at')}"
                if k not in merged:
                    item = dict(s)
                    item["cause"] = classify_sl_cause(item)
                    merged[k] = item

    items = sorted(merged.values(), key=lambda x: int(_to_float(x.get("sl_at"), 0)), reverse=True)
    if not items:
        return ["• 📓 [손절(SL) 오답 노트]: 최근 7일 내 손절 기록이 없습니다."]

    cause_counts = Counter(str(x.get("cause") or classify_sl_cause(x)) for x in items)
    tk_counts = Counter(str(x.get("track_type", "TRACK_A")) for x in items)
    top_cause, top_c_n = cause_counts.most_common(1)[0]
    top_tk, top_tk_n = tk_counts.most_common(1)[0]

    lines = [
        f"• 📓 [손절(SL) 오답 노트 & 자동 감점 학습 ({len(items)}건)]:",
        f"  ↳ 📌 주요 손절 패턴: [{top_cause}] ({top_c_n}건) | 최다 손절 트랙: {top_tk} ({top_tk_n}건)",
    ]

    for s in items[:3]:
        sym = str(s.get("symbol", "")).replace("USDT", "")
        tk = str(s.get("track_type", "TRACK_A"))
        hold_m = int(_to_float(s.get("hold_minutes"), 0))
        max_f = _to_float(s.get("max_favorable_pct"), 0.0)
        sl_p_val = int(round(_to_float(s.get("sl_pct"), 20 if tk == "TRACK_D" else 6)))
        cause = str(s.get("cause") or classify_sl_cause(s))
        lines.append(
            f"  ↳ ⚠️ ${sym} ({tk}): 보유 {hold_m}분 후 -{sl_p_val}% 손절 (최고 +{max_f:.1f}% | 원인: {cause})"
        )

    return lines
# === END OF FILE ===
