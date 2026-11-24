import os, json, time, threading
from collections import Counter
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
POSTMORTEM_DB_FILE = "postmortem_db.json"
RETENTION_SEC = 14 * 86400
ENGINE_VER = "v3"  # 구버전 로직 손절 오염 자동 격리용 버전 태그
_DB_LOCK = threading.Lock()


def _to_float(val, default=0.0):
    try:
        v = float(val)
        return default if (v != v) else v
    except Exception:
        return default


def get_kst_block(ts=None):
    dt = datetime.fromtimestamp(int(ts), tz=KST) if ts else datetime.now(KST)
    start_h = (dt.hour // 3) * 3
    return f"{start_h:02d}-{start_h + 3:02d}"


def _load_db_unlocked():
    if os.path.exists(POSTMORTEM_DB_FILE):
        try:
            with open(POSTMORTEM_DB_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                if not isinstance(data.get("records"), list):
                    data["records"] = []
                if not isinstance(data.get("wins"), list):
                    data["wins"] = []
                return data
        except Exception:
            pass
    return {"records": [], "wins": [], "updated_at": 0, "engine_ver": ENGINE_VER}


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


def merge_postmortem_dbs(local_db, remote_db):
    """
    [Git 충돌/덮어쓰기 방지] 원격과 로컬 postmortem_db.json을 ID 기준 합집합 병합
    - 동일 ID 충돌 시 autopsy_done=True 및 최신 updated 정보를 우선 보존
    """
    now_ts = int(time.time())
    l_db = local_db if isinstance(local_db, dict) else {}
    r_db = remote_db if isinstance(remote_db, dict) else {}

    rec_map = {}
    for src in (r_db.get("records") or [], l_db.get("records") or []):
        for r in src:
            if not isinstance(r, dict):
                continue
            sl_at = int(_to_float(r.get("sl_at"), 0))
            if sl_at > 0 and (now_ts - sl_at) > RETENTION_SEC:
                continue
            rid = str(r.get("id") or f"{r.get('symbol')}_{r.get('created_at')}")
            prev = rec_map.get(rid)
            if not prev:
                rec_map[rid] = dict(r)
            else:
                merged_r = dict(prev)
                merged_r.update(r)
                if prev.get("autopsy_done") and not r.get("autopsy_done"):
                    merged_r["autopsy_done"] = True
                    merged_r["autopsy_type"] = prev.get("autopsy_type", "")
                    merged_r["autopsy_post_high_pct"] = prev.get("autopsy_post_high_pct", 0.0)
                    merged_r["autopsy_post_low_pct"] = prev.get("autopsy_post_low_pct", 0.0)
                rec_map[rid] = merged_r

    win_map = {}
    for src in (r_db.get("wins") or [], l_db.get("wins") or []):
        for w in src:
            if not isinstance(w, dict):
                continue
            win_at = int(_to_float(w.get("win_at"), 0))
            if win_at > 0 and (now_ts - win_at) > RETENTION_SEC:
                continue
            wid = str(w.get("id") or f"{w.get('symbol')}_{w.get('created_at')}")
            prev_w = win_map.get(wid)
            if not prev_w or _to_float(w.get("max_favorable_pct")) >= _to_float(prev_w.get("max_favorable_pct")):
                win_map[wid] = dict(w)

    recs_sorted = sorted(rec_map.values(), key=lambda x: int(_to_float(x.get("sl_at"), 0)), reverse=True)[:250]
    wins_sorted = sorted(win_map.values(), key=lambda x: int(_to_float(x.get("win_at"), 0)), reverse=True)[:250]
    return {
        "records": recs_sorted,
        "wins": wins_sorted,
        "updated_at": max(int(_to_float(l_db.get("updated_at"), 0)), int(_to_float(r_db.get("updated_at"), 0)), now_ts),
        "engine_ver": ENGINE_VER,
    }


def classify_sl_cause(sl_item):
    rsi_v = _to_float(sl_item.get("rsi6_15m"), 50.0)
    vol_m = _to_float(sl_item.get("vol_mult"), 1.0)
    max_f = _to_float(sl_item.get("max_favorable_pct"), 0.0)
    hold_m = int(_to_float(sl_item.get("hold_minutes"), 0))
    fund_v = _to_float(sl_item.get("funding_pct"), 0.0)
    chg_24 = _to_float(sl_item.get("change_24h"), 0.0)
    btc_1h = _to_float(sl_item.get("btc_1h_pct"), 0.0)

    if fund_v >= 0.020:
        return "펀딩비 과열 롱 쏠림 함정"
    if chg_24 >= 25.0 and max_f < 3.0:
        return "기상승 고점 추격 진입"
    if rsi_v >= 70.0:
        return "단기 고점 RSI 과열 추격"
    if btc_1h <= -0.5 and hold_m <= 60:
        return "BTC 역풍 구간 진입"
    if vol_m < 1.8 and max_f < 2.0:
        return "돌파 거래량 부족(가짜 돌파)"
    if hold_m <= 45 and max_f < 1.5:
        return "진입 직후 급락(스탑헌팅/비트 연동)"
    if max_f >= 4.0:
        return f"+{max_f:.1f}% 반등 후 차익 매물 출회"
    return "비트 연동 급락 및 지지선 이탈"


def record_sl_postmortem(sl_payload):
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
        "engine_ver": ENGINE_VER,
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
        "oi_ratio": _to_float(sl_payload.get("oi_ratio"), 0.0),
        "funding_pct": _to_float(sl_payload.get("funding_pct"), 0.0),
        "change_24h": _to_float(sl_payload.get("change_24h"), 0.0),
        "btc_1h_pct": _to_float(sl_payload.get("btc_1h_pct"), 0.0),
        "kst_block": str(sl_payload.get("kst_block") or get_kst_block(c_at)),
        "cause": classify_sl_cause(sl_payload),
        "autopsy_done": False,
        "autopsy_type": "",
        "autopsy_post_high_pct": 0.0,
        "autopsy_post_low_pct": 0.0,
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
        db["records"] = recs[:250]
        db["updated_at"] = now_ts
        db["engine_ver"] = ENGINE_VER
        _save_db_unlocked(db)
    return True


def record_win_postmortem(win_payload):
    if not isinstance(win_payload, dict):
        return False

    now_ts = int(time.time())
    sym = str(win_payload.get("symbol", "")).upper().strip()
    c_at = int(_to_float(win_payload.get("created_at"), now_ts))
    win_at = int(_to_float(win_payload.get("win_at"), now_ts))
    if not sym:
        return False

    rec_id = f"{sym}_{c_at}"
    record = {
        "id": rec_id,
        "engine_ver": ENGINE_VER,
        "symbol": sym,
        "track_type": str(win_payload.get("track_type", "TRACK_A")),
        "score": int(_to_float(win_payload.get("score"), 90)),
        "created_at": c_at,
        "win_at": win_at,
        "max_favorable_pct": _to_float(win_payload.get("max_favorable_pct")),
        "vol_mult": _to_float(win_payload.get("vol_mult"), 1.0),
        "rsi6_15m": _to_float(win_payload.get("rsi6_15m"), 50.0),
        "oi_ratio": _to_float(win_payload.get("oi_ratio"), 0.0),
        "funding_pct": _to_float(win_payload.get("funding_pct"), 0.0),
        "change_24h": _to_float(win_payload.get("change_24h"), 0.0),
        "btc_1h_pct": _to_float(win_payload.get("btc_1h_pct"), 0.0),
        "kst_block": str(win_payload.get("kst_block") or get_kst_block(c_at)),
    }

    with _DB_LOCK:
        db = _load_db_unlocked()
        wins = [
            w for w in db.get("wins", [])
            if isinstance(w, dict)
            and w.get("id") != rec_id
            and (now_ts - int(_to_float(w.get("win_at"), 0))) <= RETENTION_SEC
        ]
        wins.insert(0, record)
        db["wins"] = wins[:250]
        db["updated_at"] = now_ts
        db["engine_ver"] = ENGINE_VER
        _save_db_unlocked(db)
    return True


def run_post_sl_autopsy(fetch_klines_fn, max_checks=4):
    """
    [손절 사후 부검 (Post-SL Rebound Autopsy)]
    - 15분봉 limit=100 (25시간 전체 커버)로 수정하여 16~24시간 경과 종목의 초기 반등 누락 버그 해결
    """
    if not callable(fetch_klines_fn):
        return 0

    now_ts = int(time.time())
    with _DB_LOCK:
        db = _load_db_unlocked()
        candidates = [
            dict(r) for r in db.get("records", [])
            if isinstance(r, dict)
            and not r.get("autopsy_done")
            and 7200 <= (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 86400
        ][:max_checks]

    if not candidates:
        return 0

    updates = {}
    for rec in candidates:
        sym = str(rec.get("symbol", "")).upper()
        sl_at = int(_to_float(rec.get("sl_at"), 0))
        entry_p = _to_float(rec.get("entry_price"), 0.0)
        sl_trig = _to_float(rec.get("sl_trigger_price") or rec.get("stop_price"), entry_p * 0.94)
        if not sym or entry_p <= 0 or sl_trig <= 0 or sl_at <= 0:
            continue

        try:
            # 15m × 100봉 = 25시간 전체 커버 (기존 60봉=15시간 절단 버그 수정)
            df_15m = fetch_klines_fn(sym, interval="15", limit=100)
            if df_15m is None or df_15m.empty or "timestamp" not in df_15m.columns:
                continue
            post_df = df_15m[df_15m["timestamp"] >= (sl_at * 1000)]
            if len(post_df) < 4:
                continue

            post_high = _to_float(post_df["high"].max(), sl_trig)
            post_low = _to_float(post_df["low"].min(), sl_trig)
            rebound_from_entry = round(((post_high - entry_p) / entry_p) * 100.0, 1)
            rebound_from_sl = round(((post_high - sl_trig) / sl_trig) * 100.0, 1)
            drop_from_sl = round(((post_low - sl_trig) / sl_trig) * 100.0, 1)

            elapsed_after_sl = now_ts - sl_at
            autopsy_type = ""
            done = False

            if rebound_from_entry >= 12.0 or rebound_from_sl >= 16.0:
                autopsy_type = "EARLY_ENTRY"
                done = True
            elif drop_from_sl <= -10.0:
                autopsy_type = "TOXIC_BREAKOUT"
                done = True
            elif elapsed_after_sl >= 21600:
                autopsy_type = "NEUTRAL_CHOP"
                done = True

            if done:
                updates[rec["id"]] = {
                    "autopsy_done": True,
                    "autopsy_type": autopsy_type,
                    "autopsy_post_high_pct": rebound_from_entry,
                    "autopsy_post_low_pct": drop_from_sl,
                }
        except Exception:
            continue

    if updates:
        with _DB_LOCK:
            db = _load_db_unlocked()
            changed = 0
            for r in db.get("records", []):
                if isinstance(r, dict) and r.get("id") in updates:
                    r.update(updates[r["id"]])
                    changed += 1
            if changed > 0:
                db["updated_at"] = now_ts
                _save_db_unlocked(db)
            return changed
    return 0


def _time_decay_weight(event_ts, now_ts):
    """시간 감쇠 가중치: 최근 3일 1.0 / 4~7일 0.6"""
    age_sec = max(0, now_ts - int(_to_float(event_ts, 0)))
    return 1.0 if age_sec <= (3 * 86400) else 0.6


def _compute_toxic_patterns(recs, wins, now_ts):
    """
    [오답노트 v3 · 베이지안 평활 + 시간 감쇠 + 엔진 버전 격리 5D 독성 패턴 탐지]
    - 구버전 로직 오염 데이터(engine_ver != 'v3') 자동 배제
    - 소표본 우연 과민 감점 방지: 최소 표본 >= 5건 & 손절 >= 3건 & 평활 손실률 >= 65%
    """
    r_7d = [
        r for r in recs
        if isinstance(r, dict)
        and r.get("engine_ver") == ENGINE_VER
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 604800
    ]
    w_7d = [
        w for w in wins
        if isinstance(w, dict)
        and w.get("engine_ver") == ENGINE_VER
        and (now_ts - int(_to_float(w.get("win_at"), 0))) <= 604800
    ]

    def _loss_weight(r):
        decay_w = _time_decay_weight(r.get("sl_at"), now_ts)
        at = str(r.get("autopsy_type", ""))
        base_w = 0.5 if at == "EARLY_ENTRY" else (1.2 if at == "TOXIC_BREAKOUT" else 1.0)
        return base_w * decay_w

    def _win_weight(w):
        return _time_decay_weight(w.get("win_at"), now_ts)

    rules = [
        ("funding_overheat", "펀딩비 과열(>=+0.02%)", 8, lambda x: _to_float(x.get("funding_pct"), 0.0) >= 0.020),
        ("pump_chase", "기상승 추격(24h>=+25%)", 8, lambda x: _to_float(x.get("change_24h"), 0.0) >= 25.0),
        ("low_oi_ratio", "저OI 빈수급(OI비중<8%)", 7, lambda x: 0.0 < _to_float(x.get("oi_ratio"), 0.0) < 8.0),
        ("btc_headwind", "BTC 역풍 진입(1H<-0.5%)", 7, lambda x: _to_float(x.get("btc_1h_pct"), 0.0) < -0.5),
    ]

    active_toxic = {}
    for key, label, pen, pred in rules:
        matched_sl = [r for r in r_7d if pred(r)]
        matched_win = [w for w in w_7d if pred(w)]
        raw_tot = len(matched_sl) + len(matched_win)
        eff_loss = sum(_loss_weight(r) for r in matched_sl)
        eff_win = sum(_win_weight(w) for w in matched_win)
        # 베이지안 라플라스 평활 (사전 승률 50%, 가중치 3.0)
        smoothed_lr = (eff_loss + 1.5) / (eff_loss + eff_win + 3.0)
        if raw_tot >= 5 and len(matched_sl) >= 3 and smoothed_lr >= 0.65:
            active_toxic[key] = {
                "key": key,
                "label": label,
                "penalty": pen,
                "loss_rate_pct": round(smoothed_lr * 100.0, 1),
                "losses": len(matched_sl),
                "wins": len(matched_win),
            }

    # 5번째 차원: 취약 KST 3시간 세션 블록 (최소 5건, 손절 3건+, 베이지안 평활 >= 65%)
    blocks = set(
        [str(r.get("kst_block") or get_kst_block(r.get("created_at"))) for r in r_7d]
        + [str(w.get("kst_block") or get_kst_block(w.get("created_at"))) for w in w_7d]
    )
    worst_blk, worst_rate, worst_l, worst_w = None, 0.0, 0, 0
    for blk in blocks:
        b_sls = [r for r in r_7d if str(r.get("kst_block") or get_kst_block(r.get("created_at"))) == blk]
        b_wins = [w for w in w_7d if str(w.get("kst_block") or get_kst_block(w.get("created_at"))) == blk]
        if (len(b_sls) + len(b_wins)) >= 5 and len(b_sls) >= 3:
            el = sum(_loss_weight(r) for r in b_sls)
            ew = sum(_win_weight(w) for w in b_wins)
            lr = (el + 1.5) / (el + ew + 3.0)
            if lr >= 0.65 and lr > worst_rate:
                worst_blk, worst_rate, worst_l, worst_w = blk, lr, len(b_sls), len(b_wins)

    if worst_blk:
        active_toxic["weak_session"] = {
            "key": "weak_session",
            "block": worst_blk,
            "label": f"취약 시간대({worst_blk}시 KST)",
            "penalty": 4,
            "loss_rate_pct": round(worst_rate * 100.0, 1),
            "losses": worst_l,
            "wins": worst_w,
        }

    return active_toxic


def get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult, extra_meta=None):
    """[오답노트 v3 통합 자동 감점 엔진 (구버전 격리 · 베이지안 평활 · 최대 16점)]"""
    sym = str(symbol or "").upper().strip()
    tk = str(track_type or "TRACK_A").upper().strip()
    rsi_v = _to_float(rsi6_15m, 50.0)
    vol_v = _to_float(vol_mult, 1.0)
    meta = extra_meta if isinstance(extra_meta, dict) else {}
    now_ts = int(time.time())

    with _DB_LOCK:
        db = _load_db_unlocked()
    recs = [r for r in db.get("records", []) if isinstance(r, dict) and r.get("engine_ver") == ENGINE_VER]
    wins = [w for w in db.get("wins", []) if isinstance(w, dict) and w.get("engine_ver") == ENGINE_VER]
    if not recs:
        return 0

    penalty = 0

    # 1) 최근 48시간 내 동일 종목 반복 손절 감점 (헛털림 EARLY_ENTRY면 완화)
    recent_sym_sls = [
        r for r in recs
        if str(r.get("symbol", "")).upper() == sym
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 172800
    ]
    if recent_sym_sls:
        sym_pen = sum(3 if r.get("autopsy_type") == "EARLY_ENTRY" else 6 for r in recent_sym_sls)
        penalty += min(10, sym_pen)

    # 2) 최근 7일 내 동일 트랙의 과열/저거래량 손절 패턴 학습 감점 (최소 3건 이상 누적 시)
    recent_tk_sls = [
        r for r in recs
        if str(r.get("track_type", "")).upper() == tk
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 604800
    ]
    if len(recent_tk_sls) >= 3:
        overheat_cnt = sum(1 for r in recent_tk_sls if _to_float(r.get("rsi6_15m"), 50.0) >= 68.0)
        lowvol_cnt = sum(1 for r in recent_tk_sls if _to_float(r.get("vol_mult"), 1.0) < 1.8)
        if rsi_v >= 68.0 and overheat_cnt >= 3:
            penalty += 4
        if vol_v < 1.8 and lowvol_cnt >= 3:
            penalty += 4

    # 3) 48시간 내 사후 부검 '구조 붕괴(TOXIC_BREAKOUT)' 2회 이상 발생 트랙 경계 감점
    recent_toxic_bo = [
        r for r in recent_tk_sls
        if r.get("autopsy_type") == "TOXIC_BREAKOUT"
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 172800
    ]
    if len(recent_toxic_bo) >= 2:
        penalty += 4

    # 4) 5차원 독성 패턴(베이지안 평활) 자동 감점
    toxic_map = _compute_toxic_patterns(recs, wins, now_ts)
    if meta and toxic_map:
        fund_v = _to_float(meta.get("funding_pct"), 0.0)
        chg_24 = _to_float(meta.get("change_24h"), 0.0)
        oi_r = _to_float(meta.get("oi_ratio"), 0.0)
        btc_1h = _to_float(meta.get("btc_1h_pct"), 0.0)
        kst_blk = str(meta.get("kst_block") or get_kst_block(now_ts))

        if "funding_overheat" in toxic_map and fund_v >= 0.020:
            penalty += toxic_map["funding_overheat"]["penalty"]
        if "pump_chase" in toxic_map and chg_24 >= 25.0:
            penalty += toxic_map["pump_chase"]["penalty"]
        if "low_oi_ratio" in toxic_map and 0.0 < oi_r < 8.0:
            penalty += toxic_map["low_oi_ratio"]["penalty"]
        if "btc_headwind" in toxic_map and btc_1h < -0.5 and tk not in ("TRACK_E", "TRACK_F"):
            penalty += toxic_map["btc_headwind"]["penalty"]
        if "weak_session" in toxic_map and kst_blk == toxic_map["weak_session"].get("block"):
            penalty += toxic_map["weak_session"]["penalty"]

    return int(min(16, max(0, penalty)))


def build_daily_postmortem_summary(sls_recent=None):
    """일간·주간 리포트용 오답노트 v3 대시보드 요약"""
    now_ts = int(time.time())
    merged = {}

    with _DB_LOCK:
        db = _load_db_unlocked()
    recs = db.get("records", [])
    wins = db.get("wins", [])

    for r in recs:
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
        return ["• 📓 [오답노트 v3]: 최근 7일 내 손절 기록이 없습니다."]

    cause_counts = Counter(str(x.get("cause") or classify_sl_cause(x)) for x in items)
    tk_counts = Counter(str(x.get("track_type", "TRACK_A")) for x in items)
    top_cause, top_c_n = cause_counts.most_common(1)[0]
    top_tk, top_tk_n = tk_counts.most_common(1)[0]

    v3_sl_cnt = sum(1 for x in items if x.get("engine_ver") == ENGINE_VER)
    v3_win_cnt = sum(1 for w in wins if isinstance(w, dict) and w.get("engine_ver") == ENGINE_VER and (now_ts - int(_to_float(w.get("win_at"), 0))) <= 604800)

    toxic_map = _compute_toxic_patterns(list(merged.values()), wins, now_ts)
    toxic_sorted = sorted(toxic_map.values(), key=lambda x: (x["loss_rate_pct"], x["losses"]), reverse=True)[:2]

    early_cnt = sum(1 for x in items if x.get("autopsy_type") == "EARLY_ENTRY")
    toxic_bo_cnt = sum(1 for x in items if x.get("autopsy_type") == "TOXIC_BREAKOUT")
    pending_cnt = sum(1 for x in items if not x.get("autopsy_done"))

    lines = [
        f"• 📓 [오답노트 v3 자가 학습 & 베이지안 감점 ({len(items)}건 | v3 학습표본 {v3_win_cnt}승 {v3_sl_cnt}패)]:",
        f"  ↳ 📌 주요 손절 원인: [{top_cause}] ({top_c_n}건) | 최다 손절 트랙: {top_tk} ({top_tk_n}건)",
    ]

    if toxic_sorted:
        tx_str = " / ".join(f"{t['label']} (평활손실률 {t['loss_rate_pct']:.0f}% · -{t['penalty']}점)" for t in toxic_sorted)
        lines.append(f"  ↳ 🛡️ 작동 중인 독성 패턴 감점: {tx_str}")
    else:
        lines.append("  ↳ 🛡️ 작동 중인 독성 패턴 감점: v3 표본 축적 중 / 임계치 초과 패턴 없음")

    lines.append(
        f"  ↳ 🔬 손절 사후 부검(25h): 스탑헌팅 헛털림 {early_cnt}건 vs 구조 붕괴 {toxic_bo_cnt}건 (대기 {pending_cnt}건)"
    )

    for s in items[:3]:
        sym = str(s.get("symbol", "")).replace("USDT", "")
        tk = str(s.get("track_type", "TRACK_A"))
        hold_m = int(_to_float(s.get("hold_minutes"), 0))
        max_f = _to_float(s.get("max_favorable_pct"), 0.0)
        sl_p_val = int(round(_to_float(s.get("sl_pct"), 20 if tk == "TRACK_D" else 6)))
        cause = str(s.get("cause") or classify_sl_cause(s))
        at = s.get("autopsy_type", "")
        at_tag = " [부검:헛털림]" if at == "EARLY_ENTRY" else (" [부검:구조붕괴]" if at == "TOXIC_BREAKOUT" else "")
        lines.append(
            f"  ↳ ⚠️ ${sym} ({tk}): 보유 {hold_m}분 후 -{sl_p_val}% 손절 (최고 +{max_f:.1f}% | {cause}{at_tag})"
        )

    return lines
# === END OF FILE ===
