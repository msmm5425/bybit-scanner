import os, json, time, threading
from collections import Counter
from datetime import datetime, timedelta, timezone

import config as cfg
from config import KST, POSTMORTEM_DB_FILE, to_float as _to_float

RETENTION_SEC = 14 * 86400
ENGINE_VER = "v3"
MAX_PENALTY_CAP = 8
LOWVOL_EXEMPT_TRACKS = {"TRACK_D", "TRACK_C", "TRACK_F", "TRACK_G"}

_DB_LOCK = threading.RLock()
_DB_MEM_CACHE = {"mtime": -1.0, "ts": 0.0, "data": None}
_TOXIC_MEM_CACHE = {"updated_at": -1, "ts": 0.0, "data": {}}


def get_kst_block(ts=None):
    dt = datetime.fromtimestamp(int(ts), tz=KST) if ts else datetime.now(KST)
    start_h = (dt.hour // 3) * 3
    return f"{start_h:02d}-{start_h + 3:02d}"


def invalidate_cache():
    """Git 동기화 직후 메모리 캐시를 즉시 무효화해 최신 원격 DB 반영"""
    with _DB_LOCK:
        _DB_MEM_CACHE.update({"mtime": -1.0, "ts": 0.0, "data": None})
        _TOXIC_MEM_CACHE.update({"updated_at": -1, "ts": 0.0, "data": {}})


def _load_db_unlocked(use_cache=True):
    now = time.time()
    file_exists = os.path.exists(POSTMORTEM_DB_FILE)
    mtime = os.path.getmtime(POSTMORTEM_DB_FILE) if file_exists else -1.0

    if use_cache and _DB_MEM_CACHE["data"] is not None:
        if _DB_MEM_CACHE["mtime"] == mtime and (now - _DB_MEM_CACHE["ts"]) < 120.0:
            return _DB_MEM_CACHE["data"]

    data = {"records": [], "wins": [], "updated_at": 0, "engine_ver": ENGINE_VER}
    if file_exists:
        try:
            with open(POSTMORTEM_DB_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                if not isinstance(raw.get("records"), list):
                    raw["records"] = []
                if not isinstance(raw.get("wins"), list):
                    raw["wins"] = []
                data = raw
        except Exception:
            pass

    _DB_MEM_CACHE.update({"mtime": mtime, "ts": now, "data": data})
    return data


def _save_db_unlocked(data):
    tmp = f"{POSTMORTEM_DB_FILE}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, POSTMORTEM_DB_FILE)
        mtime = os.path.getmtime(POSTMORTEM_DB_FILE) if os.path.exists(POSTMORTEM_DB_FILE) else time.time()
        _DB_MEM_CACHE.update({"mtime": mtime, "ts": time.time(), "data": data})
        _TOXIC_MEM_CACHE.update({"updated_at": -1, "ts": 0.0, "data": {}})
    except Exception:
        pass


def merge_postmortem_dbs(local_db, remote_db):
    """원격과 로컬 postmortem_db.json ID 기준 합집합 병합 (내용 무변경 시 updated_at 유지로 중복 커밋 차단)"""
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

    r_recs = r_db.get("records") if isinstance(r_db.get("records"), list) else []
    r_wins = r_db.get("wins") if isinstance(r_db.get("wins"), list) else []
    base_upd = max(int(_to_float(l_db.get("updated_at"), 0)), int(_to_float(r_db.get("updated_at"), 0)))
    content_changed = (recs_sorted != r_recs) or (wins_sorted != r_wins)
    final_upd = now_ts if (content_changed or base_upd <= 0) else base_upd

    return {
        "records": recs_sorted,
        "wins": wins_sorted,
        "updated_at": final_upd,
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
    if rsi_v >= 72.0:
        return "단기 고점 RSI 과열 추격"
    if btc_1h <= -1.0 and hold_m <= 60:
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
        "score": int(_to_float(sl_payload.get("score"), 88)),
        "entry_price": _to_float(sl_payload.get("entry_price")),
        "stop_price": _to_float(sl_payload.get("stop_price")),
        "sl_trigger_price": _to_float(sl_payload.get("sl_trigger_price")),
        "sl_pct": int(round(_to_float(sl_payload.get("sl_pct"), 6))),
        "actual_sl_pct": round(_to_float(sl_payload.get("actual_sl_pct"), -_to_float(sl_payload.get("sl_pct"), 6)), 2),
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
        db = dict(_load_db_unlocked(use_cache=False))
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
        "score": int(_to_float(win_payload.get("score"), 88)),
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
        db = dict(_load_db_unlocked(use_cache=False))
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
    """손절 사후 부검 (25시간 커버 · EARLY_ENTRY +6%/+12%, TOXIC_BREAKOUT -5%)"""
    if not callable(fetch_klines_fn):
        return 0

    now_ts = int(time.time())
    with _DB_LOCK:
        db = _load_db_unlocked(use_cache=False)
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

            if rebound_from_entry >= 6.0 or rebound_from_sl >= 12.0:
                autopsy_type = "EARLY_ENTRY"
                done = True
            elif drop_from_sl <= -5.0:
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
            db = dict(_load_db_unlocked(use_cache=False))
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
    age_sec = max(0, now_ts - int(_to_float(event_ts, 0)))
    return 1.0 if age_sec <= (3 * 86400) else 0.6


def _compute_toxic_patterns(recs, wins, now_ts, db_updated_at=-1):
    """베이지안 평활 + 시간 감쇠 + 엔진 버전 격리 5D 독성 패턴 탐지 (감점폭 균형 정규화)"""
    if db_updated_at >= 0 and _TOXIC_MEM_CACHE["updated_at"] == db_updated_at and (time.time() - _TOXIC_MEM_CACHE["ts"]) < 120.0:
        return _TOXIC_MEM_CACHE["data"]

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
        ("funding_overheat", "펀딩비 과열(>=+0.02%)", 5, lambda x: _to_float(x.get("funding_pct"), 0.0) >= 0.020),
        ("pump_chase", "기상승 추격(24h>=+25%)", 5, lambda x: _to_float(x.get("change_24h"), 0.0) >= 25.0),
        ("low_oi_ratio", "저OI 빈수급(OI비중<8%)", 4, lambda x: 0.0 < _to_float(x.get("oi_ratio"), 0.0) < 8.0),
        ("btc_headwind", "BTC 역풍 진입(1H<-1.0%)", 4, lambda x: _to_float(x.get("btc_1h_pct"), 0.0) < -1.0),
    ]

    active_toxic = {}
    for key, label, pen, pred in rules:
        matched_sl = [r for r in r_7d if pred(r)]
        matched_win = [w for w in w_7d if pred(w)]
        raw_tot = len(matched_sl) + len(matched_win)
        eff_loss = sum(_loss_weight(r) for r in matched_sl)
        eff_win = sum(_win_weight(w) for w in matched_win)
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
            "penalty": 2,
            "loss_rate_pct": round(worst_rate * 100.0, 1),
            "losses": worst_l,
            "wins": worst_w,
        }

    if db_updated_at >= 0:
        _TOXIC_MEM_CACHE.update({"updated_at": db_updated_at, "ts": time.time(), "data": active_toxic})
    return active_toxic


def get_setup_penalty(symbol, track_type, rsi6_15m, vol_mult, extra_meta=None):
    """
    [오답노트 v3 통합 자동 감점 엔진 — 최대 8점 캡 · 정상 저거래량 오폭 차단 · 과열 승·패 대칭 검증]
    """
    sym = str(symbol or "").upper().strip()
    tk = str(track_type or "TRACK_A").upper().strip()
    rsi_v = _to_float(rsi6_15m, 50.0)
    vol_v = _to_float(vol_mult, 1.0)
    meta = extra_meta if isinstance(extra_meta, dict) else {}
    now_ts = int(time.time())

    with _DB_LOCK:
        db = _load_db_unlocked(use_cache=True)
        db_upd = int(_to_float(db.get("updated_at"), 0))
        recs = [r for r in db.get("records", []) if isinstance(r, dict) and r.get("engine_ver") == ENGINE_VER]
        wins = [w for w in db.get("wins", []) if isinstance(w, dict) and w.get("engine_ver") == ENGINE_VER]
        if not recs:
            return 0
        toxic_map = _compute_toxic_patterns(recs, wins, now_ts, db_updated_at=db_upd)

    penalty = 0

    recent_sym_sls = [
        r for r in recs
        if str(r.get("symbol", "")).upper() == sym
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 172800
    ]
    if recent_sym_sls:
        sym_pen = sum(2 if r.get("autopsy_type") == "EARLY_ENTRY" else 4 for r in recent_sym_sls)
        penalty += min(6, sym_pen)

    recent_tk_sls = [
        r for r in recs
        if str(r.get("track_type", "")).upper() == tk
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 604800
    ]
    recent_tk_wins = [
        w for w in wins
        if str(w.get("track_type", "")).upper() == tk
        and (now_ts - int(_to_float(w.get("win_at"), 0))) <= 604800
    ]

    if len(recent_tk_sls) >= 3:
        # [2차 보완] 동일 조건 승리 표본과 대칭 비교(손절 > 승리 × 2) 시에만 감점 적용
        overheat_cnt = sum(1 for r in recent_tk_sls if _to_float(r.get("rsi6_15m"), 50.0) >= 70.0)
        overheat_win_cnt = sum(1 for w in recent_tk_wins if _to_float(w.get("rsi6_15m"), 50.0) >= 70.0)
        if rsi_v >= 70.0 and overheat_cnt >= 3 and overheat_cnt > (overheat_win_cnt * 2):
            penalty += 3
        if tk not in LOWVOL_EXEMPT_TRACKS:
            lowvol_cnt = sum(1 for r in recent_tk_sls if _to_float(r.get("vol_mult"), 1.0) < 1.8)
            lowvol_win_cnt = sum(1 for w in recent_tk_wins if _to_float(w.get("vol_mult"), 1.0) < 1.8)
            if vol_v < 1.8 and lowvol_cnt >= 3 and lowvol_cnt > (lowvol_win_cnt * 2):
                penalty += 3

    recent_toxic_bo = [
        r for r in recent_tk_sls
        if r.get("autopsy_type") == "TOXIC_BREAKOUT"
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 172800
    ]
    if len(recent_toxic_bo) >= 3:
        penalty += 2

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
        if "btc_headwind" in toxic_map and btc_1h < -1.0 and tk not in ("TRACK_E", "TRACK_F"):
            penalty += toxic_map["btc_headwind"]["penalty"]
        if "weak_session" in toxic_map and kst_blk == toxic_map["weak_session"].get("block"):
            penalty += toxic_map["weak_session"]["penalty"]

    return int(min(MAX_PENALTY_CAP, max(0, penalty)))


def build_daily_postmortem_summary(sls_recent=None):
    """주간 리포트용 오답노트 v3 전용 대시보드 요약 (구버전 sl_history 혼용 제거)"""
    now_ts = int(time.time())

    with _DB_LOCK:
        db = _load_db_unlocked(use_cache=False)
    recs = db.get("records", [])
    wins = db.get("wins", [])

    v3_items = [
        r for r in recs
        if isinstance(r, dict)
        and r.get("engine_ver") == ENGINE_VER
        and (now_ts - int(_to_float(r.get("sl_at"), 0))) <= 604800
    ]
    v3_items.sort(key=lambda x: int(_to_float(x.get("sl_at"), 0)), reverse=True)

    v3_wins = [
        w for w in wins
        if isinstance(w, dict)
        and w.get("engine_ver") == ENGINE_VER
        and (now_ts - int(_to_float(w.get("win_at"), 0))) <= 604800
    ]

    if not v3_items:
        return [f"• 📓 [오답노트 v3]: 최근 7일 v3 학습 표본 {len(v3_wins)}승 0패 (손절 기록 없음)"]

    cause_counts = Counter(str(x.get("cause") or classify_sl_cause(x)) for x in v3_items)
    tk_counts = Counter(str(x.get("track_type", "TRACK_A")) for x in v3_items)
    top_cause, top_c_n = cause_counts.most_common(1)[0]
    top_tk, top_tk_n = tk_counts.most_common(1)[0]

    toxic_map = _compute_toxic_patterns(v3_items, v3_wins, now_ts, db_updated_at=-1)
    toxic_sorted = sorted(toxic_map.values(), key=lambda x: (x["loss_rate_pct"], x["losses"]), reverse=True)[:2]

    early_cnt = sum(1 for x in v3_items if x.get("autopsy_type") == "EARLY_ENTRY")
    toxic_bo_cnt = sum(1 for x in v3_items if x.get("autopsy_type") == "TOXIC_BREAKOUT")
    chop_cnt = sum(1 for x in v3_items if x.get("autopsy_type") == "NEUTRAL_CHOP")
    pending_cnt = sum(1 for x in v3_items if not x.get("autopsy_done"))

    lines = [
        f"• 📓 [오답노트 v3 자가 학습 & 베이지안 감점 (최근 7일 v3 표본 {len(v3_wins)}승 {len(v3_items)}패 · 최대 -{MAX_PENALTY_CAP}점)]:",
        f"  ↳ 📌 주요 손절 원인: [{top_cause}] ({top_c_n}건) | 최다 손절 트랙: {top_tk} ({top_tk_n}건)",
    ]

    if toxic_sorted:
        tx_str = " / ".join(f"{t['label']} (평활손실률 {t['loss_rate_pct']:.0f}% · -{t['penalty']}점)" for t in toxic_sorted)
        lines.append(f"  ↳ 🛡️ 작동 중인 독성 패턴 감점: {tx_str}")
    else:
        lines.append("  ↳ 🛡️ 작동 중인 독성 패턴 감점: v3 표본 축적 중 / 임계치 초과 패턴 없음")

    lines.append(
        f"  ↳ 🔬 손절 사후 부검(25h): 스탑헌팅 헛털림 {early_cnt}건 | 구조 붕괴 {toxic_bo_cnt}건 | 횡보 {chop_cnt}건 (대기 {pending_cnt}건)"
    )

    for s in v3_items[:3]:
        sym = str(s.get("symbol", "")).replace("USDT", "")
        tk = str(s.get("track_type", "TRACK_A"))
        hold_m = int(_to_float(s.get("hold_minutes"), 0))
        max_f = _to_float(s.get("max_favorable_pct"), 0.0)
        act_sl = _to_float(s.get("actual_sl_pct"), -_to_float(s.get("sl_pct"), 20 if tk == "TRACK_D" else 6))
        cause = str(s.get("cause") or classify_sl_cause(s))
        at = s.get("autopsy_type", "")
        at_tag = " [부검:헛털림]" if at == "EARLY_ENTRY" else (" [부검:구조붕괴]" if at == "TOXIC_BREAKOUT" else "")
        lines.append(
            f"  ↳ ⚠️ ${sym} ({tk}): 보유 {hold_m}분 후 {act_sl:+.2f}% 손절 (최고 +{max_f:.1f}% | {cause}{at_tag})"
        )

    return lines
# === END OF FILE ===
