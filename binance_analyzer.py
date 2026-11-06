import os
import datetime
import ccxt
import pandas as pd
import numpy as np

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

SKIP_PREFIXES = ("1000000", "100000", "10000", "1000")


def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("텔레그램 설정 누락")
        return
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    for i in range(0, len(text), 3500):
        try:
            requests.post(
                url,
                json={"chat_id": TELEGRAM_CHAT_ID, "text": text[i:i + 3500]},
                timeout=15
            )
        except Exception as e:
            print(f"텔레그램 전송 오류: {e}")


def fmt_price(val, ref):
    if ref >= 1000:
        return f"{val:.2f}"
    elif ref >= 10:
        return f"{val:.3f}"
    elif ref >= 1:
        return f"{val:.4f}"
    elif ref >= 0.01:
        return f"{val:.5f}"
    else:
        return f"{val:.6f}"


def main():
    kst = datetime.timezone(datetime.timedelta(hours=9))
    now_str = datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST")

    try:
        # CCXT를 이용한 바이비트 선물(Linear) API 직접 연동
        exchange = ccxt.bybit({
            'enableRateLimit': True,
            'options': {'defaultType': 'linear'}
        })

        tickers = exchange.fetch_tickers()
        valid = []

        for symbol, x in tickers.items():
            if not symbol.endswith('/USDT'):
                continue
            
            base = symbol.split('/')[0]
            if any(base.startswith(p) for p in SKIP_PREFIXES):
                continue

            info = x.get('info', {})
            last_p = float(x.get('last') or 0)
            chg = float(info.get('price24hPcnt') or x.get('percentage') or 0) * 100.0
            turnover = float(info.get('turnover24h') or 0)
            funding = float(info.get('fundingRate') or 0) * 100.0
            oi_val = float(info.get('openInterestValue') or 0)

            if last_p <= 0:
                continue

            valid.append({
                "symbol": base + "USDT",
                "ticker": base,
                "ccxt_sym": symbol,
                "price": last_p,
                "chg": chg,
                "turnover": turnover,
                "funding": funding,
                "oi_val": oi_val
            })

        if not valid:
            send_telegram("⚠️ [바이비트 분석] 틱커 데이터를 불러오지 못했습니다.")
            return

        valid.sort(key=lambda k: k["chg"], reverse=True)
        top5 = valid[:5]

        results = []
        for item in top5:
            try:
                # API 직접 연동으로 5분봉 150개 가져오기
                ohlcv = exchange.fetch_ohlcv(item["ccxt_sym"], timeframe='5m', limit=150)
                if len(ohlcv) < 30:
                    continue

                df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = df[c].astype(float)

                df["ma5"] = df["close"].rolling(5).mean()
                df["ma20"] = df["close"].rolling(20).mean()
                df["vma20"] = df["volume"].rolling(20).mean()

                # 간단한 초동 발화점(수급 + 상승률 폭발) 탐색
                df["ret"] = (df["close"] - df["open"]) / (df["open"] + 1e-9) * 100.0
                df["vol_ratio"] = df["volume"] / (df["vma20"].shift(1) + 1e-9)

                search_zone = df.iloc[-60:].copy()
                if search_zone.empty:
                    continue

                score = search_zone["ret"] * search_zone["vol_ratio"]
                ig_idx = score.idxmax() if score.notna().sum() > 0 else search_zone.index[-1]
                
                ig_candle = df.loc[ig_idx]
                pre_candle = df.loc[max(0, ig_idx - 1)]

                ig_time = datetime.datetime.fromtimestamp(
                    ig_candle["ts"] / 1000.0, kst
                ).strftime("%m/%d %H:%M")

                ma_short_ok = bool(pre_candle["ma5"] > pre_candle["ma20"])
                vol_mult = float(ig_candle["vol_ratio"])
                if np.isnan(vol_mult):
                    vol_mult = 1.0

                results.append({
                    "ticker": item["ticker"],
                    "symbol": item["symbol"],
                    "price": item["price"],
                    "chg": item["chg"],
                    "turnover_m": item["turnover"] / 1_000_000.0,
                    "ig_time": ig_time,
                    "ig_price": float(ig_candle["open"]),
                    "ma_short_ok": ma_short_ok,
                    "vol_mult": vol_mult,
                    "funding": item["funding"],
                    "oi_val_m": item["oi_val"] / 1_000_000.0
                })
            except Exception as ex:
                print(f"{item['ticker']} 분석 중 예외: {ex}")

        if not results:
            send_telegram("⚠️ [바이비트 분석] 분석된 종목 결과가 없습니다.")
            return

        lines = [f"📊 [바이비트 실시간 상승률 Top 5 초동 해부 리포트] ({now_str})\n"]
        for idx, r in enumerate(results, 1):
            s_tag = "✅ 단기 정배열" if r["ma_short_ok"] else "❌ 혼조"
            p = r["price"]
            lines.append("━━━━━━━━━━━━━━")
            lines.append(f"{idx}️⃣ ${r['ticker']} (+{r['chg']:.1f}% | 현재가 {fmt_price(p, p)} USDT)")
            lines.append(f"• 거래대금: {r['turnover_m']:,.1f}M | 펀딩비: {r['funding']:+.4f}%")
            lines.append(f"• [초동 발화 시점]: {r['ig_time']} KST (시드가격: {fmt_price(r['ig_price'], p)})")
            lines.append(f"• [초동 컨펌]: 5분봉 이평({s_tag}) | 거래량 [{r['vol_mult']:.1f}배 폭발]")

        send_telegram("\n".join(lines))

    except Exception as e:
        send_telegram(f"⚠️ [바이비트 API 직접 연동 오류]: {str(e)}")


if __name__ == "__main__":
    main()
