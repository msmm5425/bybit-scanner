import os
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def test_telegram():
    print(f"DEBUG: TELEGRAM_TOKEN exists? {bool(TELEGRAM_TOKEN)}")
    print(f"DEBUG: TELEGRAM_CHAT_ID exists? {bool(TELEGRAM_CHAT_ID)}")
    print(f"DEBUG: CHAT_ID value: [{TELEGRAM_CHAT_ID}]")

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: 깃허브 Secrets에 토큰 또는 채팅 ID가 등록되지 않았습니다!")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": "🔔 [디버깅 테스트] 깃허브와 텔레그램 연동 테스트 메시지입니다!"
    }
    
    try:
        res = requests.post(url, json=payload, timeout=15)
        print(f"Telegram API Response Code: {res.status_code}")
        print(f"Telegram API Response Body: {res.text}")
        if res.status_code == 200:
            print("SUCCESS: 텔레그램 메시지 전송 성공!")
        else:
            print("FAIL: 텔레그램 서버에서 전송을 거부했습니다. (위 Response Body 확인 필요)")
    except Exception as e:
        print(f"EXCEPTION: 전송 중 에러 발생: {e}")

if __name__ == "__main__":
    test_telegram()
