import os
import requests

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

def send_telegram(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("ERROR: 깃허브 Secrets에 토큰 또는 채팅 ID가 없습니다.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        print(f"Telegram API Response Code: {res.status_code}")
        print(f"Telegram API Response Body: {res.text}")
    except Exception as e:
        print(f"EXCEPTION: {e}")

def main():
    print("스캐너 실행 시작...")
    # 강제 테스트 메시지 전송
    send_telegram("🔔 [긴급 테스트] 깃허브 Actions에서 보내는 텔레그램 연동 테스트 메시지입니다!")
    print("테스트 메시지 전송 함수 호출 완료.")

if __name__ == "__main__":
    main()
