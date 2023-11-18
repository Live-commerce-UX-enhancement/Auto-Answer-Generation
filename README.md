# Auto-Chat-Classification-and-Answers-for-Live-Commerce

1. AWS EC2 인스턴스 생성
2. 생성한 EC2 인스턴스 접속 후 repository git clone 
3. requirements 설치
    
    ```bash
    $ pip install -r requirements.txt
    ```
    
4. server 실행
    
    ```bash
    $ uvicorn main:app --reload --host=0.0.0.0 --port=8000
    ```
