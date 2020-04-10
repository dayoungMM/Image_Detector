# Keras를 이용한 사진분류

### [1단계] 사진 가져오기

1. API 등록

-  Google Cloud Platform (https://console.cloud.google.com) 
  - 사용자인증정보 - API 키 만들기 (키 기억하기)
-  Custom Search Engine 추가 (https://cse.google.com/cse/all) 
  - 검색할 사이트 등록
  - 검색엔진 ID (기억하기)
  - 이미지 검색 ON
  - 전체 웹 검색 ON

2. 환경설정

   - 패키지 설치
     - pip install pylint
     - pip install requests
     - pip install python-dotenv
     - pip install google-api-python-client
     - pip install opencv-python
   - settings.env
   - settings.py

3. python 코드 실행

   > (가상환경) python 파일명 키워드1,키워드2

   - API 코드
   - 얼굴 검출 프로그램(Haar Cascade)
     - https://docs.opencv.org/4.1.0/dc/d88/tutorial_traincascade.html
     - 머신러닝 기반 오브젝트 검출 알고리즘
     - 2001년 논문 "Rapid Object Detection using a Boosted Cascade of Simple Features"에서 Paul Viola와 Michael Jones가 제안한 특징(feature)을 기반으 로 비디오 또는 이미지에서 오브젝트를 검출하기 위해 사용되는 알고리즘 



## [2단계] 이미지에서 얼굴 추출

- graphviz (네트워크 그래프 그려주는 패키지)
  - 2.38 stable

https://graphviz.gitlab.io/_pages/Download/Download_windows.html