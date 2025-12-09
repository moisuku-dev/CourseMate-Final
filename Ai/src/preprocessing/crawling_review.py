import sys
from selenium.webdriver.common.by import By
import pandas as pd 
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import time
import datetime
from selenium.webdriver.chrome.service import Service

# https://m.map.naver.com/ 여기서 크롤링 데이터 확보

# url 입력
url = input("크롤링할 URL을 입력하세요 (예: https://m.place...): ").strip()
if not url:
    print("URL이 입력되지 않았습니다. 프로그램을 종료합니다.")
    sys.exit()
url = url + '?entry=pll'

# Webdriver headless mode setting
options = webdriver.ChromeOptions()
# options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")
options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36")

now = datetime.datetime.now() 

# 크롤링 시작
driver = None
try:
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.get(url)
    
    driver.implicitly_wait(10) 
    
    # 이름/주소 수집 로직 
    store_name = "수집실패" 
    address = "수집실패"
    try:
        print("--- 1. 페이지 초기 로딩 대기 (5초)... ---")
        time.sleep(5)
        
        store_name = driver.find_element(By.CSS_SELECTOR, 'span.GHAhO').text
        address = driver.find_element(By.CSS_SELECTOR, 'span.LDgIH').text
        print(f"--- 2. 가게 이름 수집: {store_name} ---")
        print(f"--- 2. 주소 수집: {address} ---")

        review_tab = driver.find_element(By.XPATH, '//a[@role="tab"][contains(., "리뷰")]')
        review_tab.click()
        print("--- 3. 리뷰 탭으로 이동 완료. 5초 대기... ---")
        time.sleep(5)
        
    except Exception as e:
        print(f"--- 가게 이름/주소 수집 중 에러: {e} ---")
    # 이름/주소 수집 끝


    body = driver.find_element(By.TAG_NAME, 'body')
    
    # 50번 '더보기' 루프
    print("--- 4. '더보기' 루프 시작 (최대 50회) ---")
    try:
        for i in range(50): 
            print(f"{i+1}번째 '더보기' 시도...")
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(1.0) # (intercepted 에러 잡는 1초 대기)
            # 버튼 요소를 먼저 찾고
            more_button = driver.find_element(By.CSS_SELECTOR, 'a.fvwqf')
            # 자바스크립트로 강제로 눌러버림 
            driver.execute_script("arguments[0].click();", more_button)
            time.sleep(0.4)
    except Exception as e:
        print(f"'더보기' 버튼 클릭 중단 (이유: {e})")
    else :
        print("최대 50회 '더보기' 클릭 완료")
    
    
    print("크롤링 완료. 5초간 최종 페이지 로딩 대기...")
    time.sleep(5)
    
    html = driver.page_source # 리뷰 페이지의 HTML을 변수에 저장
    driver.quit() # 드라이버 종료

    # CSV로 수집 데이터 저장
    
    # '깨끗한' 데이터를 담을 빈 리스트 생성
    review_data_list = []

    # 리뷰 수집 로직 - BeautifulSoup
    bs = BeautifulSoup(html, 'lxml')
    reviews = bs.select('li.place_apply_pui.EjjAW')

    if not reviews:
        print("--- 경고: 리뷰를 하나도 찾지 못했습니다. (0개 수집) ---")
    print(f"--- 총 {len(reviews)}개의 리뷰를 찾았습니다. ---")

    # 리뷰를 하나씩 돌면서 'dict'로 만들고, '가게이름/주소'를 가져옴
    for r in reviews:
        nickname = r.select_one('span.pui__NMi-Dp')
        content = r.select_one('div.pui__vn15t2 > a')
        
        nickname = nickname.text.strip() if nickname else 'N/A'
        content = content.text.strip() if content else 'N/A'

        # 'dict' 생성
        review_dict = {
            'store_name': store_name, # 크롤링 초반에 가져온 'store_name' 변수
            'address': address,     # 크롤링 초반에 가져온 'address' 변수
            'nickname': nickname,
            'content': content
        }
        
        # 'dict'를 리스트에 추가
        review_data_list.append(review_dict)
    
    # 리스트를 DataFrame으로 변환
    df = pd.DataFrame(review_data_list)
    
    # CSV 파일로 바로 저장
    file_name = f'crawling_{store_name}'+'.csv'
    # encoding='utf-8-sig'로 해야 엑셀에서 한글 안 깨짐
    df.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print(f"--- {file_name} 크롤링 csv 저장 완료 ---")

except Exception as e:
    print(f"--- 전체 프로세스 중 에러 발생: {e} ---")
    if driver:
        driver.quit() 
        
    print(f"--- 에러 발생. 파일 저장 안 함 ---")