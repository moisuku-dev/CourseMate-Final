import pandas as pd
import re
from konlpy.tag import Okt  
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline
import warnings
import time
import os
import sys 
from tqdm import tqdm
tqdm.pandas() 

# --- 환경 설정 ---
warnings.filterwarnings("ignore")
# 1. 입력 파일 경로 받기
while True:
    input_path = input("처리할 크롤링 CSV 파일 경로를 입력하세요: ").strip()
    # 윈도우에서 '경로 복사'하면 생기는 따옴표 제거
    input_path = input_path.replace('"', '').replace("'", "")
    if os.path.exists(input_path):
        INPUT_FILE = input_path
        break
    else:
        print("파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        # 다시 입력받기 위해 루프

# 2. 경로에서 '가게 이름' 추출하는 로직
file_name_with_ext = os.path.basename(INPUT_FILE) # 파일명만 가져오기 
file_name_no_ext = os.path.splitext(file_name_with_ext)[0] # 확장자 제거 
store_name = file_name_no_ext.replace("crawling_", "") # 접두사 제거 
# 출력 파일 이름 설정
OUTPUT_FILE = f"inputdata_{store_name}.csv"


# --- 기본 함수 정의 ---
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9\s.,?!]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text, okt_instance):
    if not text: return []
    try:
        pos = okt_instance.pos(text, stem=True)
        keywords = [word for word, tag in pos if tag in ['Noun', 'Adjective'] and len(word) > 1]
        return keywords
    except Exception as e:
        return []

def get_sentiment(text, sentiment_pipeline):
    """(작업 3) 감성 분석"""
    if not text: return 'Neutral', 0.0
    
    # 512 토큰 길이 제한 처리
    truncated_text = text[:510]
    
    try:
        result = sentiment_pipeline(truncated_text)[0]
        res_label = result['label']
        
        # LABEL_1 또는 POSITIVE면 긍정, 아니면 부정
        if res_label == 'POSITIVE' or res_label == 'LABEL_1':
            label = 'Positive'
        else:
            label = 'Negative'
            
        score = result['score']
        return label, score
    except Exception as e:
        print(f"감성 분석 에러: {e}")
        return 'Error', 0.0

def get_top_keywords(corpus, vectorizer):
    if not corpus: return pd.DataFrame(columns=['keyword', 'score'])
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()
        total_scores = tfidf_matrix.sum(axis=0).A1
        keyword_df = pd.DataFrame({'keyword': feature_names, 'score': total_scores})
        return keyword_df.sort_values(by='score', ascending=False)
    except Exception as e:
        print(f"  - TF-IDF 에러: {e}")
        return pd.DataFrame(columns=['keyword', 'score'])

# --- 메인 코드 실행 ---
print(f"--- 데이터 전처리 및 분석 시작 (입력: {INPUT_FILE}) ---")
start_time = time.time()

# 1. AI 모델 로드
print("[1/5] 모델 로딩: KoNLPy(Okt) 로딩 중...")

try:
    okt = Okt() 
except Exception as e:
    print(f"  - [치명적 에러] Okt() 로딩 실패. 'JAVA_HOME'과 'Path' 설정을 다시 확인하고 PC를 재부팅하세요. | {e}")
    sys.exit() 


print("[1/5] 모델 로딩: BERT 감성 분석 모델 로딩 중 ...")
sentiment_pipeline = pipeline("sentiment-analysis", model="matthewburke/korean_sentiment")

# 2. 데이터 로드
print(f"[2/5] 데이터 로딩: {INPUT_FILE} CSV 읽는 중...")
try:
    df = pd.read_csv(INPUT_FILE) 
    
    df = df.dropna(subset=['content'])
    review_count = len(df)
    print(f"  - 총 {review_count}개의 리뷰를 찾았습니다.")

except FileNotFoundError:
    print(f"  - 에러: {INPUT_FILE}을 찾을 수 없습니다.")
    sys.exit()
except Exception as e:
    print(f"  - 에러: 파일을 읽는 중 문제가 발생했습니다. (CSV 파일 구조 확인) | {e}")
    sys.exit()

# 3. 4가지 작업 순차 실행
print("[3/5] (작업 1) 텍스트 정제 실행 중...")
df['cleaned_content'] = df['content'].apply(clean_text)

print("[3/5] (작업 2) 형태소 분석 실행 중...")
df['tokenized_words'] = df['cleaned_content'].progress_apply(lambda x: tokenize_text(x, okt))

print("[3/5] (작업 3) 감성 분석 실행 중...")
df[['sentiment', 'sentiment_score']] = df['cleaned_content'].progress_apply(lambda x: pd.Series(get_sentiment(x, sentiment_pipeline)))

# 4. (작업 4) 핵심 특징 추출 (TF-IDF)
print("[4/5] (작업 4) 핵심 키워드 추출 실행 중 (TF-IDF)...")
positive_reviews = df[df['sentiment'] == 'Positive']
print(f"  - 긍정 리뷰 {len(positive_reviews)}개 / 전체 {review_count}개")

positive_corpus = positive_reviews['tokenized_words'].apply(lambda x: ' '.join(x))
tfidf_vect = TfidfVectorizer(min_df=2) 
top_keywords_df = get_top_keywords(positive_corpus.tolist(), tfidf_vect)

print("\n--- [결과] Top 20 긍정 핵심 키워드 ---")
print(top_keywords_df.head(20).to_string(index=False))
print("--------------------------------------")

# 5. 결과 저장
print(f"[5/5] 결과 저장: {OUTPUT_FILE} 파일 생성 중...")
output_columns = [
    'store_name',
    'address',
    'nickname', 
    'content',            # 원본 리뷰
    'cleaned_content',    # (작업 1) 정제된 리뷰
    'tokenized_words',    # (작업 2) 키워드 리스트
    'sentiment',          # (작업 3) 긍정/부정
    'sentiment_score'     # (작업 3) 긍정/부정 점수
]
df_final = df[output_columns]
df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig') 

# --- (4/4) 완료 ---
end_time = time.time()
print(f"\n--- 작업 완료 (총 {end_time - start_time:.2f}초) ---")
print(f"  - 개별 리뷰 분석 결과: {OUTPUT_FILE}에 저장됨")
print(f"  - 관광지 핵심 키워드: 위 터미널에 출력됨")