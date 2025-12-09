import pandas as pd
import glob
import os
import sys
import time

# ==========================================================
# 설정: 합칠 파일들의 패턴
# ==========================================================
FILE_PATTERN = "Ai/data/inputdata_*.csv"   # 파일 디렉토리는 각자 맞춰서 넣기
OUTPUT_FILE = "Ai/data/total_merged_data.csv" # 결과 파일 이름

# ==========================================================
# 메인 로직
# ==========================================================
print("CSV 병합기")


# 1. 파일 찾기
# 현재 폴더에서 패턴에 맞는 파일 검색
all_files = glob.glob(FILE_PATTERN)

file_count = len(all_files)
print(f"감지된 파일 개수: {file_count}개")

if file_count == 0:
    print("'inputdata_*.csv' 파일을 찾을 수 없습니다.")
    print("-> 이 코드를 csv 파일들이 있는 폴더에 넣고 실행해주세요.")
    sys.exit()

# 2. 하나씩 읽어서 리스트에 담기
print("파일 병합을 시작합니다...")
start_time = time.time()

df_list = []

for filename in all_files:
    try:
        # 파일 읽기 (인코딩 에러 방지용 utf-8-sig)
        df = pd.read_csv(filename, encoding='utf-8-sig')
        
        # 데이터가 비어있지 않은 경우에만 추가
        if not df.empty:
            # (옵션) 어떤 가게 데이터인지 구분이 필요할까봐 파일명 컬럼 추가 (필요 없으면 주석 처리)
            # df['source_file'] = os.path.basename(filename) 
            df_list.append(df)
            
    except Exception as e:
        print(f"[Skip] 파일 읽기 실패: {filename} ({e})")

# 3. 합치기 (Concatenate)
if df_list:
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 4. 저장하기
    combined_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    end_time = time.time()
    print(f"병합 완료!")
    print(f"총 합쳐진 리뷰 개수: {len(combined_df)}개")
    print(f"저장된 파일: {OUTPUT_FILE}")
    print(f"소요 시간: {end_time - start_time:.2f}초")
    print("--------------------------------------------------")
    
    # AI 팀원이 만든 tags.json 등과 연동하기 위해 컬럼명 체크
    print("[참고] 합쳐진 데이터 컬럼:", list(combined_df.columns))

else:
    print("합칠 데이터가 없습니다.")