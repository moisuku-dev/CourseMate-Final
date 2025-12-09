# -*- coding: utf-8 -*-
import pandas as pd
import os

# ==========================================
# 설정
# ==========================================
INPUT_FILE = 'spot_tag_scores.csv'       # 원본 파일 (장소별 모든 태그 점수)
OUTPUT_FILE = 'top2_tags_summary.csv'    # 결과 파일 (상위 2개 태그)

def extract_top2():
    # 1. 파일 확인
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 오류: '{INPUT_FILE}' 파일이 없습니다.")
        return

    print(f"📂 '{INPUT_FILE}' 로딩 중...")
    df = pd.read_csv(INPUT_FILE, encoding="cp949")


    # 2. 상위 2개 태그 추출 로직
    top2_data = []

    print("🚀 태그 분석 및 추출 중...")
    for index, row in df.iterrows():
        store = row['store_name']
        
        # store_name을 제외한 나머지 컬럼(태그들)을 딕셔너리로 변환
        scores = row.drop('store_name').to_dict()
        
        # 점수 높은 순서대로 정렬 (내림차순)
        sorted_tags = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # 상위 2개 뽑기 (데이터가 부족할 경우 대비)
        first_tag = sorted_tags[0] if len(sorted_tags) > 0 else ("없음", 0.0)
        second_tag = sorted_tags[1] if len(sorted_tags) > 1 else ("없음", 0.0)

        top2_data.append({
            'store_name': store,
            '1st_tag': first_tag[0],
            '1st_score': round(first_tag[1], 2), # 소수점 2자리 반올림
            '2nd_tag': second_tag[0],
            '2nd_score': round(second_tag[1], 2)
        })

    # 3. 데이터프레임 생성 및 저장
    result_df = pd.DataFrame(top2_data)
    result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print("-" * 30)
    print(f"🎉 변환 완료! '{OUTPUT_FILE}' 파일이 생성되었습니다.")
    print("-" * 30)
    print(result_df.head()) # 결과 미리보기

if __name__ == "__main__":
    extract_top2()