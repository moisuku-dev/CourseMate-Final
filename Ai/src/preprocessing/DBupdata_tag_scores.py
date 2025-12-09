import pandas as pd
import pymysql  # mariadb 대신 pymysql 사용
import sys
import os

# ==========================================
# 1. 설정 (DB 접속 정보)
# ==========================================
DB_CONFIG = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': '1234',
    'database': 'coursemate',
    'charset': 'utf8mb4' # 한글 깨짐 방지용 필수 설정
}

# CSV 파일 경로 (경로가 안 맞으면 절대 경로로 수정하세요)
CSV_FILE = r"Backend\csv_\spot_tag_scores.csv"

def update_scores():
    print("🚀 [시작] DB 태그 점수 최신화 작업을 시작합니다...")
    
    # 1. CSV 로드
    if not os.path.exists(CSV_FILE):
        print(f"❌ 오류: CSV 파일을 찾을 수 없습니다: {CSV_FILE}")
        return

    df_csv = pd.read_csv(CSV_FILE)
    print(f"✅ CSV 파일 로드 완료: {len(df_csv)}개 장소 데이터")

    conn = None
    try:
        # [변경 1] pymysql.connect 사용
        conn = pymysql.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # 2. 기존 테이블 초기화
        print("🛠️ 테이블 초기화 중...")
        cursor.execute("DROP TABLE IF EXISTS `spot_tag_scores`")
        cursor.execute("""
            CREATE TABLE `spot_tag_scores` (
                `ID` INT AUTO_INCREMENT PRIMARY KEY,
                `SPOT_ID` VARCHAR(50) NOT NULL,
                `TAG_NAME` VARCHAR(50) NOT NULL,
                `SCORE` FLOAT DEFAULT 0,
                INDEX `IDX_SPOT` (`SPOT_ID`),
                INDEX `IDX_TAG` (`TAG_NAME`)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb3;
        """)

        # 3. DB에서 관광지 정보 가져오기
        cursor.execute("SELECT SPOT_ID, NAME FROM tour_spot")
        rows = cursor.fetchall()
        
        name_to_id = {row[1]: row[0] for row in rows}
        print(f"✅ DB 관광지 목록 로드 완료: {len(name_to_id)}개 매핑 준비됨")

        # 4. 데이터 변환
        print("🔄 데이터 변환 및 DB 삽입 중...")
        insert_data = []
        matched_count = 0
        unmatched_list = []

        for index, row in df_csv.iterrows():
            store_name = row['store_name']
            
            if store_name in name_to_id:
                spot_id = name_to_id[store_name]
                matched_count += 1
                
                for tag in df_csv.columns:
                    if tag == 'store_name': continue
                    
                    score = row[tag]
                    if score > 0:
                        insert_data.append((spot_id, tag, float(score)))
            else:
                unmatched_list.append(store_name)

        # 5. 대량 삽입
        if insert_data:
            # [변경 2] 물음표(?) 대신 %s 사용해야 함! (매우 중요)
            sql = "INSERT INTO spot_tag_scores (SPOT_ID, TAG_NAME, SCORE) VALUES (%s, %s, %s)"
            
            cursor.executemany(sql, insert_data)
            conn.commit()
            
            print(f"🎉 성공! 총 {len(insert_data)}개의 태그 점수가 저장되었습니다.")
            print(f"   - 매칭된 관광지: {matched_count}개")
        else:
            print("⚠️ 저장할 데이터가 없습니다.")

    except pymysql.MySQLError as e: # [변경 3] 에러 처리 변경
        print(f"🔥 DB 에러 발생: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    update_scores()