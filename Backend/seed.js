const mariadb = require('mariadb');
const fs = require('fs');
const csv = require('csv-parser');
const iconv = require('iconv-lite'); // 한글 깨짐 방지
require('dotenv').config();

const pool = mariadb.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  connectionLimit: 5
});

async function importScores() {
  const results = [];
  console.log("🚀 태그 점수 데이터 로딩 시작...");

  // CSV 파일 읽기 (spot_tag_scores.csv 사용)
  // 윈도우에서 작성된 CSV라면 euc-kr 인코딩 처리 필요할 수 있음
  fs.createReadStream('csv_/spot_tag_scores.csv') // 경로 확인 필요!
    .pipe(iconv.decodeStream('euc-kr')) // 한글 깨짐 발생 시 주석 해제
    .pipe(csv())
    .on('data', (data) => results.push(data))
    .on('end', async () => {
      let conn;
      try {
        conn = await pool.getConnection();
        
        // 1. TOUR_SPOT 테이블에서 이름과 ID 매핑 정보 가져오기
        // (이름으로 ID를 찾아야 하니까요)
        const spots = await conn.query("SELECT SPOT_ID, NAME FROM TOUR_SPOT");
        const spotMap = {};
        spots.forEach(s => spotMap[s.NAME] = s.SPOT_ID);

        console.log(`✅ DB에서 ${spots.length}개의 관광지 ID 정보를 가져왔습니다.`);

        // 2. CSV 데이터를 DB에 삽입 (기존 데이터 초기화 후 삽입)
        await conn.query("TRUNCATE TABLE SPOT_TAG_SCORES"); // 깔끔하게 비우고 시작
        
        let insertCount = 0;
        let skippedCount = 0;

        for (const row of results) {
          // CSV 컬럼명 확인 (BOM 문자 제거 등)
          const storeName = row['store_name'] || row['﻿store_name']; 
          const spotId = spotMap[storeName];

          if (!spotId) {
            // DB에 없는 장소면 스킵 (이름이 약간 다를 수 있음)
            // console.log(`⚠️ 스킵: '${storeName}' (DB에 SPOT_ID 없음)`);
            skippedCount++;
            continue;
          }

          // 컬럼들을 순회하며 태그 점수 저장 (store_name 제외)
          for (const [key, value] of Object.entries(row)) {
            if (key.includes('store_name')) continue;

            const tagName = key.replace('#', '').trim(); // '#' 제거
            const score = parseFloat(value);

            // 점수가 0보다 큰 유의미한 데이터만 저장
            if (score > 0) {
              await conn.query(
                "INSERT INTO SPOT_TAG_SCORES (SPOT_ID, TAG_NAME, SCORE) VALUES (?, ?, ?)",
                [spotId, tagName, score]
              );
              insertCount++;
            }
          }
        }
        
        console.log(`🎉 데이터 입력 완료!`);
        console.log(`- 저장된 점수 데이터: ${insertCount}건`);
        console.log(`- 매칭 실패로 스킵된 장소: ${skippedCount}곳`);

      } catch (err) {
        console.error("❌ 오류 발생:", err);
      } finally {
        if (conn) conn.end();
        process.exit();
      }
    });
}

importScores();