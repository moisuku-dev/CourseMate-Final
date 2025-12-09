const mariadb = require('mariadb');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const iconv = require('iconv-lite');
require('dotenv').config();

// 데이터 파일 경로 (server-api 기준)
const DATA_DIR = path.join(__dirname, '/data');

const pool = mariadb.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  connectionLimit: 5
});

// 파일 하나 읽어서 DB에 넣는 함수
function processFile(filePath, spotMap, conn) {
  return new Promise((resolve, reject) => {
    const records = [];
    
    // 1. CSV 읽기 (인코딩 대응)
    fs.createReadStream(filePath)
      //.pipe(iconv.decodeStream('euc-kr')) // 윈도우 한글 깨짐 방지
      .pipe(csv())
      .on('data', (data) => {
        // 컬럼명 대소문자/BOM 대응
        const storeName = data.store_name || data.STORE_NAME || data['﻿store_name'];
        const content = data.cleaned_content || data.CONTENT;
        
        // 필수 데이터가 있는 경우만 추가
        if (storeName && content) {
          records.push({
            storeName: storeName,
            nickname: data.nickname || data.NICKNAME || 'Anonymous',
            content: content,
            sentiment: data.sentiment || data.SENTIMENT || 'Neutral',
            scoreRaw: data.sentiment_score || data.SENTIMENT_SCORE || '0',
            keywords: data.tokenized_words || data.TOKENIZED_WORDS || '[]'
          });
        }
      })
      .on('end', async () => {
        try {
          let count = 0;
          // 2. DB 저장 (Batch 처리 권장하지만 여기선 심플하게 loop)
          for (const row of records) {
            const spotId = spotMap[row.storeName];
            
            // DB에 없는 장소면 저장 불가 -> 스킵
            if (!spotId) continue;

            // 점수 변환 ("99.5%" -> 0.995)
            let score = 0.0;
            if (row.scoreRaw) {
              let num = parseFloat(row.scoreRaw.replace('%', ''));
              if (num > 1.0) num = num / 100.0; // 100점 만점이면 1.0으로 정규화
              score = num;
            }

            // 감성 라벨 정리
            let sentimentVal = row.sentiment;
            if (sentimentVal.includes('Positive')) sentimentVal = 'Positive';
            else if (sentimentVal.includes('Negative')) sentimentVal = 'Negative';

            await conn.query(
              `INSERT INTO CRAWLED_REVIEW 
               (SPOT_ID, NICKNAME, CONTENT, SENTIMENT, SENTIMENT_SCORE, KEYWORDS) 
               VALUES (?, ?, ?, ?, ?, ?)`,
              [
                spotId,
                row.nickname,
                row.content,
                sentimentVal,
                score,
                row.keywords
              ]
            );
            count++;
          }
          resolve(count);
        } catch (err) {
          reject(err);
        }
      })
      .on('error', (err) => reject(err));
  });
}

async function importCrawledReviews() {
  let conn;
  try {
    console.log("🚀 크롤링 리뷰 데이터 적재 시작...");
    
    // 1. 파일 목록 확인
    if (!fs.existsSync(DATA_DIR)) {
      throw new Error(`데이터 폴더를 찾을 수 없습니다: ${DATA_DIR}`);
    }
    const files = fs.readdirSync(DATA_DIR).filter(file => file.startsWith('inputdata_') && file.endsWith('.csv'));
    console.log(`📂 총 ${files.length}개의 리뷰 파일 발견`);

    conn = await pool.getConnection();

    // 2. 장소 이름 -> ID 매핑 정보 로드 (속도 향상)
    const spots = await conn.query("SELECT SPOT_ID, NAME FROM TOUR_SPOT");
    const spotMap = {};
    spots.forEach(s => spotMap[s.NAME] = s.SPOT_ID);
    console.log(`✅ DB 매핑 준비 완료 (${spots.length}개 장소)`);

    // 3. 기존 데이터 초기화 (선택 사항: 중복 방지용)
    await conn.query("TRUNCATE TABLE CRAWLED_REVIEW"); 
    console.log("🧹 기존 테이블 초기화 완료");

    // 4. 파일별 처리 실행
    let totalInserted = 0;
    for (const file of files) {
      const filePath = path.join(DATA_DIR, file);
      // 진행 상황 로그
      // process.stdout.write(`processing ${file}... `);
      const inserted = await processFile(filePath, spotMap, conn);
      totalInserted += inserted;
      // console.log(`${inserted}건 저장`);
    }

    console.log(`\n🎉 모든 작업 완료! 총 ${totalInserted}개의 리뷰가 저장되었습니다.`);

  } catch (err) {
    console.error("❌ 오류 발생:", err);
  } finally {
    if (conn) conn.end();
    process.exit();
  }
}

importCrawledReviews();