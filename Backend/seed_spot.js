const mariadb = require('mariadb');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const iconv = require('iconv-lite');
const axios = require('axios');
require('dotenv').config();

// ⚠️ 네이버 지도 API 키 설정 (없으면 좌표가 0.0으로 저장됩니다)
const NAVER_CLIENT_ID = 'C7jCvHz37Ft84mqqF8Gf';
const NAVER_CLIENT_SECRET = 'ujAnCZ1DAU';

// 데이터 파일이 있는 경로 (server-ai/data 폴더를 가리키도록 수정)
// 현재 파일 위치가 server-api/ 루트라고 가정할 때:
const DATA_DIR = path.join(__dirname, '/data');

const pool = mariadb.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  connectionLimit: 5
});

// 주소 -> 좌표 변환 함수 (Geocoding)
async function getGeocode(address) {
  if (!NAVER_CLIENT_ID || NAVER_CLIENT_ID === 'C7jCvHz37Ft84mqqF8Gf') {
    return { lat: 0.0, lng: 0.0 }; // 키가 없으면 기본값 반환
  }
  
  try {
    const url = `https://naveropenapi.apigw.ntruss.com/map-geocode/v2/geocode?query=${encodeURI(address)}`;
    const response = await axios.get(url, {
      headers: {
        'X-NCP-APIGW-API-KEY-ID': NAVER_CLIENT_ID,
        'X-NCP-APIGW-API-KEY': NAVER_CLIENT_SECRET
      }
    });

    if (response.data.addresses.length > 0) {
      return {
        lat: parseFloat(response.data.addresses[0].y),
        lng: parseFloat(response.data.addresses[0].x)
      };
    }
  } catch (error) {
    console.error(`❌ 지오코딩 실패 (${address}):`, error.message);
  }
  return { lat: 0.0, lng: 0.0 };
}

// 개별 CSV 파일에서 첫 번째 줄(주소 정보)만 읽는 함수
// 기존 readPlaceInfo 함수를 이걸로 덮어쓰세요
function readPlaceInfo(filePath) {
  return new Promise((resolve, reject) => {
    let placeInfo = null;
    let isFirstRow = true;

    // ✨ 수정: iconv 제거 (UTF-8로 읽기)
    fs.createReadStream(filePath)
      .pipe(csv())
      .on('data', (data) => {
        // 디버깅: 첫 번째 파일의 첫 번째 줄만 로그로 출력해 봄
        if (isFirstRow && !placeInfo) {
           // console.log(`[Debug] 읽은 데이터 예시:`, data); // 필요시 주석 해제
           isFirstRow = false;
        }

        if (!placeInfo) {
          // 컬럼명 확인 (대소문자, 공백 제거 등 유연하게 처리)
          const name = data.store_name || data.STORE_NAME || data['﻿store_name']; // BOM 문자 대응
          const address = data.address || data.ADDRESS;

          if (name && address) {
            placeInfo = { name, address };
          }
        }
      })
      .on('end', () => resolve(placeInfo))
      .on('error', (err) => reject(err));
  });
}
async function importTourSpots() {
  let conn;
  try {
    console.log("🚀 관광지 데이터 마이그레이션 시작...");
    
    if (!fs.existsSync(DATA_DIR)) {
      throw new Error(`데이터 폴더를 찾을 수 없습니다: ${DATA_DIR}`);
    }
    const files = fs.readdirSync(DATA_DIR).filter(file => file.startsWith('inputdata_') && file.endsWith('.csv'));
    console.log(`📂 총 ${files.length}개의 데이터 파일을 발견했습니다.`);

    conn = await pool.getConnection();
    
    // 🔥 [수정 포인트 1] 반복문 시작 전에 현재 DB에 있는 총 개수를 딱 한 번만 가져옵니다.
    // 이렇게 "기준점"을 잡아둬야 중복 더하기가 발생하지 않습니다.
    const initialRows = await conn.query("SELECT COUNT(*) as cnt FROM TOUR_SPOT");
    const initialCount = Number(initialRows[0].cnt);

    let count = 0; // 새로 추가되는 개수 카운트

    for (const file of files) {
      const filePath = path.join(DATA_DIR, file);
      const info = await readPlaceInfo(filePath);
      
      if (info && info.name && info.address) {
        // 중복 확인
        const exists = await conn.query("SELECT SPOT_ID FROM TOUR_SPOT WHERE NAME = ?", [info.name]);
        if (exists.length > 0) {
          console.log(`⏩ 스킵: ${info.name} (이미 존재함)`);
          continue;
        }

        // 좌표 변환
        const location = await getGeocode(info.address);
        
        // 🔥 [수정 포인트 2] 기준점(initialCount) + 현재순서(count) + 1
        // 예: 기존 0개일 때 -> 0 + 0 + 1 = SPOT001
        // 예: 기존 0개일 때 -> 0 + 1 + 1 = SPOT002 ... 순차적으로 생성됨
        const currentIdNum = initialCount + count + 1;
        const nextId = `SPOT${String(currentIdNum).padStart(3, '0')}`;

        // DB 저장
        await conn.query(
          `INSERT INTO TOUR_SPOT (SPOT_ID, NAME, ADDRESS) 
           VALUES (?, ?, ?)`, // ⚠️ 주의: location.lat, lng 컬럼이 DB에 있다면 추가해야 함 (현재 쿼리엔 빠져있음)
          [
            nextId,
            info.name,
            info.address,
            // location.lat, location.lng (필요시 추가)
          ]
        );
        
        console.log(`✅ 저장 완료: [${nextId}] ${info.name}`);
        count++; // 카운트 증가
      }
    }

    console.log(`🎉 총 ${count}개의 새로운 관광지 정보를 TOUR_SPOT 테이블에 저장했습니다!`);

  } catch (err) {
    console.error("❌ 오류 발생:", err);
  } finally {
    if (conn) conn.end();
  }
}

importTourSpots();