const mariadb = require('mariadb');
const fs = require('fs');
const path = require('path');
require('dotenv').config();

// 사진이 들어있는 폴더 경로
const UPLOADS_DIR = path.join(__dirname, 'images');

const pool = mariadb.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  connectionLimit: 5
});

async function importPhotos() {
  let conn;
  try {
    console.log("📸 사진 데이터 검증 및 DB 연결 시작...");

    if (!fs.existsSync(UPLOADS_DIR)) {
      throw new Error(`uploads 폴더가 없습니다: ${UPLOADS_DIR}`);
    }

    // 1. 파일 목록 가져오기
    const files = fs.readdirSync(UPLOADS_DIR);
    console.log(`📂 폴더에서 총 ${files.length}개의 파일을 발견했습니다.`);

    conn = await pool.getConnection();

    // 2. DB에서 모든 관광지 정보 가져오기
    const spots = await conn.query("SELECT SPOT_ID, NAME FROM TOUR_SPOT");
    const spotMap = {};
    const allSpotNames = new Set(); // 전체 장소 이름 집합
    
    spots.forEach(s => {
      spotMap[s.NAME] = s.SPOT_ID;
      allSpotNames.add(s.NAME);
    });
    console.log(`✅ DB에서 총 ${spots.length}개의 관광지 정보를 가져왔습니다.`);

    // 추적용 집합 (Set)
    const matchedFiles = new Set(); // DB에 들어간 파일들
    const matchedSpots = new Set(); // 사진이 등록된 장소들

    let successCount = 0;

    // 3. 매칭 및 저장 시작
    for (const file of files) {
      // 파일명에서 확장자 제거 (예: "서울랜드.jpg" -> "서울랜드")
      const name = path.parse(file).name; 
      const spotId = spotMap[name];

      if (spotId) {
        // 이미 등록된 사진인지 확인 (중복 방지 - 필요시 주석 해제)
        // const exist = await conn.query("SELECT 1 FROM PHOTO WHERE IMG_URL = ?", [`/uploads/${file}`]);
        // if (exist.length > 0) { matchedFiles.add(file); matchedSpots.add(name); continue; }

        const photoId = 'P_' + spotId + '_' + Date.now(); 
        const imgUrl = `/images/${file}`; 
        
        // PHOTO 테이블 저장 (3개 컬럼 버전)
        await conn.query(
          `INSERT INTO PHOTO (PHOTO_ID, SPOT_ID, IMG_URL) 
           VALUES (?, ?, ?)`,
          [photoId, spotId, imgUrl]
        );
        
        // 성공 목록에 추가
        matchedFiles.add(file);
        matchedSpots.add(name);
        successCount++;
      }
    }

    console.log(`\n🎉 작업 완료! 총 ${successCount}건 연결 성공.`);
    console.log("====================================================");

    // 4. [리포트 A] 파일은 있는데 DB에 못 들어간 사진들 (이름 불일치)
    const unmatchedFiles = files.filter(f => !matchedFiles.has(f));
    
    if (unmatchedFiles.length > 0) {
      console.log(`⚠️ [파일O -> DBX] 이름이 안 맞아서 못 올린 사진 (${unmatchedFiles.length}개):`);
      unmatchedFiles.forEach(f => console.log(`   - ${f}`));
      console.log("   👉 팁: 파일명을 DB 장소명과 똑같이(띄어쓰기 포함) 수정하세요.");
    } else {
      console.log("✅ 모든 사진 파일이 DB와 완벽하게 매칭되었습니다!");
    }

    console.log("----------------------------------------------------");

    // 5. [리포트 B] DB에는 있는데 사진이 없는 장소들 (사진 누락)
    const missingSpots = spots.filter(s => !matchedSpots.has(s.NAME)).map(s => s.NAME);
    
    if (missingSpots.length > 0) {
      console.log(`⚠️ [DB O -> 사진X] 사진이 아직 없는 관광지 (${missingSpots.length}곳):`);
      missingSpots.forEach(s => console.log(`   - ${s}`));
      console.log("   👉 팁: 이 장소들의 사진을 구해서 uploads 폴더에 넣어주세요.");
    } else {
      console.log("✅ 모든 관광지에 사진이 등록되었습니다!");
    }
    console.log("====================================================");

  } catch (err) {
    console.error("❌ 오류 발생:", err);
  } finally {
    if (conn) conn.end();
  }
}

importPhotos();