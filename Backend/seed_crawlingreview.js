const mariadb = require('mariadb');
const { v4: uuidv4 } = require('uuid');

// ▼▼▼ 본인의 DB 정보로 꼭 수정하세요! ▼▼▼
const pool = require('./database');
// ▲▲▲ 수정 끝 ▲▲▲

async function migrateData() {
  let conn;
  try {
    conn = await pool.getConnection();
    console.log("DB 연결 성공. 마이그레이션 시작...");

    // 1. CRAWLED_REVIEW에서 스팟별로 5개씩 가져오기
    // 날짜 컬럼이 없으므로, 정렬 기준을 'CONTENT(내용)'로 임의 지정하여 에러를 방지합니다.
    // (어차피 날짜가 없으니 어떤 게 최신인지 모르므로 아무거나 5개 가져옵니다)
    const selectQuery = `
      SELECT * FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY SPOT_ID ORDER BY CONTENT) as rn
        FROM CRAWLED_REVIEW
      ) t
      WHERE t.rn <= 5
    `;

    const crawledRows = await conn.query(selectQuery);
    console.log(`총 ${crawledRows.length}개의 리뷰를 이동시킵니다.`);

    await conn.beginTransaction();

    for (const row of crawledRows) {
      // 2-1. 가짜 유저 생성 (봇 계정)
      const fakeUserId = 'bot_' + uuidv4().replace(/-/g, '').substring(0, 10);
      const fakeEmail = `${fakeUserId}@crawled.temp`;
      const nickname = row.NICKNAME || '익명';

      await conn.query(`
        INSERT INTO USER (USER_ID, NAME, EMAIL, PASSWORD, JOIN_DATE)
        VALUES (?, ?, ?, 'blocked', NOW())
      `, [fakeUserId, nickname, fakeEmail]);

      // 2-2. 리뷰 데이터 생성
      const reviewId = 'REV' + uuidv4().replace(/-/g, '').substring(0, 12);
      
      // ★ [수정됨] 원본 날짜가 없으므로 '현재 시간'을 넣습니다.
      const regDate = new Date(); 

      await conn.query(`
        INSERT INTO REVIEW (REVIEW_ID, USER_ID, SPOT_ID, CONTENT, RATING, REG_DATE)
        VALUES (?, ?, ?, ?, ?, ?)
      `, [
        reviewId,
        fakeUserId,
        row.SPOT_ID,
        row.CONTENT,
        row.RATING || 5, // 평점이 없으면 기본 5점
        regDate          // 현재 시간 입력
      ]);
    }

    await conn.commit();
    console.log("✅ 마이그레이션 성공! 모든 데이터가 현재 날짜로 이동되었습니다.");

  } catch (err) {
    if (conn) await conn.rollback();
    console.error("❌ 에러 발생:", err);
  } finally {
    if (conn) conn.end();
    process.exit();
  }
}

migrateData();