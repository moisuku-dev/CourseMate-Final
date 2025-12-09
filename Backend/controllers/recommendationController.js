const mariadb = require('mariadb');
require('dotenv').config();

const pool = require('../database');

// 🤖 [Core Logic] 하이브리드 추천 엔진
async function mockAIRecommendation(conn, userTags, region, excludeIds = []) {
  console.log(`[Simple Hybrid Recommender] 지역: ${region}, 태그: ${userTags}`);

  if (!userTags || userTags.length === 0) {
    userTags = ['좋다', '추천', '만족']; 
  }

  // 제외할 ID 조건 생성
  let excludeCondition = "";
  if (excludeIds.length > 0) {
    const idsString = excludeIds.map(id => `'${id}'`).join(",");
    excludeCondition = `AND main.SPOT_ID NOT IN (${idsString})`;
  }

  // 태그 검색 조건 생성
  const crawlConditions = userTags.map(tag => `c.KEYWORDS LIKE '%${tag.replace('#', '')}%'`).join(' OR ');
  const userConditions = userTags.map(tag => `r.CONTENT LIKE '%${tag.replace('#', '')}%'`).join(' OR ');

  // ✅ [수정된 쿼리] 쉼표 제거 완료 & 문법 오류 수정됨
  const query = `
    SELECT 
      main.SPOT_ID, main.NAME, main.ADDRESS,
      (
        (SELECT COUNT(*) FROM CRAWLED_REVIEW c WHERE c.SPOT_ID = main.SPOT_ID AND (${crawlConditions})) * 2 
        + 
        (SELECT COUNT(*) FROM REVIEW r WHERE r.SPOT_ID = main.SPOT_ID AND (${userConditions})) * 3
      ) as score
    FROM TOUR_SPOT main
    WHERE main.ADDRESS LIKE ? ${excludeCondition}
    ORDER BY score DESC
    LIMIT 5
  `;

  // ✅ [수정된 파라미터] item.spotId (x) -> %region% (o)
  const rows = await conn.query(query, [`%${region}%`]);

  return rows.map(row => {
    // BigInt 처리
    const scoreAsNumber = Number(row.score); 
    const logScore = Math.log(scoreAsNumber + 1); 
    const finalScore = 0.5 + (logScore * 0.1); 
    
    return {
      spotId: row.SPOT_ID.toString(), // BigInt -> String 변환
      spotName: row.NAME,
      address: row.ADDRESS,
      matchScore: Math.min(finalScore, 0.99).toFixed(2),
      features: ["#AI추천", "#취향저격"]
    };
  });
}

// 1. 추천 API 컨트롤러
exports.getRecommendations = async (req, res) => {
  let conn;
  try {
    // 로그인 체크
    if (!req.user || !req.user.userId) {
      return res.status(401).json({ result_code: 401, result_msg: "로그인이 필요합니다." });
    }
    
    const userId = req.user.userId;
    const region = req.query.region;
    
    // 지역 정보 체크
    if (!region) {
      return res.status(400).json({ result_code: 400, result_msg: "지역(region) 정보가 필요합니다." });
    }

    conn = await pool.getConnection();

    // 사용자 취향 태그 조회
    const prefRows = await conn.query(
      "SELECT t.TAG_NAME FROM USER_PREFERENCE up JOIN TAG t ON up.TAG_ID = t.TAG_ID WHERE up.USER_ID = ?", 
      [userId]
    );
    const userTags = prefRows.map(row => row.TAG_NAME.replace('#', ''));

    // 추천 함수 실행
    const course = await mockAIRecommendation(conn, userTags, region);

    if (course.length === 0) {
       return res.status(200).json({ result_code: 200, result_msg: "추천 결과가 없습니다.", course: [] });
    }

    res.status(200).json({
      result_code: 200,
      result_msg: "AI 코스 추천 성공",
      course: course
    });

  } catch (err) {
    console.error("추천 API 에러:", err); // 에러 로그 출력
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};