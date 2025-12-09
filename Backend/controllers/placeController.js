const mariadb = require('mariadb');
require('dotenv').config();

// db_config 파일이 있다면 require로 불러오세요. 여기선 pool 직접 선언 예시.
const pool = mariadb.createPool({
  host: process.env.DB_HOST,
  port: process.env.DB_PORT,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_NAME,
  connectionLimit: 5
});

// 1. 관광지 목록 조회
exports.getPlaces = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    const { category, keyword, region } = req.query;

    let sql = "SELECT SPOT_ID, NAME, ADDRESS FROM TOUR_SPOT WHERE 1=1";
    let params = [];

    if (region) { sql += " AND ADDRESS LIKE ?"; params.push(`%${region}%`); }


    // 정렬: 평점이 없으므로 이름순 또는 ID순으로 변경
    sql += " ORDER BY NAME ASC LIMIT 100"; 

    const rows = await conn.query(sql, params);

    res.status(200).json({
      result_code: 200,
      result_msg: "관광지 목록 조회 성공",
      totalCount: rows.length,
      places: rows // 여기엔 별점이 없습니다. (목록 로딩 속도 위해 제외)
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 2. 관광지 상세 정보 조회 (✨ 평점 실시간 계산)
exports.getPlaceDetail = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    const { id } = req.params;

    // (1) 기본 정보 조회
    const placeRows = await conn.query(
      "SELECT SPOT_ID, NAME, ADDRESS FROM TOUR_SPOT WHERE SPOT_ID = ?", 
      [id]
    );

    if (placeRows.length === 0) {
      return res.status(404).json({ result_code: 404, result_msg: "존재하지 않는 관광지입니다." });
    }
    const place = placeRows[0];

    // (2) ✨ 실시간 평점 계산 (REVIEW 테이블)
    const ratingRow = await conn.query(
      "SELECT AVG(RATING) as avgRating, COUNT(*) as reviewCount FROM REVIEW WHERE SPOT_ID = ?", 
      [id]
    );
    // 평점이 없으면 0.0, 소수점 1자리까지
    place.avgRating = ratingRow[0].avgRating ? Number(ratingRow[0].avgRating).toFixed(1) : "0.0";
    place.reviewCount = Number(ratingRow[0].reviewCount);

    // (3) 사진 목록 조회
    const photoRows = await conn.query("SELECT IMG_URL FROM PHOTO WHERE SPOT_ID = ?", [id]);
    place.photos = photoRows.map(p => p.IMG_URL);

    // (4) 대표 태그 조회
    const tagRows = await conn.query(
      "SELECT TAG_NAME FROM SPOT_TAG_SCORES WHERE SPOT_ID = ? ORDER BY SCORE DESC LIMIT 5",
      [id]
    );
    place.topTags = tagRows.map(t => '#' + t.TAG_NAME);

    res.status(200).json({
      result_code: 200,
      result_msg: "관광지 상세 정보 조회 성공",
      place: place
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};