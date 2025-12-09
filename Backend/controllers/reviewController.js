const mariadb = require('mariadb');
require('dotenv').config();

const pool = require('../database');

// 1. 특정 관광지의 리뷰 조회 (비로그인도 가능)
exports.getReviewsByPlace = async (req, res) => {
  let conn;
  try {
    const { id } = req.params;
    conn = await pool.getConnection();

    const query = `
      SELECT r.REVIEW_ID, r.RATING, r.CONTENT, r.SENTIMENT, r.REG_DATE, u.NAME as nickname
      FROM REVIEW r
      JOIN USER u ON r.USER_ID = u.USER_ID
      WHERE r.SPOT_ID = ?
      ORDER BY r.REG_DATE DESC
    `;
    const rows = await conn.query(query, [id]);

    res.status(200).json({ result_code: 200, result_msg: "성공", reviews: rows });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 2. 리뷰 작성 (로그인 필수 ✨)
exports.createReview = async (req, res) => {
  let conn;
  try {
    // ✨ [수정] 토큰 ID 사용
    if (!req.user || !req.user.userId) {
      return res.status(401).json({ result_code: 401, result_msg: "로그인이 필요합니다." });
    }
    const userId = req.user.userId;

    // ★ [핵심 수정] body에 없으면 URL 파라미터(req.params.id)에서 가져옵니다.
    // 프론트엔드가 POST /places/:id/reviews 로 요청하기 때문입니다.
    const spotId = req.body.spotId || req.params.id; 
    
    const { rating, content } = req.body;
    
    // spotId가 여전히 없으면 에러 처리
    if (!spotId) {
        return res.status(400).json({ result_code: 400, result_msg: "관광지 ID(spotId)가 필요합니다." });
    }

    // 감성 분석 로직 (3점 이상 긍정, 미만 부정)
    const sentiment = rating >= 3 ? 'P' : 'N';
    const reviewId = 'REV' + Date.now();

    conn = await pool.getConnection();
    await conn.query(
      "INSERT INTO REVIEW (REVIEW_ID, USER_ID, SPOT_ID, RATING, CONTENT, SENTIMENT, REG_DATE) VALUES (?, ?, ?, ?, ?, ?, NOW())",
      [reviewId, userId, spotId, rating, content, sentiment]
    );

    res.status(200).json({ result_code: 200, result_msg: "리뷰 등록 성공" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 3. 리뷰 삭제 (본인만 가능 ✨)
exports.deleteReview = async (req, res) => {
  let conn;
  try {
    if (!req.user || !req.user.userId) {
      return res.status(401).json({ result_code: 401, result_msg: "로그인이 필요합니다." });
    }
    const userId = req.user.userId;
    const { reviewId } = req.params;

    conn = await pool.getConnection();
    // 본인이 쓴 글인지 확인 후 삭제
    const result = await conn.query("DELETE FROM REVIEW WHERE REVIEW_ID = ? AND USER_ID = ?", [reviewId, userId]);

    if (result.affectedRows === 0) {
      return res.status(403).json({ result_code: 403, result_msg: "권한이 없거나 존재하지 않는 리뷰입니다." });
    }

    res.status(200).json({ result_code: 200, result_msg: "리뷰 삭제 성공" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 4. 리뷰 수정 (PUT /api/reviews/:reviewId)
exports.updateReview = async (req, res) => {
  let conn;
  try {
    const { reviewId } = req.params;
    const { rating, content } = req.body;
    
    // 수정 시에도 평점에 따라 감성을 다시 계산
    const newRating = parseFloat(rating);
    const newSentiment = newRating >= 3 ? 'P' : 'N';

    conn = await pool.getConnection();

    // 내용, 평점, 감성 모두 업데이트
    const result = await conn.query(
      "UPDATE REVIEW SET RATING = ?, CONTENT = ?, SENTIMENT = ? WHERE REVIEW_ID = ?",
      [newRating, content, newSentiment, reviewId]
    );

    if (result.affectedRows === 0) {
      return res.status(404).json({ result_code: 404, result_msg: "리뷰가 존재하지 않습니다." });
    }

    res.status(200).json({
      result_code: 200,
      result_msg: "리뷰 수정 및 감성 업데이트 성공",
      sentiment: newSentiment
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 5. 관광지 사진 목록 조회 (GET /api/places/:id/photos)
exports.getPlacePhotos = async (req, res) => {
  let conn;
  try {
    const { id } = req.params; // spotId
    conn = await pool.getConnection();

    // 경량화된 PHOTO 테이블 조회 (PHOTO_ID, IMG_URL)
    const query = "SELECT PHOTO_ID, IMG_URL FROM PHOTO WHERE SPOT_ID = ?";
    const rows = await conn.query(query, [id]);

    res.status(200).json({
      result_code: 200,
      result_msg: "사진 목록 조회 성공",
      photos: rows.map(row => ({
        photoId: row.PHOTO_ID,
        url: row.IMG_URL
      }))
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};