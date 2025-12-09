const mariadb = require('mariadb');
require('dotenv').config();

const pool = require('../database');

// 1. 관리자 로그인 (POST /api/admin/login) 
exports.adminLogin = async (req, res) => {
  const { adminId, password } = req.body;
  let conn;
  try {
    conn = await pool.getConnection();
    
    // DB에서 해당 ID의 관리자 정보 조회
    const rows = await conn.query("SELECT * FROM admin WHERE ADMIN_ID = ?", [adminId]);
    
    // 1. ID가 DB에 없는 경우
    if (rows.length === 0) {
      return res.status(200).json({ result_code: 101, result_msg: "존재하지 않는 ID입니다." });
    }

    // 2. 비밀번호 비교 (단순 텍스트 비교)
    if (rows[0].PASSWORD !== password) {
      return res.status(200).json({ result_code: 101, result_msg: "비밀번호가 틀렸습니다." });
    }

    // 3. 로그인 성공! (토큰 발급)
    res.status(200).json({
      result_code: 200,
      result_msg: "로그인 성공",
      token: "admin_token_" + rows[0].ADMIN_ID, // 임시 토큰
      name: rows[0].NAME
    });

  } catch (err) {
    console.error("🔥 로그인 에러:", err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 2. 대시보드 통계 조회 (GET /api/admin/dashboard) [cite: 240]
exports.getDashboardStats = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    const userCount = await conn.query("SELECT COUNT(*) as cnt FROM USER");
    const reviewCount = await conn.query("SELECT COUNT(*) as cnt FROM REVIEW");
    const spotCount = await conn.query("SELECT COUNT(*) as cnt FROM TOUR_SPOT");

    res.status(200).json({
      result_code: 200,
      result_msg: "통계 조회 성공",
      stats: {
        totalUsers: Number(userCount[0].cnt),
        totalReviews: Number(reviewCount[0].cnt),
        totalSpots: Number(spotCount[0].cnt)
      }
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 3. 회원 목록 조회 (GET /api/admin/users) [cite: 301]
exports.getAllUsers = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    const rows = await conn.query("SELECT USER_ID, NAME, EMAIL, IS_ACTIVE, JOIN_DATE FROM USER ORDER BY JOIN_DATE DESC");
    res.status(200).json({ result_code: 200, users: rows });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 4. 회원 상태 변경(정지/해제) (PUT /api/admin/users/:userId/status) [cite: 301]
exports.changeUserStatus = async (req, res) => {
  let conn;
  try {
    const { userId } = req.params;
    const { isActive } = req.body; // 'Y' or 'N'
    conn = await pool.getConnection();
    await conn.query("UPDATE USER SET IS_ACTIVE = ? WHERE USER_ID = ?", [isActive, userId]);
    res.status(200).json({ result_code: 200, result_msg: "회원 상태 변경 성공" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 5. 관광지 등록 (POST /api/admin/places) [cite: 305]
exports.createPlace = async (req, res) => {
  let conn;
  try {
    // 1. 프론트엔드에서 보낸 값 받기
    const { spotId, name, address, category, latitude, longitude } = req.body;
    
    // 2. 빈 값("")이나 undefined가 오면 NULL로 변환하는 안전장치
    // (JS에서 빈 문자열 ""은 false로 취급되므로, 삼항 연산자로 쉽게 처리 가능)
    const safeAddress = address && address.trim() !== "" ? address : null;
    const safeCategory = category && category.trim() !== "" ? category : null;
    
    // 위도/경도는 숫자가 0일 수도 있으니, 빈 문자열("")이거나 null일 때만 null로 처리
    const safeLat = (latitude === "" || latitude === null || latitude === undefined) ? null : latitude;
    const safeLon = (longitude === "" || longitude === null || longitude === undefined) ? null : longitude;

    conn = await pool.getConnection();
    
    // 3. DB에 NULL로 저장
    await conn.query(
      "INSERT INTO TOUR_SPOT (SPOT_ID, NAME, ADDRESS, CATEGORY, LATITUDE, LONGITUDE) VALUES (?, ?, ?, ?, ?, ?)",
      [spotId, name, safeAddress, safeCategory, safeLat, safeLon]
    );

    res.status(200).json({ result_code: 200, result_msg: "관광지 등록 성공" });

  } catch (err) {
    console.error("🔥 관광지 등록 에러:", err);
    res.status(500).json({ result_code: 500, result_msg: "DB 저장 실패" });
  } finally {
    if (conn) conn.end();
  }
};

// 6. 관광지 삭제 (DELETE /api/admin/places/:id) [cite: 305]
exports.deletePlace = async (req, res) => {
  let conn;
  try {
    const { id } = req.params;
    conn = await pool.getConnection();
    await conn.query("DELETE FROM TOUR_SPOT WHERE SPOT_ID = ?", [id]);
    res.status(200).json({ result_code: 200, result_msg: "관광지 삭제 성공" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 7. 전체 리뷰 조회 (GET /api/admin/reviews) [cite: 308]
exports.getAllReviews = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    // 어떤 유저가 어디에 썼는지 알기 위해 JOIN
    const query = `
      SELECT r.REVIEW_ID, u.NAME as writer, ts.NAME as spotName, r.CONTENT, r.RATING, r.REG_DATE 
      FROM REVIEW r
      JOIN USER u ON r.USER_ID = u.USER_ID
      JOIN TOUR_SPOT ts ON r.SPOT_ID = ts.SPOT_ID
      ORDER BY r.REG_DATE DESC
    `;
    const rows = await conn.query(query);
    res.status(200).json({ result_code: 200, reviews: rows });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 8. 리뷰 삭제 (DELETE /api/admin/reviews/:reviewId) [cite: 308]
exports.deleteReviewAdmin = async (req, res) => {
  // 로직은 일반 리뷰 삭제와 같지만, 관리자 권한으로 수행한다는 점이 다름
  let conn;
  try {
    const { reviewId } = req.params;
    conn = await pool.getConnection();
    await conn.query("DELETE FROM REVIEW WHERE REVIEW_ID = ?", [reviewId]);
    res.status(200).json({ result_code: 200, result_msg: "관리자 권한으로 리뷰 삭제 성공" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// 9. 문의 답변 등록 (POST /api/admin/inquiries/:id/answer) [cite: 318]
exports.answerInquiry = async (req, res) => {
  let conn;
  try {
    const { id } = req.params;
    const { answerContent } = req.body;
    conn = await pool.getConnection();
    
    // 답변 내용 업데이트 및 상태를 '완료'로 변경
    await conn.query(
      "UPDATE INQUIRY SET ANSWER_CONTENT = ?, ANSWER_DATE = NOW(), STATUS = '완료' WHERE INQUIRY_ID = ?",
      [answerContent, id]
    );
    res.status(200).json({ result_code: 200, result_msg: "답변 등록 성공" });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.end();
  }
};

// ... (기존 코드 아래에 추가)

// 10. [관리자용] 전체 관광지 목록 조회
exports.getAllPlaces = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    
    // 필요한 정보만 선택해서 조회 (ID, 이름, 주소, 카테고리 등)
    const query = `
      SELECT SPOT_ID, NAME, ADDRESS, CATEGORY
      FROM TOUR_SPOT 
      ORDER BY NAME ASC
    `;
    const rows = await conn.query(query);

    res.status(200).json({
      result_code: 200,
      result_msg: "관광지 목록 조회 성공",
      places: rows.map(row => ({
        id: row.SPOT_ID,
        name: row.NAME,
        address: row.ADDRESS,
        category: row.CATEGORY
      }))
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// ============================================================
// 11. 관리자 설정 조회 (GET /api/admin/settings)
// ============================================================
exports.getAdminSettings = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    const rows = await conn.query("SELECT * FROM admin_settings");
    
    // DB 데이터를 JSON 객체로 변환
    const settings = {};
    rows.forEach(r => { settings[r.SETTING_KEY] = r.SETTING_VALUE; });
    
    // 프론트엔드 호환성을 위해 형변환 (String -> Number/Boolean)
    if(settings.loginFailedLimit) settings.loginFailedLimit = Number(settings.loginFailedLimit);
    if(settings.lockMinutes) settings.lockMinutes = Number(settings.lockMinutes);
    if(settings.allowNewAdmins) settings.allowNewAdmins = (settings.allowNewAdmins === 'true');

    res.status(200).json({ result_code: 200, settings });
  } catch (err) {
    console.error("🔥 설정 조회 에러:", err);
    res.status(200).json({ result_code: 200, settings: {} }); // 에러 나도 빈 객체 반환해서 멈춤 방지
  } finally {
    if (conn) conn.end();
  }
};

// ============================================================
// 12. 관리자 설정 저장 (PUT /api/admin/settings)
// ============================================================
exports.updateAdminSettings = async (req, res) => {
  let conn;
  try {
    const payload = req.body; // { loginFailedLimit: 5, ... }
    conn = await pool.getConnection();
    
    // 들어온 설정값들을 하나씩 DB에 저장 (없으면 넣고, 있으면 수정)
    for (const [key, value] of Object.entries(payload)) {
      const strVal = String(value); // 안전하게 문자열로 변환
      await conn.query(
        "INSERT INTO admin_settings (SETTING_KEY, SETTING_VALUE) VALUES (?, ?) ON DUPLICATE KEY UPDATE SETTING_VALUE = ?",
        [key, strVal, strVal]
      );
    }
    
    // ★ 응답 필수 (이게 없으면 프론트가 멈춤)
    res.status(200).json({ result_code: 200, result_msg: "설정이 저장되었습니다." });
  } catch (err) {
    console.error("🔥 설정 저장 에러:", err);
    res.status(500).json({ result_code: 500, result_msg: "저장 실패" });
  } finally {
    if (conn) conn.end();
  }
};

// ★ [신규 추가] 11. 문의 상세 조회 (GET /api/admin/inquiries/:id)
exports.getInquiryDetail = async (req, res) => {
  let conn;
  try {
    const { id } = req.params;
    conn = await pool.getConnection();

    // 문의 내용과 작성자 정보를 함께 조회
    const query = `
      SELECT 
        i.INQUIRY_ID, i.TITLE, i.CONTENT, i.STATUS, i.REG_DATE, 
        i.ANSWER_CONTENT, i.ANSWER_DATE,
        u.NAME as writerName, u.EMAIL as writerEmail
      FROM INQUIRY i
      LEFT JOIN USER u ON i.USER_ID = u.USER_ID
      WHERE i.INQUIRY_ID = ?
    `;
    const rows = await conn.query(query, [id]);

    if (rows.length === 0) {
      return res.status(404).json({ result_code: 404, result_msg: "해당 문의를 찾을 수 없습니다." });
    }

    // 프론트엔드에서 쓰기 편하게 소문자로 변환해서 응답
    const item = rows[0];
    const inquiry = {
      id: item.INQUIRY_ID,
      title: item.TITLE,
      content: item.CONTENT,
      status: item.STATUS,
      regDate: item.REG_DATE,
      answerContent: item.ANSWER_CONTENT,
      answerDate: item.ANSWER_DATE,
      writerName: item.writerName,
      writerEmail: item.writerEmail,
    };

    res.status(200).json({ result_code: 200, inquiry });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// [신규 추가] 13. 사용자 문의 삭제 (DELETE /api/admin/inquiries/:id)
exports.deleteInquiry = async (req, res) => {
  let conn;
  try {
    const { id } = req.params;
    conn = await pool.getConnection();

    // 문의글 삭제 (DB)
    const result = await conn.query("DELETE FROM INQUIRY WHERE INQUIRY_ID = ?", [id]);

    if (result.affectedRows === 0) {
      return res.status(404).json({ result_code: 404, result_msg: "삭제할 문의가 없습니다." });
    }

    res.status(200).json({ result_code: 200, result_msg: "문의가 삭제되었습니다." });
  } catch (err) {
    console.error("🔥 문의 삭제 에러:", err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};