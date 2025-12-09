const mariadb = require('mariadb');
require('dotenv').config();

const pool = require('../database');

// 1. 공지사항 목록 조회 (로그인 불필요)
exports.getNotices = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();
    const rows = await conn.query("SELECT NOTICE_ID, TITLE, REG_DATE FROM NOTICE ORDER BY REG_DATE DESC");

    res.status(200).json({
      result_code: 200,
      result_msg: "공지사항 목록 조회 성공",
      notices: rows.map(row => ({
        id: row.NOTICE_ID,
        title: row.TITLE,
        regDate: row.REG_DATE
      }))
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 2. 공지사항 상세 조회
exports.getNoticeDetail = async (req, res) => {
  let conn;
  try {
    const { id } = req.params;
    conn = await pool.getConnection();
    const rows = await conn.query("SELECT * FROM NOTICE WHERE NOTICE_ID = ?", [id]);

    if (rows.length === 0) {
      return res.status(404).json({ result_code: 404, result_msg: "공지사항이 없습니다." });
    }

    // ★ [핵심] 대문자 DB 컬럼명을 소문자로 변환해서 응답
    const notice = rows[0];
    const formattedNotice = {
        id: notice.NOTICE_ID,
        title: notice.TITLE,
        content: notice.CONTENT,
        regDate: notice.REG_DATE
    };

    res.status(200).json({
      result_code: 200,
      result_msg: "상세 조회 성공",
      notice: formattedNotice // 변환된 데이터 전달
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 3. 문의/피드백 등록 (POST)
exports.createInquiry = async (req, res) => {
  let conn;
  try {
    // ✨ [수정] 토큰에서 ID 꺼내기
    if (!req.user || !req.user.userId) {
      return res.status(401).json({ result_code: 401, result_msg: "로그인이 필요합니다." });
    }
    const userId = req.user.userId;
    const { title, content } = req.body;

    const inquiryId = 'INQ' + Date.now();

    conn = await pool.getConnection();

    await conn.query(
      "INSERT INTO INQUIRY (INQUIRY_ID, USER_ID, TITLE, CONTENT, STATUS, REG_DATE) VALUES (?, ?, ?, ?, ?, NOW())",
      [inquiryId, userId, title, content, '대기']
    );

    res.status(200).json({
      result_code: 200,
      result_msg: "문의 등록 성공",
      inquiryId: inquiryId
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 4. 내 문의 내역 조회 (GET)
exports.getMyFeedbacks = async (req, res) => {
  let conn;
  try {
    // ✨ [수정] 토큰에서 ID 꺼내기
    if (!req.user || !req.user.userId) {
      return res.status(401).json({ result_code: 401, result_msg: "로그인이 필요합니다." });
    }
    const userId = req.user.userId;

    conn = await pool.getConnection();

    const rows = await conn.query(
      "SELECT INQUIRY_ID, TITLE, STATUS, REG_DATE, ANSWER_DATE, ANSWER_CONTENT FROM INQUIRY WHERE USER_ID = ? ORDER BY REG_DATE DESC",
      [userId]
    );

    res.status(200).json({
      result_code: 200,
      result_msg: "내 문의 조회 성공",
      feedbacks: rows
    });
  } catch (err) {
    console.error(err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    if (conn) conn.end();
  }
};

// 5. [관리자용] 전체 문의 조회 (이미 잘 구현되어 있을 경우 유지)
exports.getAllInquiries = async (req, res) => {
  let conn;
  try {
    conn = await pool.getConnection();

    // 1. USER 테이블을 백틱(`)으로 감싸기
    // 2. u.NAME이 맞는지 확인 (만약 닉네임 컬럼이 u.NICKNAME이라면 수정 필요)
    const query = `
      SELECT 
        i.INQUIRY_ID as id, 
        u.NAME as userNickname, 
        i.TITLE as title, 
        i.CONTENT as content, 
        i.STATUS as status, 
        i.REG_DATE as createdAt
      FROM INQUIRY i
      LEFT JOIN \`USER\` u ON i.USER_ID = u.USER_ID 
      ORDER BY i.REG_DATE DESC
    `;

    const rows = await conn.query(query);
    
    res.status(200).json({ result_code: 200, feedbacks: rows });
  } catch (err) {
    console.error("SQL 에러 발생:", err); // 서버 콘솔에서 이 에러를 꼭 확인하세요!
    res.status(500).json({ result_code: 500 });
  } finally {
    if (conn) conn.release();
  }
};

// 6. [관리자용] 공지사항 등록
exports.createNotice = async (req, res) => {
    let conn;
    try {
        const { title, content } = req.body;
        const noticeId = 'NOTI' + Date.now();
        conn = await pool.getConnection();
        await conn.query("INSERT INTO NOTICE (NOTICE_ID, TITLE, CONTENT, REG_DATE) VALUES (?, ?, ?, NOW())", [noticeId, title, content]);
        res.status(200).json({ result_code: 200, result_msg: "등록 성공" });
    } catch (e) {
        console.error(e);
        res.status(500).json({ result_code: 500 });
    } finally {
        if(conn) conn.end();
    }
};

// 7. [관리자용] 공지사항 삭제
exports.deleteNotice = async (req, res) => {
    let conn;
    try {
        const { id } = req.params;
        conn = await pool.getConnection();
        await conn.query("DELETE FROM NOTICE WHERE NOTICE_ID = ?", [id]);
        res.status(200).json({ result_code: 200, result_msg: "삭제 성공" });
    } catch (e) {
        console.error(e);
        res.status(500).json({ result_code: 500 });
    } finally {
        if(conn) conn.end();
    }
};

// 8. [관리자용] 공지사항 수정
exports.updateNotice = async (req, res) => {
  let conn;
  try {
    // 1. URL 파라미터로 공지사항 ID(NOTICE_ID) 받기
    const { id } = req.params;

    // 2. Body에서 수정할 제목(TITLE)과 내용(CONTENT) 받기
    const { title, content } = req.body;

    // 유효성 검사
    if (!title || !content) {
      return res.status(400).json({ result_code: 400, result_msg: "제목과 내용을 입력해주세요." });
    }

    conn = await pool.getConnection();

    // 3. DB 업데이트 실행 (이미지의 컬럼명 NOTICE_ID, TITLE, CONTENT 사용)
    const query = "UPDATE NOTICE SET TITLE = ?, CONTENT = ? WHERE NOTICE_ID = ?";
    const result = await conn.query(query, [title, content, id]);

    // 4. 결과 확인 (수정된 행이 0개면 ID가 잘못된 것)
    // mariadb 라이브러리는 result.affectedRows 로 확인 가능
    if (result.affectedRows === 0) {
      return res.status(404).json({ result_code: 404, result_msg: "해당 공지사항을 찾을 수 없습니다." });
    }

    res.status(200).json({ result_code: 200, result_msg: "공지사항 수정 성공" });

  } catch (err) {
    console.error("공지사항 수정 에러:", err);
    res.status(500).json({ result_code: 500, result_msg: "서버 오류" });
  } finally {
    // pool 연결 반납 (release 사용 권장)
    if (conn) conn.release();
  }
};