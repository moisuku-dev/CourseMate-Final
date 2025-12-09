const express = require('express');
const mariadb = require('mariadb');
const cors = require('cors');
require('dotenv').config();

// 라우터 불러오기
const authRoutes = require('./routes/authRoutes');
const placeRoutes = require('./routes/placeRoutes');
const reviewRoutes = require('./routes/reviewRoutes');
const userRoutes = require('./routes/userRoutes');
const recommendationRoutes = require('./routes/recommendationRoutes');
const communityRoutes = require('./routes/communityRoutes');
const adminRoutes = require('./routes/adminRoutes');
const path = require('path');
const app = express();
const port = process.env.PORT || 3000;

// 미들웨어 설정
app.use(cors()); 
app.use(express.json()); 
app.use('/images', express.static(path.join(__dirname, 'images')));

// DB 연결
const pool = require('./database');

async function testDbConnection() {
  let conn;
  try {
    conn = await pool.getConnection();
    console.log("✅ MariaDB 연결 성공! (Database: " + process.env.DB_NAME + ")");
  } catch (err) {
    console.error("❌ MariaDB 연결 실패:", err);
  } finally {
    if (conn) conn.end();
  }
}
testDbConnection();

// ✨ [중요] 라우트 설정은 반드시 서버 시작(listen) 전에 해야 합니다.
app.use('/api/auth', authRoutes);
app.use('/api/places', placeRoutes);
app.use('/api', reviewRoutes); // '/api/places/:id/reviews' 등
app.use('/api/users', userRoutes);
app.use('/api/recommendations', recommendationRoutes);
app.use('/api/community', communityRoutes);
app.use('/api/admin', adminRoutes);

// 기본 접속 테스트
app.get('/', (req, res) => {
  res.send('CourseMate 백엔드 서버가 정상 작동 중입니다.');
});

// 서버 시작 (코드 맨 마지막에 한 번만!)
app.listen(port, '0.0.0.0', () => {
  console.log(`🚀 Server running at http://localhost:${port}`);
});