const jwt = require('jsonwebtoken');
require('dotenv').config();

// ★ 중요: 환경변수가 없으면 임시키를 쓰도록 변수에 담음
const SERVER_SECRET = process.env.JWT_SECRET || 'my_super_secret_key_1234';

exports.verifyToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  
  // [디버깅 1] 헤더 값 확인
  console.log(`[1] 받은 헤더: ${authHeader}`);

  // 헤더가 없거나, 'Bearer ' 형식이 아닌 경우 체크
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
     console.log(`[Error] 헤더가 없거나 Bearer 형식이 아님`);
     return res.status(401).json({ result_code: 401, result_msg: "토큰 형식이 잘못됨" });
  }

  const token = authHeader.split(' ')[1];
  
  // [디버깅 2] 토큰 분리 확인
  console.log(`[2] 분리된 토큰: ${token ? token.substring(0, 20) + '...' : '없음'}`);

  if (!token) {
    console.log(`[Error] 토큰 값이 비어있음`);
    return res.status(401).json({ result_code: 401, result_msg: "토큰 없음" });
  }

  try {
    console.log(`[3] 검증에 사용하는 키: ${SERVER_SECRET}`);

    const decoded = jwt.verify(token, SERVER_SECRET);
    
    // ★ [수정] 컨트롤러가 보통 req.user를 쓰므로 이름을 맞춰줍니다.
    req.user = decoded; 
    // 혹은 호환성을 위해 둘 다 넣어주세요.
    req.decoded = decoded; 

    console.log(`[Success] 인증 성공! User ID: ${req.user.userId}`); // 로그도 req.user로 확인
    return next();

  } catch (error) {
    // ★ [디버깅 4] 에러 정체 확인 (여기를 봐야 함!)
    console.log(`[Error] 검증 실패 상세 내용:`, error.name, error.message);

    if (error.name === 'TokenExpiredError') {
      return res.status(419).json({ code: 419, message: '토큰 만료' });
    }
    
    return res.status(401).json({ code: 401, message: '유효하지 않은 토큰 (검증 실패)' });
  }
};