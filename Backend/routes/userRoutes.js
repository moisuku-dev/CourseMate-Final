const express = require('express');
const router = express.Router();
const userController = require('../controllers/userController');
const { verifyToken } = require('../middleware/authMiddleware');

// ✨ [안전 장치] 함수인지 검사하고 연결하는 도우미 함수
function safeConnect(method, path, ...handlers) {
  handlers.forEach((h, index) => {
    if (typeof h !== 'function') {
      console.error(`\n🚨 [심각한 오류 발견]`);
      console.error(`👉 라우트: ${method.toUpperCase()} ${path}`);
      console.error(`👉 위치: ${index + 1}번째 핸들러`);
      console.error(`👉 문제: 함수가 와야 하는데 '${h}' (타입: ${typeof h})가 왔습니다.`);
      console.error(`👉 힌트: 컨트롤러나 미들웨어 이름에 오타가 있는지, 파일이 잘 저장됐는지 확인하세요!\n`);
    }
  });
  // 문제 없으면 실제 라우터에 연결
  router[method](path, ...handlers);
}

console.log("🔍 [라우터 점검] 라우트 연결을 시작합니다...");

// 1. 내 정보 조회
safeConnect('get', '/me/settings', verifyToken, userController.getMySettings);

// 2. 내 정보 수정
safeConnect('put', '/me/settings', verifyToken, userController.updateMyInfo);

// 3. 회원 탈퇴
safeConnect('delete', '/me', verifyToken, userController.deleteAccount);

// 4. 전체 태그 조회
safeConnect('get', '/tags', userController.getAllTags);

// 5. 내 취향 태그 조회
safeConnect('get', '/me/preferences', verifyToken, userController.getMyPreferences);

// 6. 내 취향 태그 설정
safeConnect('post', '/me/preferences', verifyToken, userController.setMyPreferences);

// ★ 7. 내 리뷰 조회 (새로 추가)
safeConnect('get', '/me/reviews', verifyToken, userController.getMyReviews);

// ★ [신규 추가] 8. 위시리스트 조회
safeConnect('get', '/me/wishlist', verifyToken, userController.getWishlist);

// ★ [신규 추가] 9. 위시리스트 추가/삭제 (토글)
safeConnect('post', '/me/wishlist', verifyToken, userController.toggleWishlist);

// ★ [신규 추가] 10. 위시리스트 삭제 (ID로 명시적 삭제)
safeConnect('delete', '/me/wishlist/:placeId', verifyToken, userController.removeWishlist);

console.log("✅ [라우터 점검] 모든 라우트가 정상적으로 연결되었습니다.");

module.exports = router;