const express = require('express');
const router = express.Router();
const adminController = require('../controllers/adminController');
const communityController = require('../controllers/communityController');



// 1. 관리자 로그인
router.post('/login', adminController.adminLogin);

// 2. 대시보드 통계
router.get('/dashboard', adminController.getDashboardStats);

// 3. 회원 관리
router.get('/users', adminController.getAllUsers);
router.put('/users/:userId/status', adminController.changeUserStatus);

// 4. 관광지 관리
router.post('/places', adminController.createPlace);
router.delete('/places/:id', adminController.deletePlace);

// 5. 리뷰 관리
router.get('/reviews', adminController.getAllReviews);
router.delete('/reviews/:reviewId', adminController.deleteReviewAdmin);

// 6. 문의 답변
router.post('/inquiries/:id/answer', adminController.answerInquiry);

// ★ [신규 추가] 문의 상세 조회
router.get('/inquiries/:id', adminController.getInquiryDetail);

// 문의 삭제 라우터
router.delete('/inquiries/:id', adminController.deleteInquiry);

router.get('/feedbacks', communityController.getAllInquiries);

router.post('/notice', communityController.createNotice);

router.delete('/notices/:id', communityController.deleteNotice);

router.get('/places', adminController.getAllPlaces);

// 관리자 설정 조회 및 저장
router.get('/settings', adminController.getAdminSettings);
router.put('/settings', adminController.updateAdminSettings);

// 공지사항 수정
router.put('/notices/:id', communityController.updateNotice);

module.exports = router;