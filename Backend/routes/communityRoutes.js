const express = require('express');
const router = express.Router();
const communityController = require('../controllers/communityController');

const { verifyToken } = require('../middleware/authMiddleware');

// 1. 공지사항 목록 조회
router.get('/notices', communityController.getNotices);

// 2. 공지사항 상세 조회
router.get('/notices/:id', communityController.getNoticeDetail);

// 3. 문의 등록
router.post('/feedback', verifyToken, communityController.createInquiry);

// 4. 내 문의 내역 조회
router.get('/feedback/me', verifyToken, communityController.getMyFeedbacks);

module.exports = router;