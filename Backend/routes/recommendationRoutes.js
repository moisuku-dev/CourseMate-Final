const express = require('express');
const router = express.Router();
const recommendationController = require('../controllers/recommendationController');

const { verifyToken } = require('../middleware/authMiddleware');

// 1. AI 맞춤 관광 코스 추천
// 경로: /api/recommendations
router.get('/', verifyToken, recommendationController.getRecommendations);



module.exports = router;