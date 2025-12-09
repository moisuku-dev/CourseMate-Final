// api/recommendations.js
import request from "./client";

/**
 * AI 맞춤 관광 코스 추천: GET /api/recommendations
 */
export async function fetchRecommendations(region) {
  // region이 없으면 기본값으로 '서울'을 넣거나, 에러를 방지하기 위해 처리
  const targetRegion = region || "서울"; 
  
  // URL 쿼리 파라미터로 region 전달
  return request(`/recommendations?region=${encodeURIComponent(targetRegion)}`);
}
/**
 * 다른 관광 코스 선택(재추천): GET /api/recommendations/retry
 */
export async function retryRecommendations() {
  return request("/recommendations/retry");
}
