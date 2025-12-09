// src/api/adminReviews.js
import request from "./client.js";

/**
 * 전체 리뷰 조회
 * GET /api/admin/reviews
 */
export async function fetchAdminReviews() {
  return request("/api/admin/reviews", {
    method: "GET",
    auth: true,
  });
}

/**
 * 리뷰 삭제 (관리자 권한)
 * DELETE /api/admin/reviews/:reviewId
 */
export async function deleteAdminReview(reviewId) {
  return request(`/api/admin/reviews/${reviewId}`, {
    method: "DELETE",
    auth: true,
  });
}
