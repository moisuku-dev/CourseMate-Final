// api/reviews.js
import request from "./client";

/**
 * 특정 관광지 리뷰 목록: GET /api/places/{id}/reviews
 */
export function fetchPlaceReviews(placeId) {
  return request(`/places/${placeId}/reviews`);
}

/**
 * 리뷰 작성: POST /api/places/{id}/reviews
 * payload: { rating: number, content: string }
 */
export function createReview(placeId, payload) {
  return request(`/places/${placeId}/reviews`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/**
 * 내 리뷰 수정: PUT /api/reviews/{reviewId}
 * payload: { rating?: number, content?: string }
 */
export function updateReview(reviewId, payload) {
  return request(`/reviews/${reviewId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/**
 * 내 리뷰 삭제: DELETE /api/reviews/{reviewId}
 */
export function deleteReview(reviewId) {
  return request(`/reviews/${reviewId}`, {
    method: "DELETE",
  });
}
