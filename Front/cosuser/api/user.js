// api/user.js
import request from "./client";

/** 내 취향 태그 조회: GET /api/users/me/preferences */
export async function fetchPreferences() {
  return request("/users/me/preferences");
}

/** 내 취향 태그 설정/수정: POST /api/users/me/preferences */
export async function updatePreferences(tags) {
  // tags: string[]
  return request("/users/me/preferences", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tags }),
  });
}

/** 내 위시리스트 조회: GET /api/users/me/wishlist */
export async function fetchWishlist() {
  return request("/users/me/wishlist");
}

/** 위시리스트에 관광지 추가: POST /api/users/me/wishlist */
export async function addToWishlist(placeId) {
  return request("/users/me/wishlist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ placeId }),
  });
}

/** 위시리스트에서 관광지 삭제: DELETE /api/users/me/wishlist/{placeId} */
export async function removeFromWishlist(placeId) {
  return request(`/users/me/wishlist/${placeId}`, {
    method: "DELETE",
  });
}

/** 내가 작성한 리뷰/별점 내역 조회: GET /api/users/me/reviews */
export async function fetchMyReviews() {
  return request("/users/me/reviews");
}

/** 내 계정 설정 조회: GET /api/users/me/settings */
export async function fetchSettings() {
  return request("/users/me/settings");
}

/** 내 계정 정보 수정: PUT /api/users/me/settings */
export async function updateSettings(payload) {
  return request("/users/me/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** 회원탈퇴: DELETE /api/users/me */
export async function deleteAccount() {
  return request("/users/me", {
    method: "DELETE",
  });
}
