// src/api/adminPlaces.js
import request from "./client.js";

/**
 * 관광지 등록
 * POST /api/admin/places
 * Body: { spotId, name, address, category, latitude, longitude }
 */
export async function createAdminPlace(payload) {
  return request("/api/admin/places", {
    method: "POST",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

/**
 * 관광지 삭제
 * DELETE /api/admin/places/:id
 */
export async function deleteAdminPlace(id) {
  return request(`/api/admin/places/${id}`, {
    method: "DELETE",
    auth: true,
  });
}

// ✨ [추가된 부분] 이 함수가 없어서 에러가 난 것입니다.
/**
 * [관리자용] 전체 관광지 목록 조회
 * GET /api/admin/places
 * Response: { result_code, places: [...] }
 */
export async function fetchAdminPlaces() {
  const res = await request("/api/admin/places", {
    method: "GET",
    auth: true,
  });
  // 백엔드 응답 구조({ result_code, places: [...] })에서 places 배열만 반환
  return res?.places || [];
}