// src/api/adminSettings.js
import request from "./client.js";

/**
 * 관리자 설정 조회
 * 응답 예시:
 * {
 *   loginFailedLimit: number,
 *   lockMinutes: number,
 *   allowNewAdmins: boolean
 * }
 */
export async function fetchAdminSettings() {
  return request("/api/admin/settings", {
    method: "GET",
    auth: true,
  });
}

/**
 * 관리자 설정 저장
 * @param {{ loginFailedLimit: number, lockMinutes: number, allowNewAdmins: boolean }} payload
 */
export async function updateAdminSettings(payload) {
  return request("/api/admin/settings", {
    method: "PUT",
    auth: true,
    body: JSON.stringify(payload),
  });
}
