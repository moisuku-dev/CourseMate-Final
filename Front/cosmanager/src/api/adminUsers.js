// src/api/adminUsers.js
import request from "./client.js";

/**
 * 전체 회원 목록 조회
 * GET /api/admin/users
 * Response: { result_code, users: [...] }
 */
export async function fetchAdminUsers() {
  return request("/api/admin/users", {
    method: "GET",
    auth: true,
  });
}

/**
 * 회원 상태 변경 (정지/해제)
 * PUT /api/admin/users/:userId/status
 * Body: { isActive: 'Y' | 'N' }
 */
export async function changeUserStatus(userId, isActive) {
  return request(`/api/admin/users/${userId}/status`, {
    method: "PUT",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ isActive }),
  });
}
