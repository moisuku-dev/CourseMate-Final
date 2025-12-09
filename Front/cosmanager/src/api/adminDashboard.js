// src/api/adminDashboard.js
import request from "./client.js";

/**
 * 대시보드 통계 조회
 * GET /api/admin/dashboard
 * Response: {
 *   result_code,
 *   result_msg,
 *   stats: { totalUsers, totalReviews, totalSpots }
 * }
 */
export async function fetchDashboardStats() {
  return request("/api/admin/dashboard", {
    method: "GET",
    auth: true,
  });
}
