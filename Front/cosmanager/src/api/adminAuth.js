// src/api/adminAuth.js
import request from "./client.js";

/**
 * 관리자 로그인
 * 실제 백엔드 엔드포인트: POST /api/admin/login
 * Body: { adminId, password }
 * Response: { result_code, result_msg, token }
 */
export async function adminLogin(payload) {
  const res = await request("/api/admin/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  // result_code 200 이고 token 이 있을 때만 성공으로 간주
  if (res && res.result_code === 200 && res.token) {
    return {
      // 현재 백엔드에서 id/name 을 따로 내려주지 않으므로
      // 기본 관리자 프로필로 매핑
      id: "admin",
      name: "관리자",
      token: res.token,
    };
  }

  // 실패 시에는 상위에서 에러로 처리할 수 있게 그대로 반환
  throw new Error(res?.result_msg || "관리자 로그인에 실패했습니다.");
}
