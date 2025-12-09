// src/api/auth.js
import request from "./client";

/** 회원가입: POST /api/auth/signup */
export async function signUp(payload) {
  return request("/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

/** 로그인: POST /api/auth/login */
export async function login({ email, password }) {
  return request("/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

/** 로그아웃: POST /api/auth/logout*/
export async function logout() {
  return request("/auth/logout", {
    method: "POST",
  });
}

/** 아이디 찾기: POST /api/auth/find-id */
export async function findId({ name, contact }) {
  return request("/auth/find-id", {
    method: "POST",
    body: JSON.stringify({ name, contact }),
  });
}

/** 비밀번호 재설정: POST /api/auth/reset-password */
export async function resetPassword({ email, newPassword }) {
  return request("/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ email, new_password: newPassword }),
  });
}

/**
 * 내 계정 설정 조회: GET /api/users/me/settings
 */
export async function fetchMe() {
  const data = await request("/users/me/settings");
  return data.setting; // {name, email, age, gender, is_active}
}

/** (로그인 상태에서) 비밀번호 변경: POST /auth/change-password */
export async function changePassword(payload) {
  // payload 예: { currentPassword, newPassword }
  return request("/auth/change-password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}