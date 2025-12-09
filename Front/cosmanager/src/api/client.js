// src/api/client.js
const BASE_URL =
  import.meta.env.VITE_ADMIN_API_BASE_URL || "http://localhost:3000";

const STORAGE_KEY = "cosmate_admin_auth";

/**
 * 관리자 API 공통 요청 함수
 * @param {string} path - 예: "/admin/users"
 * @param {RequestInit & { auth?: boolean }} options
 *   - auth: true이면 Authorization 헤더 자동으로 붙임
 */
async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;

  const defaultHeaders = {
    "Content-Type": "application/json",
  };

  let headers = { ...defaultHeaders, ...(options.headers || {}) };

  // auth 옵션이 true면 로컬스토리지의 토큰을 Authorization에 넣음
  if (options.auth) {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed?.token) {
          headers = {
            ...headers,
            Authorization: `Bearer ${parsed.token}`,
          };
        }
      }
    } catch (e) {
      console.error("Failed to read admin token from storage:", e);
    }
  }

  const res = await fetch(url, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    const error = new Error(`API Error: ${res.status}`);
    error.status = res.status;
    error.body = errorText;
    throw error;
  }

  const text = await res.text();
  if (!text) return null;

  try {
    return JSON.parse(text);
  } catch (e) {
    console.error("Failed to parse JSON response:", e, text);
    throw e;
  }
}

export default request;
