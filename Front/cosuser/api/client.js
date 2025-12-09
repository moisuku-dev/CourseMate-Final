// api/client.js
//import * as SecureStore from "expo-secure-store";
import { getToken } from "../utils/storage"; // (추가) 방금 만든 파일 불러오기

const API_BASE_URL = "http://localhost:3000/api"; //< 실제 주소로 변경 필요

/**
 * 공통 요청 함수
 * @param {string} path - 예: "/auth/login"
 * @param {RequestInit} options
 */
export default async function request(path, options = {}) {
  const url = `${API_BASE_URL}${path}`;

  const defaultHeaders = {
    "Content-Type": "application/json",
  };

  let authHeader = {};
  try {
    const token = await getToken("cosmate_user_token");
    if (token) {
      authHeader = { Authorization: `Bearer ${token}` };
    }
  } catch (e) {
    console.warn("Failed to read token from SecureStore", e);
  }

  const config = {
    method: "GET",
    ...options,
    headers: {
      ...defaultHeaders,
      ...(options.headers || {}),
      ...authHeader,
    },
  };

  const res = await fetch(url, config);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `Request failed: ${res.status}`);
  }

  // no-content 방어
  const text = await res.text();
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    return text;
  }
}
