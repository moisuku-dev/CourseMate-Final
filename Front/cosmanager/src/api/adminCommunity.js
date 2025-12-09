/*
// src/api/adminCommunity.js
import request from "./client.js";


export async function answerInquiry(inquiryId, payload) {
  return request(`/api/admin/inquiries/${inquiryId}/answer`, {
    method: "POST",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

// TODO:
//  - 문의 목록 조회: GET /api/admin/inquiries
//  - 문의 상세 조회: GET /api/admin/inquiries/:id
// 가 백엔드에 추가되면, 위 스펙에 맞는 함수를 여기에 정의하고
// AdminInquiryListPage / AdminInquiryDetailPage 에서 연동해야 합니다.
*/

// src/api/adminCommunity.js
import request from "./client.js";

/**
 * 공지사항 목록 조회
 */
export async function fetchNotices() {
  const res = await request("/api/community/notices", { // ✨ /api/admin... 경로 확인
    method: "GET",
    auth: true,
  });
  // 백엔드 응답에서 notices 배열만 반환
  return res?.notices || []; 
}

/**
 * 공지사항 생성
 * @param {{ title: string, content: string }} payload
 */
export async function createNotice(payload) {
  return request("/api/admin/notice", {
    method: "POST",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

/**
 * 공지사항 수정
 * @param {number|string} id
 * @param {{ title: string, content: string }} payload
 */
export async function updateNotice(id, payload) {
  return request(`/api/admin/notices/${id}`, {
    method: "PUT",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

/**
 * 공지사항 삭제
 * @param {number|string} id
 */
export async function deleteNotice(id) {
  return request(`/api/admin/notices/${id}`, {
    method: "DELETE",
    auth: true,
  });
}

/**
 * 이용자 의견 목록 조회
 */
export async function fetchFeedbacks() {
  const res = await request("/api/admin/feedbacks", {
    method: "GET",
    auth: true,
  });
  // 백엔드 응답에서 feedbacks 배열만 반환
  return res?.feedbacks || [];
}

/**
 * 의견 상태 변경 (예: 처리중/완료 등)
 * @param {number|string} feedbackId
 * @param {string} status
 */
export async function updateFeedbackStatus(feedbackId, status) {
  return request(`/api/community/feedback/${feedbackId}/status`, {
    method: "PUT",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ status }),
  });
}

/**
 * 의견 삭제
 * @param {number|string} feedbackId
 */
export async function deleteFeedback(feedbackId) {
  return request(`/api/community/feedback/${feedbackId}`, {
    method: "DELETE",
    auth: true,
  });
}

/**
 * 문의 답변 등록
 * 실제 백엔드 엔드포인트 예시:
 * POST /api/admin/inquiries/:id/answer
 * Body: { answerContent }
 */
export async function answerInquiry(inquiryId, payload) {
  return request(`/api/admin/inquiries/${inquiryId}/answer`, {
    method: "POST",
    auth: true,
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });
}

// ★ [신규 추가] 문의 상세 조회 함수
export async function fetchInquiryDetail(inquiryId) {
  const res = await request(`/api/admin/inquiries/${inquiryId}`, {
    method: "GET",
    auth: true,
  });
  return res?.inquiry || null;
}

/**
 * 7. 문의 삭제 (★ 신규 추가)
 * 아까 백엔드에 추가한 router.delete('/inquiries/:id', ...) 와 연결됩니다.
 */
export async function deleteInquiry(id) {
  return request(`/api/admin/inquiries/${id}`, {
    method: "DELETE",
    auth: true,
  });
}


