import React, { useEffect, useState } from "react";
import AdminSidebar from "../components/AdminSidebar.jsx";
import {
  fetchNotices,
  createNotice,
  updateNotice,
  deleteNotice,
  fetchFeedbacks,
  deleteInquiry, // ★ [수정] deleteFeedback -> deleteInquiry로 이름 통일
} from "../api/adminCommunity.js";

const EMPTY_NOTICE_FORM = {
  title: "",
  content: "",
};

const AdminCommunityPage = () => {
  // ==============================
  // 상태 관리
  // ==============================
  const [notices, setNotices] = useState([]);
  const [noticesLoading, setNoticesLoading] = useState(false);
  const [noticeForm, setNoticeForm] = useState(EMPTY_NOTICE_FORM);
  const [editingNoticeId, setEditingNoticeId] = useState(null);

  const [feedbacks, setFeedbacks] = useState([]);
  const [feedbacksLoading, setFeedbacksLoading] = useState(false);

  // 초기 로딩
  useEffect(() => {
    loadNotices();
    loadFeedbacks();
  }, []);

  // ==============================
  // 1. 공지사항 관련 로직
  // ==============================
  const loadNotices = async () => {
    setNoticesLoading(true);
    try {
      const data = await fetchNotices();
      const mapped = (Array.isArray(data) ? data : []).map((n) => ({
        id: n.id || n.NOTICE_ID,
        title: n.title || n.TITLE,
        content: n.content || n.CONTENT || "",
        regDate: n.regDate || n.REG_DATE,
      }));
      setNotices(mapped);
    } catch (e) {
      console.error(e);
      alert("공지사항 목록을 불러오지 못했습니다.");
    } finally {
      setNoticesLoading(false);
    }
  };

  const handleNoticeChange = (e) => {
    const { name, value } = e.target;
    setNoticeForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleNoticeSubmit = async (e) => {
    e.preventDefault();
    if (!noticeForm.title || !noticeForm.content) {
      return alert("제목과 내용을 모두 입력해주세요.");
    }

    try {
      if (editingNoticeId) {
        await updateNotice(editingNoticeId, noticeForm);
        alert("공지사항이 수정되었습니다.");
      } else {
        await createNotice(noticeForm);
        alert("새 공지사항이 등록되었습니다.");
      }
      
      setNoticeForm(EMPTY_NOTICE_FORM);
      setEditingNoticeId(null);
      loadNotices();
    } catch (e) {
      console.error(e);
      alert("처리 중 오류가 발생했습니다.");
    }
  };

  const handleEditNotice = (notice) => {
    setEditingNoticeId(notice.id);
    setNoticeForm({
      title: notice.title,
      content: notice.content,
    });
  };

  const handleCancelEdit = () => {
    setEditingNoticeId(null);
    setNoticeForm(EMPTY_NOTICE_FORM);
  };

  const handleDeleteNotice = async (id) => {
    if (!window.confirm("정말 이 공지사항을 삭제하시겠습니까?")) return;
    try {
      await deleteNotice(id);
      loadNotices();
    } catch (e) {
      console.error(e);
      alert("삭제 실패");
    }
  };

  // ==============================
  // 2. 이용자 문의(피드백) 관련 로직
  // ==============================
  const loadFeedbacks = async () => {
    setFeedbacksLoading(true);
    try {
      const data = await fetchFeedbacks();
      const mapped = (Array.isArray(data) ? data : []).map((f) => ({
        id: f.id || f.INQUIRY_ID,
        title: f.title || f.TITLE,
        content: f.content || f.CONTENT,
        status: f.status || f.STATUS,
        createdAt: f.createdAt || f.regDate || f.REG_DATE,
      }));
      setFeedbacks(mapped);
    } catch (e) {
      console.error(e);
    } finally {
      setFeedbacksLoading(false);
    }
  };

  // ★ [핵심 수정] 함수 이름도 deleteInquiry 로 맞춤
  const handleDeleteInquiry = async (item) => {
    if (!window.confirm("정말 이 문의 내역을 삭제하시겠습니까?")) return;
    
    const targetId = item.id; 

    try {
      // ★ 여기가 문제였습니다. deleteInquiry로 확실하게 호출합니다.
      await deleteInquiry(targetId); 
      alert("삭제되었습니다.");
      loadFeedbacks(); 
    } catch (e) {
      console.error(e);
      alert("삭제 실패: " + (e.message || "오류"));
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "#020617" }}>
      <AdminSidebar />
      <div style={{ flex: 1, padding: "24px 28px", color: "#f8fafc" }}>
        <h1 style={{ fontSize: "24px", fontWeight: "700", marginBottom: "20px" }}>커뮤니티 관리</h1>

        {/* --- 섹션 1: 공지사항 관리 --- */}
        <section style={{ marginBottom: "40px", backgroundColor: "#1e293b", padding: "20px", borderRadius: "12px" }}>
          <h2 style={{ fontSize: "18px", marginBottom: "15px", borderBottom: "1px solid #334155", paddingBottom: "10px" }}>
            공지사항 관리
          </h2>
          
          <form onSubmit={handleNoticeSubmit} style={{ marginBottom: "20px", display: "flex", flexDirection: "column", gap: "10px" }}>
            <input name="title" placeholder="제목" value={noticeForm.title} onChange={handleNoticeChange} style={inputStyle} />
            <textarea name="content" placeholder="내용" value={noticeForm.content} onChange={handleNoticeChange} rows={3} style={{ ...inputStyle, resize: "vertical" }} />
            <div style={{ display: "flex", gap: "10px" }}>
              <button type="submit" style={btnPrimary}>
                {editingNoticeId ? "공지사항 수정 저장" : "공지사항 등록"}
              </button>
              {editingNoticeId && (
                <button type="button" onClick={handleCancelEdit} style={btnSecondary}>취소</button>
              )}
            </div>
          </form>

          {noticesLoading ? <p>로딩 중...</p> : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
                <thead>
                  <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155" }}>
                    <th style={{ padding: "10px", textAlign: "left" }}>ID</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>제목</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>날짜</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>관리</th>
                  </tr>
                </thead>
                <tbody>
                  {notices.map((n) => (
                    <tr key={n.id} style={{ borderBottom: "1px solid #334155" }}>
                      <td style={{ padding: "10px" }}>{n.id}</td>
                      <td style={{ padding: "10px", fontWeight: "bold" }}>{n.title}</td>
                      <td style={{ padding: "10px", color: "#cbd5e1" }}>{new Date(n.regDate).toLocaleDateString()}</td>
                      <td style={{ padding: "10px", display: "flex", gap: "6px" }}>
                        <button onClick={() => handleEditNotice(n)} style={btnEdit}>수정</button>
                        <button onClick={() => handleDeleteNotice(n.id)} style={btnDelete}>삭제</button>
                      </td>
                    </tr>
                  ))}
                  {notices.length === 0 && <tr><td colSpan="4" style={{ padding: "20px", textAlign: "center", color: "#64748b" }}>등록된 공지사항이 없습니다.</td></tr>}
                </tbody>
              </table>
            </div>
          )}
        </section>

        {/* --- 섹션 2: 이용자 문의 관리 --- */}
        <section style={{ backgroundColor: "#1e293b", padding: "20px", borderRadius: "12px" }}>
          <h2 style={{ fontSize: "18px", marginBottom: "15px", borderBottom: "1px solid #334155", paddingBottom: "10px" }}>
            이용자 문의 관리
          </h2>

          {feedbacksLoading ? <p>로딩 중...</p> : (
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px" }}>
                <thead>
                  <tr style={{ color: "#94a3b8", borderBottom: "1px solid #334155" }}>
                    <th style={{ padding: "10px", textAlign: "left" }}>ID</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>제목</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>내용</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>상태</th>
                    <th style={{ padding: "10px", textAlign: "left" }}>관리</th>
                  </tr>
                </thead>
                <tbody>
                  {feedbacks.map((f) => (
                    <tr key={f.id} style={{ borderBottom: "1px solid #334155" }}>
                      <td style={{ padding: "10px" }}>{f.id}</td>
                      <td style={{ padding: "10px", fontWeight: "bold" }}>{f.title}</td>
                      <td style={{ padding: "10px", maxWidth: "300px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f.content}</td>
                      <td style={{ padding: "10px" }}>
                        {f.status === '완료' ? <span style={{color: "#22c55e", fontWeight: "bold"}}>답변완료</span> : <span style={{color: "#f59e0b", fontWeight: "bold"}}>대기중</span>}
                      </td>
                      <td style={{ padding: "10px" }}>
                        {/* ★ [수정] 버튼 클릭 시 handleDeleteInquiry 호출 */}
                        <button onClick={() => handleDeleteInquiry(f)} style={btnDelete}>삭제</button>
                      </td>
                    </tr>
                  ))}
                  {feedbacks.length === 0 && <tr><td colSpan="5" style={{ padding: "20px", textAlign: "center", color: "#64748b" }}>등록된 문의 내역이 없습니다.</td></tr>}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

// 스타일 정의
const inputStyle = { padding: "10px", borderRadius: "6px", border: "1px solid #475569", backgroundColor: "#0f172a", color: "white" };
const btnPrimary = { padding: "8px 16px", backgroundColor: "#3b82f6", color: "white", borderRadius: "6px", border: "none", cursor: "pointer", fontWeight: "600" };
const btnSecondary = { padding: "8px 16px", backgroundColor: "#64748b", color: "white", borderRadius: "6px", border: "none", cursor: "pointer", fontWeight: "600" };
const btnEdit = { padding: "4px 8px", backgroundColor: "#0ea5e9", color: "white", borderRadius: "4px", border: "none", cursor: "pointer", fontSize: "12px" };
const btnDelete = { padding: "4px 8px", backgroundColor: "#ef4444", color: "white", borderRadius: "4px", border: "none", cursor: "pointer", fontSize: "12px" };

export default AdminCommunityPage;