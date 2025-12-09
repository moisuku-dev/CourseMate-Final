// src/pages/AdminInquiryDetailPage.jsx
import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import AdminSidebar from "../components/AdminSidebar.jsx";
import { answerInquiry, fetchInquiryDetail } from "../api/adminCommunity.js"; // fetchInquiryDetail 추가

const AdminInquiryDetailPage = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  
  // 데이터 상태 관리
  const [inquiry, setInquiry] = useState(null);
  const [loading, setLoading] = useState(true);

  const [answer, setAnswer] = useState("");
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  // ★ 1. 페이지 로드 시 상세 정보 가져오기
  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const data = await fetchInquiryDetail(id);
        if (data) {
          setInquiry(data);
          // 이미 답변이 있다면 답변란에 채워두기
          if (data.answerContent) {
            setAnswer(data.answerContent);
          }
        } else {
          setMessage("문의 정보를 불러올 수 없습니다.");
        }
      } catch (e) {
        console.error(e);
        setMessage("오류가 발생했습니다.");
      } finally {
        setLoading(false);
      }
    };
    loadData();
  }, [id]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");
    if (!answer.trim()) {
      setMessage("답변 내용을 입력해주세요.");
      return;
    }
    setSaving(true);
    try {
      const res = await answerInquiry(id, { answerContent: answer });
      if (res && res.result_code === 200) {
        setMessage("답변 등록에 성공했습니다.");
        // 성공 후 상태 업데이트 (완료 표시 등)
        setInquiry(prev => ({ ...prev, status: '완료', answerContent: answer }));
      } else {
        setMessage(res?.result_msg || "답변 등록에 실패했습니다.");
      }
    } catch (e) {
      console.error(e);
      setMessage("답변 등록 중 오류가 발생했습니다.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "#020617" }}>
      <AdminSidebar />
      <main style={{ flex: 1, padding: "24px 28px" }}>
        <button
          type="button"
          onClick={() => navigate(-1)}
          style={{
            marginBottom: "12px",
            padding: "6px 12px",
            borderRadius: "99px",
            border: "1px solid #334155",
            backgroundColor: "transparent",
            color: "#94a3b8",
            fontSize: "13px",
            cursor: "pointer",
          }}
        >
          ← 목록으로
        </button>

        <h1 style={{ fontSize: "24px", fontWeight: "700", color: "#f9fafb", marginBottom: "20px" }}>
          문의 답변 관리
        </h1>

        {loading ? (
          <div style={{ color: "#94a3b8" }}>로딩 중...</div>
        ) : inquiry ? (
          <>
            {/* 문의 내용 카드 */}
            <div
              style={{
                backgroundColor: "#1e293b",
                borderRadius: "12px",
                padding: "24px",
                border: "1px solid #334155",
                marginBottom: "24px",
              }}
            >
              <div style={{ marginBottom: "16px", borderBottom: "1px solid #334155", paddingBottom: "16px" }}>
                <h2 style={{ fontSize: "18px", fontWeight: "600", color: "#f9fafb", marginBottom: "8px" }}>
                  {inquiry.title}
                </h2>
                <div style={{ fontSize: "13px", color: "#94a3b8", display: "flex", gap: "12px" }}>
                  <span>작성자: {inquiry.writerName} ({inquiry.writerEmail})</span>
                  <span>|</span>
                  <span>작성일: {new Date(inquiry.regDate).toLocaleString()}</span>
                  <span>|</span>
                  <span style={{ 
                    color: inquiry.status === '완료' ? '#4ade80' : '#fbbf24',
                    fontWeight: "600"
                  }}>
                    {inquiry.status}
                  </span>
                </div>
              </div>
              <div style={{ color: "#e2e8f0", fontSize: "15px", lineHeight: "1.6", whiteSpace: "pre-wrap" }}>
                {inquiry.content}
              </div>
            </div>

            {/* 답변 입력 폼 */}
            <form
              onSubmit={handleSubmit}
              style={{
                backgroundColor: "#1e293b",
                borderRadius: "12px",
                padding: "24px",
                border: "1px solid #334155",
              }}
            >
              <h3 style={{ fontSize: "16px", fontWeight: "600", color: "#f9fafb", marginBottom: "12px" }}>
                관리자 답변
              </h3>
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                rows={6}
                style={{
                  width: "100%",
                  resize: "vertical",
                  padding: "12px",
                  borderRadius: "8px",
                  border: "1px solid #475569",
                  backgroundColor: "#0f172a",
                  color: "#f1f5f9",
                  fontSize: "14px",
                  outline: "none",
                  marginBottom: "12px"
                }}
                placeholder="답변 내용을 입력하세요..."
              />
              
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: "14px", color: message.includes("성공") ? "#4ade80" : "#f87171" }}>
                  {message}
                </span>
                <button
                  type="submit"
                  disabled={saving}
                  style={{
                    padding: "10px 20px",
                    borderRadius: "8px",
                    border: "none",
                    backgroundColor: saving ? "#475569" : "#4f46e5",
                    color: "#fff",
                    fontSize: "14px",
                    fontWeight: "600",
                    cursor: saving ? "default" : "pointer",
                    transition: "background-color 0.2s"
                  }}
                >
                  {saving ? "등록 중..." : "답변 등록하기"}
                </button>
              </div>
            </form>
          </>
        ) : (
          <div style={{ color: "#f87171" }}>문의 정보를 찾을 수 없습니다.</div>
        )}
      </main>
    </div>
  );
};

export default AdminInquiryDetailPage;