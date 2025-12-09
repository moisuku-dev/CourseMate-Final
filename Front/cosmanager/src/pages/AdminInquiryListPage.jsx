// src/pages/AdminInquiryListPage.jsx
import React, { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios"; // axios가 없다면 fetch로 대체 가능
import AdminSidebar from "../components/AdminSidebar.jsx";

const AdminInquiryListPage = () => {
  const navigate = useNavigate();
  
  // 1. 문의 목록 상태 관리
  const [inquiries, setInquiries] = useState([]);
  const [loading, setLoading] = useState(true);

  // 2. API 호출하여 목록 가져오기
  useEffect(() => {
    const fetchInquiries = async () => {
      try {
        // 백엔드 라우트 경로에 맞춰 수정해주세요 (예: /api/admin/inquiries)
        const response = await axios.get(`http://localhost:3000/api/admin/feedbacks?t=${new Date().getTime()}`);

        console.log("프론트 응답 데이터 확인:", response.data); // 이걸 꼭 확인하세요!
        
        // 백엔드에서 보낸 구조: { result_code: 200, feedbacks: [...] }
        if (response.data.result_code === 200) {
          setInquiries(response.data.feedbacks);
        }
      } catch (error) {
        console.error("문의 목록 조회 실패:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchInquiries();
  }, []);

  // 3. 상세 페이지 이동 핸들러 (실제 ID 사용)
  const handleInquiryClick = (inquiryId) => {
    // 예: /admin/inquiries/1, /admin/inquiries/15 등 DB ID가 들어감
    navigate(`/admin/inquiries/${inquiryId}`);
  };

  // 날짜 포맷팅 헬퍼 함수
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
    });
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "#020617" }}>
      <AdminSidebar />
      <main style={{ flex: 1, padding: "24px 28px" }}>
        <h1 style={{ fontSize: "22px", fontWeight: "700", color: "#f9fafb", marginBottom: "4px" }}>
          문의 관리
        </h1>
        <p style={{ fontSize: "13px", color: "#9ca3af", marginBottom: "24px" }}>
          사용자가 남긴 문의 내역을 확인하고 관리합니다.
        </p>

        {/* 목록 영역 */}
        <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
          
          {/* 로딩 중일 때 */}
          {loading && <p style={{ color: "#9ca3af" }}>목록을 불러오는 중...</p>}

          {/* 데이터가 없을 때 */}
          {!loading && inquiries.length === 0 && (
            <div style={{ color: "#9ca3af", padding: "20px", border: "1px dashed #374151", borderRadius: "8px" }}>
              등록된 문의가 없습니다.
            </div>
          )}

          {/* 데이터가 있을 때 리스트 렌더링 */}
          {!loading && inquiries.map((item) => (
            <div
              key={item.id}
              onClick={() => handleInquiryClick(item.id)}
              style={{
                backgroundColor: "#111827", // 카드 배경색 (조금 더 밝게)
                borderRadius: "12px",
                padding: "20px",
                border: "1px solid #1f2937",
                cursor: "pointer",
                transition: "background-color 0.2s",
              }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#1f2937")}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#111827")}
            >
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ 
                  fontSize: "12px", 
                  color: item.status === "답변완료" ? "#34d399" : "#f87171", // 상태별 색상 구분
                  fontWeight: "bold",
                  backgroundColor: "rgba(255,255,255,0.05)",
                  padding: "2px 8px",
                  borderRadius: "4px"
                }}>
                  {item.status || "대기중"}
                </span>
                <span style={{ fontSize: "12px", color: "#6b7280" }}>
                  {formatDate(item.createdAt)}
                </span>
              </div>
              
              <h3 style={{ fontSize: "16px", fontWeight: "600", color: "#f3f4f6", marginBottom: "4px" }}>
                {item.title}
              </h3>
              
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <p style={{ 
                  fontSize: "13px", 
                  color: "#9ca3af", 
                  overflow: "hidden", 
                  textOverflow: "ellipsis", 
                  whiteSpace: "nowrap", 
                  maxWidth: "80%" 
                }}>
                  {item.content}
                </p>
                <span style={{ fontSize: "12px", color: "#6b7280" }}>
                  작성자: {item.userNickname || "알 수 없음"}
                </span>
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
};

export default AdminInquiryListPage;