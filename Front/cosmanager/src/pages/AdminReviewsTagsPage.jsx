// src/pages/AdminReviewsTagsPage.jsx
import React, { useEffect, useState } from "react";
import AdminSidebar from "../components/AdminSidebar.jsx";
import { fetchAdminReviews, deleteAdminReview } from "../api/adminReviews.js";

const AdminReviewsTagsPage = () => {
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState("");

  const loadReviews = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetchAdminReviews();
      if (res && res.result_code === 200 && Array.isArray(res.reviews)) {
        setReviews(res.reviews);
      } else {
        setError(res?.result_msg || "리뷰 목록 조회에 실패했습니다.");
      }
    } catch (e) {
      console.error(e);
      setError("리뷰 목록 조회 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadReviews();
  }, []);

  const handleDelete = async (review) => {
    if (!window.confirm("정말 이 리뷰를 삭제하시겠습니까?")) return;
    setDeletingId(review.REVIEW_ID);
    try {
      const res = await deleteAdminReview(review.REVIEW_ID);
      if (res && res.result_code === 200) {
        await loadReviews();
      } else {
        alert(res?.result_msg || "리뷰 삭제에 실패했습니다.");
      }
    } catch (e) {
      console.error(e);
      alert("리뷰 삭제 중 오류가 발생했습니다.");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "#020617" }}>
      <AdminSidebar />
      <main style={{ flex: 1, padding: "24px 28px" }}>
        <h1 style={{ fontSize: "22px", fontWeight: "700", color: "#f9fafb", marginBottom: "4px" }}>
          리뷰 관리
        </h1>
        <p style={{ fontSize: "13px", color: "#9ca3af", marginBottom: "16px" }}>
          사용자들이 작성한 리뷰를 조회하고, 부적절한 리뷰를 삭제할 수 있습니다.
        </p>

        {loading && (
          <div style={{ color: "#e5e7eb", fontSize: "14px" }}>
            리뷰를 불러오는 중입니다...
          </div>
        )}
        {error && (
          <div style={{ color: "#fecaca", fontSize: "13px", marginBottom: "8px" }}>
            {error}
          </div>
        )}

        {!loading && reviews.length === 0 && (
          <div style={{ fontSize: "14px", color: "#9ca3af" }}>등록된 리뷰가 없습니다.</div>
        )}

        {reviews.length > 0 && (
          <div
            style={{
              marginTop: "8px",
              backgroundColor: "#020617",
              borderRadius: "12px",
              overflow: "hidden",
              border: "1px solid rgba(148,163,184,0.35)",
              boxShadow: "0 8px 24px rgba(15,23,42,0.65)",
            }}
          >
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: "13px",
                color: "#e5e7eb",
              }}
            >
              <thead style={{ backgroundColor: "#020617" }}>
                <tr>
                  <Th>리뷰 ID</Th>
                  <Th>작성자</Th>
                  <Th>관광지</Th>
                  <Th>평점</Th>
                  <Th>내용</Th>
                  <Th>작성일</Th>
                  <Th>관리</Th>
                </tr>
              </thead>
              <tbody>
                {reviews.map((r) => (
                  <tr key={r.REVIEW_ID} style={{ borderTop: "1px solid rgba(31,41,55,0.8)" }}>
                    <Td>{r.REVIEW_ID}</Td>
                    <Td>{r.writer}</Td>
                    <Td>{r.spotName}</Td>
                    <Td>{r.RATING}</Td>
                    <Td>
                      <div
                        style={{
                          maxWidth: "320px",
                          whiteSpace: "nowrap",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                        }}
                      >
                        {r.CONTENT}
                      </div>
                    </Td>
                    <Td>{r.REG_DATE}</Td>
                    <Td>
                      <button
                        type="button"
                        onClick={() => handleDelete(r)}
                        disabled={deletingId === r.REVIEW_ID}
                        style={{
                          padding: "4px 10px",
                          borderRadius: "9999px",
                          border: "none",
                          fontSize: "12px",
                          cursor: deletingId === r.REVIEW_ID ? "default" : "pointer",
                          backgroundColor: "#ef4444",
                          color: "#0b1120",
                          fontWeight: "600",
                        }}
                      >
                        {deletingId === r.REVIEW_ID ? "삭제 중..." : "삭제"}
                      </button>
                    </Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
    </div>
  );
};

const Th = ({ children }) => (
  <th
    style={{
      textAlign: "left",
      padding: "10px 12px",
      fontWeight: "600",
      color: "#9ca3af",
    }}
  >
    {children}
  </th>
);

const Td = ({ children }) => (
  <td style={{ padding: "8px 12px", color: "#e5e7eb" }}>{children}</td>
);

export default AdminReviewsTagsPage;
