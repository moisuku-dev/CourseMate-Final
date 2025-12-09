// src/pages/AdminUsersPage.jsx
import React, { useEffect, useState } from "react";
import AdminSidebar from "../components/AdminSidebar.jsx";
import { fetchAdminUsers, changeUserStatus } from "../api/adminUsers.js";

const AdminUsersPage = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [savingId, setSavingId] = useState(null);
  const [error, setError] = useState("");

  const loadUsers = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetchAdminUsers();
      if (res && res.result_code === 200 && Array.isArray(res.users)) {
        setUsers(res.users);
      } else {
        setError(res?.result_msg || "회원 목록 조회에 실패했습니다.");
      }
    } catch (e) {
      console.error(e);
      setError("회원 목록 조회 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsers();
  }, []);

  const handleToggleStatus = async (user) => {
    const next = user.IS_ACTIVE === "Y" ? "N" : "Y";
    setSavingId(user.USER_ID);
    try {
      const res = await changeUserStatus(user.USER_ID, next);
      if (res && res.result_code === 200) {
        await loadUsers();
      } else {
        alert(res?.result_msg || "회원 상태 변경에 실패했습니다.");
      }
    } catch (e) {
      console.error(e);
      alert("회원 상태 변경 중 오류가 발생했습니다.");
    } finally {
      setSavingId(null);
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "#020617" }}>
      <AdminSidebar />
      <main style={{ flex: 1, padding: "24px 28px" }}>
        <h1 style={{ fontSize: "22px", fontWeight: "700", color: "#f9fafb", marginBottom: "4px" }}>
          회원 관리
        </h1>
        <p style={{ fontSize: "13px", color: "#9ca3af", marginBottom: "16px" }}>
          서비스에 가입된 회원의 정보를 조회하고, 활동 상태를 관리합니다.
        </p>

        {loading && (
          <div style={{ color: "#e5e7eb", fontSize: "14px" }}>
            회원 목록을 불러오는 중입니다...
          </div>
        )}
        {error && (
          <div style={{ color: "#fecaca", fontSize: "13px", marginBottom: "8px" }}>
            {error}
          </div>
        )}

        {!loading && users.length === 0 && (
          <div style={{ fontSize: "14px", color: "#9ca3af" }}>회원이 없습니다.</div>
        )}

        {users.length > 0 && (
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
                  <Th>ID</Th>
                  <Th>이름</Th>
                  <Th>이메일</Th>
                  <Th>상태</Th>
                  <Th>가입일</Th>
                  <Th>관리</Th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.USER_ID} style={{ borderTop: "1px solid rgba(31,41,55,0.8)" }}>
                    <Td>{u.USER_ID}</Td>
                    <Td>{u.NAME}</Td>
                    <Td>{u.EMAIL}</Td>
                    <Td>
                      <span
                        style={{
                          display: "inline-block",
                          padding: "2px 8px",
                          borderRadius: "9999px",
                          fontSize: "11px",
                          backgroundColor:
                            u.IS_ACTIVE === "Y" ? "rgba(22,163,74,0.15)" : "rgba(239,68,68,0.12)",
                          color: u.IS_ACTIVE === "Y" ? "#4ade80" : "#fca5a5",
                        }}
                      >
                        {u.IS_ACTIVE === "Y" ? "활성" : "정지"}
                      </span>
                    </Td>
                    <Td>{u.JOIN_DATE}</Td>
                    <Td>
                      <button
                        type="button"
                        onClick={() => handleToggleStatus(u)}
                        disabled={savingId === u.USER_ID}
                        style={{
                          padding: "4px 10px",
                          borderRadius: "9999px",
                          border: "none",
                          fontSize: "12px",
                          cursor: savingId === u.USER_ID ? "default" : "pointer",
                          backgroundColor:
                            u.IS_ACTIVE === "Y" ? "#f97316" : "#22c55e",
                          color: "#0b1120",
                          fontWeight: "600",
                        }}
                      >
                        {savingId === u.USER_ID
                          ? "저장 중..."
                          : u.IS_ACTIVE === "Y"
                          ? "정지"
                          : "해제"}
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

export default AdminUsersPage;
