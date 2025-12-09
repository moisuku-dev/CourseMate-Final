// src/components/AdminSidebar.jsx
import React from "react";
import { NavLink } from "react-router-dom";

const linkStyle = ({ isActive }) => ({
  display: "block",
  padding: "8px 12px",
  fontSize: "14px",
  fontWeight: isActive ? "700" : "400",
  textDecoration: "none",
  borderRadius: "6px",
  backgroundColor: isActive ? "#222" : "transparent",
});

const AdminSidebar = () => {
  return (
    <aside
      style={{
        width: "220px",
        backgroundColor: "#000",
        color: "#f9fafb",
        padding: "16px",
        minHeight: "100vh",
      }}
    >
      <div style={{ marginBottom: "16px" }}>
        <div style={{ fontSize: "18px", fontWeight: "800" }}>COSMATE Admin</div>
        <div style={{ fontSize: "12px", color: "#9ca3af", marginTop: "2px" }}>
          관리자 콘솔
        </div>
      </div>

      <section style={{ marginBottom: "16px" }}>
        <div style={{ fontWeight: "600", marginBottom: "4px", fontSize: "13px" }}>
          메인
        </div>
        <NavLink to="/admin/dashboard" style={linkStyle}>
          대시보드
        </NavLink>
      </section>

      <section style={{ marginBottom: "16px" }}>
        <div style={{ fontWeight: "600", marginBottom: "4px", fontSize: "13px" }}>
          사용자 관리
        </div>
        <NavLink to="/admin/users" style={linkStyle}>
          회원 관리
        </NavLink>
      </section>

      <section style={{ marginBottom: "16px" }}>
        <div style={{ fontWeight: "600", marginBottom: "4px", fontSize: "13px" }}>
          콘텐츠 관리
        </div>
        <NavLink to="/admin/places" style={linkStyle}>
          관광지 관리
        </NavLink>
        <NavLink to="/admin/reviews" style={linkStyle}>
          리뷰 관리
        </NavLink>
        <NavLink to="/admin/community" style={linkStyle}>
          커뮤니티 관리
        </NavLink>
        <NavLink to="/admin/inquiries" style={linkStyle}>
          문의 관리
        </NavLink>
      </section>

      <section>
        <div style={{ fontWeight: "600", marginBottom: "4px", fontSize: "13px" }}>
          환경설정
        </div>
        <NavLink to="/admin/settings" style={linkStyle}>
          설정
        </NavLink>
      </section>
    </aside>
  );
};

export default AdminSidebar;
