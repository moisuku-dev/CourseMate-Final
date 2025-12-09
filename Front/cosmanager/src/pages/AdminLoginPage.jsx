// src/pages/AdminLoginPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { adminLogin } from "../api/adminAuth.js";
import { useAuth } from "../context/AuthContext.jsx";

const AdminLoginPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [form, setForm] = useState({ adminId: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!form.adminId || !form.password) {
      setError("아이디와 비밀번호를 모두 입력해주세요.");
      return;
    }

    setLoading(true);
    try {
      const data = await adminLogin({
        adminId: form.adminId,
        password: form.password,
      });
      // data: { id, name, token }
      login(data);
      navigate("/admin/dashboard");
    } catch (err) {
      console.error(err);
      setError("로그인에 실패했습니다. 아이디/비밀번호를 확인해주세요.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#050505",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          width: "360px",
          padding: "24px",
          borderRadius: "12px",
          border: "1px solid #333",
          backgroundColor: "#111",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "16px" }}>
          <div style={{ fontSize: "24px", fontWeight: "800" }}>
            코스메이트 관리자
          </div>
          <div style={{ fontSize: "12px", color: "#aaa", marginTop: "4px" }}>
            관리자 로그인
          </div>
        </div>

        <label
          style={{ display: "block", marginBottom: "12px", fontSize: "13px" }}
        >
          아이디
          <input
            type="text"
            name="adminId"
            value={form.adminId}
            onChange={handleChange}
            style={{
              marginTop: "4px",
              width: "100%",
              padding: "8px 10px",
              borderRadius: "8px",
              border: "1px solid #444",
              backgroundColor: "#000",
              color: "#f5f5f5",
            }}
          />
        </label>

        <label
          style={{ display: "block", marginBottom: "12px", fontSize: "13px" }}
        >
          비밀번호
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={handleChange}
            style={{
              marginTop: "4px",
              width: "100%",
              padding: "8px 10px",
              borderRadius: "8px",
              border: "1px solid #444",
              backgroundColor: "#000",
              color: "#f5f5f5",
            }}
          />
        </label>

        {error && (
          <div
            style={{
              color: "#ff6b6b",
              fontSize: "12px",
              marginBottom: "8px",
            }}
          >
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "none",
            backgroundColor: loading ? "#444" : "#f5f5f5",
            color: "#000",
            fontWeight: "700",
            marginTop: "4px",
          }}
        >
          {loading ? "로그인 중..." : "로그인"}
        </button>
      </form>
    </div>
  );
};

export default AdminLoginPage;

/*
//테스트로그인
// src/pages/AdminLoginPage.jsx
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";

const AdminLoginPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [form, setForm] = useState({ adminId: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!form.adminId || !form.password) {
      setError("아이디와 비밀번호를 모두 입력해주세요.");
      return;
    }

    setLoading(true);
    try {
      const fakeAdmin = {
        id: "admin",
        name: "테스트 관리자",
        token: "dummy-admin-token",
      };

      login(fakeAdmin);

      navigate("/admin/dashboard", { replace: true });
    } catch (err) {
      console.error(err);
      setError("로그인 처리 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#050505",
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          width: "360px",
          padding: "24px",
          borderRadius: "12px",
          border: "1px solid #333",
          backgroundColor: "#111",
        }}
      >
        <div style={{ textAlign: "center", marginBottom: "16px" }}>
          <div style={{ fontSize: "24px", fontWeight: "800" }}>
            코스메이트 관리자
          </div>
          <div style={{ fontSize: "12px", color: "#aaa", marginTop: "4px" }}>
            관리자 로그인
          </div>
        </div>

        <label style={{ display: "block", marginBottom: "12px", fontSize: "13px" }}>
          아이디
          <input
            type="text"
            name="adminId"
            value={form.adminId}
            onChange={handleChange}
            style={{
              marginTop: "4px",
              width: "100%",
              padding: "8px 10px",
              borderRadius: "8px",
              border: "1px solid #444",
              backgroundColor: "#000",
              color: "#f5f5f5",
            }}
          />
        </label>

        <label style={{ display: "block", marginBottom: "12px", fontSize: "13px" }}>
          비밀번호
          <input
            type="password"
            name="password"
            value={form.password}
            onChange={handleChange}
            style={{
              marginTop: "4px",
              width: "100%",
              padding: "8px 10px",
              borderRadius: "8px",
              border: "1px solid #444",
              backgroundColor: "#000",
              color: "#f5f5f5",
            }}
          />
        </label>

        {error && (
          <div style={{ color: "#ff6b6b", fontSize: "12px", marginBottom: "8px" }}>
            {error}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "none",
            backgroundColor: loading ? "#444" : "#f5f5f5",
            color: "#000",
            fontWeight: "700",
            marginTop: "4px",
          }}
        >
          {loading ? "로그인 중..." : "로그인"}
        </button>
      </form>
    </div>
  );
};

export default AdminLoginPage;
*/