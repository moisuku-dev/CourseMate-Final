// src/pages/AdminSettingsPage.jsx
import React, { useEffect, useState } from "react";
import AdminSidebar from "../components/AdminSidebar.jsx";
import { fetchAdminSettings, updateAdminSettings } from "../api/adminSettings.js";

const AdminSettingsPage = () => {
  const [settings, setSettings] = useState({
    loginFailedLimit: 5,
    lockMinutes: 30,
    allowNewAdmins: true,
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const loadSettings = async () => {
    setLoading(true);
    setError("");
    setMessage("");
    try {
      const data = await fetchAdminSettings();
      if (data) setSettings(data);
    } catch (e) {
      console.error(e);
      setError("");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadSettings();
  }, []);

  const handleChangeNumber = (field, value) => {
    const num = Number(value);
    setSettings((prev) => ({
      ...prev,
      [field]: Number.isNaN(num) ? prev[field] : num,
    }));
  };

  const handleChangeCheckbox = (field, checked) => {
    setSettings((prev) => ({
      ...prev,
      [field]: checked,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setMessage("");
    try {
      await updateAdminSettings(settings);
      setMessage("설정이 저장되었습니다.");
    } catch (e) {
      console.error(e);
      setError("설정 저장에 실패했습니다.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <AdminSidebar />
      <div style={{ flex: 1, padding: "24px", maxWidth: "640px" }}>
        <h1 style={{ fontSize: "20px", marginBottom: "12px" }}>환경설정</h1>

        {loading && <div style={{ fontSize: "13px" }}>설정 불러오는 중...</div>}
        {error && (
          <div style={{ fontSize: "13px", color: "#ff6b6b", marginBottom: "8px" }}>
            {error}
          </div>
        )}
        {message && (
          <div style={{ fontSize: "13px", color: "#8ce99a", marginBottom: "8px" }}>
            {message}
          </div>
        )}

        <form
          onSubmit={handleSubmit}
          style={{
            borderRadius: "10px",
            border: "1px solid #333",
            padding: "12px",
            backgroundColor: "#111",
          }}
        >
          <h2 style={{ fontSize: "16px", marginTop: 0, marginBottom: "8px" }}>로그인 정책</h2>
          <div style={{ marginBottom: "10px" }}>
            <label style={{ fontSize: "13px" }}>
              로그인 실패 허용 횟수
              <input
                type="number"
                min={1}
                max={20}
                value={settings.loginFailedLimit}
                onChange={(e) => handleChangeNumber("loginFailedLimit", e.target.value)}
                style={{
                  marginTop: "4px",
                  width: "100%",
                  maxWidth: "200px",
                  padding: "6px 8px",
                  borderRadius: "6px",
                  border: "1px solid #444",
                  backgroundColor: "#000",
                  color: "#f5f5f5",
                  fontSize: "13px",
                }}
              />
            </label>
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label style={{ fontSize: "13px" }}>
              계정 잠금 시간(분)
              <input
                type="number"
                min={1}
                max={1440}
                value={settings.lockMinutes}
                onChange={(e) => handleChangeNumber("lockMinutes", e.target.value)}
                style={{
                  marginTop: "4px",
                  width: "100%",
                  maxWidth: "200px",
                  padding: "6px 8px",
                  borderRadius: "6px",
                  border: "1px solid #444",
                  backgroundColor: "#000",
                  color: "#f5f5f5",
                  fontSize: "13px",
                }}
              />
            </label>
          </div>

          <div style={{ marginBottom: "10px" }}>
            <label
              style={{
                fontSize: "13px",
                display: "inline-flex",
                alignItems: "center",
              }}
            >
              <input
                type="checkbox"
                checked={settings.allowNewAdmins}
                onChange={(e) => handleChangeCheckbox("allowNewAdmins", e.target.checked)}
                style={{ marginRight: "6px" }}
              />
              다른 관리자가 새 관리자 계정 추가 가능
            </label>
          </div>

          <div style={{ marginTop: "12px", display: "flex", gap: "8px" }}>
            <button
              type="submit"
              disabled={saving}
              style={{
                padding: "6px 12px",
                borderRadius: "6px",
                border: "1px solid #444",
                backgroundColor: saving ? "#444" : "#f5f5f5",
                color: "#000",
                fontSize: "13px",
                fontWeight: "600",
              }}
            >
              {saving ? "저장 중..." : "저장"}
            </button>
            <button
              type="button"
              onClick={loadSettings}
              style={{
                padding: "6px 12px",
                borderRadius: "6px",
                border: "1px solid #444",
                backgroundColor: "#111",
                fontSize: "13px",
              }}
            >
              다시 불러오기
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AdminSettingsPage;
