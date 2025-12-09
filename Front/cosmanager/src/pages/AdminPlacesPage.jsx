// src/pages/AdminPlacesPage.jsx
import React, { useEffect, useState } from "react";
import AdminSidebar from "../components/AdminSidebar.jsx";
import {
  createAdminPlace,
  deleteAdminPlace,
  fetchAdminPlaces, // ✨ 새로 추가한 API 함수 임포트
} from "../api/adminPlaces.js";

const EMPTY_PLACE_FORM = {
  spotId: "",
  name: "",
  address: "",
  category: "관광지", // 기본값
  latitude: "",
  longitude: "",
};

const AdminPlacesPage = () => {
  // 상태 관리
  const [places, setPlaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  
  const [form, setForm] = useState(EMPTY_PLACE_FORM);

  // 1. 목록 불러오기 함수
  const loadPlaces = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await fetchAdminPlaces(); // API 호출
      setPlaces(Array.isArray(data) ? data : []);
    } catch (err) {
      console.error(err);
      setError("관광지 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  // 2. 화면 켜질 때 목록 조회
  useEffect(() => {
    loadPlaces();
  }, []);

  // 입력 폼 핸들러
  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  // 등록 핸들러
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!form.spotId || !form.name) {
      alert("ID와 이름은 필수입니다.");
      return;
    }
    try {
      await createAdminPlace(form);
      alert("관광지가 등록되었습니다.");
      setForm(EMPTY_PLACE_FORM); // 폼 초기화
      loadPlaces(); // 목록 새로고침
    } catch (err) {
      console.error(err);
      alert("등록 실패: " + err.message);
    }
  };

  // 삭제 핸들러
  const handleDelete = async (id) => {
    if (!window.confirm("정말 삭제하시겠습니까?")) return;
    try {
      await deleteAdminPlace(id);
      loadPlaces(); // 목록 새로고침
    } catch (err) {
      console.error(err);
      alert("삭제 실패: " + err.message);
    }
  };

  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>
      <AdminSidebar />
      <div style={{ flex: 1, padding: "24px" }}>
        <h1 style={{ fontSize: "20px", marginBottom: "12px" }}>관광지 관리</h1>

        {/* --- 등록 폼 --- */}
        <section
          style={{
            marginBottom: "24px",
            padding: "16px",
            border: "1px solid #333",
            borderRadius: "8px",
            backgroundColor: "#111",
          }}
        >
          <h2 style={{ fontSize: "16px", marginBottom: "12px" }}>새 관광지 등록</h2>
          <form onSubmit={handleSubmit} style={{ display: "grid", gap: "8px" }}>
            <div style={{ display: "flex", gap: "8px" }}>
              <input
                name="spotId"
                placeholder="ID (예: SPOT001)"
                value={form.spotId}
                onChange={handleChange}
                style={{ flex: 1, padding: "6px", backgroundColor: "#000", color: "#fff", border: "1px solid #444" }}
              />
              <input
                name="name"
                placeholder="이름"
                value={form.name}
                onChange={handleChange}
                style={{ flex: 2, padding: "6px", backgroundColor: "#000", color: "#fff", border: "1px solid #444" }}
              />
            </div>
            <input
              name="address"
              placeholder="주소"
              value={form.address}
              onChange={handleChange}
              style={{ width: "100%", padding: "6px", backgroundColor: "#000", color: "#fff", border: "1px solid #444" }}
            />
            <div style={{ display: "flex", gap: "8px" }}>
              <select
                name="category"
                value={form.category}
                onChange={handleChange}
                style={{ padding: "6px", backgroundColor: "#000", color: "#fff", border: "1px solid #444" }}
              >
                <option value="관광지">관광지</option>
                <option value="식당">식당</option>
                <option value="숙소">숙소</option>
                <option value="카페">카페</option>
              </select>
              <input
                name="latitude"
                placeholder="위도"
                value={form.latitude}
                onChange={handleChange}
                style={{ flex: 1, padding: "6px", backgroundColor: "#000", color: "#fff", border: "1px solid #444" }}
              />
              <input
                name="longitude"
                placeholder="경도"
                value={form.longitude}
                onChange={handleChange}
                style={{ flex: 1, padding: "6px", backgroundColor: "#000", color: "#fff", border: "1px solid #444" }}
              />
            </div>
            <button
              type="submit"
              style={{
                marginTop: "8px",
                padding: "8px",
                backgroundColor: "#f5f5f5",
                color: "#000",
                fontWeight: "bold",
                border: "none",
                borderRadius: "4px",
                cursor: "pointer",
              }}
            >
              등록하기
            </button>
          </form>
        </section>

        {/* --- 목록 표시 영역 (수정된 부분) --- */}
        <section>
          <h2 style={{ fontSize: "16px", marginBottom: "8px" }}>등록된 관광지 목록</h2>
          
          {loading && <p>불러오는 중...</p>}
          {error && <p style={{ color: "red" }}>{error}</p>}

          {!loading && !error && (
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: "13px",
                  minWidth: "600px",
                }}
              >
                <thead>
                  <tr style={{ borderBottom: "1px solid #444" }}>
                    <th style={{ padding: "8px", textAlign: "left" }}>ID</th>
                    <th style={{ padding: "8px", textAlign: "left" }}>카테고리</th>
                    <th style={{ padding: "8px", textAlign: "left" }}>이름</th>
                    <th style={{ padding: "8px", textAlign: "left" }}>주소</th>
                    <th style={{ padding: "8px", textAlign: "center" }}>관리</th>
                  </tr>
                </thead>
                <tbody>
                  {places.map((place) => (
                    <tr key={place.id} style={{ borderBottom: "1px solid #222" }}>
                      <td style={{ padding: "8px" }}>{place.id}</td>
                      <td style={{ padding: "8px" }}>{place.category}</td>
                      <td style={{ padding: "8px", fontWeight: "bold" }}>{place.name}</td>
                      <td style={{ padding: "8px", color: "#aaa" }}>{place.address}</td>
                      <td style={{ padding: "8px", textAlign: "center" }}>
                        <button
                          onClick={() => handleDelete(place.id)}
                          style={{
                            padding: "4px 8px",
                            backgroundColor: "#4a0000",
                            color: "#ffaaaa",
                            border: "1px solid #ff4444",
                            borderRadius: "4px",
                            cursor: "pointer",
                            fontSize: "12px",
                          }}
                        >
                          삭제
                        </button>
                      </td>
                    </tr>
                  ))}
                  {places.length === 0 && (
                    <tr>
                      <td colSpan="5" style={{ padding: "16px", textAlign: "center", color: "#666" }}>
                        등록된 관광지가 없습니다.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
};

export default AdminPlacesPage;