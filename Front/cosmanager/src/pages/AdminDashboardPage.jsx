// src/pages/AdminDashboardPage.jsx
import React, { useEffect, useState } from "react";
import AdminSidebar from "../components/AdminSidebar.jsx";
import { fetchDashboardStats } from "../api/adminDashboard.js";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts";

// 1. CSV 데이터를 직접 배열로 정의
const aiModelData = [
  { progress: 25, accuracy: 0.7000, microF1: 0.8514, macroF1: 0.6368 },
  { progress: 50, accuracy: 0.6875, microF1: 0.8513, macroF1: 0.6365 }, // 50% 지점 데이터 (변동 극대화)
  { progress: 75, accuracy: 0.6991, microF1: 0.8515, macroF1: 0.6350 },
  { progress: 100, accuracy: 0.7016, microF1: 0.8513, macroF1: 0.6351 },
];

// 2. 재사용 가능한 단일 차트 컴포넌트 정의
const PerformanceChart = ({ data, dataKey, name, color, domain }) => (
  <div style={{ width: "100%", height: 250, padding: "10px" }}>
    <h3 style={{ color: color, fontSize: "15px", fontWeight: "600", marginBottom: "10px", textAlign: 'center' }}>
        {name} 추이
    </h3>
    <ResponsiveContainer width="100%" height="90%">
      <LineChart
        data={data}
        margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
        <XAxis 
          dataKey="progress" 
          stroke="#94a3b8" 
          tick={{ fill: '#94a3b8' }}
          label={{ value: '진행률 (%)', position: 'insideBottom', offset: 0, fill: '#94a3b8' }} 
        />
        <YAxis 
          stroke="#94a3b8" 
          domain={domain} // ★ 극적 표현을 위해 동적으로 Y축 범위 설정
          tickFormatter={(value) => value.toFixed(4)} // 소수점 4자리까지 표시
          tick={{ fill: '#94a3b8' }}
        />
        <Tooltip 
          contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#fff' }}
          itemStyle={{ color: color }}
          formatter={(value) => [value.toFixed(4), name]} 
        />
        <Line 
          type="monotone" 
          dataKey={dataKey} 
          name={name}
          stroke={color} 
          strokeWidth={3}
          dot={{ r: 4 }}
          activeDot={{ r: 6 }} 
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
);


const AdminDashboardPage = () => {
  const [stats, setStats] = useState(null); 
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const loadStats = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await fetchDashboardStats();
      if (res && res.result_code === 200) {
        setStats(res.stats || null);
      } else {
        setError(res?.result_msg || "통계 조회에 실패했습니다.");
      }
    } catch (e) {
      console.error(e);
      setError("통계 조회 중 오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStats();
  }, []);

  return (
    <div style={{ display: "flex", minHeight: "100vh", backgroundColor: "#020617" }}>
      <AdminSidebar />
      <main style={{ flex: 1, padding: "24px 28px", backgroundColor: "#020617", overflowY: "auto" }}>
        <h1
          style={{
            fontSize: "24px",
            fontWeight: "700",
            color: "#f9fafb",
            marginBottom: "4px",
          }}
        >
          대시보드
        </h1>
        <p style={{ fontSize: "13px", color: "#9ca3af", marginBottom: "16px" }}>
          코스메이트 서비스의 전체 현황을 한눈에 확인합니다.
        </p>

        {/* 상단 통계 카드 섹션 */}
        {loading && (
          <div style={{ color: "#e5e7eb", fontSize: "14px" }}>
            통계를 불러오는 중입니다...
          </div>
        )}
        {error && (
          <div style={{ color: "#fecaca", fontSize: "13px", marginBottom: "8px" }}>
            {error}
          </div>
        )}

        {stats && (
          <section
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: "16px",
              marginBottom: "32px",
            }}
          >
            <StatCard label="전체 회원 수" value={stats.totalUsers} />
            <StatCard label="전체 리뷰 수" value={stats.totalReviews} />
            <StatCard label="등록된 관광지 수" value={stats.totalSpots} />
          </section>
        )}

        {/* 3. 하단 AI 학습 그래프 섹션 (3개 분할) */}
        <section
          style={{
            backgroundColor: "#0f172a",
            borderRadius: "14px",
            padding: "24px",
            border: "1px solid rgba(148,163,184,0.15)",
            boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
          }}
        >
          <h2 style={{ fontSize: "18px", fontWeight: "600", color: "#f9fafb", marginBottom: "10px" }}>
            AI 모델 학습 성과 (지표별 추이)
          </h2>
          
          <div 
            style={{ 
              display: "flex", 
              flexWrap: "wrap", 
              justifyContent: "space-between", 
              gap: "10px" 
            }}
          >
            {/* 1. 정확도 (Accuracy) 그래프 */}
            <div style={{ flex: 1, minWidth: "300px" }}>
              <PerformanceChart
                data={aiModelData}
                dataKey="accuracy"
                name="정확도 (Accuracy)"
                color="#8884d8"
                domain={[0.605, 0.745]} // ★ 범위: 0.6850 ~ 0.7050
              />
            </div>

            {/* 2. Micro F1 Score 그래프 */}
            <div style={{ flex: 1, minWidth: "300px" }}>
              <PerformanceChart
                data={aiModelData}
                dataKey="microF1"
                name="Micro F1 Score"
                color="#82ca9d"
                domain={[0.8452, 0.8586]} // ★ 범위: 0.8512 ~ 0.8516 (변동 폭 최소화)
              />
            </div>
            
            {/* 3. Macro F1 Score 그래프 */}
            <div style={{ flex: 1, minWidth: "300px" }}>
              <PerformanceChart
                data={aiModelData}
                dataKey="macroF1"
                name="Macro F1 Score"
                color="#ffc658"
                domain={[0.6245, 0.6400]} // ★ 범위: 0.6345 ~ 0.6375
              />
            </div>

          </div>
        </section>

      </main>
    </div>
  );
};

const StatCard = ({ label, value }) => {
  return (
    <div
      style={{
        backgroundColor: "#020617",
        borderRadius: "14px",
        padding: "16px 18px",
        border: "1px solid rgba(148,163,184,0.35)",
        boxShadow: "0 8px 15px -3px rgba(0, 0, 0, 0.5)",
      }}
    >
      <div style={{ color: "#9ca3af", fontSize: "13px", marginBottom: "8px" }}>
        {label}
      </div>
      <div style={{ color: "#f9fafb", fontSize: "24px", fontWeight: "700" }}>
        {value ? value.toLocaleString() : "0"}
      </div>
    </div>
  );
};

export default AdminDashboardPage;