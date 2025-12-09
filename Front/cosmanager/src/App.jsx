// src/App.jsx
import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./context/AuthContext.jsx";

import AdminLoginPage from "./pages/AdminLoginPage.jsx";
import AdminDashboardPage from "./pages/AdminDashboardPage.jsx";
import AdminUsersPage from "./pages/AdminUsersPage.jsx";
import AdminPlacesPage from "./pages/AdminPlacesPage.jsx";
import AdminReviewsTagsPage from "./pages/AdminReviewsTagsPage.jsx";
import AdminCommunityPage from "./pages/AdminCommunityPage.jsx";
import AdminSettingsPage from "./pages/AdminSettingsPage.jsx";
import AdminInquiryListPage from "./pages/AdminInquiryListPage.jsx";
import AdminInquiryDetailPage from "./pages/AdminInquiryDetailPage.jsx";

const App = () => {
  const { isLoggedIn } = useAuth();

  const requireLogin = (element) =>
    isLoggedIn ? element : <Navigate to="/admin/login" replace />;

  return (
    <Routes>
      {/* 관리자 로그인 */}
      <Route path="/admin/login" element={<AdminLoginPage />} />

      {/* 관리자 메인/서브 페이지 (로그인 필요) */}
      <Route
        path="/admin/dashboard"
        element={requireLogin(<AdminDashboardPage />)}
      />
      <Route
        path="/admin/users"
        element={requireLogin(<AdminUsersPage />)}
      />
      <Route
        path="/admin/places"
        element={requireLogin(<AdminPlacesPage />)}
      />
      <Route
        path="/admin/reviews"
        element={requireLogin(<AdminReviewsTagsPage />)}
      />
      <Route
        path="/admin/community"
        element={requireLogin(<AdminCommunityPage />)}
      />
      <Route
        path="/admin/inquiries"
        element={requireLogin(<AdminInquiryListPage />)}
      />
      <Route
        path="/admin/inquiries/:id"
        element={requireLogin(<AdminInquiryDetailPage />)}
      />
      <Route
        path="/admin/settings"
        element={requireLogin(<AdminSettingsPage />)}
      />

      {/* 그 외 경로는 로그인으로 재경로설정 */}
      <Route path="*" element={<Navigate to="/admin/login" replace />} />
    </Routes>
  );
};

export default App;
