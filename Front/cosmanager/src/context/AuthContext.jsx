// src/context/AuthContext.jsx
import React, { createContext, useContext, useEffect, useState } from "react";

const STORAGE_KEY = "cosmate_admin_auth";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [admin, setAdmin] = useState(null); // { id, name, token }

  // 앱 시작 시 로컬스토리지에서 로그인 정보 복원
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setAdmin(parsed);
      }
    } catch (e) {
      console.error("Failed to load admin auth from storage:", e);
    }
  }, []);

  const login = (adminData) => {
    setAdmin(adminData);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(adminData));
    } catch (e) {
      console.error("Failed to save admin auth to storage:", e);
    }
  };

  const logout = () => {
    setAdmin(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.error("Failed to remove admin auth from storage:", e);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        admin,
        isLoggedIn: !!admin,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
