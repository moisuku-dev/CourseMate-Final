// context/AuthContext.js
// 수정된 버전: SecureStore 대신 utils/storage.js 사용 (웹/앱 호환)

import React, {
  createContext,
  useState,
  useEffect,
  useCallback,
} from "react";
// expo-secure-store는 제거하거나 주석 처리
// import * as SecureStore from "expo-secure-store"; 

// ✨ [핵심 수정] 웹과 앱 모두 지원하는 저장소 유틸리티 불러오기
import { saveToken, removeToken, getToken } from "../utils/storage";
import { login as loginApi, fetchMe } from "../api/auth";

const TOKEN_KEY = "cosmate_user_token";

export const AuthContext = createContext({
  user: null,
  token: null,
  loading: true,
  login: async () => {},
  logout: async () => {},
  refreshUser: async () => {},
});

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  /**
   * 앱 시작 시:
   * 1) 저장소(웹: localStorage, 앱: SecureStore)에서 토큰 로드
   * 2) 토큰 있으면 fetchMe()로 내 정보 조회
   */
  useEffect(() => {
    let isMounted = true;

    (async () => {
      try {
        // 🔍 [수정] SecureStore.getItemAsync -> getToken
        const storedToken = await getToken(TOKEN_KEY);

        if (!storedToken) {
          if (isMounted) {
            setToken(null);
            setUser(null);
          }
          return;
        }

        if (isMounted) {
          setToken(storedToken);
        }

        try {
          // 토큰으로 내 정보 가져오기
          const me = await fetchMe();
          if (isMounted) {
            // fetchMe()는 api/auth.js 기준으로 data.setting 반환 ({ name, email, ... })
            setUser(me || null);
          }
        } catch (e) {
          console.warn("fetchMe failed, clearing token", e);
          // 🔍 [수정] 내 정보 조회 실패 시 토큰 삭제: SecureStore.deleteItemAsync -> removeToken
          await removeToken(TOKEN_KEY);
          if (isMounted) {
            setToken(null);
            setUser(null);
          }
        }
      } catch (e) {
        console.warn("Failed to load token", e);
        if (isMounted) {
          setToken(null);
          setUser(null);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    })();

    return () => {
      isMounted = false;
    };
  }, []);

  /**
   * 로그인
   * - login(email, password)
   * - 또는 login({ email, password }) 둘 다 지원
   */
  const handleLogin = useCallback(async (arg1, arg2) => {
    let email;
    let password;

    if (typeof arg1 === "string") {
      email = arg1;
      password = arg2;
    } else if (arg1 && typeof arg1 === "object") {
      email = arg1.email;
      password = arg1.password;
    }

    if (!email || !password) {
      throw new Error("이메일과 비밀번호를 입력해주세요.");
    }

    // 실제 로그인 API 호출
    const res = await loginApi({ email, password });

    // 토큰 필드 찾기 (token 또는 accessToken)
    const nextToken = res?.token ?? res?.accessToken ?? null;

    if (!nextToken) {
      throw new Error("로그인 응답에 토큰이 없습니다. 백엔드 응답 형식을 확인하세요.");
    }

    // 🔍 [수정] 토큰 저장: SecureStore.setItemAsync -> saveToken
    try {
      await saveToken(TOKEN_KEY, String(nextToken));
    } catch (e) {
      console.warn("Failed to save token", e);
    }

    setToken(String(nextToken));

    // 유저 정보 세팅 (응답에 없으면 fetchMe 호출)
    if (res?.user || res?.setting) {
      const baseUser = res.user || res.setting;
      setUser(baseUser);
    } else {
      try {
        const me = await fetchMe();
        setUser(me || null);
      } catch (e) {
        console.warn("fetchMe after login failed", e);
        setUser(null);
      }
    }

    return res;
  }, []);

  /**
   * 로그아웃
   * - 저장소에서 토큰 삭제 및 상태 초기화
   */
  const handleLogout = useCallback(async () => {
    try {
      // 🔍 [수정] 토큰 삭제: SecureStore.deleteItemAsync -> removeToken
      await removeToken(TOKEN_KEY);
    } catch (e) {
      console.warn("Failed to delete token on logout", e);
    } finally {
      setUser(null);
      setToken(null);
    }
  }, []);

  /**
   * 내 정보 새로고침
   */
  const refreshUser = useCallback(async () => {
    try {
      const me = await fetchMe();
      setUser(me || null);
      return me;
    } catch (e) {
      console.warn("refreshUser failed", e);
      return null;
    }
  }, []);

  const value = {
    user,
    token,
    loading,
    login: handleLogin,
    logout: handleLogout,
    refreshUser,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}