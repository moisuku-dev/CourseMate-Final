// app/my-page.js
import React from "react";
import { View, Text, TouchableOpacity, Alert } from "react-native";
import { useRouter } from "expo-router";
import { useAuth } from "../hooks/useAuth";

export default function MyPageScreen() {
  const router = useRouter();
  const { user, logout } = useAuth();

  // ★ [핵심 수정] 로그아웃 처리 함수 추가
  const handleLogout = async () => {
    try {
      await logout(); // 1. 로그아웃 실행 (토큰 삭제 등)
      
      // 2. 로그인 화면으로 강제 이동 (뒤로가기 방지 위해 replace 사용)
      router.replace("/login"); 
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "로그아웃 중 문제가 발생했습니다.");
    }
  };

  const MenuItem = ({ label, onPress }) => (
    <TouchableOpacity
      onPress={onPress}
      style={{
        paddingVertical: 14,
        borderBottomWidth: 1,
        borderBottomColor: "#eee",
      }}
    >
      <Text style={{ fontSize: 16 }}>{label}</Text>
    </TouchableOpacity>
  );

  return (
    <View style={{ flex: 1, padding: 16, backgroundColor: '#fff' }}>
      <View style={{ marginBottom: 24 }}>
        <Text style={{ fontSize: 22, fontWeight: "700" }}>마이페이지</Text>
        <Text style={{ marginTop: 8, fontSize: 16 }}>
          {user?.name || "코스메이트 사용자"}
        </Text>
        <Text style={{ marginTop: 2, color: "#666" }}>{user?.email}</Text>
      </View>

      <MenuItem
        label="위시리스트"
        onPress={() => router.push("/wishlist")}
      />
      <MenuItem
        label="내 리뷰 / 별점 내역"
        onPress={() => router.push("/my-reviews")}
      />
      <MenuItem
        label="내 취향 태그 설정"
        onPress={() => router.push("/preferences")}
      />
      <MenuItem
        label="AI 맞춤 관광 코스 추천"
        onPress={() => router.push("/recommendations")}
      />
      <MenuItem
        label="환경 설정"
        onPress={() => router.push("/settings")}
      />
      <MenuItem
        label="공지 / 문의 (커뮤니티)"
        onPress={() => router.push("/community")}
      />

      <TouchableOpacity
        // ★ [핵심 수정] 여기서 handleLogout 함수 호출
        onPress={handleLogout}
        style={{
          marginTop: 24,
          borderWidth: 1,
          borderColor: "#EF4444",
          borderRadius: 8,
          paddingVertical: 12,
          alignItems: "center",
        }}
      >
        <Text style={{ color: "#EF4444", fontWeight: "600" }}>로그아웃</Text>
      </TouchableOpacity>
    </View>
  );
}