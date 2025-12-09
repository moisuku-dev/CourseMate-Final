// app/settings.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  Switch,
  TextInput,
  TouchableOpacity,
  Alert,
  Platform, // ★ 1. Platform 추가
} from "react-native";
import { useRouter } from "expo-router"; // ★ useRouter 추가
import { fetchSettings, updateSettings, deleteAccount } from "../api/user";
import { useAuth } from "../hooks/useAuth";

export default function SettingsScreen() {
  const router = useRouter(); // 라우터 초기화
  const { logout } = useAuth();

  const [loading, setLoading] = useState(true);

  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [autoLogin, setAutoLogin] = useState(true);
  const [notification, setNotification] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchSettings();
        const s = data?.setting || data || {};
        setName(s.name || "");
        setAge(s.age != null ? String(s.age) : "");
        setGender(s.gender || "");
        // DB에 컬럼이 없다면 기본값 true로 설정
        setAutoLogin(s.autoLogin ?? true);
        setNotification(s.notification ?? true);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const onSave = async () => {
    try {
      // API 호출 (이름, 나이, 성별, 설정값 등 전송)
      await updateSettings({ 
        name, 
        age: Number(age), 
        gender, 
        autoLogin, 
        notification 
      });

      // ★ 2. 웹/앱 분기 알림
      if (Platform.OS === 'web') {
        window.alert("완료: 설정이 저장되었습니다.");
        router.replace("/my-page"); // 마이페이지로 이동
      } else {
        Alert.alert("완료", "설정이 저장되었습니다.", [
            { text: "확인", onPress: () => router.replace("/my-page") }
        ]);
      }
    } catch (e) {
      console.error(e);
      const msg = "설정 저장 실패";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
    }
  };

  const onDeleteAccount = () => {
    const msg = "정말로 탈퇴하시겠습니까? 이 작업은 되돌릴 수 없습니다.";
    
    // ★ 3. 회원탈퇴 확인 팝업 (웹/앱 분기)
    if (Platform.OS === 'web') {
      if (window.confirm(msg)) {
        handleRealDelete();
      }
    } else {
      Alert.alert("회원 탈퇴", msg, [
        { text: "취소", style: "cancel" },
        { text: "탈퇴", style: "destructive", onPress: handleRealDelete },
      ]);
    }
  };

  const handleRealDelete = async () => {
    try {
      // 주의: 백엔드 deleteAccount가 비밀번호를 요구한다면, 
      // 여기서 비밀번호 입력 모달을 띄우거나 API를 수정해야 합니다.
      // 현재는 일단 호출만 하도록 작성됨.
      await deleteAccount(); 
      
      if (Platform.OS === 'web') {
        window.alert("탈퇴가 완료되었습니다.");
        router.replace("/login");
      } else {
        Alert.alert("완료", "탈퇴가 완료되었습니다.", [
          { text: "확인", onPress: () => router.replace("/login") }
        ]);
      }
      logout(); // 로그아웃 처리
    } catch (e) {
      console.error(e);
      const msg = "탈퇴 처리에 실패했습니다. (비밀번호 확인 필요)";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
    }
  };

  return (
    <View style={{ flex: 1, padding: 16, backgroundColor: '#fff' }}>
      <Text style={{ fontSize: 20, fontWeight: "700", marginBottom: 20 }}>
        환경 설정
      </Text>

      <Text style={label}>이름</Text>
      <TextInput style={input} value={name} onChangeText={setName} />

      <Text style={label}>나이</Text>
      <TextInput 
        style={input} 
        value={age} 
        onChangeText={setAge} 
        keyboardType="numeric" 
      />

      <Text style={label}>성별 (M/F)</Text>
      <TextInput style={input} value={gender} onChangeText={setGender} />

      <View
        style={{
          marginTop: 24,
          paddingTop: 16,
          borderTopWidth: 1,
          borderTopColor: "#eee",
        }}
      >
        <Text style={{ fontSize: 16, fontWeight: "600", marginBottom: 8 }}>
          로그인 & 서비스 설정
        </Text>

        <View style={row}>
          <Text style={{ fontSize: 15 }}>자동 로그인</Text>
          <Switch value={autoLogin} onValueChange={setAutoLogin} />
        </View>

        <View style={row}>
          <Text style={{ fontSize: 15 }}>알림 받기</Text>
          <Switch value={notification} onValueChange={setNotification} />
        </View>
      </View>

      {/* 저장 버튼 */}
      <TouchableOpacity onPress={onSave} style={btnPrimary}>
        <Text style={{ color: "#fff", fontWeight: "600" }}>설정 저장</Text>
      </TouchableOpacity>

      {/* 회원탈퇴 */}
      <TouchableOpacity onPress={onDeleteAccount} style={btnDanger}>
        <Text style={{ color: "#EF4444", fontWeight: "600" }}>회원탈퇴</Text>
      </TouchableOpacity>
    </View>
  );
}

const label = {
  marginTop: 12,
  marginBottom: 4,
  fontWeight: "500",
  color: "#333"
};

const input = {
  borderWidth: 1,
  borderColor: "#ccc",
  borderRadius: 8,
  padding: 10,
  fontSize: 14,
  backgroundColor: "#f9f9f9"
};

const row = {
  marginTop: 12,
  flexDirection: "row",
  justifyContent: "space-between",
  alignItems: "center",
};

const btnPrimary = {
  marginTop: 30,
  backgroundColor: "#4F46E5",
  paddingVertical: 14,
  borderRadius: 8,
  alignItems: "center",
};

const btnDanger = {
  marginTop: 16,
  paddingVertical: 14,
  borderRadius: 8,
  borderWidth: 1,
  borderColor: "#EF4444",
  alignItems: "center",
};