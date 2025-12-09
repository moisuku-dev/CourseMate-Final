// app/find-id.js
import React, { useState } from "react";
import {
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from "react-native";
import { useRouter } from "expo-router";
import ScreenContainer from "../components/ScreenContainer";
import COLORS from "../constants/colors";
import { findId } from "../api/auth";

export default function FindIdScreen() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    if (!name.trim() || !email.trim()) {
      Alert.alert("알림", "이름과 이메일을 모두 입력해 주세요.");
      return;
    }

    try {
      setLoading(true);
      const data = await findId({ name: name.trim(), email: email.trim() });

      if (data.result_code && data.result_code !== 200) {
        Alert.alert("알림", data.result_msg || "아이디 찾기에 실패했습니다.");
        return;
      }

      const foundId = data.foundId || data.userId || data.loginId;
      if (foundId) {
        setResult(`가입하신 아이디는 "${foundId}" 입니다.`);
      } else {
        setResult("해당 정보로 가입된 아이디가 없습니다.");
      }
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "아이디 찾기 중 문제가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScreenContainer>
      <Text style={styles.title}>아이디 찾기</Text>

      <Text style={styles.label}>이름</Text>
      <TextInput
        style={styles.input}
        value={name}
        onChangeText={setName}
        placeholder="이름을 입력하세요"
      />

      <Text style={styles.label}>이메일</Text>
      <TextInput
        style={styles.input}
        value={email}
        onChangeText={setEmail}
        placeholder="가입하신 이메일"
        keyboardType="email-address"
        autoCapitalize="none"
      />

      <TouchableOpacity
        style={[styles.button, loading && styles.buttonDisabled]}
        onPress={onSubmit}
        disabled={loading}
      >
        <Text style={styles.buttonText}>
          {loading ? "조회 중..." : "아이디 찾기"}
        </Text>
      </TouchableOpacity>

      {!!result && <Text style={styles.result}>{result}</Text>}

      <TouchableOpacity onPress={() => router.back()} style={styles.secondary}>
        <Text style={styles.secondaryText}>로그인 화면으로 돌아가기</Text>
      </TouchableOpacity>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  title: {
    fontSize: 20,
    fontWeight: "700",
    color: COLORS.text,
    marginBottom: 20,
  },
  label: {
    fontSize: 14,
    fontWeight: "500",
    color: COLORS.text,
    marginBottom: 6,
    marginTop: 12,
  },
  input: {
    borderWidth: 1,
    borderColor: COLORS.border || "#ddd",
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 14,
    backgroundColor: COLORS.backgroundSecondary || "#fff",
  },
  button: {
    marginTop: 24,
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: "center",
    backgroundColor: COLORS.primary || "#4F46E5",
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: COLORS.onPrimary || "#fff",
    fontWeight: "600",
    fontSize: 15,
  },
  result: {
    marginTop: 20,
    color: COLORS.text,
  },
  secondary: {
    marginTop: 16,
    alignItems: "center",
  },
  secondaryText: {
    color: COLORS.muted,
  },
});
