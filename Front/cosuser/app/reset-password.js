// app/reset-password.js
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
import { resetPassword } from "../api/auth";

export default function ResetPasswordScreen() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [code, setCode] = useState(""); // 인증코드 쓰면
  const [newPassword, setNewPassword] = useState("");
  const [newPasswordCheck, setNewPasswordCheck] = useState("");
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    if (!email.trim() || !newPassword.trim() || !newPasswordCheck.trim()) {
      Alert.alert("알림", "이메일과 새 비밀번호를 모두 입력해 주세요.");
      return;
    }
    if (newPassword !== newPasswordCheck) {
      Alert.alert("알림", "새 비밀번호가 일치하지 않습니다.");
      return;
    }

    try {
      setLoading(true);
      const payload = {
        email: email.trim(),
        newPassword: newPassword.trim(),
      };
      if (code.trim()) payload.code = code.trim();

      const data = await resetPassword(payload);

      if (data.result_code && data.result_code !== 200) {
        Alert.alert("오류", data.result_msg || "비밀번호 재설정에 실패했습니다.");
        return;
      }

      Alert.alert("완료", "비밀번호가 재설정되었습니다.", [
        { text: "확인", onPress: () => router.replace("/login") },
      ]);
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "비밀번호 재설정 중 문제가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScreenContainer>
      <Text style={styles.title}>비밀번호 재설정</Text>

      <Text style={styles.label}>이메일</Text>
      <TextInput
        style={styles.input}
        value={email}
        onChangeText={setEmail}
        placeholder="가입하신 이메일"
        keyboardType="email-address"
        autoCapitalize="none"
      />

      <Text style={styles.label}>인증 코드 (선택)</Text>
      <TextInput
        style={styles.input}
        value={code}
        onChangeText={setCode}
        placeholder="이메일로 받은 코드가 있다면 입력"
      />

      <Text style={styles.label}>새 비밀번호</Text>
      <TextInput
        style={styles.input}
        secureTextEntry
        value={newPassword}
        onChangeText={setNewPassword}
        placeholder="새 비밀번호"
      />

      <Text style={styles.label}>새 비밀번호 확인</Text>
      <TextInput
        style={styles.input}
        secureTextEntry
        value={newPasswordCheck}
        onChangeText={setNewPasswordCheck}
        placeholder="새 비밀번호 다시 입력"
      />

      <TouchableOpacity
        style={[styles.button, loading && styles.buttonDisabled]}
        onPress={onSubmit}
        disabled={loading}
      >
        <Text style={styles.buttonText}>
          {loading ? "처리 중..." : "비밀번호 재설정"}
        </Text>
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
});
