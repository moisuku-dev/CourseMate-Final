// app/login.js
import React, { useState } from "react";
import { View, Text, TextInput, StyleSheet, TouchableOpacity } from "react-native";
import { useRouter, Link } from "expo-router";
import ScreenContainer from "../components/ScreenContainer";
import CommonButton from "../components/CommonButton";
import COLORS from "../constants/colors";
import { useAuth } from "../hooks/useAuth";

export default function LoginScreen() {
  const router = useRouter();
  const { login } = useAuth();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async () => {
    setError("");
    setSubmitting(true);
    try {
      await login(email, password);
      router.replace("/"); // 로그인 성공 시 홈으로
    } catch (e) {
      console.warn(e);
      setError("로그인에 실패했습니다. 아이디/비밀번호를 확인해 주세요.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ScreenContainer>
      <View style={styles.header}>
        <Text style={styles.title}>COSMATE</Text>
        <Text style={styles.subtitle}>여행 코스를 추천받아보세요!</Text>
      </View>

      <View style={styles.form}>
        <Text style={styles.label}>이메일</Text>
        <TextInput
          style={styles.input}
          placeholder="example@email.com"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
        <Text style={styles.label}>비밀번호</Text>
        <TextInput
          style={styles.input}
          placeholder="비밀번호"
          secureTextEntry
          value={password}
          onChangeText={setPassword}
        />

        {error ? <Text style={styles.error}>{error}</Text> : null}

        <CommonButton
          title="로그인"
          onPress={handleLogin}
          disabled={submitting || !email || !password}
        />

        <Link href="/sign-up" asChild>
          <TouchableOpacity>
            <Text style={styles.linkText}>아직 회원이 아니신가요? 회원가입</Text>
          </TouchableOpacity>
        </Link>
      </View>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  header: {
    marginBottom: 32,
  },
  title: {
    fontSize: 28,
    fontWeight: "800",
    color: COLORS.primary,
  },
  subtitle: {
    marginTop: 8,
    fontSize: 14,
    color: COLORS.muted,
  },
  form: {
    marginTop: 8,
  },
  label: {
    fontSize: 14,
    color: COLORS.text,
    marginTop: 12,
    marginBottom: 4,
  },
  input: {
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 14,
  },
  error: {
    marginTop: 8,
    color: COLORS.danger,
    fontSize: 13,
  },
  linkText: {
    marginTop: 16,
    fontSize: 13,
    color: COLORS.primary,
    textAlign: "center",
  },
});
