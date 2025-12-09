// app/sign-up.js
import React, { useState } from "react";
import { 
  View, 
  Text, 
  TextInput, 
  StyleSheet, 
  TouchableOpacity, 
  ScrollView,
  KeyboardAvoidingView,
  Platform
} from "react-native";
import { useRouter, Link } from "expo-router";
import ScreenContainer from "../components/ScreenContainer";
import CommonButton from "../components/CommonButton";
import COLORS from "../constants/colors";
import { signUp } from "../api/auth";

export default function SignUpScreen() {
  const router = useRouter();
  
  // 1. 상태 변수 추가 (이름, 나이, 성별)
  const [email, setEmail] = useState("");
  const [name, setName] = useState(""); // nickname -> name으로 변경 (DB 컬럼명 일치)
  const [password, setPassword] = useState("");
  const [passwordCheck, setPasswordCheck] = useState("");
  const [age, setAge] = useState("");     // 나이
  const [gender, setGender] = useState(""); // 성별 (M 또는 F)

  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  const handleSignUp = async () => {
    // 유효성 검사
    if (!email || !name || !password || !passwordCheck || !age || !gender) {
      setError("모든 필드를 입력해 주세요.");
      return;
    }

    if (password !== passwordCheck) {
      setError("비밀번호가 서로 일치하지 않습니다.");
      return;
    }

    setError("");
    setSubmitting(true);

    try {
      // 2. 백엔드로 보낼 데이터 구성
      // DB 오류 로그를 보면 파라미터 순서가 [USER_ID, EMAIL, PASSWORD, NAME, GENDER, AGE] 입니다.
      // 따라서 아래 키값들을 백엔드 컨트롤러가 req.body에서 꺼내 쓸 수 있어야 합니다.
      const payload = {
        email,
        password,
        name,   // 백엔드가 expecting 'name' (not nickname)
        age: Number(age),
        gender, // 'M' or 'F'
      };

      await signUp(payload);
      
      // 회원가입 성공 시 로그인 화면으로 이동
      router.replace("/login");
    } catch (e) {
      console.warn(e);
      setError("회원가입에 실패했습니다. 입력을 확인하거나 이미 존재하는 이메일일 수 있습니다.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ScreenContainer>
      <KeyboardAvoidingView 
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={{ flex: 1 }}
      >
        <ScrollView contentContainerStyle={{ paddingBottom: 40 }} showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <Text style={styles.title}>회원가입</Text>
            <Text style={styles.subtitle}>COSMATE와 함께 여행 코스를 만들어 보세요</Text>
          </View>

          <View style={styles.form}>
            {/* 이메일 */}
            <Text style={styles.label}>이메일</Text>
            <TextInput
              style={styles.input}
              placeholder="example@email.com"
              autoCapitalize="none"
              keyboardType="email-address"
              value={email}
              onChangeText={setEmail}
            />

            {/* 이름 (기존 닉네임) */}
            <Text style={styles.label}>이름</Text>
            <TextInput
              style={styles.input}
              placeholder="사용할 이름을 입력하세요"
              value={name}
              onChangeText={setName}
            />

            {/* 비밀번호 */}
            <Text style={styles.label}>비밀번호</Text>
            <TextInput
              style={styles.input}
              placeholder="비밀번호"
              secureTextEntry
              value={password}
              onChangeText={setPassword}
            />

            {/* 비밀번호 확인 */}
            <Text style={styles.label}>비밀번호 확인</Text>
            <TextInput
              style={styles.input}
              placeholder="비밀번호 확인"
              secureTextEntry
              value={passwordCheck}
              onChangeText={setPasswordCheck}
            />

            {/* 나이 & 성별 (한 줄에 배치) */}
            <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
              
              {/* 나이 입력 */}
              <View style={{ flex: 0.4 }}>
                <Text style={styles.label}>나이</Text>
                <TextInput
                  style={styles.input}
                  placeholder="예: 24"
                  keyboardType="numeric"
                  value={age}
                  onChangeText={setAge}
                  maxLength={3}
                />
              </View>

              {/* 성별 선택 */}
              <View style={{ flex: 0.55 }}>
                <Text style={styles.label}>성별</Text>
                <View style={styles.genderContainer}>
                  <TouchableOpacity
                    style={[
                      styles.genderButton,
                      gender === "M" && styles.genderButtonSelected,
                    ]}
                    onPress={() => setGender("M")}
                  >
                    <Text
                      style={[
                        styles.genderText,
                        gender === "M" && styles.genderTextSelected,
                      ]}
                    >
                      남성
                    </Text>
                  </TouchableOpacity>

                  <TouchableOpacity
                    style={[
                      styles.genderButton,
                      gender === "F" && styles.genderButtonSelected,
                    ]}
                    onPress={() => setGender("F")}
                  >
                    <Text
                      style={[
                        styles.genderText,
                        gender === "F" && styles.genderTextSelected,
                      ]}
                    >
                      여성
                    </Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>

            {error ? <Text style={styles.error}>{error}</Text> : null}

            <CommonButton
              title="회원가입 완료"
              onPress={handleSignUp}
              disabled={
                submitting ||
                !email ||
                !name ||
                !password ||
                !passwordCheck ||
                !age ||
                !gender
              }
            />

            <Link href="/login" asChild>
              <TouchableOpacity>
                <Text style={styles.linkText}>이미 계정이 있으신가요? 로그인</Text>
              </TouchableOpacity>
            </Link>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  header: {
    marginBottom: 24,
    marginTop: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: "700",
    color: COLORS.text,
  },
  subtitle: {
    marginTop: 6,
    fontSize: 13,
    color: COLORS.muted,
  },
  form: {
    marginTop: 4,
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
    backgroundColor: '#fff',
  },
  // 성별 선택 스타일
  genderContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
  },
  genderButton: {
    flex: 0.48,
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 10,
    paddingVertical: 10,
    alignItems: "center",
    backgroundColor: '#fff',
  },
  genderButtonSelected: {
    borderColor: COLORS.primary,
    backgroundColor: "#EEF2FF", // 선택되었을 때 배경색 (연한 파랑)
  },
  genderText: {
    fontSize: 14,
    color: COLORS.text,
  },
  genderTextSelected: {
    color: COLORS.primary,
    fontWeight: "600",
  },
  error: {
    marginTop: 12,
    color: COLORS.danger,
    fontSize: 13,
    textAlign: "center",
  },
  linkText: {
    marginTop: 20,
    fontSize: 13,
    color: COLORS.primary,
    textAlign: "center",
  },
});