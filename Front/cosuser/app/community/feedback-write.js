// app/community/feedback-write.js
import React, { useState } from "react";
import { 
  View, 
  Text, 
  TextInput, 
  TouchableOpacity, 
  Alert, 
  StyleSheet,
  Platform, // Platform 추가
  KeyboardAvoidingView,
  ScrollView
} from "react-native";
import { createFeedback } from "../../api/community";
import { useRouter } from "expo-router";

export default function FeedbackWrite() {
  const router = useRouter();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [sending, setSending] = useState(false);

  const submit = async () => {
    // 1. 입력값 검증
    if (!title.trim() || !content.trim()) {
      const msg = "제목과 내용을 모두 입력하세요.";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("알림", msg);
      return;
    }

    try {
      setSending(true);
      const data = await createFeedback({ title, content });

      if (data.result_code === 200) {
        // ★ 2. 웹/앱 분기 처리 (성공 시 이동)
        if (Platform.OS === 'web') {
            window.alert("문의가 등록되었습니다.");
            router.back(); // 뒤로 가기
        } else {
            Alert.alert("등록 완료", "문의가 성공적으로 등록되었습니다.", [
              { text: "확인", onPress: () => router.back() },
            ]);
        }
      } else {
        const msg = data.result_msg || "등록 실패";
        Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
      }
    } catch (e) {
      console.error(e);
      const msg = "문의 등록 중 오류가 발생했습니다.";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
    } finally {
      setSending(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.headerTitle}>문의 보내기</Text>

      <KeyboardAvoidingView behavior={Platform.OS === "ios" ? "padding" : "height"} style={{flex: 1}}>
        <ScrollView contentContainerStyle={{paddingBottom: 20}}>
          <Text style={styles.label}>제목</Text>
          <TextInput
            placeholder="문의 제목을 입력하세요"
            style={styles.input}
            value={title}
            onChangeText={setTitle}
          />

          <Text style={styles.label}>내용</Text>
          <TextInput
            placeholder="문의 내용을 자세히 적어주세요"
            style={[styles.input, styles.textArea]}
            value={content}
            onChangeText={setContent}
            multiline
            textAlignVertical="top"
          />

          <TouchableOpacity
            style={[styles.btn, sending && styles.btnDisabled]}
            onPress={submit}
            disabled={sending}
          >
            <Text style={styles.btnText}>
              {sending ? "전송 중..." : "등록하기"}
            </Text>
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 20, backgroundColor: '#fff' },
  headerTitle: { fontSize: 22, fontWeight: "700", marginBottom: 20, color: "#111" },
  label: { fontSize: 14, fontWeight: "600", marginBottom: 8, color: "#374151" },
  input: {
    borderWidth: 1,
    borderColor: "#E5E7EB",
    borderRadius: 8,
    padding: 12,
    fontSize: 15,
    marginBottom: 16,
    backgroundColor: "#F9FAFB",
    color: "#111"
  },
  textArea: {
    height: 150,
  },
  btn: {
    marginTop: 10,
    backgroundColor: "#4F46E5",
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: "center",
  },
  btnDisabled: {
    backgroundColor: "#A5B4FC",
  },
  btnText: { color: "#fff", fontWeight: "600", fontSize: 16 },
});