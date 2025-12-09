// app/review-write.js
import React, { useState } from "react";
import {
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  Platform, // ★ 1. Platform 추가
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import ScreenContainer from "../components/ScreenContainer";
import COLORS from "../constants/colors";
import { createReview } from "../api/reviews";

export default function ReviewWriteScreen() {
  const { placeId, placeName } = useLocalSearchParams();
  const router = useRouter();

  const [rating, setRating] = useState("5");
  const [content, setContent] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const onSubmit = async () => {
    const numericRating = Number(rating);

    if (!placeId) {
      const msg = "관광지 정보가 없습니다. 다시 시도해 주세요.";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
      return;
    }
    if (!numericRating || numericRating < 1 || numericRating > 5) {
      const msg = "별점은 1~5 사이 숫자로 입력해 주세요.";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("알림", msg);
      return;
    }
    if (!content.trim()) {
      const msg = "리뷰 내용을 입력해 주세요.";
      Platform.OS === 'web' ? window.alert(msg) : Alert.alert("알림", msg);
      return;
    }

    try {
      setSubmitting(true);
      
      // 리뷰 작성 API 호출
      await createReview(placeId, {
        rating: numericRating,
        content: content.trim(),
      });

      // ★ 2. 웹/앱 분기 처리
      if (Platform.OS === 'web') {
        window.alert("완료: 리뷰가 등록되었습니다.");
        router.replace(`/place/${placeId}`); // 바로 이동
      } else {
        Alert.alert("완료", "리뷰가 등록되었습니다.", [
          {
            text: "확인",
            onPress: () => router.replace(`/place/${placeId}`),
          },
        ]);
      }
      
    } catch (e) {
      console.error(e);
      const msg = "리뷰 등록 중 문제가 발생했습니다.";
      Platform.OS === 'web' ? window.alert("오류: " + msg) : Alert.alert("오류", msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ScreenContainer>
      <Text style={styles.title}>
        {placeName ? `${placeName} 리뷰 작성` : "리뷰 작성"}
      </Text>

      <Text style={styles.label}>별점 (1 ~ 5)</Text>
      <TextInput
        style={styles.input}
        keyboardType="numeric"
        value={rating}
        onChangeText={setRating}
        placeholder="예: 5"
        maxLength={1} // 한 자리만 입력받게 제한
      />

      <Text style={styles.label}>리뷰 내용</Text>
      <TextInput
        style={[styles.input, styles.textArea]}
        value={content}
        onChangeText={setContent}
        placeholder="여행 경험을 자세히 적어 주세요."
        multiline
        textAlignVertical="top"
      />

      <TouchableOpacity
        style={[styles.button, submitting && styles.buttonDisabled]}
        onPress={onSubmit}
        disabled={submitting}
      >
        <Text style={styles.buttonText}>
          {submitting ? "등록 중..." : "리뷰 등록"}
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
  textArea: {
    height: 150,
    marginTop: 4,
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