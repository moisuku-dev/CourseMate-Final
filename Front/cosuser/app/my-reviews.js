// app/my-reviews.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  FlatList,
  TouchableOpacity,
  Alert,
  Platform,
} from "react-native";
import { fetchMyReviews } from "../api/user";
import { deleteReview } from "../api/reviews";

export default function MyReviewsScreen() {
  const [loading, setLoading] = useState(true);
  const [reviews, setReviews] = useState([]);

  const load = async () => {
    try {
      setLoading(true);
      const data = await fetchMyReviews();
      // data.reviews 배열을 가져옵니다.
      setReviews(data?.reviews || data || []);
    } catch (e) {
      console.error(e);
      // 웹에서는 Alert가 다르게 동작할 수 있으므로 콘솔에도 출력
      if (Platform.OS === 'web') {
        alert("내 리뷰를 불러오는 중 문제가 발생했습니다.");
      } else {
        Alert.alert("오류", "내 리뷰를 불러오는 중 문제가 발생했습니다.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // 실제 삭제 요청을 수행하는 함수 분리
  const executeDelete = async (reviewId) => {
    try {
      console.log("Deleting review:", reviewId); // 디버깅용 로그
      await deleteReview(reviewId);
      await load(); // 목록 새로고침
      
      if (Platform.OS === 'web') {
        alert("삭제되었습니다.");
      } else {
        Alert.alert("성공", "리뷰가 삭제되었습니다.");
      }
    } catch (e) {
      console.error(e);
      if (Platform.OS === 'web') {
        alert("리뷰 삭제에 실패했습니다.");
      } else {
        Alert.alert("오류", "리뷰 삭제에 실패했습니다.");
      }
    }
  };

  const onDelete = (reviewId) => {
    // ID가 제대로 넘어왔는지 확인
    if (!reviewId) {
      console.error("삭제할 리뷰 ID가 없습니다.");
      return;
    }

    // ★ [수정] 웹(Web) 환경 대응
    if (Platform.OS === "web") {
      // 웹 브라우저용 확인창
      const ok = window.confirm("해당 리뷰를 삭제하시겠습니까?");
      if (ok) {
        executeDelete(reviewId);
      }
    } else {
      // 앱(Android/iOS)용 확인창
      Alert.alert("삭제 확인", "해당 리뷰를 삭제하시겠습니까?", [
        { text: "취소", style: "cancel" },
        {
          text: "삭제",
          style: "destructive",
          onPress: () => executeDelete(reviewId),
        },
      ]);
    }
  };

  if (loading) {
    return (
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: "700", marginBottom: 12 }}>
        내가 작성한 리뷰
      </Text>

      <FlatList
        data={reviews}
        //  DB에서 REVIEW_ID(대문자)로 올 수 있으므로 둘 다 체크
        keyExtractor={(item) => String(item.reviewId || item.REVIEW_ID || item.id)}
        ListEmptyComponent={<Text>작성한 리뷰가 없습니다.</Text>}
        renderItem={({ item }) => {
          // ID 추출 로직 강화
          // DB가 대문자(REVIEW_ID)로 주는지 소문자로 주는지 모르므로 둘 다 체크
          const currentId = item.reviewId || item.REVIEW_ID || item.id;

          return (
            <View
              style={{
                paddingVertical: 10,
                borderBottomWidth: 1,
                borderBottomColor: "#eee",
              }}
            >
              <Text style={{ fontWeight: "600" }}>
                {item.placeName || item.PLACE_NAME || "관광지"} · ★ {item.rating || item.RATING}
              </Text>
              <Text style={{ marginTop: 4 }}>{item.content || item.CONTENT}</Text>
              <View
                style={{
                  marginTop: 6,
                  flexDirection: "row",
                  justifyContent: "flex-end",
                }}
              >
                <TouchableOpacity
                  // 수정된 ID 사용
                  onPress={() => onDelete(currentId)}
                  style={{
                    paddingHorizontal: 10,
                    paddingVertical: 6,
                    borderRadius: 6,
                    borderWidth: 1,
                    borderColor: "#EF4444",
                  }}
                >
                  <Text style={{ color: "#EF4444" }}>삭제</Text>
                </TouchableOpacity>
              </View>
            </View>
          );
        }}
      />
    </View>
  );
}