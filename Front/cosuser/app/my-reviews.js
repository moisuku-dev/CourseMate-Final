// app/my-reviews.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  FlatList,
  TouchableOpacity,
  Alert,
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
      setReviews(data?.reviews || data || []);
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "내 리뷰를 불러오는 중 문제가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const onDelete = (reviewId) => {
    Alert.alert("삭제 확인", "해당 리뷰를 삭제하시겠습니까?", [
      { text: "취소", style: "cancel" },
      {
        text: "삭제",
        style: "destructive",
        onPress: async () => {
          try {
            await deleteReview(reviewId);
            await load();
          } catch (e) {
            console.error(e);
            Alert.alert("오류", "리뷰 삭제에 실패했습니다.");
          }
        },
      },
    ]);
  };

  if (loading) {
    return (
      <View
        style={{ flex: 1, justifyContent: "center", alignItems: "center" }}
      >
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
        keyExtractor={(item) => String(item.reviewId || item.id)}
        ListEmptyComponent={<Text>작성한 리뷰가 없습니다.</Text>}
        renderItem={({ item }) => (
          <View
            style={{
              paddingVertical: 10,
              borderBottomWidth: 1,
              borderBottomColor: "#eee",
            }}
          >
            <Text style={{ fontWeight: "600" }}>
              {item.placeName || "관광지"} · ★ {item.rating}
            </Text>
            <Text style={{ marginTop: 4 }}>{item.content}</Text>
            <View
              style={{
                marginTop: 6,
                flexDirection: "row",
                justifyContent: "flex-end",
              }}
            >
              <TouchableOpacity
                onPress={() => onDelete(item.reviewId || item.id)}
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
        )}
      />
    </View>
  );
}
