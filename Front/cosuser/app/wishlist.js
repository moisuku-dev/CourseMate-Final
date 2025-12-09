// app/wishlist.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  FlatList,
  TouchableOpacity,
  Alert,
  Platform, // ★ 1. Platform 추가
} from "react-native";
import { useRouter } from "expo-router";
import { fetchWishlist, removeFromWishlist } from "../api/user";

export default function WishlistScreen() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [wishlist, setWishlist] = useState([]);

  const load = async () => {
    try {
      setLoading(true);
      const data = await fetchWishlist();
      setWishlist(data?.wishlist || data || []);
    } catch (e) {
      console.error(e);
      // 에러 메시지도 플랫폼에 따라 다르게 표시
      if (Platform.OS === 'web') {
        window.alert("오류: 위시리스트를 불러오는 중 문제가 발생했습니다.");
      } else {
        Alert.alert("오류", "위시리스트를 불러오는 중 문제가 발생했습니다.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  // ★ 2. 삭제 함수 수정 (웹/앱 분기)
  const onRemove = async (placeId) => {
    // 🌐 웹 환경
    if (Platform.OS === 'web') {
      const ok = window.confirm("위시리스트에서 제거하시겠습니까?");
      if (ok) {
        try {
          await removeFromWishlist(placeId);
          await load(); // 목록 새로고침
        } catch (e) {
          console.error(e);
          window.alert("오류: 제거에 실패했습니다.");
        }
      }
    } 
    // 📱 앱 환경 (Android/iOS)
    else {
      Alert.alert("삭제", "위시리스트에서 제거하시겠습니까?", [
        { text: "취소", style: "cancel" },
        {
          text: "제거",
          style: "destructive",
          onPress: async () => {
            try {
              await removeFromWishlist(placeId);
              await load();
            } catch (e) {
              console.error(e);
              Alert.alert("오류", "제거에 실패했습니다.");
            }
          },
        },
      ]);
    }
  };

  const renderItem = ({ item }) => (
    <TouchableOpacity
      onPress={() => router.push(`/place/${item.placeId}`)}
      style={{
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: "#eee",
      }}
    >
      <Text style={{ fontSize: 16, fontWeight: "600" }}>
        {item.placeName || item.name}
      </Text>
      <Text style={{ color: "#666", marginTop: 2 }}>{item.address}</Text>
      <View
        style={{
          marginTop: 6,
          flexDirection: "row",
          justifyContent: "flex-end",
        }}
      >
        <TouchableOpacity
          onPress={() => onRemove(item.placeId)}
          style={{
            paddingHorizontal: 10,
            paddingVertical: 6,
            borderRadius: 6,
            borderWidth: 1,
            borderColor: "#EF4444",
          }}
        >
          <Text style={{ color: "#EF4444" }}>제거</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );

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
        위시리스트
      </Text>

      <FlatList
        data={wishlist}
        keyExtractor={(item) => String(item.placeId)}
        renderItem={renderItem}
        ListEmptyComponent={<Text>위시리스트가 비어 있습니다.</Text>}
      />
    </View>
  );
}