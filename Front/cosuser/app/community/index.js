// app/community/index.js
import React, { useEffect, useState } from "react";
import { View, Text, FlatList, TouchableOpacity, ActivityIndicator, Alert, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import { fetchNotices } from "../../api/community";
import COLORS from "../../constants/colors"; // 색상 상수 사용 권장

export default function CommunityMain() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [notices, setNotices] = useState([]);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);
        const data = await fetchNotices();
        if (data.result_code === 200) {
          // 서버가 보내주는 데이터 확인: id, title, regDate
          setNotices(data.notices || []);
        } else {
          Alert.alert("오류", data.result_msg);
        }
      } catch (e) {
        Alert.alert("오류", "공지사항 조회 실패");
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const renderItem = ({ item }) => (
    <TouchableOpacity
      style={styles.itemContainer}
      // ★ [수정] item.noticeId -> item.id (서버가 주는 이름)
      onPress={() => router.push(`/community/${item.id}`)}
    >
      <Text style={styles.itemTitle}>{item.title}</Text>
      <Text style={styles.itemDate}>
        {new Date(item.regDate).toLocaleString()}
      </Text>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.header}>공지사항</Text>

      {loading ? (
        <ActivityIndicator size="large" color="#4F46E5" />
      ) : (
        <FlatList
          data={notices}
          renderItem={renderItem}
          // ★ [수정] 키 추출도 item.id로 변경
          keyExtractor={(item) => String(item.id)}
          ListEmptyComponent={<Text style={styles.emptyText}>공지사항이 없습니다.</Text>}
        />
      )}

      <TouchableOpacity
        style={styles.primaryBtn}
        onPress={() => router.push("/community/feedback-write")}
      >
        <Text style={styles.btnText}>문의 보내기</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={styles.secondaryBtn}
        onPress={() => router.push("/community/my-feedback")}
      >
        <Text style={styles.btnTextBlack}>내 문의 내역</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: '#fff' },
  header: { fontSize: 20, fontWeight: "700", marginBottom: 12 },
  itemContainer: { paddingVertical: 12, borderBottomWidth: 1, borderColor: "#eee" },
  itemTitle: { fontSize: 16, fontWeight: "600", marginBottom: 4 },
  itemDate: { fontSize: 12, color: "#666" },
  emptyText: { textAlign: 'center', marginTop: 20, color: '#999' },
  primaryBtn: {
    marginTop: 20,
    padding: 14,
    backgroundColor: "#4F46E5",
    borderRadius: 8,
    alignItems: "center",
  },
  secondaryBtn: {
    marginTop: 10,
    padding: 14,
    backgroundColor: "#f3f4f6",
    borderRadius: 8,
    alignItems: "center",
  },
  btnText: { color: "#fff", fontWeight: "600" },
  btnTextBlack: { color: "#333", fontWeight: "600" },
});