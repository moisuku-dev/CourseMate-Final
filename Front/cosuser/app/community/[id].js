// app/community/[id].js
import React, { useEffect, useState } from "react";
import { useLocalSearchParams } from "expo-router";
import { View, Text, ScrollView, ActivityIndicator, Alert, StyleSheet } from "react-native";
import { fetchNoticeDetail } from "../../api/community";

export default function NoticeDetail() {
  const { id } = useLocalSearchParams();
  const [notice, setNotice] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        console.log("공지사항 ID 요청:", id); // ID가 제대로 넘어오는지 확인
        const data = await fetchNoticeDetail(id);
        
        if (data.result_code === 200) {
          // 백엔드에서 소문자로 변환해줬지만, 혹시 몰라 대문자도 대비
          const raw = data.notice;
          setNotice({
              ...raw,
              title: raw.title || raw.TITLE,
              content: raw.content || raw.CONTENT,
              regDate: raw.regDate || raw.REG_DATE
          });
        } else {
          Alert.alert("오류", data.result_msg);
        }
      } catch (e) {
        Alert.alert("오류", "공지사항 조회 실패");
      } finally {
        setLoading(false);
      }
    };
    if (id) load(); // ID가 있을 때만 로드
  }, [id]);

  if (loading) return <View style={styles.center}><ActivityIndicator size="large" color="#4F46E5"/></View>;

  if (!notice)
    return <View style={styles.center}><Text>공지사항 정보를 찾을 수 없습니다.</Text></View>;

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>{notice.title}</Text>
        <Text style={styles.date}>
          {new Date(notice.regDate).toLocaleString()}
        </Text>
      </View>

      <View style={styles.contentContainer}>
        <Text style={styles.content}>{notice.content}</Text>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#fff' },
    center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
    header: {
        padding: 20,
        borderBottomWidth: 1,
        borderBottomColor: '#eee',
        backgroundColor: '#f9fafb'
    },
    title: { fontSize: 20, fontWeight: "700", color: '#111', marginBottom: 8 },
    date: { fontSize: 13, color: "#666" },
    contentContainer: { padding: 20 },
    content: { fontSize: 16, lineHeight: 24, color: '#333' }
});