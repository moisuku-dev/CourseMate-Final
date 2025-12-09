// app/community/my-feedback.js
import React, { useEffect, useState } from "react";
import { 
  View, 
  Text, 
  ActivityIndicator, 
  FlatList, 
  Alert, 
  StyleSheet,
  Platform 
} from "react-native";
import { fetchMyFeedbacks } from "../../api/community";

export default function MyFeedback() {
  const [loading, setLoading] = useState(true);
  const [list, setList] = useState([]);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchMyFeedbacks();
        
        if (data.result_code === 200) {
          const rawList = data.feedbacks || [];
          
          // ★ [핵심 수정] DB 컬럼명(ANSWER_CONTENT)을 정확히 연결
          const formattedList = rawList.map(item => ({
            id: item.INQUIRY_ID || item.inquiryId || item.id,
            title: item.TITLE || item.title,
            content: item.CONTENT || item.content,
            status: item.STATUS || item.status || '대기중',
            // ▼ 수정된 부분: ANSWER_CONTENT 추가
            answer: item.ANSWER_CONTENT || item.answerContent || item.answer,
            regDate: item.REG_DATE || item.regDate || item.createdAt
          }));
          
          setList(formattedList);
        } else {
          const msg = data.result_msg || "조회 실패";
          Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
        }
      } catch (e) {
        console.error(e);
        const msg = "문의 내역을 불러오지 못했습니다.";
        Platform.OS === 'web' ? window.alert(msg) : Alert.alert("오류", msg);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" color="#4F46E5" />
      </View>
    );
  }

  const renderItem = ({ item }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.title}>{item.title}</Text>
        <Text style={[
          styles.status, 
          item.status === '완료' || item.status === '답변완료' ? styles.statusDone : styles.statusPending
        ]}>
          {item.status}
        </Text>
      </View>
      
      <Text style={styles.date}>
        {item.regDate ? new Date(item.regDate).toLocaleString() : ""}
      </Text>
      
      <Text style={styles.content} numberOfLines={3}>{item.content}</Text>

      {/* 답변이 있을 경우 표시 */}
      {item.answer ? (
        <View style={styles.answerBox}>
          <View style={{flexDirection:'row', justifyContent:'space-between', marginBottom: 4}}>
             <Text style={styles.answerLabel}>└ 관리자 답변:</Text>
             {/* 답변 날짜가 있다면 표시 (선택사항) */}
             {/* <Text style={{fontSize:11, color:'#6b7280'}}>답변일</Text> */}
          </View>
          <Text style={styles.answerText}>{item.answer}</Text>
        </View>
      ) : null}
    </View>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.headerTitle}>내 문의 내역</Text>

      <FlatList
        data={list}
        keyExtractor={(item) => String(item.id)}
        renderItem={renderItem}
        contentContainerStyle={{ paddingBottom: 20 }}
        ListEmptyComponent={
          <View style={styles.center}>
            <Text style={styles.emptyText}>등록된 문의가 없습니다.</Text>
          </View>
        }
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: '#fff' },
  center: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 },
  headerTitle: { fontSize: 20, fontWeight: "700", marginBottom: 16, color: "#111" },
  card: {
    padding: 16,
    borderRadius: 12,
    backgroundColor: "#fff",
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    // 그림자
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 2,
  },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  title: { fontSize: 16, fontWeight: "600", color: "#111", flex: 1 },
  status: { fontSize: 12, fontWeight: "600", paddingHorizontal: 8, paddingVertical: 2, borderRadius: 4, overflow: 'hidden' },
  statusPending: { color: "#D97706", backgroundColor: "#FEF3C7" }, // 대기중 (노랑)
  statusDone: { color: "#059669", backgroundColor: "#D1FAE5" },    // 완료 (초록)
  date: { fontSize: 12, color: "#9ca3af", marginBottom: 8 },
  content: { fontSize: 14, color: "#4b5563", lineHeight: 20 },
  emptyText: { color: "#9ca3af", fontSize: 16 },
  
  answerBox: {
    marginTop: 12,
    padding: 12,
    backgroundColor: "#F3F4F6",
    borderRadius: 8,
    borderLeftWidth: 3,     // 왼쪽 강조선 추가
    borderLeftColor: "#4F46E5"
  },
  answerLabel: { fontSize: 13, fontWeight: "700", color: "#4F46E5" },
  answerText: { fontSize: 13, color: "#374151", marginTop: 2, lineHeight: 18 }
});