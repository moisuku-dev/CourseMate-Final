// app/preferences.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  TouchableOpacity,
  FlatList,
  Alert,
  Platform, // ★ 1. Platform 추가
} from "react-native";
import { useRouter } from "expo-router";
import { fetchPreferences, updatePreferences } from "../api/user";

const ALL_TAGS = [
  "#맛집", "#가성비", "#양푸짐", "#디저트맛집", "#분위기깡패",
  "#뷰맛집", "#야경명소", "#사진명소", "#힙한", "#조용한",
  "#레트로", "#주차편함", "#대중교통편리", "#넓은공간", "#청결한",
  "#친절해요", "#웨이팅필수", "#혼자가기좋은", "#직원친절", "#무료입장",
  "#데이트코스", "#가족과함께", "#아이와함께", "#반려동물동반", "#친구랑",
  "#힐링산책", "#빵지순례", "#실내데이트", "#이색체험", "#카공하기좋은",  
];

export default function PreferencesScreen() {
  const router = useRouter();
  const [selected, setSelected] = useState([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchPreferences();
        const tags = data?.tags || data || [];
        setSelected(tags);
      } catch (e) {
        console.error(e);
      }
    };
    load();
  }, []);

  const toggleTag = (tag) => {
    if (selected.includes(tag)) {
      setSelected(selected.filter((t) => t !== tag));
    } else {
      setSelected([...selected, tag]);
    }
  };

  // ★ 2. onSave 함수 수정 (웹/앱 분기 처리)
  const onSave = async () => {
    try {
      setSaving(true);
      await updatePreferences(selected);
      
      if (Platform.OS === 'web') {
        // 🌐 웹 환경: 브라우저 기본 알림창 사용
        // window.alert는 확인 버튼만 있고 누르면 바로 다음 코드가 실행됨
        window.alert("완료: 취향 태그가 저장되었습니다.");
        router.replace("/");
      } else {
        // 📱 앱 환경: React Native Alert 사용
        Alert.alert("완료", "취향 태그가 저장되었습니다.", [
          {
            text: "확인",
            onPress: () => router.replace("/"),
          },
        ]);
      }
      
    } catch (e) {
      console.error(e);
      // 에러 메시지도 웹/앱 구분
      if (Platform.OS === 'web') {
        window.alert("오류: 취향 태그 저장 실패");
      } else {
        Alert.alert("오류", "취향 태그 저장 실패");
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 20, fontWeight: "700", marginBottom: 8 }}>
        내 취향 태그
      </Text>
      <Text style={{ color: "#666", marginBottom: 12 }}>
        본인의 여행 스타일에 맞는 태그를 선택해 주세요.
      </Text>

      <FlatList
        data={ALL_TAGS}
        keyExtractor={(item) => item}
        numColumns={2}
        columnWrapperStyle={{ justifyContent: "space-between" }}
        renderItem={({ item }) => {
          const active = selected.includes(item);
          return (
            <TouchableOpacity
              onPress={() => toggleTag(item)}
              style={{
                flex: 1,
                margin: 4,
                paddingVertical: 10,
                borderRadius: 999,
                borderWidth: 1,
                borderColor: active ? "#4F46E5" : "#ccc",
                backgroundColor: active ? "#EEF2FF" : "#fff",
                alignItems: "center",
              }}
            >
              <Text
                style={{
                  color: active ? "#4F46E5" : "#333",
                  fontWeight: active ? "600" : "400",
                }}
              >
                {item}
              </Text>
            </TouchableOpacity>
          );
        }}
      />

      <TouchableOpacity
        onPress={onSave}
        disabled={saving}
        style={{
          marginTop: 20,
          backgroundColor: "#4F46E5",
          paddingVertical: 12,
          borderRadius: 8,
          alignItems: "center",
        }}
      >
        <Text style={{ color: "#fff", fontWeight: "600" }}>
          {saving ? "저장 중..." : "저장하기"}
        </Text>
      </TouchableOpacity>
    </View>
  );
}