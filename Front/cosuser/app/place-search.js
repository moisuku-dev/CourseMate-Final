// app/place-search.js
import React, { useEffect, useState } from "react";
import { View, Text, TextInput, FlatList, StyleSheet } from "react-native";
import { useRouter } from "expo-router";
import ScreenContainer from "../components/ScreenContainer";
import COLORS from "../constants/colors";
import PlaceCard from "../components/PlaceCard";
import { fetchPlaces } from "../api/places";

export default function PlaceSearchScreen() {
  const router = useRouter();
  const [keyword, setKeyword] = useState("");
  const [places, setPlaces] = useState([]);
  const [loading, setLoading] = useState(false);

  const loadPlaces = async () => {
    setLoading(true);
    try {
      const data = await fetchPlaces({ keyword });
      
      console.log("서버 데이터 원본:", JSON.stringify(data, null, 2));

      let rawList = [];
      // 1. 데이터 위치 찾기
      if (Array.isArray(data)) {
        rawList = data;
      } else if (data && Array.isArray(data.places)) {
        rawList = data.places;
      } else if (data && Array.isArray(data.data)) {
        rawList = data.data;
      }

      // 2. ★ [핵심] 대문자 키를 소문자로 변환 (Mapping) ★
      const formattedList = rawList.map((item) => ({
        id: item.SPOT_ID || item.spotId || item.id, // SPOT_ID를 id로 변환
        name: item.NAME || item.name,               // NAME을 name으로 변환
        address: item.ADDRESS || item.address,      // ADDRESS를 address로 변환
        thumbnail: item.THUMBNAIL || item.thumbnail,// 이미지 필드가 있다면
        ...item // 혹시 모를 다른 데이터도 유지
      }));

      console.log("변환된 리스트 개수:", formattedList.length);
      setPlaces(formattedList);

    } catch (e) {
      console.warn(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPlaces();
  }, []);

  return (
    <ScreenContainer>
      <Text style={styles.title}>관광지 검색</Text>

      <TextInput
        style={styles.input}
        placeholder="지역명, 관광지명, 태그 등으로 검색"
        value={keyword}
        onChangeText={setKeyword}
        onSubmitEditing={loadPlaces}
      />

      <FlatList
        data={places}
        // 이제 id가 확실히 존재하므로 안심하고 사용 가능
        keyExtractor={(item) => String(item.id)}
        refreshing={loading}
        onRefresh={loadPlaces}
        renderItem={({ item }) => (
          <PlaceCard
            place={item}
            // 상세 페이지로 갈 때도 변환된 id 사용
            onPress={() => router.push(`/place/${item.id}`)}
          />
        )}
        ListEmptyComponent={
          !loading && (
            <Text style={styles.empty}>검색 결과가 없습니다.</Text>
          )
        }
      />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  title: {
    fontSize: 20,
    fontWeight: "700",
    marginBottom: 12,
    color: COLORS.text,
  },
  input: {
    borderWidth: 1,
    borderColor: COLORS.border,
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    fontSize: 14,
    marginBottom: 12,
  },
  empty: {
    marginTop: 16,
    textAlign: "center",
    color: COLORS.muted,
  },
});