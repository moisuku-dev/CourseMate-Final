// components/PlaceCard.js
import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, Image } from "react-native";
import COLORS from "../constants/colors";

// ★ [핵심] 백엔드 서버 주소 설정 (api/client.js와 동일하게)
// 로컬 테스트(iOS/Web): "http://localhost:3000"
// 안드로이드 에뮬레이터: "http://10.0.2.2:3000"
const SERVER_URL = "http://localhost:3000";

export default function PlaceCard({ place, onPress }) {
  
  // 이미지 경로 생성 함수
  const getImageUrl = () => {
    // 1. 데이터에 썸네일 경로가 있는 경우
    if (place.thumbnail) {
      // http로 시작하면 외부 링크이므로 그대로 사용
      if (place.thumbnail.startsWith('http')) return place.thumbnail;
      // 아니면 서버 주소를 앞에 붙임
      return `${SERVER_URL}${place.thumbnail.startsWith('/') ? '' : '/'}${place.thumbnail}`;
    }

    // 2. 데이터에 썸네일이 없는 경우 -> 이름으로 이미지 추측 (images 폴더 규칙)
    // 예: "강릉선교장" -> "http://localhost:3000/images/강릉선교장.jpg"
    if (place.name) {
      return `${SERVER_URL}/images/${encodeURIComponent(place.name)}.jpg`;
    }

    return null;
  };

  const imageUrl = getImageUrl();

  return (
    <TouchableOpacity style={styles.card} onPress={onPress}>
      {/* 이미지가 있을 때만 렌더링 */}
      {imageUrl && (
        <Image 
          source={{ uri: imageUrl }} 
          style={styles.thumbnail}
          resizeMode="cover"
        />
      )}
      
      <View style={styles.info}>
        <Text style={styles.name}>{place.name}</Text>
        
        {place.tags && (
          <Text style={styles.tags}>
            {Array.isArray(place.tags) ? place.tags.join(" · ") : place.tags}
          </Text>
        )}
        
        {place.shortDescription && (
          <Text style={styles.desc} numberOfLines={2}>
            {place.shortDescription}
          </Text>
        )}
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: "row",
    borderWidth: 1,
    borderColor: COLORS.border,
    backgroundColor: COLORS.card,
    borderRadius: 16,
    padding: 12,
    marginVertical: 6,
  },
  thumbnail: {
    width: 72,
    height: 72,
    borderRadius: 12,
    marginRight: 12,
    backgroundColor: '#eee', // 이미지 로딩 전 회색 배경
  },
  info: {
    flex: 1,
    justifyContent: 'center',
  },
  name: {
    fontSize: 16,
    fontWeight: "600",
    color: COLORS.text,
    marginBottom: 4,
  },
  tags: {
    fontSize: 12,
    color: COLORS.muted,
    marginTop: 2,
  },
  desc: {
    fontSize: 12,
    color: COLORS.muted,
    marginTop: 4,
  },
});