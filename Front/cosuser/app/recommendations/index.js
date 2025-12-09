// app/recommendations/index.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  FlatList,
  TouchableOpacity,
  Alert,
  Image,
  Platform,
  StyleSheet,
  Modal,
  ScrollView
} from "react-native";
import { useRouter } from "expo-router";
import * as Location from 'expo-location';

import { useAuth } from "../../hooks/useAuth";
import { fetchRecommendations, retryRecommendations } from "../../api/recommendations";
import { addToWishlist } from "../../api/user";

const SERVER_URL = Platform.select({
  ios: 'http://localhost:3000',
  android: 'http://10.0.2.2:3000',
  web: 'http://localhost:3000',
  default: 'http://localhost:3000',
});

const REGIONS = [
  "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
  "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"
];

// ★ 비상용 매핑 추가 (홈 화면과 동일)
const CITY_TO_REGION_MAP = {
  "김포": "경기", "수원": "경기", "성남": "경기", "고양": "경기", "용인": "경기", "부천": "경기",
  "안산": "경기", "안양": "경기", "남양주": "경기", "화성": "경기", "평택": "경기", "의정부": "경기",
  "시흥": "경기", "파주": "경기", "광명": "경기", "광주": "경기", "군포": "경기", "오산": "경기",
  "이천": "경기", "양주": "경기", "안성": "경기", "구리": "경기", "포천": "경기", "의왕": "경기",
  "하남": "경기", "여주": "경기", "양평": "경기", "동두천": "경기", "과천": "경기", "가평": "경기", "연천": "경기",
  "창원": "경남", "진주": "경남", "통영": "경남", "사천": "경남", "김해": "경남", "밀양": "경남", "거제": "경남", "양산": "경남",
  "포항": "경북", "경주": "경북", "김천": "경북", "안동": "경북", "구미": "경북", "영주": "경북", "영천": "경북", "상주": "경북", "문경": "경북", "경산": "경북",
  "춘천": "강원", "원주": "강원", "강릉": "강원", "동해": "강원", "태백": "강원", "속초": "강원", "삼척": "강원",
  "전주": "전북", "군산": "전북", "익산": "전북", "정읍": "전북", "남원": "전북", "김제": "전북",
  "목포": "전남", "여수": "전남", "순천": "전남", "나주": "전남", "광양": "전남",
  "청주": "충북", "충주": "충북", "제천": "충북",
  "천안": "충남", "공주": "충남", "보령": "충남", "아산": "충남", "서산": "충남", "논산": "충남", "계룡": "충남", "당진": "충남",
  "제주": "제주", "서귀포": "제주"
};

export default function RecommendationScreen() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [recs, setRecs] = useState([]);
  const [currentRegion, setCurrentRegion] = useState("서울");
  const [modalVisible, setModalVisible] = useState(false);

  const formatData = (data) => {
    let list = [];
    if (data && Array.isArray(data.course)) {
      list = data.course;
    } else if (Array.isArray(data)) {
      list = data;
    }

    return list.map((item) => ({
      id: item.spotId || item.placeId || item.id,
      name: item.spotName || item.placeName || item.name,
      address: item.address,
      tags: item.features || item.tags,
      matchScore: item.matchScore,
      thumbnail: `${SERVER_URL}/images/${encodeURIComponent(item.spotName || item.name)}.jpg`
    }));
  };

  const getLocation = async () => {
    try {
      let { status } = await Location.requestForegroundPermissionsAsync();
      
      if (status !== 'granted') {
        Alert.alert('권한 거부', '위치 권한이 거부되어 기본 지역(서울)으로 설정됩니다.');
        return '서울';
      }

      let location = await Location.getCurrentPositionAsync({});
      let fullAddress = "서울";

      if (Platform.OS === 'web') {
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${location.coords.latitude}&lon=${location.coords.longitude}&accept-language=ko`,
            { headers: { 'User-Agent': 'CourseMate/1.0' } }
          );
          if (!response.ok) throw new Error("Network response was not ok");

          const data = await response.json();
          if (data && data.display_name) {
             fullAddress = data.display_name; // 전체 주소 사용
             console.log("웹 전체 주소:", fullAddress);
          }
        } catch (err) {
          console.log("Web reverse geocoding failed:", err);
        }
      } 
      else {
        let address = await Location.reverseGeocodeAsync({
          latitude: location.coords.latitude,
          longitude: location.coords.longitude
        });
        if (address && address.length > 0) {
          const a = address[0];
          fullAddress = `${a.region} ${a.city} ${a.district}`;
        }
      }

      // 1. 1차 매칭 (전체 주소에서 경기, 서울 등 찾기)
      let foundRegion = REGIONS.find(r => fullAddress.includes(r));

      // 2. 2차 매칭 (매핑 테이블 사용)
      if (!foundRegion) {
        for (const [city, region] of Object.entries(CITY_TO_REGION_MAP)) {
          if (fullAddress.includes(city)) {
            foundRegion = region;
            break;
          }
        }
      }

      const simpleRegion = foundRegion || "서울";
      setCurrentRegion(simpleRegion);
      return simpleRegion;
    } catch (error) {
      console.log("위치 가져오기 실패:", error);
    }
    return '서울';
  };

  const load = async (regionToUse) => {
    try {
      setLoading(true);
      const data = await fetchRecommendations(regionToUse); 
      setRecs(formatData(data));
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "추천 코스를 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const init = async () => {
      const detected = await getLocation();
      load(detected);
    };
    init();
  }, []);

  const handleRegionChange = (newRegion) => {
    setCurrentRegion(newRegion);
    setModalVisible(false);
    load(newRegion);
  };

  const onRetry = async () => {
    try {
      setLoading(true);
      const data = await fetchRecommendations(currentRegion); 
      setRecs(formatData(data));
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "다른 코스 추천에 실패했습니다.");
    } finally {
      setLoading(false);
    }
  };

  const onAddWishlist = async (placeId) => {
    if (!placeId) return;
    try {
      await addToWishlist(placeId);
      if(Platform.OS === 'web') window.alert("위시리스트에 추가되었습니다.");
      else Alert.alert("알림", "위시리스트에 추가되었습니다.");
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "위시리스트 추가 실패");
    }
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#4F46E5" />
        <Text style={{ marginTop: 10, color: "#666" }}>
          AI가 {currentRegion} 지역 코스를 분석 중입니다...
        </Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.headerTitle}>AI 추천 코스 결과</Text>

      <TouchableOpacity 
        style={styles.locationButton} 
        onPress={() => setModalVisible(true)}
      >
        <Text style={styles.locationText}>📍 지역: {currentRegion} (변경)</Text>
      </TouchableOpacity>

      <FlatList
        data={recs}
        keyExtractor={(item) => String(item.id)}
        ListEmptyComponent={
          <View style={styles.centerContainer}>
            <Text style={styles.emptyText}>추천 결과가 없습니다.</Text>
          </View>
        }
        renderItem={({ item }) => (
          <View style={styles.card}>
            <TouchableOpacity
              onPress={() => router.push(`/place/${item.id}`)}
              style={styles.cardTouchArea} // ★ [디자인] 가로 배치 적용
            >
              {/* 왼쪽: 사진 */}
              <Image
                source={{ uri: item.thumbnail }}
                style={styles.cardImage}
                resizeMode="cover"
              />
              
              {/* 오른쪽: 정보 */}
              <View style={styles.cardContent}>
                <View style={styles.cardHeader}>
                  <Text style={styles.placeName} numberOfLines={1}>{item.name}</Text>
                </View>
                
                <Text style={styles.address} numberOfLines={1}>{item.address}</Text>

                {item.matchScore && (
                    <Text style={styles.matchScore}>
                      {Math.round(Number(item.matchScore) * 100)}% 일치
                    </Text>
                )}

                {item.tags && (
                  <View style={styles.tagContainer}>
                    {(Array.isArray(item.tags) ? item.tags : [item.tags]).slice(0, 3).map(
                      (tag, idx) => (
                        <Text key={idx} style={styles.tag}>#{tag}</Text>
                      )
                    )}
                  </View>
                )}
              </View>
            </TouchableOpacity>

            <TouchableOpacity
              onPress={() => onAddWishlist(item.id)}
              style={styles.wishButton}
            >
              <Text style={styles.wishButtonText}>♥ 위시리스트 담기</Text>
            </TouchableOpacity>
          </View>
        )}
        contentContainerStyle={{ paddingBottom: 20 }}
      />

      <TouchableOpacity onPress={onRetry} style={styles.retryButton}>
        <Text style={styles.retryButtonText}>다른 코스 추천 받기</Text>
      </TouchableOpacity>

      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>지역 선택</Text>
            <ScrollView style={{ maxHeight: 300 }}>
              {REGIONS.map((region) => (
                <TouchableOpacity
                  key={region}
                  style={[
                    styles.modalItem,
                    currentRegion === region && styles.modalItemSelected
                  ]}
                  onPress={() => handleRegionChange(region)}
                >
                  <Text style={[
                    styles.modalItemText,
                    currentRegion === region && { color: '#4F46E5', fontWeight: 'bold' }
                  ]}>
                    {region}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity
              style={styles.modalCloseBtn}
              onPress={() => setModalVisible(false)}
            >
              <Text style={{ color: '#fff', fontWeight: '600' }}>닫기</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: "#fff" },
  centerContainer: { flex: 1, justifyContent: "center", alignItems: "center", padding: 20 },
  headerTitle: { fontSize: 20, fontWeight: "bold", marginBottom: 10, color: "#111827" },
  emptyText: { fontSize: 16, color: "#6b7280", textAlign: "center" },
  
  // ★ [디자인] 가로 배치 카드 스타일
  card: {
    backgroundColor: "#fff",
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: "#e5e7eb",
    overflow: 'hidden',
    elevation: 2,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
  },
  cardTouchArea: {
    flexDirection: 'row', // 가로 배치 핵심
    alignItems: 'center',
    padding: 12,
  },
  cardImage: {
    width: 100, // 왼쪽 고정 너비
    height: 100,
    borderRadius: 8,
    backgroundColor: '#eee',
  },
  cardContent: {
    flex: 1,
    marginLeft: 12, // 이미지와의 간격
    justifyContent: 'center',
  },
  cardHeader: {
    marginBottom: 4,
  },
  placeName: {
    fontSize: 16,
    fontWeight: "600",
    color: "#111827",
  },
  address: {
    fontSize: 13,
    color: "#6b7280",
    marginBottom: 6,
  },
  matchScore: {
    fontSize: 12,
    color: "#4F46E5",
    fontWeight: "700",
    marginBottom: 6,
  },
  tagContainer: {
    flexDirection: "row",
    flexWrap: "wrap",
  },
  tag: {
    fontSize: 11,
    color: "#6F42C1",
    backgroundColor: "#F3E8FF",
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    marginRight: 4,
    marginBottom: 4,
    overflow: "hidden",
  },
  wishButton: {
    borderTopWidth: 1,
    borderTopColor: '#f3f4f6',
    paddingVertical: 10,
    alignItems: 'center',
    backgroundColor: '#fafafa',
  },
  wishButtonText: {
    color: "#4F46E5",
    fontWeight: "600",
    fontSize: 13,
  },
  retryButton: {
    marginTop: 8,
    paddingVertical: 14,
    borderRadius: 12,
    backgroundColor: "#6366F1",
    alignItems: "center",
  },
  retryButtonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 16,
  },
  locationButton: {
    backgroundColor: '#F3F4F6', padding: 12, borderRadius: 8, marginBottom: 16,
    alignItems: 'center', borderWidth: 1, borderColor: '#E5E7EB'
  },
  locationText: { color: '#4F46E5', fontWeight: '600' },
  modalOverlay: {
    flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: 'rgba(0,0,0,0.5)'
  },
  modalContent: {
    width: '80%', backgroundColor: 'white', borderRadius: 16, padding: 20, elevation: 5
  },
  modalTitle: { fontSize: 18, fontWeight: 'bold', marginBottom: 12, textAlign: 'center' },
  modalItem: { paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#eee' },
  modalItemSelected: { backgroundColor: '#EEF2FF' },
  modalItemText: { fontSize: 16, textAlign: 'center', color: '#333' },
  modalCloseBtn: {
    marginTop: 16, backgroundColor: '#4F46E5', padding: 12, borderRadius: 8, alignItems: 'center'
  }
});