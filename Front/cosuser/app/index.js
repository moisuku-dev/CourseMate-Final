// app/index.js
import React, { useEffect, useState } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  FlatList,
  TouchableOpacity,
  Image,
  Platform,
  Alert,
  Modal,
  ScrollView,
  StyleSheet
} from "react-native";
import { useRouter, Redirect } from "expo-router";
import * as Location from 'expo-location'; 
import { useAuth } from "../hooks/useAuth";
import { fetchRecommendations } from "../api/recommendations";

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

// ★ [추가] 도시 이름으로 지역을 못 찾을 경우를 대비한 비상용 매핑
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

export default function HomeScreen() {
  const router = useRouter();
  const { user, loading } = useAuth();
  
  const [recsLoading, setRecsLoading] = useState(false);
  const [recs, setRecs] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [currentRegion, setCurrentRegion] = useState("서울"); 
  const [modalVisible, setModalVisible] = useState(false); 

  const getLocation = async () => {
    try {
      setRecsLoading(true);
      let { status } = await Location.requestForegroundPermissionsAsync();
      
      if (status !== 'granted') {
        if (Platform.OS === 'web') window.alert("위치 권한이 거부되어 서울로 설정합니다.");
        return '서울';
      }

      let location = await Location.getCurrentPositionAsync({});
      let fullAddress = "";

      if (Platform.OS === 'web') {
        try {
          const response = await fetch(
            `https://nominatim.openstreetmap.org/reverse?format=json&lat=${location.coords.latitude}&lon=${location.coords.longitude}&accept-language=ko`,
            { headers: { 'User-Agent': 'CourseMate/1.0' } }
          );
          
          if (!response.ok) throw new Error("Network response was not ok");
          
          const data = await response.json();
          // ★ [핵심] display_name(전체 주소)을 사용해 검색 범위를 넓힘
          if (data && data.display_name) {
             fullAddress = data.display_name;
             // window.alert(`[웹] 감지된 주소: ${fullAddress}`); // 디버깅용 (확인 후 주석처리)
          }
        } catch (err) {
          console.warn("Web geocoding error:", err);
        }
      } 
      else {
        let address = await Location.reverseGeocodeAsync({
          latitude: location.coords.latitude,
          longitude: location.coords.longitude
        });
        if (address && address.length > 0) {
          // 앱에서는 region, city, district 등을 다 합침
          const a = address[0];
          fullAddress = `${a.region} ${a.city} ${a.district} ${a.subregion}`;
        }
      }
      
      // 1. REGIONS 배열에서 직접 매칭 ("경기", "서울" 등 포함 여부 확인)
      let foundRegion = REGIONS.find(r => fullAddress.includes(r));

      // 2. 못 찾았다면 도시 이름 매핑 시도 ("김포" -> "경기")
      if (!foundRegion) {
        for (const [city, region] of Object.entries(CITY_TO_REGION_MAP)) {
          if (fullAddress.includes(city)) {
            foundRegion = region;
            break;
          }
        }
      }

      const finalRegion = foundRegion || "서울";
      setCurrentRegion(finalRegion);
      return finalRegion;

    } catch (error) {
      console.log("위치 가져오기 실패:", error);
    }
    return '서울';
  };

  const loadRecommendations = async (regionToUse) => {
    try {
      setRecsLoading(true);
      const data = await fetchRecommendations(regionToUse);
      
      if (data && data.course) {
        const formattedRecs = data.course.map(item => ({
            id: item.spotId,
            name: item.spotName,
            address: item.address,
            matchScore: item.matchScore,
            tags: item.features,
            thumbnail: `${SERVER_URL}/images/${encodeURIComponent(item.spotName)}.jpg`
        }));
        setRecs(formattedRecs);
      } else {
        setRecs([]);
      }
    } catch (e) {
      console.error(e);
      setErrorMsg("추천 코스를 불러오지 못했습니다.");
    } finally {
      setRecsLoading(false);
    }
  };

  useEffect(() => {
    if (!user) return;
    const init = async () => {
      const detected = await getLocation();
      loadRecommendations(detected);
    };
    init();
  }, [user]);

  const handleRegionChange = (newRegion) => {
    setCurrentRegion(newRegion);
    setModalVisible(false);
    loadRecommendations(newRegion); 
  };

  if (loading) {
    return <View style={styles.center}><ActivityIndicator /></View>;
  }

  if (!user) {
    return <Redirect href="/login" />;
  }

  const renderRecItem = ({ item }) => (
    <TouchableOpacity
      style={styles.card}
      onPress={() => router.push(`/place/${item.id}`)}
    >
      <Image
        source={{ uri: item.thumbnail }}
        style={styles.cardImage}
        resizeMode="cover"
      />
      <View style={styles.cardContent}>
        <Text style={styles.cardTitle} numberOfLines={1}>{item.name}</Text>
        <Text style={styles.cardAddress} numberOfLines={1}>{item.address}</Text>
        {item.matchScore && (
            <Text style={styles.matchScore}>
            🤖 {Math.round(item.matchScore * 100)}%
            </Text>
        )}
        {item.tags && (
        <View style={styles.tagContainer}>
            {(Array.isArray(item.tags) ? item.tags : [item.tags]).slice(0,2).map((tag, idx) => (
                <Text key={idx} style={styles.tag}>#{tag}</Text>
            ))}
        </View>
        )}
      </View>
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <View style={{ marginBottom: 16 }}>
        <Text style={styles.greeting}>
          안녕하세요, {user?.name || "코스메이트"}님 👋
        </Text>
        <Text style={styles.subGreeting}>
          리뷰 기반 AI 맞춤 관광지 추천 서비스, 코스메이트입니다.
        </Text>
      </View>

      <TouchableOpacity 
        style={styles.locationButton} 
        onPress={() => setModalVisible(true)}
      >
        <Text style={styles.locationText}>📍 현재 지역: {currentRegion} (변경)</Text>
      </TouchableOpacity>

      <View style={{ flexDirection: "row", justifyContent: "space-between", marginBottom: 16 }}>
        <TouchableOpacity style={cardBtn} onPress={() => router.push("/preferences")}>
          <Text style={cardBtnTitle}>취향 태그</Text>
          <Text style={cardBtnDesc}>나만의 여행 스타일 설정</Text>
        </TouchableOpacity>
        <TouchableOpacity style={cardBtn} onPress={() => router.push("/recommendations")}>
          <Text style={cardBtnTitle}>AI 추천 코스</Text>
          <Text style={cardBtnDesc}>맞춤 관광 코스 받기</Text>
        </TouchableOpacity>
      </View>

      <TouchableOpacity style={mainBtn} onPress={() => router.push("/place-search")}>
        <Text style={{ color: "#fff", fontWeight: "600" }}>관광지 찾으러 가기</Text>
      </TouchableOpacity>

      <TouchableOpacity style={subBtn} onPress={() => router.push("/community")}>
        <Text style={{ fontWeight: "500" }}>공지 / 문의(커뮤니티) 보기</Text>
      </TouchableOpacity>

      <View style={{ marginTop: 8, marginBottom: 8 }}>
        <Text style={styles.sectionTitle}>
          오늘의 추천 코스 ({currentRegion})
        </Text>
        <Text style={{ color: "#6b7280", marginBottom: 8 }}>
          취향 태그를 기반으로 추천된 관광지들이에요.
        </Text>
      </View>

      {recsLoading ? (
        <ActivityIndicator size="large" color="#4F46E5"/>
      ) : recs.length === 0 ? (
        <Text style={styles.emptyText}>
          {errorMsg ? errorMsg : "해당 지역에 대한 추천 결과가 없습니다."}
        </Text>
      ) : (
        <FlatList
          data={recs}
          keyExtractor={(item) => String(item.id)}
          horizontal
          showsHorizontalScrollIndicator={false}
          renderItem={renderRecItem}
          contentContainerStyle={{ paddingBottom: 20, paddingRight: 16 }}
        />
      )}

      <TouchableOpacity
        style={floatingBtn}
        onPress={() => router.push("/my-page")}
      >
        <Text style={{ color: "#fff", fontWeight: "600" }}>마이페이지</Text>
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
  container: { flex: 1, padding: 16, backgroundColor: '#fff' },
  center: { flex: 1, justifyContent: "center", alignItems: "center" },
  greeting: { fontSize: 22, fontWeight: "700" },
  subGreeting: { marginTop: 6, color: "#4b5563" },
  
  // ★ [디자인 수정] 홈 화면 카드는 가로 공간이 좁으므로 세로 배치 유지하되 깔끔하게
  card: {
    borderRadius: 12, borderWidth: 1, borderColor: "#e5e7eb", marginRight: 12,
    width: 200, backgroundColor: 'white', overflow: 'hidden', paddingBottom: 12,
  },
  cardImage: { width: '100%', height: 120, backgroundColor: '#eee' },
  cardContent: { paddingHorizontal: 10, paddingTop: 10 },
  cardTitle: { fontSize: 16, fontWeight: "600", marginBottom: 2 },
  cardAddress: { fontSize: 12, color: "#4b5563", marginBottom: 4 },
  matchScore: { fontSize: 12, color: "#4F46E5", fontWeight: '600', marginBottom: 4 },
  tagContainer: { flexDirection: 'row', flexWrap: 'wrap' },
  tag: { color: "#6F42C1", fontSize: 11, marginRight: 4, backgroundColor: "#F3E8FF", paddingHorizontal: 6, borderRadius: 4 },
  
  sectionTitle: { fontSize: 18, fontWeight: "600", marginBottom: 4 },
  emptyText: { color: "#9ca3af", padding: 20, textAlign: 'center' },
  locationButton: {
    backgroundColor: '#F3F4F6', padding: 10, borderRadius: 8, marginBottom: 16,
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

const cardBtn = { flex: 1, marginRight: 8, padding: 12, borderRadius: 12, backgroundColor: "#EEF2FF" };
const cardBtnTitle = { fontSize: 15, fontWeight: "600" };
const cardBtnDesc = { marginTop: 4, fontSize: 12, color: "#4b5563" };
const mainBtn = { padding: 14, borderRadius: 999, backgroundColor: "#4F46E5", alignItems: "center", marginBottom: 12 };
const subBtn = { padding: 12, borderRadius: 999, borderWidth: 1, borderColor: "#e5e7eb", alignItems: "center", marginBottom: 16 };
const floatingBtn = { position: "absolute", right: 16, bottom: 16, paddingVertical: 10, paddingHorizontal: 16, borderRadius: 999, backgroundColor: "#111827", elevation: 5, shadowColor: "#000", shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.25, shadowRadius: 3.84 };