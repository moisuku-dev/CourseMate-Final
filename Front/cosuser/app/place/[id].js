// app/place/[id].js
import React, { useEffect, useState, useMemo } from "react";
import {
  View,
  Text,
  ActivityIndicator,
  ScrollView,
  TouchableOpacity,
  Image,
  FlatList,
  Alert,
  Linking,
  StyleSheet,
  Platform,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import { fetchPlaceDetail, fetchPlacePhotos } from "../../api/places";
import { fetchPlaceReviews } from "../../api/reviews";
import {
  fetchWishlist,
  addToWishlist,
  removeFromWishlist,
} from "../../api/user";
import COLORS from "../../constants/colors";
import ScreenContainer from "../../components/ScreenContainer";

// 서버 주소 설정
const SERVER_URL = Platform.select({
  ios: 'http://localhost:3000',
  android: 'http://10.0.2.2:3000',
  web: 'http://localhost:3000',
  default: 'http://localhost:3000',
});

export default function PlaceDetailScreen() {
  const { id } = useLocalSearchParams();
  const router = useRouter();

  const [place, setPlace] = useState(null);
  const [photos, setPhotos] = useState([]);
  const [reviews, setReviews] = useState([]);
  const [wishlist, setWishlist] = useState([]);
  const [loading, setLoading] = useState(true);
  const [wishlistLoading, setWishlistLoading] = useState(false);

  const isInWishlist = useMemo(() => {
    return wishlist.some((item) => String(item.placeId) === String(id));
  }, [wishlist, id]);

  useEffect(() => {
    const load = async () => {
      try {
        setLoading(true);

        const [placeRes, photoRes, reviewRes] = await Promise.all([
          fetchPlaceDetail(id),
          fetchPlacePhotos(id),
          fetchPlaceReviews(id),
        ]);

        // 1. 관광지 정보 매핑
        const rawPlace = placeRes.place || placeRes;
        if (rawPlace) {
            setPlace({
                ...rawPlace,
                name: rawPlace.NAME || rawPlace.name,
                address: rawPlace.ADDRESS || rawPlace.address,
                description: rawPlace.DESCRIPTION || rawPlace.description,
                avgRating: rawPlace.avgRating, // Controller에서 계산해서 줌
                reviewCount: rawPlace.reviewCount
            });
        }

        // 2. 사진 데이터 설정
        setPhotos(photoRes?.photos || photoRes || []);
        
        // 3. ★ [핵심 수정] 리뷰 데이터 매핑 (대문자 -> 소문자 변환)
        const rawReviews = reviewRes?.reviews || reviewRes || [];
        const formattedReviews = rawReviews.map(r => ({
            id: r.REVIEW_ID || r.reviewId || r.id,
            rating: r.RATING || r.rating,
            content: r.CONTENT || r.content, // ★ 여기서 내용을 확실히 연결
            nickname: r.nickname || r.NICKNAME || r.userName || "익명",
            regDate: r.REG_DATE || r.regDate
        }));
        setReviews(formattedReviews);

        // 4. 위시리스트 로딩 (실패해도 무시)
        try {
            const wishRes = await fetchWishlist();
            setWishlist(wishRes?.wishlist || wishRes || []);
        } catch (wishError) {
            console.warn("위시리스트 로딩 실패:", wishError);
        }

      } catch (e) {
        console.error("데이터 로딩 실패:", e);
        Alert.alert("오류", "관광지 정보를 불러오는 중 문제가 발생했습니다.");
      } finally {
        setLoading(false);
      }
    };
    if (id) load();
  }, [id]);

  const toggleWishlist = async () => {
    try {
      setWishlistLoading(true);
      if (isInWishlist) {
        await removeFromWishlist(id);
      } else {
        await addToWishlist(id);
      }
      const updated = await fetchWishlist();
      setWishlist(updated?.wishlist || updated || []);
    } catch (e) {
      console.error(e);
      Alert.alert("오류", "위시리스트 기능을 사용할 수 없습니다.");
    } finally {
      setWishlistLoading(false);
    }
  };

  const openNaverMap = () => {
    if (!place) return;
    const query = encodeURIComponent(place.name || place.placeName || "");
    const url = `https://map.naver.com/v5/search/${query}`;
    Linking.openURL(url).catch(() => {
      Alert.alert("오류", "네이버 지도를 열 수 없습니다.");
    });
  };

  const blogList = useMemo(() => {
    if (Array.isArray(place?.blogs)) return place.blogs;
    if (Array.isArray(place?.referenceBlogs)) return place.referenceBlogs;
    if (Array.isArray(place?.blogUrls)) {
      return place.blogUrls.map((url, idx) => ({
        id: idx,
        title: url,
        url,
      }));
    }
    return [];
  }, [place]);

  const openBlog = (url) => {
    if (!url) return;
    Linking.openURL(url).catch(() => {
      Alert.alert("오류", "블로그 링크를 열 수 없습니다.");
    });
  };

  // 이미지 URL 생성 헬퍼
  const getImageUrl = (url) => {
    if (!url) return null;
    if (url.startsWith('http')) return url;
    const cleanUrl = url.startsWith('/') ? url : `/${url}`;
    return `${SERVER_URL}${cleanUrl}`;
  };

  if (loading) {
      return (
          <ScreenContainer>
              <ActivityIndicator size="large" color={COLORS.primary} />
          </ScreenContainer>
      )
  }

  if (!place) {
    return (
      <ScreenContainer>
          <Text style={{textAlign: 'center', marginTop: 20}}>관광지 정보를 찾을 수 없습니다.</Text>
      </ScreenContainer>
    );
  }

  return (
    <ScreenContainer>
      <ScrollView>
        {/* 상단 정보 */}
        <View style={styles.headerRow}>
          <View style={{ flex: 1 }}>
            <Text style={styles.title}>
              {place.name || "관광지"}
            </Text>
            {!!place.address && (
              <Text style={styles.address}>{place.address}</Text>
            )}
          </View>

          <TouchableOpacity
            onPress={toggleWishlist}
            disabled={wishlistLoading}
            style={[
              styles.wishButton,
              { backgroundColor: isInWishlist ? "#F97316" : COLORS.primary },
            ]}
          >
            <Text style={styles.wishButtonText}>
              {isInWishlist ? "제거" : "찜하기"}
            </Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity onPress={openNaverMap} style={styles.naverButton}>
          <Text style={styles.naverButtonText}>네이버 지도에서 길찾기</Text>
        </TouchableOpacity>

        {/* 사진 섹션 */}
        <Text style={styles.sectionTitle}>사진</Text>
        {photos.length > 0 ? (
          <FlatList
            data={photos}
            keyExtractor={(item, idx) => String(item.photoId || idx)}
            horizontal
            showsHorizontalScrollIndicator={false}
            renderItem={({ item }) => (
              <Image
                source={{ uri: getImageUrl(item.url || item.photoUrl || item.IMG_URL || item) }}
                style={styles.photo}
              />
            )}
          />
        ) : (
          <Text style={styles.emptyText}>등록된 사진이 없습니다.</Text>
        )}

        {/* 소개 */}
        {!!place.description && (
          <>
            <Text style={styles.sectionTitle}>소개</Text>
            <Text style={styles.description}>{place.description}</Text>
          </>
        )}
        
        {/* 평점 정보 */}
        <View style={{marginTop: 16}}>
            <Text style={styles.sectionTitle}>평점</Text>
            <Text style={{fontSize: 16, color: '#333'}}>
                ⭐ {place.avgRating || "0.0"} ({place.reviewCount || 0}개의 리뷰)
            </Text>
        </View>

        {/* 리뷰 헤더 및 작성 버튼 */}
        <View style={styles.reviewHeaderRow}>
          <Text style={styles.sectionTitle}>리뷰</Text>
          <TouchableOpacity
            onPress={() =>
              router.push({
                pathname: "/review-write",
                params: {
                  placeId: id,
                  placeName: place.name,
                },
              })
            }
            style={styles.reviewButton}
          >
            <Text style={styles.reviewButtonText}>리뷰 작성</Text>
          </TouchableOpacity>
        </View>

        {/* ★ 리뷰 목록 (수정된 데이터 사용) */}
        {reviews.length === 0 ? (
          <Text style={styles.emptyText}>아직 등록된 리뷰가 없습니다.</Text>
        ) : (
          reviews.map((r) => (
            <View
              key={r.id}
              style={styles.reviewItem}
            >
              <Text style={styles.reviewTitle}>
                ★ {r.rating} / 5{" "}
                <Text style={styles.reviewUser}>
                  {r.nickname}
                </Text>
              </Text>
              {/* ★ 여기서 내용(content)이 표시됩니다 */}
              <Text style={styles.reviewContent}>{r.content}</Text>
              <Text style={styles.reviewDate}>
                {r.regDate ? new Date(r.regDate).toLocaleDateString() : ""}
              </Text>
            </View>
          ))
        )}
        <View style={{height: 50}} />
      </ScrollView>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  headerRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 12,
  },
  title: {
    fontSize: 22,
    fontWeight: "700",
    color: COLORS.text,
  },
  address: {
    marginTop: 4,
    color: COLORS.muted,
  },
  wishButton: {
    marginLeft: 12,
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    alignSelf: "flex-start",
  },
  wishButtonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 14,
  },
  naverButton: {
    marginTop: 4,
    marginBottom: 16,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: COLORS.primary,
    alignSelf: "flex-start",
  },
  naverButtonText: {
    color: COLORS.primary,
    fontWeight: "600",
  },
  sectionTitle: {
    marginTop: 24,
    marginBottom: 8,
    fontSize: 18,
    fontWeight: "700",
    color: COLORS.text,
  },
  photo: {
    width: 240,
    height: 160,
    borderRadius: 12,
    marginRight: 12,
    backgroundColor: "#eee",
  },
  emptyText: {
    color: COLORS.muted,
    fontSize: 14,
  },
  description: {
    color: COLORS.text,
    lineHeight: 22,
    fontSize: 15,
  },
  blogItem: {
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#f3f4f6",
  },
  blogTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: COLORS.text,
    marginBottom: 4,
  },
  blogLink: {
    fontSize: 13,
    color: COLORS.muted,
  },
  reviewHeaderRow: {
    marginTop: 24,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  reviewButton: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
    backgroundColor: COLORS.primary,
  },
  reviewButtonText: {
    color: "#fff",
    fontWeight: "600",
    fontSize: 14,
  },
  reviewItem: {
    paddingVertical: 16,
    borderBottomWidth: 1,
    borderBottomColor: "#f3f4f6",
  },
  reviewTitle: {
    fontWeight: "600",
    color: COLORS.text,
    marginBottom: 6,
    fontSize: 15,
  },
  reviewUser: {
    fontWeight: "400",
    color: "#9ca3af",
    fontSize: 14,
    marginLeft: 8,
  },
  reviewContent: {
    color: "#4b5563",
    fontSize: 15,
    lineHeight: 22,
  },
  reviewDate: {
    marginTop: 6,
    fontSize: 12,
    color: "#9ca3af",
  }
});