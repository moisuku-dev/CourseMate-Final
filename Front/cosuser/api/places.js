// api/places.js
import request from "./client";

/**
 * 관광지 목록 검색: GET /api/places
 * @param {{ keyword?: string, page?: number, size?: number }} params
 */
export function fetchPlaces(params = {}) {
  const search = new URLSearchParams();
  if (params.keyword) search.set("keyword", params.keyword);
  if (params.page != null) search.set("page", String(params.page));
  if (params.size != null) search.set("size", String(params.size));

  const qs = search.toString();
  const path = qs ? `/places?${qs}` : "/places";

  return request(path);
}

/**
 * 관광지 상세: GET /api/places/{id}
 */
export function fetchPlaceDetail(id) {
  return request(`/places/${id}`);
}

/**
 * 관광지 사진 목록: GET /api/places/{id}/photos
 */
export function fetchPlacePhotos(id) {
  return request(`/places/${id}/photos`);
}
