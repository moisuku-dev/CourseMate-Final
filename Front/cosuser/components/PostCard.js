// components/PostCard.js
import React from "react";
import { View, Text, StyleSheet, TouchableOpacity } from "react-native";
import COLORS from "../constants/colors";

export default function PostCard({ post, onPress }) {
  return (
    <TouchableOpacity style={styles.card} onPress={onPress}>
      <Text style={styles.title} numberOfLines={1}>
        {post.title}
      </Text>
      <Text style={styles.meta}>
        {post.author} · {post.createdAt}
      </Text>
      <Text style={styles.preview} numberOfLines={2}>
        {post.preview}
      </Text>
      <View style={styles.footer}>
        <Text style={styles.footerText}>❤️ {post.likes}</Text>
        <Text style={styles.footerText}>댓글 {post.commentCount}</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    padding: 14,
    borderRadius: 16,
    backgroundColor: COLORS.card,
    marginVertical: 6,
  },
  title: {
    fontSize: 15,
    fontWeight: "600",
    color: COLORS.text,
  },
  meta: {
    marginTop: 4,
    fontSize: 11,
    color: COLORS.muted,
  },
  preview: {
    marginTop: 8,
    fontSize: 13,
    color: COLORS.text,
  },
  footer: {
    marginTop: 10,
    flexDirection: "row",
    justifyContent: "space-between",
  },
  footerText: {
    fontSize: 11,
    color: COLORS.muted,
  },
});
