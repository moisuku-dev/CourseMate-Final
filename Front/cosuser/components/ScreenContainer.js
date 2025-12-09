// components/ScreenContainer.js
import React from "react";
import { SafeAreaView, View, StyleSheet } from "react-native";
import COLORS from "../constants/colors";

export default function ScreenContainer({ children, style }) {
  return (
    <SafeAreaView style={styles.safe}>
      <View style={[styles.container, style]}>{children}</View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: COLORS.background,
  },
  container: {
    flex: 1,
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: COLORS.background,
  },
});
