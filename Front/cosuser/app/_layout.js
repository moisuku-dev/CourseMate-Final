// app/_layout.js
import React from "react";
import { Stack } from "expo-router";
import { ActivityIndicator, View, StyleSheet } from "react-native";
import { AuthProvider } from "../context/AuthContext";
import { useAuth } from "../hooks/useAuth";

function AuthGate() {
  const { loading, token } = useAuth();

  if (loading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (!token) {
    return (
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="login" />
        <Stack.Screen name="sign-up" />
      </Stack>
    );
  }

  return (
    <Stack
      screenOptions={{
        headerShown: true,
        headerTitle: "코스메이트",            
        headerBackButtonDisplayMode: "minimal", 
        headerTintColor: "#4F46E5",
      }}
    >
      <Stack.Screen name="index" />
      <Stack.Screen name="place-search" />
      <Stack.Screen name="place/[id]" />
      <Stack.Screen name="review-write" />
      <Stack.Screen name="wishlist" />
      <Stack.Screen name="my-page" />
      <Stack.Screen name="community" />
      <Stack.Screen name="community/[id]" />
      <Stack.Screen name="settings" />
    </Stack>
  );
}

export default function RootLayout() {
  return (
    <AuthProvider>
      <AuthGate />
    </AuthProvider>
  );
}

const styles = StyleSheet.create({
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
  },
});

