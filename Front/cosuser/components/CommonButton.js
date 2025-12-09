// components/CommonButton.js
import React from "react";
import { TouchableOpacity, Text, StyleSheet } from "react-native";
import COLORS from "../constants/colors";

export default function CommonButton({ title, onPress, variant = "filled", disabled }) {
  const isFilled = variant === "filled";

  return (
    <TouchableOpacity
      style={[
        styles.button,
        isFilled ? styles.filled : styles.outlined,
        disabled && styles.disabled,
      ]}
      onPress={onPress}
      disabled={disabled}
    >
      <Text
        style={[
          styles.text,
          isFilled ? styles.textFilled : styles.textOutlined,
        ]}
      >
        {title}
      </Text>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  button: {
    height: 48,
    borderRadius: 12,
    alignItems: "center",
    justifyContent: "center",
    marginVertical: 6,
    borderWidth: 1,
  },
  filled: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },
  outlined: {
    backgroundColor: "transparent",
    borderColor: COLORS.primary,
  },
  disabled: {
    opacity: 0.5,
  },
  text: {
    fontSize: 16,
    fontWeight: "600",
  },
  textFilled: {
    color: "#ffffff",
  },
  textOutlined: {
    color: COLORS.primary,
  },
});
