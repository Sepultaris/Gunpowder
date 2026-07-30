#pragma once

#include <cmath>

namespace gunpowder {

struct Vec2 {
    float x = 0.0F;
    float y = 0.0F;

    Vec2& operator+=(Vec2 other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vec2& operator-=(Vec2 other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vec2& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }
};

inline Vec2 operator+(Vec2 lhs, Vec2 rhs) { return lhs += rhs; }
inline Vec2 operator-(Vec2 lhs, Vec2 rhs) { return lhs -= rhs; }
inline Vec2 operator*(Vec2 value, float scalar) { return value *= scalar; }
inline Vec2 operator*(float scalar, Vec2 value) { return value *= scalar; }

inline float length(Vec2 value) {
    return std::sqrt(value.x * value.x + value.y * value.y);
}

inline float dot(Vec2 first, Vec2 second) {
    return first.x * second.x + first.y * second.y;
}

inline Vec2 normalized(Vec2 value) {
    const float magnitude = length(value);
    return magnitude > 0.0001F ? value * (1.0F / magnitude) : Vec2{1.0F, 0.0F};
}

} // namespace gunpowder
