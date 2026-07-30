#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec4 inColor;
layout(location = 2) in vec3 inLightingData;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 worldPosition;
layout(location = 2) out float lightingStrength;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
    worldPosition = inLightingData.xy;
    lightingStrength = inLightingData.z;
}
