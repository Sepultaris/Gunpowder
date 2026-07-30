#version 450

layout(binding = 0) uniform usampler2D materialTexture;
layout(binding = 2) uniform usampler2D visibilityTexture;
layout(binding = 4) uniform usampler2D derivedTexture;
layout(binding = 15) uniform sampler2D giTexture;
layout(std430, binding = 12) readonly buffer SceneLightBuffer {
    vec4 playerLight;
    vec4 fireLights[64];
    uvec4 lightCounts;
} sceneLights;

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 worldPosition;
layout(location = 2) in float lightingStrength;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Camera {
    vec4 view;
    vec2 worldSize;
    uint debugMode;
    float time;
    vec4 atmosphere;
    vec4 celestial;
    vec4 materialSurface;
    vec4 materialDetail;
    vec4 fluidSurface;
    vec4 fluidMotion;
} camera;

vec3 sunlightColor(float phase) {
    float angle = (phase - 0.25) * 6.28318530718;
    float height = sin(angle);
    float warm = 1.0 - smoothstep(0.03, 0.58, max(height, 0.0));
    return mix(vec3(1.00, 0.95, 0.78),
               vec3(1.00, 0.34, 0.095), warm);
}

vec3 skylightColor(float daylight) {
    float highSun =
        smoothstep(0.08, 0.82, clamp(daylight, 0.0, 1.0));
    return mix(vec3(1.00, 0.48, 0.20),
               vec3(0.48, 0.68, 1.00), highSun);
}

float sunVisibilityAt(ivec2 cell, ivec2 extent) {
    cell = clamp(cell, ivec2(0), extent - ivec2(1));
    return float(texelFetch(derivedTexture, cell, 0).g) / 255.0;
}

float skyExposureAt(ivec2 cell, ivec2 extent) {
    cell = clamp(cell, ivec2(0), extent - ivec2(1));
    return float(texelFetch(derivedTexture, cell, 0).b) / 255.0;
}

float filteredSkyExposure(vec2 texturePosition) {
    ivec2 extent = textureSize(derivedTexture, 0);
    vec2 centered = texturePosition - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centered));
    vec2 blend = fract(centered);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float top = mix(
        skyExposureAt(baseCell, extent),
        skyExposureAt(baseCell + ivec2(1, 0), extent),
        blend.x);
    float bottom = mix(
        skyExposureAt(baseCell + ivec2(0, 1), extent),
        skyExposureAt(baseCell + ivec2(1, 1), extent),
        blend.x);
    return mix(top, bottom, blend.y);
}

float filteredSunVisibility(vec2 texturePosition) {
    ivec2 extent = textureSize(derivedTexture, 0);
    vec2 centered = texturePosition - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centered));
    vec2 blend = fract(centered);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float top = mix(
        sunVisibilityAt(baseCell, extent),
        sunVisibilityAt(baseCell + ivec2(1, 0), extent),
        blend.x);
    float bottom = mix(
        sunVisibilityAt(baseCell + ivec2(0, 1), extent),
        sunVisibilityAt(baseCell + ivec2(1, 1), extent),
        blend.x);
    return mix(top, bottom, blend.y);
}

float pointLight(vec2 position, vec4 light) {
    if (light.w <= 0.0 || light.z <= 0.0) {
        return 0.0;
    }
    float distanceToLight = distance(position, light.xy);
    float falloff =
        clamp(1.0 - distanceToLight / light.z, 0.0, 1.0);
    falloff = falloff * falloff * (3.0 - 2.0 * falloff);
    return falloff * light.w;
}

vec4 lightingAt(ivec2 cell, ivec2 extent) {
    cell = clamp(cell, ivec2(0), extent - ivec2(1));
    uvec4 packed = texelFetch(visibilityTexture, cell, 0);
    uint packedGi = (packed.b << 4u) | (packed.a >> 4u);
    return vec4(
        vec2(packed.rg) / 255.0,
        float(packedGi) / 4095.0,
        float(packed.a & 15u));
}

vec4 filteredLighting(vec2 texturePosition) {
    ivec2 extent = textureSize(visibilityTexture, 0);
    vec2 centered = texturePosition - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centered));
    vec2 blend = fract(centered);
    blend = blend * blend * (3.0 - 2.0 * blend);
    vec4 top = mix(
        lightingAt(baseCell, extent),
        lightingAt(baseCell + ivec2(1, 0), extent),
        blend.x);
    vec4 bottom = mix(
        lightingAt(baseCell + ivec2(0, 1), extent),
        lightingAt(baseCell + ivec2(1, 1), extent),
        blend.x);
    return mix(top, bottom, blend.y);
}

vec3 giAt(ivec2 cell, ivec2 extent) {
    cell = clamp(cell, ivec2(0), extent - ivec2(1));
    return texelFetch(giTexture, cell, 0).rgb;
}

vec3 filteredGi(vec2 texturePosition) {
    ivec2 extent = textureSize(giTexture, 0);
    vec2 centered = texturePosition - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centered));
    vec2 blend = fract(centered);
    blend = blend * blend * (3.0 - 2.0 * blend);
    vec3 top = mix(
        giAt(baseCell, extent),
        giAt(baseCell + ivec2(1, 0), extent),
        blend.x);
    vec3 bottom = mix(
        giAt(baseCell + ivec2(0, 1), extent),
        giAt(baseCell + ivec2(1, 1), extent),
        blend.x);
    return mix(top, bottom, blend.y);
}

void main() {
    if (camera.debugMode >= 2u) {
        discard;
    }
    if (lightingStrength < 0.5) {
        outColor = fragColor;
        return;
    }

    vec4 lighting = filteredLighting(
        worldPosition - floor(camera.view.xy));
    vec3 bouncedLighting = filteredGi(
        worldPosition - floor(camera.view.xy));
    float playerIllumination =
        pointLight(worldPosition, sceneLights.playerLight);
    float fireIllumination = 0.0;
    for (uint index = 0u; index < sceneLights.lightCounts.x; ++index) {
        fireIllumination +=
            pointLight(worldPosition, sceneLights.fireLights[index]);
    }
    vec3 illumination =
        camera.atmosphere.x * vec3(0.833333, 1.0, 1.5);
    illumination += playerIllumination * lighting.r *
                    vec3(1.00, 0.88, 0.69);
    illumination += min(fireIllumination, 1.35) * lighting.g *
                    vec3(1.18, 0.43, 0.10);
    illumination += bouncedLighting;
    illumination +=
        filteredSunVisibility(
            worldPosition - floor(camera.view.xy)) *
        camera.celestial.y *
        sunlightColor(camera.celestial.x);
    illumination +=
        filteredSkyExposure(
            worldPosition - floor(camera.view.xy)) *
        camera.celestial.w *
        skylightColor(camera.celestial.y);
    illumination += vec3(max(lightingStrength - 1.0, 0.0));
    if (camera.debugMode != 0) {
        illumination = vec3(1.0);
    }
    outColor = vec4(
        clamp(fragColor.rgb * illumination, 0.0, 1.0),
        fragColor.a);
}
