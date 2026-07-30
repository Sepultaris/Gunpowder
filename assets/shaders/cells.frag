#version 450

layout(binding = 0) uniform usampler2D materialTexture;
layout(binding = 2) uniform usampler2D visibilityTexture;
layout(binding = 4) uniform usampler2D derivedTexture;
layout(rgba8ui, binding = 1) uniform readonly uimage2D rawLightingImage;
layout(rgba16f, binding = 13) uniform readonly image2D rawGiImage;
layout(binding = 15) uniform sampler2D giTexture;
layout(std430, binding = 12) readonly buffer SceneLightBuffer {
    vec4 playerLight;
    vec4 fireLights[64];
    uvec4 lightCounts;
} sceneLights;

layout(location = 0) in vec2 textureCoordinate;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Camera {
    vec4 view;
    vec2 worldSize;
    uint debugMode;
    float time;
    vec4 atmosphere;
    // x: normalized time of day, y: daylight, z: sky intensity
    vec4 celestial;
    // x: specular strength, y: normal detail, z: marble polish,
    // w: dark-vein reflectivity
    vec4 materialSurface;
    // x: marble fleck density, y: marble fleck reflectivity
    vec4 materialDetail;
    // x: enabled, y: field density, z: kernel radius, w: edge softness
    vec4 fluidSurface;
    // x: flow stretch, y: pool flattening, z: normal strength
    vec4 fluidMotion;
} camera;

float hash21(vec2 point) {
    vec3 value = fract(vec3(point.xyx) * 0.1031);
    value += dot(value, value.yzx + 33.33);
    return fract((value.x + value.y) * value.z);
}

float cellHash(ivec2 cell) {
    return hash21(vec2(cell));
}

float marbleFleck(ivec2 cell) {
    // A second, decorrelated cell hash gives the marble sparse mineral
    // inclusions without a texture lookup. Because it is indexed by the
    // absolute world cell, the flecks remain fixed while the camera moves.
    return step(
        1.0 - clamp(camera.materialDetail.x, 0.0, 0.25),
        hash21(vec2(cell) + vec2(71.3, 29.7)));
}

float smoothNoise(vec2 point) {
    vec2 base = floor(point);
    vec2 fraction = fract(point);
    fraction = fraction * fraction * (3.0 - 2.0 * fraction);
    float bottom = mix(hash21(base),
                       hash21(base + vec2(1.0, 0.0)), fraction.x);
    float top = mix(hash21(base + vec2(0.0, 1.0)),
                    hash21(base + vec2(1.0, 1.0)), fraction.x);
    return mix(bottom, top, fraction.y);
}

float organicNoise(vec2 point) {
    float result = 0.0;
    float weight = 0.56;
    mat2 rotation = mat2(0.80, 0.60, -0.60, 0.80);
    for (int octave = 0; octave < 4; ++octave) {
        result += smoothNoise(point) * weight;
        point = rotation * point * 2.03 + vec2(7.1, 3.7);
        weight *= 0.48;
    }
    return result;
}

vec3 palette(uint material, ivec2 cell) {
    vec2 point = vec2(cell);
    float fine = cellHash(cell);
    switch (material) {
        case 1: {
            float broad = organicNoise(point * 0.075);
            float detail =
                organicNoise(point * 0.23 + vec2(13.0, 5.0));
            vec3 darkSoil = vec3(0.20, 0.105, 0.045);
            vec3 ochreSoil = vec3(0.46, 0.285, 0.105);
            vec3 soil = mix(darkSoil, ochreSoil,
                            smoothstep(0.24, 0.82, broad));
            soil *= 0.84 + detail * 0.27;
            soil += vec3(0.10, 0.065, 0.018) * step(0.97, fine);
            return soil;
        }
        case 2: {
            float detail =
                organicNoise(point * 0.23 + vec2(13.0, 5.0));
            float dune = organicNoise(point * vec2(0.11, 0.075) +
                                      vec2(2.0, 17.0));
            vec3 sand = mix(vec3(0.49, 0.33, 0.105),
                            vec3(0.84, 0.67, 0.275), dune);
            sand *= 0.90 + detail * 0.16;
            sand += vec3(0.16, 0.12, 0.035) * step(0.88, fine);
            return sand;
        }
        case 3: {
            float formation = organicNoise(point * 0.085);
            float veinField = abs(organicNoise(point * 0.13 +
                                               vec2(31.0, 9.0)) - 0.52);
            float vein = 1.0 - smoothstep(0.025, 0.085, veinField);
            vec3 rock = mix(vec3(0.105, 0.115, 0.135),
                            vec3(0.31, 0.32, 0.35), formation);
            return rock * (1.0 - vein * 0.42) +
                   vec3(fine * 0.025);
        }
        case 4: {
            float current = organicNoise(point * 0.12 +
                                         vec2(camera.time * 0.18, 0.0));
            return mix(vec3(0.025, 0.20, 0.43),
                       vec3(0.075, 0.45, 0.78), current) +
                   vec3(0.0, fine * 0.025, fine * 0.055);
        }
        case 5: {
            float sheen = organicNoise(point * 0.15 +
                                       vec2(camera.time * 0.06,
                                            -camera.time * 0.035));
            return mix(vec3(0.095, 0.055, 0.105),
                       vec3(0.27, 0.095, 0.29), sheen * 0.58);
        }
        case 6: {
            float flicker = organicNoise(
                point * vec2(0.34, 0.19) +
                vec2(camera.time * 1.7, -camera.time * 2.8));
            return mix(vec3(1.0, 0.12, 0.01),
                       vec3(1.0, 0.78, 0.08), flicker);
        }
        case 7: {
            float detail =
                organicNoise(point * 0.23 + vec2(13.0, 5.0));
            float billow = organicNoise(
                point * 0.095 +
                vec2(camera.time * 0.025, -camera.time * 0.055));
            vec3 smoke = vec3(0.19, 0.205, 0.235) +
                         vec3(billow * 0.13 + detail * 0.025);
            return mix(vec3(0.018, 0.025, 0.042), smoke, 0.82);
        }
        case 8: {
            float vapor = organicNoise(
                point * 0.11 +
                vec2(camera.time * 0.02, -camera.time * 0.08));
            return mix(vec3(0.43, 0.49, 0.53),
                       vec3(0.78, 0.84, 0.87), vapor);
        }
        case 9: {
            float grain = organicNoise(vec2(point.x * 0.07,
                                            point.y * 0.29));
            float knot = smoothstep(0.79, 0.94,
                                    organicNoise(point * 0.105 +
                                                 vec2(9.0, 22.0)));
            vec3 timber = mix(vec3(0.24, 0.105, 0.028),
                              vec3(0.57, 0.31, 0.075), grain);
            return timber * (1.0 - knot * 0.34) +
                   vec3(fine * 0.025);
        }
        case 10: {
            float body = organicNoise(
                point * 0.052 + vec2(18.0, 4.0));
            // Gently warp the vein coordinate with the broad body pattern.
            // This avoids straight procedural stripes while retaining long,
            // recognizable marble veins.
            vec2 warpedPoint =
                point * vec2(0.105, 0.078) +
                vec2(body * 2.7, -body * 1.9) +
                vec2(41.0, 27.0);
            float veinField = abs(
                organicNoise(warpedPoint) - 0.49);
            float veinCloud =
                1.0 - smoothstep(0.026, 0.092, veinField);
            float darkVein =
                1.0 - smoothstep(0.006, 0.030, veinField);

            vec3 warmMarble = mix(
                vec3(0.31, 0.285, 0.235),
                vec3(0.72, 0.665, 0.545),
                smoothstep(0.18, 0.86, body));
            warmMarble = mix(
                warmMarble, vec3(0.105, 0.112, 0.118),
                veinCloud * 0.46);
            warmMarble = mix(
                warmMarble, vec3(0.035, 0.042, 0.052),
                darkVein * 0.88);

            float fleck = marbleFleck(cell);
            float warmFleck =
                step(0.5, hash21(vec2(cell) + vec2(13.1, 91.7)));
            vec3 fleckColor = mix(
                vec3(0.42, 0.47, 0.53),
                vec3(0.58, 0.43, 0.19),
                warmFleck);
            return mix(warmMarble, fleckColor, fleck * 0.84);
        }
        case 11: {
            float brushed =
                organicNoise(point * vec2(0.055, 0.42));
            float oxidation =
                smoothstep(0.70, 0.92,
                           organicNoise(point * 0.095 +
                                        vec2(23.0, 11.0)));
            vec3 steel = mix(vec3(0.16, 0.19, 0.22),
                             vec3(0.49, 0.54, 0.58), brushed);
            return mix(steel, vec3(0.27, 0.13, 0.055),
                       oxidation * 0.42);
        }
        default: {
            // Empty space only needs a faint low-frequency variation. A
            // single smooth octave avoids running the full terrain texture
            // stack over most of the screen.
            float haze = smoothNoise(point * 0.035) * 0.0015;
            float dust = step(0.9992, fine) * 0.035;
            return vec3(0.0007 + haze, 0.0010 + haze, 0.0020 + haze) +
                   vec3(dust);
        }
    }
}

vec3 smoothLiquidPalette(uint material, vec2 worldPoint) {
    if (material == 4u) {
        // Metaball shading is evaluated continuously rather than once per
        // simulation cell. Animating every octave made its fine detail crawl
        // against display pixels during camera motion. Keep the texture in
        // absolute world space; actual transport and the reconstructed
        // surface provide the motion.
        float current =
            organicNoise(worldPoint * 0.12);
        float fine =
            smoothNoise(worldPoint * 0.58 + vec2(7.3, 19.1));
        return mix(
                   vec3(0.025, 0.20, 0.43),
                   vec3(0.075, 0.45, 0.78),
                   current) +
               vec3(0.0, fine * 0.025, fine * 0.055);
    }
    float sheen =
        organicNoise(worldPoint * 0.15 + vec2(3.7, 11.9));
    float fine = smoothNoise(
        worldPoint * 0.47 + vec2(17.0, 31.0));
    return mix(
               vec3(0.095, 0.055, 0.105),
               vec3(0.27, 0.095, 0.29),
               sheen * 0.58) *
           mix(0.96, 1.05, fine);
}

bool isSolidMaterial(uint material) {
    return material == 1 || material == 2 || material == 3 ||
           material == 9 || material == 10 || material == 11;
}

bool isLiquidMaterial(uint material) {
    return material == 4u || material == 5u;
}

vec2 unpackNormalizedPair(float packedPair) {
    uint packed = floatBitsToUint(packedPair);
    return vec2(
        float(packed & 0xffffu),
        float((packed >> 16u) & 0xffffu)) /
        65535.0;
}

float materialMatch(
    ivec2 sampleCell, ivec2 textureExtent, uint material) {
    sampleCell = clamp(
        sampleCell, ivec2(0), textureExtent - ivec2(1));
    return texelFetch(materialTexture, sampleCell, 0).r == material
               ? 1.0
               : 0.0;
}

struct MarbleDepthSample {
    float depth;
    vec2 outward;
};

MarbleDepthSample marbleDepthAt(
    ivec2 cell, ivec2 textureExtent, vec2 preferredOutward,
    int maxDistanceCells) {
    MarbleDepthSample result;
    result.depth = float(maxDistanceCells) + 0.5;
    result.outward =
        length(preferredOutward) > 0.001
            ? normalize(preferredOutward)
            : vec2(0.0, -1.0);
    const ivec2 directions[8] = ivec2[](
        ivec2(1, 0), ivec2(-1, 0),
        ivec2(0, 1), ivec2(0, -1),
        ivec2(1, 1), ivec2(-1, 1),
        ivec2(1, -1), ivec2(-1, -1));

    maxDistanceCells = clamp(maxDistanceCells, 1, 32);
    for (int distanceCells = 1;
         distanceCells <= 32; ++distanceCells) {
        if (distanceCells > maxDistanceCells) {
            break;
        }

        float openingCount = 0.0;
        vec2 openingDirection = vec2(0.0);
        vec2 bestDirection = result.outward;
        float bestAlignment = -2.0;
        for (int index = 0; index < 8; ++index) {
            ivec2 sampleCell =
                cell + directions[index] * distanceCells;
            bool outside =
                any(lessThan(sampleCell, ivec2(0))) ||
                any(greaterThanEqual(sampleCell, textureExtent));
            bool opening =
                outside ||
                texelFetch(
                    materialTexture,
                    clamp(sampleCell, ivec2(0),
                          textureExtent - ivec2(1)),
                    0).r != 10u;
            if (!opening) {
                continue;
            }

            vec2 direction =
                normalize(vec2(directions[index]));
            float alignment =
                dot(direction, result.outward);
            float directionWeight =
                0.20 + 0.80 * max(alignment, 0.0);
            openingDirection +=
                direction * directionWeight;
            openingCount += 1.0;
            if (alignment > bestAlignment) {
                bestAlignment = alignment;
                bestDirection = direction;
            }
        }

        if (openingCount > 0.0) {
            result.depth =
                float(distanceCells) -
                0.45 * clamp(openingCount * 0.125, 0.0, 1.0);
            result.outward =
                length(openingDirection) > 0.001
                    ? normalize(openingDirection)
                    : bestDirection;
            return result;
        }
    }
    return result;
}

float liquidDepthAt(
    vec2 materialPosition, ivec2 textureExtent, uint material) {
    ivec2 cell = clamp(
        ivec2(floor(materialPosition)),
        ivec2(0), textureExtent - ivec2(1));
    // Measure actual immersion in cells from the free surface downward. The
    // fractional position removes one-cell color bands across a calm pool.
    float depthCells = fract(materialPosition.y);
    for (int stepIndex = 1; stepIndex <= 24; ++stepIndex) {
        ivec2 sampleCell =
            cell + ivec2(0, -stepIndex);
        if (sampleCell.y < 0 ||
            texelFetch(materialTexture, sampleCell, 0).r != material) {
            break;
        }
        depthCells += 1.0;
    }
    return depthCells;
}

vec2 liquidOverheadAt(
    ivec2 cell, ivec2 textureExtent) {
    vec2 overhead = vec2(0.0);
    const int distances[4] = int[](1, 2, 4, 7);
    const float weights[4] =
        float[](1.0, 0.82, 0.58, 0.34);
    for (int index = 0; index < 4; ++index) {
        ivec2 sampleCell = clamp(
            cell + ivec2(0, -distances[index]),
            ivec2(0), textureExtent - ivec2(1));
        uint material =
            texelFetch(materialTexture, sampleCell, 0).r;
        if (material == 4u) {
            overhead.x = max(overhead.x, weights[index]);
        } else if (material == 5u) {
            overhead.y = max(overhead.y, weights[index]);
        }
    }
    return overhead;
}

vec2 decodedLiquidFlow(uvec4 data) {
    uint packedFlow = data.b;
    return clamp(
        (vec2(float((packedFlow >> 4u) & 15u),
              float(packedFlow & 15u)) - 7.0) / 7.0,
        vec2(-1.0), vec2(1.0));
}

struct LiquidMetaballSample {
    vec2 fields;
    vec2 waterGradient;
    vec2 oilGradient;
    vec2 waterFlow;
    vec2 oilFlow;
    vec2 flowWeights;
    uvec4 strongestWater;
    uvec4 strongestOil;
};

LiquidMetaballSample sampleLiquidMetaballs(
    vec2 position, ivec2 textureExtent) {
    LiquidMetaballSample result = LiquidMetaballSample(
        vec2(0.0), vec2(0.0), vec2(0.0),
        vec2(0.0), vec2(0.0), vec2(0.0),
        uvec4(0u), uvec4(0u));
    float strongestWaterContribution = 0.0;
    float strongestOilContribution = 0.0;
    float referenceField = 0.0;
    vec2 referenceGradient = vec2(0.0);
    ivec2 centerCell = ivec2(floor(position));
    float baseRadius =
        clamp(camera.fluidSurface.z, 0.35, 5.0);
    float density =
        clamp(camera.fluidSurface.y, 0.05, 3.0);
    float configuredPoolShape =
        clamp(camera.fluidMotion.y, 0.0, 1.0);
    float maximumConfiguredStretch =
        1.0 +
        clamp(camera.fluidMotion.x, 0.0, 3.0) * 0.82;
    float maximumKernelRadius = min(
        max(baseRadius * 1.48,
            baseRadius * maximumConfiguredStretch),
        5.0);
    int kernelRadius =
        clamp(int(ceil(maximumKernelRadius - 0.5)), 1, 5);

    // Dynamically uniform bounds keep the default radius on the original
    // 3x3 fast path. The larger neighborhoods are paid for only when the
    // artist deliberately raises the metaball radius.
    for (int offsetY = -kernelRadius;
         offsetY <= kernelRadius; ++offsetY) {
        for (int offsetX = -kernelRadius;
             offsetX <= kernelRadius; ++offsetX) {
            ivec2 sourceCell = clamp(
                centerCell + ivec2(offsetX, offsetY),
                ivec2(0), textureExtent - ivec2(1));
            vec2 sourceCenter =
                vec2(sourceCell) + vec2(0.5);

            // Normalize against the same kernel over a completely filled,
            // resting lattice. Compact metaball kernels do not naturally
            // form a partition of unity: their raw sum rises at cell centers
            // and falls between rows. That periodic field became visible as
            // coverage flicker whenever the camera moved. Dividing by this
            // reference makes uniform liquid evaluate to exactly one at
            // every sub-cell position while retaining a smooth boundary.
            vec2 referenceRadius = vec2(
                baseRadius *
                    mix(1.0, 1.48, configuredPoolShape),
                baseRadius *
                    mix(1.0, 0.68, configuredPoolShape));
            referenceRadius =
                min(referenceRadius, vec2(5.0));
            vec2 referenceDelta =
                position - sourceCenter;
            vec2 normalizedReferenceDelta =
                referenceDelta /
                max(referenceRadius, vec2(0.08));
            float referenceDistanceSquared =
                dot(normalizedReferenceDelta,
                    normalizedReferenceDelta);
            float referenceInfluence =
                max(1.0 - referenceDistanceSquared, 0.0);
            referenceField +=
                referenceInfluence *
                referenceInfluence * density;
            referenceGradient +=
                -4.0 * referenceInfluence * density *
                referenceDelta /
                max(referenceRadius * referenceRadius,
                    vec2(0.0064));

            uvec4 source =
                texelFetch(materialTexture, sourceCell, 0);
            if (!isLiquidMaterial(source.r) || source.g < 8u) {
                continue;
            }

            float fill = float(source.g) / 255.0;
            vec2 flow = decodedLiquidFlow(source);
            float speed = clamp(length(flow), 0.0, 1.0);
            // Packed flow has only fifteen signed levels. Ignore its lowest
            // bins for geometry so a resting pool cannot pulse as metadata
            // alternates between zero and one quantization step.
            float deformationSpeed =
                smoothstep(0.22, 0.62, speed);
            float resting =
                1.0 - deformationSpeed;
            float poolShape =
                configuredPoolShape * resting;
            float massRadius =
                baseRadius;
            vec2 radius = vec2(
                massRadius * mix(1.0, 1.48, poolShape),
                massRadius * mix(1.0, 0.68, poolShape));

            float verticalMotion =
                abs(flow.y) * deformationSpeed *
                clamp(camera.fluidMotion.x, 0.0, 3.0);
            float stretch = 1.0 + verticalMotion * 0.82;
            radius.y *= stretch;
            radius.x /= sqrt(stretch);
            radius = min(radius, vec2(5.0));

            sourceCenter.y += (1.0 - fill) * 0.42 * resting;
            vec2 delta = position - sourceCenter;
            vec2 normalizedDelta = delta / max(radius, vec2(0.08));
            float distanceSquared =
                dot(normalizedDelta, normalizedDelta);
            float influence =
                max(1.0 - distanceSquared, 0.0);
            float contribution =
                influence * influence * density * fill;
            if (contribution <= 0.0) {
                continue;
            }
            vec2 gradient =
                -4.0 * influence * density * fill *
                delta / max(radius * radius, vec2(0.0064));

            if (source.r == 4u) {
                result.fields.x += contribution;
                result.waterGradient += gradient;
                result.waterFlow += flow * contribution;
                result.flowWeights.x += contribution;
                if (contribution > strongestWaterContribution) {
                    strongestWaterContribution = contribution;
                    result.strongestWater = source;
                }
            } else {
                result.fields.y += contribution;
                result.oilGradient += gradient;
                result.oilFlow += flow * contribution;
                result.flowWeights.y += contribution;
                if (contribution > strongestOilContribution) {
                    strongestOilContribution = contribution;
                    result.strongestOil = source;
                }
            }
        }
    }
    float safeReferenceField =
        max(referenceField, 0.0001);
    vec2 unnormalizedFields = result.fields;
    result.fields /=
        safeReferenceField;
    result.waterGradient =
        (result.waterGradient * safeReferenceField -
         unnormalizedFields.x * referenceGradient) /
        (safeReferenceField * safeReferenceField);
    result.oilGradient =
        (result.oilGradient * safeReferenceField -
         unnormalizedFields.y * referenceGradient) /
        (safeReferenceField * safeReferenceField);
    return result;
}

struct SurfaceProperties {
    float roughness;
    float specular;
    float metallic;
};

// x: perceptual roughness, y: dielectric specular level, z: metallic.
// A compact constant table avoids another texture/descriptor read and keeps
// all surface definitions in one cache-friendly lookup.
const vec3 materialSurfaceTable[12] = vec3[](
    vec3(1.00, 0.00, 0.00), // air
    vec3(0.92, 0.28, 0.00), // dirt
    vec3(0.84, 0.32, 0.00), // sand
    vec3(0.72, 0.38, 0.03), // rock
    vec3(0.10, 0.62, 0.00), // water
    vec3(0.16, 0.58, 0.00), // oil
    vec3(1.00, 0.00, 0.00), // fire
    vec3(1.00, 0.00, 0.00), // smoke
    vec3(1.00, 0.00, 0.00), // steam
    vec3(0.68, 0.30, 0.00), // wood
    vec3(0.29, 0.74, 0.02), // polished marble
    vec3(0.24, 0.92, 0.94)  // metal
);

SurfaceProperties surfaceProperties(uint material) {
    vec3 packed = materialSurfaceTable[min(material, 11u)];
    return SurfaceProperties(packed.x, packed.y, packed.z);
}

SurfaceProperties variedSurfaceProperties(
    uint material, vec3 baseColor, ivec2 worldCell) {
    SurfaceProperties surface = surfaceProperties(material);
    if (material == 10u) {
        // The dark mineral veins polish more deeply than the pale body.
        // Crystalline flecks are nearly mirror-like and slightly metallic,
        // producing isolated colored reflections instead of uniform glitter.
        float luminance = dot(
            baseColor, vec3(0.2126, 0.7152, 0.0722));
        float darkMineral =
            1.0 - smoothstep(0.13, 0.31, luminance);
        float fleck = marbleFleck(worldCell);
        float reflectiveInclusion =
            max(darkMineral * 0.88 *
                    camera.materialSurface.w,
                fleck * camera.materialDetail.y);
        surface.roughness = mix(
            surface.roughness, 0.075,
            clamp(reflectiveInclusion, 0.0, 1.0));
        surface.specular = mix(
            surface.specular, 0.98,
            clamp(reflectiveInclusion, 0.0, 1.0));
        surface.roughness = clamp(
            surface.roughness +
                (1.0 - camera.materialSurface.z) * 0.28,
            0.035, 1.0);
        surface.specular *= clamp(
            0.25 + camera.materialSurface.z * 0.75,
            0.0, 1.75);
        surface.metallic = max(
            surface.metallic,
            fleck * 0.62 *
                clamp(camera.materialDetail.y, 0.0, 2.0));
    }
    return surface;
}

float pow5(float value) {
    float squared = value * value;
    return squared * squared * value;
}

vec3 surfaceF0(vec3 baseColor, SurfaceProperties surface) {
    vec3 dielectricF0 =
        vec3(mix(0.02, 0.08, surface.specular));
    return mix(dielectricF0, baseColor, surface.metallic);
}

// Isotropic GGX evaluated only once for each light class (player, clustered
// fire, and sun), rather than once per individual fire. This keeps metallic
// highlights practical at full display resolution.
vec3 directSpecular(
    vec2 normal2D, vec2 lightDirection2D, vec3 radiance,
    float irradiance, vec3 baseColor, SurfaceProperties surface,
    float surfaceWeight) {
    if (irradiance <= 0.0001 || surface.specular <= 0.0001 ||
        surfaceWeight <= 0.0001) {
        return vec3(0.0);
    }
    vec3 normal = normalize(vec3(normal2D, 1.10));
    vec3 lightDirection =
        normalize(vec3(lightDirection2D, 0.82));
    const vec3 viewDirection = vec3(0.0, 0.0, 1.0);
    vec3 halfway = normalize(lightDirection + viewDirection);
    float nDotL = max(dot(normal, lightDirection), 0.0);
    float nDotV = max(dot(normal, viewDirection), 0.001);
    float nDotH = max(dot(normal, halfway), 0.0);
    float vDotH = max(dot(viewDirection, halfway), 0.0);

    float alpha = max(surface.roughness * surface.roughness, 0.025);
    float alphaSquared = alpha * alpha;
    float denominator =
        nDotH * nDotH * (alphaSquared - 1.0) + 1.0;
    float distribution =
        alphaSquared /
        max(3.14159265 * denominator * denominator, 0.0001);
    float geometryK =
        (surface.roughness + 1.0) *
        (surface.roughness + 1.0) * 0.125;
    float geometryV =
        nDotV / (nDotV * (1.0 - geometryK) + geometryK);
    float geometryL =
        nDotL / (nDotL * (1.0 - geometryK) + geometryK);
    vec3 f0 = surfaceF0(baseColor, surface);
    vec3 fresnel =
        f0 + (vec3(1.0) - f0) * pow5(1.0 - vDotH);
    float ggxLobe =
        distribution * geometryV * geometryL /
        max(4.0 * nDotV, 0.001);
    // Strict GGX is often sub-pixel at this scale. A normalized broad lobe
    // preserves the roughness ordering while making highlights readable on
    // two-display-pixel material cells.
    float broadExponent =
        mix(2.5, 18.0, 1.0 - surface.roughness);
    float broadLobe =
        pow(nDotH, broadExponent) *
        (broadExponent + 2.0) * 0.105;
    float grazingLobe =
        pow5(1.0 - nDotV) *
        mix(0.12, 0.42, 1.0 - surface.roughness);
    float specularLobe = min(
        ggxLobe * 0.32 + broadLobe + grazingLobe,
        5.0);
    return radiance * fresnel * specularLobe *
           nDotL * irradiance * surfaceWeight * 1.55;
}

float pointLight(vec2 worldPosition, vec4 light) {
    if (light.w <= 0.0 || light.z <= 0.0) {
        return 0.0;
    }
    float distanceToLight = distance(worldPosition, light.xy);
    float falloff = clamp(1.0 - distanceToLight / light.z, 0.0, 1.0);
    falloff = falloff * falloff * (3.0 - 2.0 * falloff);
    return falloff * light.w;
}

float storedHazeAt(ivec2 sampleCell, ivec2 textureExtent) {
    sampleCell = clamp(sampleCell, ivec2(0),
                       textureExtent - ivec2(1));
    return float(texelFetch(derivedTexture, sampleCell, 0).r) /
           255.0;
}

float filteredHaze(vec2 position, ivec2 textureExtent) {
    // Stored values describe cell centers. Interpolating around those centers
    // removes the enlarged square voxels without softening solid materials.
    vec2 centeredPosition = position - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centeredPosition));
    vec2 blend = fract(centeredPosition);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float top = mix(
        storedHazeAt(baseCell, textureExtent),
        storedHazeAt(baseCell + ivec2(1, 0), textureExtent),
        blend.x);
    float bottom = mix(
        storedHazeAt(baseCell + ivec2(0, 1), textureExtent),
        storedHazeAt(baseCell + ivec2(1, 1), textureExtent),
        blend.x);
    return mix(top, bottom, blend.y);
}

float storedDerivedChannel(
    ivec2 sampleCell, ivec2 textureExtent, int channel) {
    sampleCell = clamp(sampleCell, ivec2(0),
                       textureExtent - ivec2(1));
    uvec4 fields = texelFetch(derivedTexture, sampleCell, 0);
    return float(fields[channel]) / 255.0;
}

float filteredDerivedChannel(
    vec2 position, ivec2 textureExtent, int channel) {
    vec2 centeredPosition = position - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centeredPosition));
    vec2 blend = fract(centeredPosition);
    blend = blend * blend * (3.0 - 2.0 * blend);
    float top = mix(
        storedDerivedChannel(
            baseCell, textureExtent, channel),
        storedDerivedChannel(
            baseCell + ivec2(1, 0), textureExtent, channel),
        blend.x);
    float bottom = mix(
        storedDerivedChannel(
            baseCell + ivec2(0, 1), textureExtent, channel),
        storedDerivedChannel(
            baseCell + ivec2(1, 1), textureExtent, channel),
        blend.x);
    return mix(top, bottom, blend.y);
}

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

vec3 proceduralSky(vec2 screenPosition, ivec2 worldCell) {
    float phase = camera.celestial.x;
    float angle = (phase - 0.25) * 6.28318530718;
    float sunHeight = sin(angle);
    float daylight = clamp(camera.celestial.y, 0.0, 1.0);
    float horizon = smoothstep(0.0, 0.92, screenPosition.y);

    vec3 nightTop = vec3(0.0015, 0.0035, 0.012);
    vec3 nightHorizon = vec3(0.010, 0.014, 0.030);
    vec3 dayTop = vec3(0.045, 0.20, 0.52);
    vec3 dayHorizon = vec3(0.50, 0.70, 0.90);
    vec3 sky = mix(
        mix(nightTop, nightHorizon, horizon),
        mix(dayTop, dayHorizon, horizon),
        daylight);

    float twilight =
        exp(-abs(sunHeight) * 8.0) *
        (1.0 - smoothstep(0.10, 0.42, abs(sunHeight)));
    sky += vec3(0.72, 0.16, 0.035) *
           twilight * smoothstep(0.35, 1.0, horizon);

    float star =
        step(0.9965, hash21(vec2(worldCell) * vec2(0.73, 1.19)));
    sky += vec3(0.62, 0.72, 0.92) *
           star * (1.0 - daylight) *
           (0.35 + 0.65 *
               hash21(vec2(worldCell) + vec2(17.0, 41.0)));

    vec2 sunPosition = vec2(
        0.5 + cos(angle) * 0.44,
        0.82 - max(sunHeight, 0.0) * 0.68);
    float sunDisk =
        1.0 - smoothstep(
            0.014, 0.030,
            distance(screenPosition, sunPosition));
    sunDisk *= smoothstep(-0.08, 0.03, sunHeight);
    sky += sunlightColor(phase) * sunDisk * 1.65;
    return sky * camera.celestial.z;
}

vec4 storedLightingAt(ivec2 sampleCell, ivec2 textureExtent) {
    sampleCell = clamp(sampleCell, ivec2(0),
                       textureExtent - ivec2(1));
    uvec4 packed =
        texelFetch(visibilityTexture, sampleCell, 0);
    uint packedGi = (packed.b << 4u) | (packed.a >> 4u);
    return vec4(
        vec2(packed.rg) / 255.0,
        float(packedGi) / 4095.0,
        float(packed.a & 15u));
}

vec4 filteredLighting(vec2 materialPosition) {
    const float lightingResolutionScale = 1.0;
    ivec2 extent = textureSize(visibilityTexture, 0);
    // Samples represent material-cell centers. Smooth interpolation prevents
    // visible steps when the grid is enlarged for a higher display resolution.
    vec2 lightingPosition =
        (materialPosition - vec2(lightingResolutionScale * 0.5)) /
        lightingResolutionScale;
    ivec2 baseCell = ivec2(floor(lightingPosition));
    vec2 blend = fract(lightingPosition);
    blend = blend * blend * (3.0 - 2.0 * blend);
    vec4 top = mix(
        storedLightingAt(baseCell, extent),
        storedLightingAt(baseCell + ivec2(1, 0), extent),
        blend.x);
    vec4 bottom = mix(
        storedLightingAt(baseCell + ivec2(0, 1), extent),
        storedLightingAt(baseCell + ivec2(1, 1), extent),
        blend.x);
    return mix(top, bottom, blend.y);
}

vec3 storedGiAt(ivec2 sampleCell, ivec2 textureExtent) {
    sampleCell = clamp(sampleCell, ivec2(0),
                       textureExtent - ivec2(1));
    return texelFetch(giTexture, sampleCell, 0).rgb;
}

vec3 filteredGi(vec2 materialPosition) {
    ivec2 extent = textureSize(giTexture, 0);
    vec2 centeredPosition = materialPosition - vec2(0.5);
    ivec2 baseCell = ivec2(floor(centeredPosition));
    vec2 blend = fract(centeredPosition);
    blend = blend * blend * (3.0 - 2.0 * blend);
    vec3 top = mix(
        storedGiAt(baseCell, extent),
        storedGiAt(baseCell + ivec2(1, 0), extent),
        blend.x);
    vec3 bottom = mix(
        storedGiAt(baseCell + ivec2(0, 1), extent),
        storedGiAt(baseCell + ivec2(1, 1), extent),
        blend.x);
    return mix(top, bottom, blend.y);
}

vec3 rawGiAt(vec2 materialPosition) {
    ivec2 extent = imageSize(rawGiImage);
    ivec2 sampleCell = clamp(
        ivec2(floor(materialPosition)), ivec2(0),
        extent - ivec2(1));
    return imageLoad(rawGiImage, sampleCell).rgb;
}

void main() {
    ivec2 size = textureSize(materialTexture, 0);
    vec2 worldPosition = textureCoordinate;
    ivec2 cell = clamp(ivec2(worldPosition),
                       ivec2(0), size - ivec2(1));
    if (camera.debugMode == 2u || camera.debugMode == 3u) {
        vec3 gi =
            camera.debugMode == 2u
                ? filteredGi(worldPosition)
                : rawGiAt(worldPosition);
        // A display gamma makes low-energy second and later bounces visible
        // without adding any direct, ambient, material, or haze lighting.
        vec3 visibleGi = pow(clamp(gi, vec3(0.0), vec3(1.0)),
                             vec3(0.60));
        outColor = vec4(visibleGi, 1.0);
        return;
    }
    uvec4 data = texelFetch(materialTexture, cell, 0);
    uint material = data.r;
    uint originalMaterial = material;
    bool visibleMaterialPixel = true;
    vec2 shadingNormal2D = vec2(0.0);
    float shadingSurfaceWeight = 0.0;
    float visualSkyMask =
        material == 0u ? float(data.g) / 255.0 : 0.0;
    bool metaballLiquidPixel = false;
    float metaballCoverage = 0.0;
    vec2 metaballNormal = vec2(0.0, -1.0);
    float metaballGradientStrength = 0.0;
    vec2 metaballFlow = vec2(0.0);
    bool metaballsEnabled =
        camera.fluidSurface.x > 0.5 &&
        (camera.debugMode == 0u ||
         camera.debugMode == 7u);

    if (metaballsEnabled &&
        (material == 0u || isLiquidMaterial(material))) {
        LiquidMetaballSample metaballs =
            sampleLiquidMetaballs(worldPosition, size);
        uint selectedMaterial =
            metaballs.fields.x >= metaballs.fields.y ? 4u : 5u;
        // Slightly favor the material already occupying the simulation cell
        // to prevent a one-frame identity shimmer at water/oil interfaces.
        if (material == 4u &&
            metaballs.fields.x >= metaballs.fields.y * 0.86) {
            selectedMaterial = 4u;
        } else if (material == 5u &&
                   metaballs.fields.y >= metaballs.fields.x * 0.86) {
            selectedMaterial = 5u;
        }
        float selectedField =
            selectedMaterial == 4u
                ? metaballs.fields.x
                : metaballs.fields.y;
        vec2 selectedGradient =
            selectedMaterial == 4u
                ? metaballs.waterGradient
                : metaballs.oilGradient;
        float edgeSoftness =
            clamp(camera.fluidSurface.w, 0.002, 0.35);
        const float surfaceThreshold = 0.42;
        // Use the analytic field slope to widen steep transitions. Unlike
        // screen derivatives, this remains defined where liquid meets a
        // divergent solid-material shader branch.
        edgeSoftness = max(
            edgeSoftness,
            min(length(selectedGradient) * 0.12, 0.18));
        metaballCoverage = smoothstep(
            surfaceThreshold - edgeSoftness,
            surfaceThreshold + edgeSoftness,
            selectedField);

        if (metaballCoverage > 0.001) {
            vec2 selectedFlowSum =
                selectedMaterial == 4u
                    ? metaballs.waterFlow
                    : metaballs.oilFlow;
            float selectedFlowWeight =
                selectedMaterial == 4u
                    ? metaballs.flowWeights.x
                    : metaballs.flowWeights.y;
            data =
                selectedMaterial == 4u
                    ? metaballs.strongestWater
                    : metaballs.strongestOil;
            material = selectedMaterial;
            metaballFlow =
                selectedFlowSum / max(selectedFlowWeight, 0.0001);
            metaballGradientStrength =
                length(selectedGradient);
            if (metaballGradientStrength > 0.0001) {
                metaballNormal = normalize(-selectedGradient);
            }
            metaballLiquidPixel = true;
        } else if (isLiquidMaterial(material)) {
            // The implicit surface is allowed to carve the square corners
            // from source liquid cells. Simulation occupancy is unchanged.
            material = 0u;
            visibleMaterialPixel = false;
        }
    }
    if (camera.debugMode == 7u) {
        vec3 fieldColor =
            material == 5u
                ? vec3(0.48, 0.12, 0.62)
                : vec3(0.05, 0.48, 0.92);
        outColor = vec4(
            mix(vec3(0.0), fieldColor, metaballCoverage),
            1.0);
        return;
    }
    // Close a one-cell gap inside a falling stream. Physics and material
    // conservation remain cell-exact; this only reconstructs its silhouette.
    if (!metaballsEnabled && material == 0u) {
        ivec2 aboveCell = max(cell + ivec2(0, -1), ivec2(0));
        ivec2 belowCell = min(cell + ivec2(0, 1), size - ivec2(1));
        uvec4 aboveData =
            texelFetch(materialTexture, aboveCell, 0);
        uvec4 belowData =
            texelFetch(materialTexture, belowCell, 0);
        bool verticalStreamGap =
            isLiquidMaterial(aboveData.r) &&
            aboveData.r == belowData.r;
        if (verticalStreamGap) {
            data = aboveData;
            data.g = 255u;
            material = aboveData.r;
        }
    }
    ivec2 worldCell = ivec2(floor(camera.view.xy)) + cell;
    vec2 absoluteMaterialPosition =
        floor(camera.view.xy) + worldPosition;
    vec2 screenPosition =
        (worldPosition - fract(camera.view.xy)) / camera.view.zw;
    vec3 color =
        metaballLiquidPixel
            ? smoothLiquidPalette(
                  material, absoluteMaterialPosition)
            : palette(material, worldCell);
    if (metaballLiquidPixel) {
        color = mix(
            palette(0u, worldCell), color,
            metaballCoverage);
    }
    if (material == 8) {
        // Steam is represented primarily by the filtered density volume.
        // Smoke retains its visible animated carrier particles.
        color = palette(0, worldCell);
    }

    if (material == 4 || material == 5) {
        float fill = float(data.g) / 255.0;
        float foam = float((data.a >> 4u) & 7u) / 7.0;
        vec2 flow =
            metaballLiquidPixel
                ? metaballFlow
                : decodedLiquidFlow(data);
        ivec2 aboveCell = max(cell + ivec2(0, -1), ivec2(0));
        ivec2 belowCell = min(cell + ivec2(0, 1), size - ivec2(1));
        ivec2 leftCell = max(cell + ivec2(-1, 0), ivec2(0));
        ivec2 rightCell = min(cell + ivec2(1, 0), size - ivec2(1));
        uvec4 aboveData = texelFetch(materialTexture, aboveCell, 0);
        uvec4 belowData = texelFetch(materialTexture, belowCell, 0);
        uvec4 leftData = texelFetch(materialTexture, leftCell, 0);
        uvec4 rightData = texelFetch(materialTexture, rightCell, 0);
        bool exposedSurface =
            cell.y == 0 || aboveData.r != material || aboveData.g < 8;
        bool verticalStream =
            abs(flow.y) > abs(flow.x) + 0.08 &&
            abs(flow.y) > 0.16 &&
            (leftData.r != material || rightData.r != material);
        if (camera.debugMode == 0 && !metaballLiquidPixel) {
            bool supported =
                belowData.r == 4u || belowData.r == 5u ||
                isSolidMaterial(belowData.r);
            bool pooledSurface = exposedSurface && supported;
            // Unsupported liquid is a falling droplet/stream and occupies a
            // complete visual cell. Only the supported top cell of a pool
            // uses fractional height. Rendering every tiny surface packet as
            // a full cell exaggerated low mass into tall water mounds.
            fill = data.g < 8
                       ? 0.0
                       : (pooledSurface
                              ? clamp(fill, 0.10, 1.0)
                              : 1.0);
            verticalStream = false;
        }
        vec2 localPosition = fract(worldPosition);
        bool liquidPixel =
            metaballLiquidPixel ||
            localPosition.y >= 1.0 - fill;
        float streamWidth = 1.0;
        float streamCenter = 0.5;
        if (!metaballLiquidPixel &&
            verticalStream && fill > 0.0) {
            // A falling cell represents a section of a continuous stream.
            // Use its mass as stream width instead of drawing the mass as a
            // short bottom-aligned bar in every grid cell.
            streamWidth = clamp(
                sqrt(fill) * (material == 4 ? 0.92 : 1.0),
                material == 4 ? 0.54 : 0.60, 1.0);
            streamCenter = clamp(0.5 + flow.x * 0.16,
                                 streamWidth * 0.5,
                                 1.0 - streamWidth * 0.5);
            bool connectedAbove =
                aboveData.r == material && aboveData.g >= 8;
            bool connectedBelow =
                belowData.r == material && belowData.g >= 8;
            float streamTop =
                connectedAbove ? 0.0 : max(0.0, 0.5 - fill * 0.65);
            float streamBottom =
                connectedBelow ? 1.0 : min(1.0, 0.5 + fill * 0.65);
            liquidPixel =
                abs(localPosition.x - streamCenter) <=
                    streamWidth * 0.5 &&
                localPosition.y >= streamTop &&
                localPosition.y <= streamBottom;
        }
        if (!liquidPixel) {
            color = palette(0, worldCell);
            visibleMaterialPixel = false;
        } else if (camera.debugMode != 0) {
            vec3 flowColor = vec3(0.5 + flow.x * 0.5,
                                  0.5 + flow.y * 0.5,
                                  0.18);
            color = mix(color, flowColor, 0.68);
        } else if (metaballLiquidPixel) {
            float edgeHighlight =
                1.0 - smoothstep(
                    0.48, 0.96, metaballCoverage);
            color +=
                (material == 4u
                     ? vec3(0.075, 0.19, 0.27)
                     : vec3(0.13, 0.092, 0.040)) *
                edgeHighlight;
        } else if (verticalStream) {
            float edgeDistance =
                streamWidth * 0.5 -
                abs(localPosition.x - streamCenter);
            float edgeHighlight =
                1.0 - smoothstep(0.0, 0.16, edgeDistance);
            color +=
                (material == 4
                     ? vec3(0.06, 0.17, 0.24)
                     : vec3(0.12, 0.085, 0.035)) *
                edgeHighlight;
        } else if (exposedSurface) {
            float surfaceDistance =
                fract(worldPosition.y) - (1.0 - fill);
            float surfaceHighlight =
                1.0 - smoothstep(0.0, 0.18, surfaceDistance);
            if (material == 4) {
                float horizontalFlow =
                    abs(flow.x) < 0.08 ? 0.0 : flow.x;
                float wave = 0.6 + 0.4 *
                    sin(float(worldCell.x) * 0.38 -
                        camera.time *
                            (1.8 + abs(horizontalFlow) * 4.2) *
                            (horizontalFlow < 0.0 ? -1.0 : 1.0));
                color += vec3(0.10, 0.25, 0.31) *
                         surfaceHighlight * wave;
            } else {
                vec3 rainbow = 0.5 + 0.5 *
                    cos(vec3(0.0, 2.1, 4.2) +
                        float(worldCell.x) * 0.16 + camera.time * 0.65);
                color = mix(color, rainbow * 0.42,
                            surfaceHighlight * 0.24);
            }
        }

        if (liquidPixel && camera.debugMode == 0) {
            if (metaballLiquidPixel) {
                // The summed field is nearly flat inside a pool. Its tiny
                // residual gradient has no meaningful surface direction and
                // used to swing as the camera crossed display pixels,
                // producing specular flashes throughout the liquid. Only
                // shade the reconstructed iso-surface, where both coverage
                // and the analytic gradient describe a reliable normal.
                float outerSurfaceBand =
                    smoothstep(0.025, 0.34, metaballCoverage);
                float innerSurfaceBand =
                    1.0 - smoothstep(
                        0.70, 0.985, metaballCoverage);
                float normalReliability =
                    smoothstep(
                        0.035, 0.16,
                        metaballGradientStrength);
                float surfaceBand =
                    outerSurfaceBand *
                    innerSurfaceBand *
                    normalReliability;
                shadingSurfaceWeight =
                    surfaceBand;
                shadingNormal2D =
                    metaballNormal *
                    camera.fluidMotion.z *
                    normalReliability;
            } else {
                shadingSurfaceWeight =
                    exposedSurface || verticalStream ? 1.0 : 0.16;
            }
            if (!metaballLiquidPixel && verticalStream) {
                float edgeSide =
                    clamp((localPosition.x - streamCenter) /
                              max(streamWidth * 0.5, 0.001),
                          -1.0, 1.0);
                shadingNormal2D =
                    normalize(vec2(edgeSide * 0.85, -0.18));
            } else if (!metaballLiquidPixel && exposedSurface) {
                float leftFill =
                    leftData.r == material
                        ? float(leftData.g) / 255.0
                        : 0.0;
                float rightFill =
                    rightData.r == material
                        ? float(rightData.g) / 255.0
                        : 0.0;
                float ripple =
                    sin((float(worldCell.x) + localPosition.x) * 0.31 -
                        camera.time * (material == 4u ? 1.4 : 0.55)) *
                    (material == 4u ? 0.11 : 0.045);
                shadingNormal2D =
                    normalize(vec2(leftFill - rightFill + ripple, -1.0));
            }
            float flowSpeed = clamp(length(flow), 0.0, 1.0);
            float visibleFlowThreshold =
                metaballLiquidPixel ? 0.24 : 0.08;
            if (flowSpeed > visibleFlowThreshold) {
                vec2 flowDirection =
                    normalize(flow + vec2(0.0001, 0.0001));
                vec2 crossDirection =
                    vec2(-flowDirection.y, flowDirection.x);
                vec2 worldPoint = vec2(worldCell) + localPosition;
                float alongFlow = dot(worldPoint, flowDirection);
                float acrossFlow = dot(worldPoint, crossDirection);
                float movingDetail =
                    metaballLiquidPixel
                        ? smoothNoise(
                              absoluteMaterialPosition * 0.13 +
                              vec2(5.1, 23.7))
                        : smoothNoise(vec2(
                              alongFlow * 0.16 -
                                  camera.time *
                                      (1.1 + flowSpeed * 2.8),
                              acrossFlow * 0.11));
                vec3 flowTint =
                    material == 4
                        ? vec3(0.07, 0.18, 0.25)
                        : vec3(0.075, 0.045, 0.025);
                color += flowTint *
                         (movingDetail - 0.38) *
                         flowSpeed * 0.72;
            }

            if (material == 4 && foam > 0.001) {
                float foamDetail =
                    metaballLiquidPixel
                        ? smoothNoise(
                              absoluteMaterialPosition * 0.31 +
                              vec2(29.7, 2.3))
                        : organicNoise(
                              (vec2(worldCell) + localPosition) * 0.43 +
                              vec2(camera.time * 0.22,
                                   -camera.time * 0.31));
                float edgeBias =
                    exposedSurface ? 1.0 :
                    (leftData.r != material ||
                     rightData.r != material ? 0.76 : 0.42);
                float foamMask =
                    foam * edgeBias *
                    smoothstep(0.38, 0.72, foamDetail);
                color = mix(color, vec3(0.68, 0.86, 0.96),
                            clamp(foamMask * 0.82, 0.0, 0.82));
            }
        }
    } else if (material == 9) {
        float charAmount = float(data.g) / 255.0;
        float heat = float(data.b) / 255.0;
        color = mix(color, vec3(0.075, 0.045, 0.025),
                    charAmount * 0.82);
        float emberPattern =
            float((worldCell.x * 19 + worldCell.y * 31) & 3) / 3.0;
        float ember = smoothstep(0.55, 0.9, heat) *
                      smoothstep(0.25, 0.8, charAmount) *
                      step(0.45, emberPattern);
        color += vec3(0.85, 0.19, 0.015) * ember;
    }

    // Irregularly emphasize exposed terrain while allowing solid interiors to
    // fall into shadow. This follows the simulated silhouette instead of
    // imposing a tile grid on it.
    if (isSolidMaterial(material)) {
        float leftSolid = isSolidMaterial(
            texelFetch(materialTexture,
                       max(cell + ivec2(-1, 0), ivec2(0)), 0).r)
                ? 1.0 : 0.0;
        float rightSolid = isSolidMaterial(
            texelFetch(materialTexture,
                       min(cell + ivec2(1, 0), size - ivec2(1)), 0).r)
                ? 1.0 : 0.0;
        float upperSolid = isSolidMaterial(
            texelFetch(materialTexture,
                       max(cell + ivec2(0, -1), ivec2(0)), 0).r)
                ? 1.0 : 0.0;
        float lowerSolid = isSolidMaterial(
            texelFetch(materialTexture,
                       min(cell + ivec2(0, 1), size - ivec2(1)), 0).r)
                ? 1.0 : 0.0;
        vec2 surfaceNormal =
            vec2(leftSolid - rightSolid,
                 upperSolid - lowerSolid);
        float edge = smoothstep(0.08, 0.72,
                                length(surfaceNormal));
        shadingNormal2D =
            length(surfaceNormal) > 0.01
                ? normalize(surfaceNormal)
                : vec2(0.0);
        shadingSurfaceWeight = mix(0.18, 1.0, edge);
        float irregularShade =
            0.68 + organicNoise(vec2(worldCell) * 0.055) * 0.17;
        color *= mix(irregularShade, 1.16, edge);
    }

    if (camera.debugMode >= 4u && camera.debugMode <= 6u) {
        SurfaceProperties debugSurface =
            variedSurfaceProperties(
                visibleMaterialPixel ? material : 0u,
                color, worldCell);
        if (camera.debugMode == 4u) {
            outColor = vec4(
                color * (1.0 - debugSurface.metallic * 0.92), 1.0);
        } else if (camera.debugMode == 5u) {
            outColor = vec4(
                vec3(debugSurface.specular) *
                    mix(vec3(1.0), color, debugSurface.metallic),
                1.0);
        } else {
            outColor = vec4(vec3(debugSurface.metallic), 1.0);
        }
        return;
    }

    vec2 absoluteWorldPosition =
        absoluteMaterialPosition;
    vec4 tracedLighting = filteredLighting(worldPosition);
    vec3 bouncedLighting = filteredGi(worldPosition);
    SurfaceProperties surface =
        variedSurfaceProperties(
            visibleMaterialPixel ? material : 0u,
            color, worldCell);
    vec3 baseColor = color;
    vec2 packedMarbleOptics =
        unpackNormalizedPair(camera.materialDetail.z);
    float marbleSubsurfaceStrength =
        packedMarbleOptics.x * 3.0;
    float marbleScatterDistance =
        packedMarbleOptics.y * 12.0;
    vec2 packedLiquidOptics =
        unpackNormalizedPair(camera.materialDetail.w);
    float liquidReflectionStrength =
        packedLiquidOptics.x * 3.0;
    float liquidSubsurfaceStrength =
        packedLiquidOptics.y * 3.0;
    vec2 packedLiquidEffects =
        unpackNormalizedPair(camera.fluidMotion.w);
    float liquidCausticStrength =
        packedLiquidEffects.x * 3.0;
    float liquidDispersionStrength =
        packedLiquidEffects.y * 2.0;
    float liquidDepth = 0.0;
    vec3 liquidTransmission = vec3(1.0);
    if (isLiquidMaterial(material)) {
        liquidDepth =
            liquidDepthAt(worldPosition, size, material);
        vec3 absorptionCoefficient =
            material == 4u
                ? vec3(0.055, 0.018, 0.006)
                : vec3(0.025, 0.075, 0.140);
        liquidTransmission = exp(
            -absorptionCoefficient *
            liquidDepth);

        vec2 opticalFlow =
            metaballLiquidPixel
                ? metaballFlow
                : decodedLiquidFlow(data);
        bool fallingColumn =
            abs(opticalFlow.y) >
                abs(opticalFlow.x) + 0.10 &&
            abs(opticalFlow.y) > 0.22;
        if (!fallingColumn) {
            // A settled pool has one coherent free surface. Residual
            // gradients in the normalized metaball lattice are not physical
            // internal surfaces; using them for specular lighting creates a
            // repeating field of pale "clouds." Replace them with a stable
            // world-space ripple normal and let reflection decay smoothly
            // through the first few cells.
            vec2 ripplePoint =
                absoluteWorldPosition *
                    vec2(0.085, 0.052) +
                vec2(camera.time * 0.010,
                     -camera.time * 0.006);
            float rippleLeft =
                smoothNoise(ripplePoint - vec2(0.32, 0.0));
            float rippleRight =
                smoothNoise(ripplePoint + vec2(0.32, 0.0));
            float rippleSlope =
                (rippleLeft - rippleRight) * 0.62;
            shadingNormal2D =
                normalize(vec2(rippleSlope, -1.0));
            shadingSurfaceWeight =
                exp(-liquidDepth / 1.45);
        }
    }
    float marbleLuminance =
        dot(baseColor, vec3(0.2126, 0.7152, 0.0722));
    float marbleDarkMineral =
        material == 10u
            ? 1.0 - smoothstep(0.13, 0.31, marbleLuminance)
            : 0.0;
    float marbleReflectiveFleck =
        material == 10u ? marbleFleck(worldCell) : 0.0;
    if (material == 10u) {
        // Unlike porous terrain, the whole visible marble face is a polished
        // surface. Keep silhouette edges strongest, but allow reflections to
        // read across broad walls and columns as well.
        shadingSurfaceWeight = max(shadingSurfaceWeight, 0.46);
    }
    if (surface.specular > 0.0001 &&
        shadingSurfaceWeight > 0.0001) {
        vec2 microNormal;
        if (metaballLiquidPixel) {
            // Continuous world-space detail moves with the liquid surface.
            // Per-cell hashes are stable for solids but visibly shimmer when
            // a smooth implicit surface crosses cell boundaries.
            vec2 detailPoint =
                absoluteWorldPosition * 0.31;
            microNormal = vec2(
                smoothNoise(detailPoint + vec2(11.7, 4.3)) - 0.5,
                smoothNoise(detailPoint + vec2(37.1, 19.6)) - 0.5);
        } else {
            microNormal = vec2(
                cellHash(worldCell) - 0.5,
                hash21(vec2(worldCell) + vec2(19.7, 43.1)) - 0.5);
        }
        float microStrength =
            mix(0.035, 0.24, 1.0 - surface.roughness);
        microStrength *= camera.materialSurface.y;
        if (material == 11u) {
            // Brushed metal favors horizontal grooves instead of isotropic
            // sparkling noise.
            microNormal.y *= 0.22;
            microStrength *= 1.35;
        } else if (material == 10u) {
            // Polished marble has subtle, isotropic waviness rather than the
            // stronger granular breakup used by rough terrain.
            microStrength *= 0.46;
        } else if (metaballLiquidPixel) {
            microStrength *= 0.34;
        }
        shadingNormal2D += microNormal * microStrength;
    }
    float playerIllumination =
        pointLight(absoluteWorldPosition, sceneLights.playerLight);
    float fireIllumination = 0.0;
    vec2 fireDirectionSum = vec2(0.0);
    for (uint index = 0u; index < sceneLights.lightCounts.x; ++index) {
        vec4 fireLight = sceneLights.fireLights[index];
        float contribution =
            pointLight(absoluteWorldPosition, fireLight);
        fireIllumination += contribution;
        vec2 direction = fireLight.xy - absoluteWorldPosition;
        float directionLength = length(direction);
        if (directionLength > 0.001) {
            fireDirectionSum +=
                direction / directionLength * contribution;
        }
    }
    float sunVisibility =
        filteredDerivedChannel(worldPosition, size, 1);
    float skyExposure =
        filteredDerivedChannel(worldPosition, size, 2);
    vec3 sunColor = sunlightColor(camera.celestial.x);
    vec3 skyColor = skylightColor(camera.celestial.y);
    float sunAngle =
        (camera.celestial.x - 0.25) * 6.28318530718;
    vec2 sunDirection =
        vec2(cos(sunAngle), -sin(sunAngle));
    vec2 playerDirection =
        sceneLights.playerLight.xy - absoluteWorldPosition;
    if (length(playerDirection) > 0.001) {
        playerDirection = normalize(playerDirection);
    }
    vec2 fireDirection =
        length(fireDirectionSum) > 0.001
            ? normalize(fireDirectionSum)
            : vec2(0.0, -1.0);
    vec3 illumination =
        camera.atmosphere.x * vec3(0.833333, 1.0, 1.5);
    illumination += playerIllumination * tracedLighting.r *
                    vec3(1.00, 0.88, 0.69);
    illumination += min(fireIllumination, 1.35) *
                    tracedLighting.g *
                    vec3(1.18, 0.43, 0.10);
    illumination += bouncedLighting;
    illumination += sunVisibility * camera.celestial.y *
                    sunColor;
    illumination += skyExposure * camera.celestial.w *
                    skyColor;
    vec3 shadowedDirectRadiance =
        playerIllumination * tracedLighting.r *
            vec3(1.00, 0.88, 0.69) +
        min(fireIllumination, 1.35) * tracedLighting.g *
            vec3(1.18, 0.43, 0.10) +
        sunVisibility * camera.celestial.y * sunColor;

    // Subsurface scattering is driven only by visible, directional light.
    // Ambient light and directionless GI cannot identify which side of a
    // surface was illuminated, so including either makes back faces glow.
    vec2 scatteringNormal =
        length(shadingNormal2D) > 0.001
            ? normalize(shadingNormal2D)
            : vec2(0.0, -1.0);
    float playerFrontLight = smoothstep(
        0.0, 0.42, dot(scatteringNormal, playerDirection));
    float fireFrontLight = smoothstep(
        0.0, 0.42, dot(scatteringNormal, fireDirection));
    float sunFrontLight = smoothstep(
        0.0, 0.42, dot(scatteringNormal, sunDirection));
    float skyFrontLight = smoothstep(
        0.0, 0.42, dot(scatteringNormal, vec2(0.0, -1.0)));
    vec3 frontFacingSubsurfaceRadiance =
        playerIllumination * tracedLighting.r *
            playerFrontLight * vec3(1.00, 0.88, 0.69) +
        min(fireIllumination, 1.35) * tracedLighting.g *
            fireFrontLight * vec3(1.18, 0.43, 0.10) +
        sunVisibility * camera.celestial.y *
            sunFrontLight * sunColor +
        skyExposure * camera.celestial.w *
            skyFrontLight * skyColor;

    vec3 specularLighting = vec3(0.0);
    if (surface.specular > 0.0001 &&
        shadingSurfaceWeight > 0.0001) {
        specularLighting += directSpecular(
            shadingNormal2D, playerDirection,
            vec3(1.00, 0.88, 0.69),
            playerIllumination * tracedLighting.r,
            baseColor, surface, shadingSurfaceWeight);

        specularLighting += directSpecular(
            shadingNormal2D, fireDirection,
            vec3(1.18, 0.43, 0.10),
            min(fireIllumination, 1.35) * tracedLighting.g,
            baseColor, surface, shadingSurfaceWeight);
        specularLighting += directSpecular(
            shadingNormal2D, sunDirection, sunColor,
            sunVisibility * camera.celestial.y,
            baseColor, surface, shadingSurfaceWeight);

        // A low-frequency sheen keeps smooth materials readable when the
        // mathematically correct highlight falls between material pixels.
        // It uses the already computed, shadowed light energy, so it neither
        // glows in darkness nor needs additional rays.
        float sheenStrength =
            mix(0.035, 0.38, 1.0 - surface.roughness) *
            surface.specular;
        specularLighting +=
            shadowedDirectRadiance *
            surfaceF0(baseColor, surface) *
            sheenStrength * shadingSurfaceWeight;
        if (material == 10u) {
            // Dark inclusions and mica crystals sit under the marble's
            // polished clear surface. Reinforce their broad reflected light
            // using existing radiance so they remain visible at pixel scale
            // without behaving like emissive dust.
            vec3 mineralTint = mix(
                vec3(0.76, 0.86, 1.0),
                max(baseColor, vec3(0.16)),
                0.22);
            float mineralReflection =
                marbleDarkMineral * 0.18 *
                    camera.materialSurface.w +
                marbleReflectiveFleck * 0.58 *
                    camera.materialDetail.y;
            specularLighting +=
                shadowedDirectRadiance * mineralTint *
                mineralReflection * shadingSurfaceWeight;
        }

        vec3 environment =
            bouncedLighting +
            skyExposure * camera.celestial.w * skyColor;
        float environmentGloss =
            mix(0.16, 1.0,
                (1.0 - surface.roughness) *
                (1.0 - surface.roughness));
        specularLighting +=
            environment * surfaceF0(baseColor, surface) *
            environmentGloss * shadingSurfaceWeight * 1.18;
        if (material == 10u) {
            float mineralEnvironment =
                marbleDarkMineral * 0.24 *
                    camera.materialSurface.w +
                marbleReflectiveFleck * 0.82 *
                    camera.materialDetail.y;
            specularLighting +=
                environment *
                mix(vec3(0.72, 0.82, 0.96),
                    max(baseColor, vec3(0.18)), 0.18) *
                mineralEnvironment * shadingSurfaceWeight;
        }
        // The game often exposes materials in nearly enclosed rooms where
        // the sampled environment is intentionally very dark. Retain a small
        // roughness-aware local reflection so metal reads as metal instead of
        // becoming a black silhouette between highlight angles.
        specularLighting +=
            illumination * surfaceF0(baseColor, surface) *
            surface.specular * shadingSurfaceWeight *
            mix(0.045, 0.30,
                (1.0 - surface.roughness) *
                (1.0 - surface.roughness));
    }

    vec3 opticalScattering = vec3(0.0);
    if (material == 10u &&
        marbleSubsurfaceStrength > 0.0001) {
        float scatterLength =
            max(marbleScatterDistance, 0.35);
        int depthSearchDistance = int(clamp(
            ceil(scatterLength * 3.0), 4.0, 32.0));
        MarbleDepthSample marbleDepth =
            marbleDepthAt(
                cell, size, scatteringNormal,
                depthSearchDistance);
        vec2 entryPosition =
            worldPosition +
            marbleDepth.outward *
                max(marbleDepth.depth - 0.35, 0.0);
        vec4 entryLighting =
            filteredLighting(entryPosition);
        float entrySunVisibility =
            filteredDerivedChannel(entryPosition, size, 1);
        float entrySkyExposure =
            filteredDerivedChannel(entryPosition, size, 2);
        float playerEntryFacing = smoothstep(
            0.0, 0.42,
            dot(marbleDepth.outward, playerDirection));
        float fireEntryFacing = smoothstep(
            0.0, 0.42,
            dot(marbleDepth.outward, fireDirection));
        float sunEntryFacing = smoothstep(
            0.0, 0.42,
            dot(marbleDepth.outward, sunDirection));
        float skyEntryFacing = smoothstep(
            0.0, 0.42,
            dot(marbleDepth.outward, vec2(0.0, -1.0)));
        vec3 marbleIncidentRadiance =
            playerIllumination * entryLighting.r *
                playerEntryFacing *
                vec3(1.00, 0.88, 0.69) +
            min(fireIllumination, 1.35) * entryLighting.g *
                fireEntryFacing *
                vec3(1.18, 0.43, 0.10) +
            entrySunVisibility * camera.celestial.y *
                sunEntryFacing * sunColor +
            entrySkyExposure * camera.celestial.w *
                skyEntryFacing * skyColor;
        float depthTransmission =
            exp(-marbleDepth.depth / scatterLength);
        float translucentMineral =
            1.0 - marbleDarkMineral * 0.78;
        vec3 marbleScatterTint =
            mix(vec3(0.92, 0.45, 0.22),
                max(baseColor, vec3(0.16)),
                0.58);
        float scatterDistanceResponse =
            smoothstep(
                0.0, 8.0, marbleScatterDistance);
        opticalScattering +=
            marbleIncidentRadiance *
            marbleScatterTint *
            marbleSubsurfaceStrength *
            translucentMineral *
            depthTransmission *
            mix(0.13, 0.19, scatterDistanceResponse);
    }

    if (isLiquidMaterial(material)) {
        vec3 liquidNormal =
            normalize(vec3(shadingNormal2D, 1.05));
        float viewFacing =
            clamp(abs(liquidNormal.z), 0.0, 1.0);
        float fresnel =
            0.018 +
            (1.0 - 0.018) *
                pow5(1.0 - viewFacing);
        vec3 environmentRadiance =
            bouncedLighting +
            skyExposure * camera.celestial.w * skyColor +
            shadowedDirectRadiance * 0.22;
        float upwardSurface =
            max(dot(
                    length(shadingNormal2D) > 0.001
                        ? normalize(shadingNormal2D)
                        : vec2(0.0, -1.0),
                    vec2(0.0, -1.0)),
                0.0);
        specularLighting +=
            environmentRadiance *
            liquidReflectionStrength *
            shadingSurfaceWeight *
            mix(0.34, 1.35, fresnel) *
            mix(0.72, 1.22, upwardSurface);
        // A broad, low-energy reflection keeps a calm surface legible even
        // when its sub-pixel normal misses every sharp direct highlight.
        specularLighting +=
            (illumination * vec3(0.22, 0.24, 0.27) +
             skyColor * camera.celestial.w * skyExposure +
             shadowedDirectRadiance * 0.28) *
            liquidReflectionStrength *
            shadingSurfaceWeight *
            upwardSurface * 0.16;

        vec3 liquidScatterTint =
            material == 4u
                ? vec3(0.055, 0.20, 0.34)
                : vec3(0.48, 0.16, 0.035);
        float liquidScatterDepth =
            1.0 - exp(
                -liquidDepth *
                (material == 4u ? 0.050 : 0.095));
        opticalScattering +=
            frontFacingSubsurfaceRadiance *
            liquidScatterTint *
            liquidSubsurfaceStrength *
            liquidScatterDepth *
            (material == 4u ? 0.055 : 0.12);

        float spectralPhase =
            absoluteWorldPosition.x * 0.34 +
            absoluteWorldPosition.y * 0.11 +
            camera.time * 0.12;
        vec3 spectralColor =
            0.5 + 0.5 *
                cos(vec3(0.0, 2.0944, 4.1888) +
                    spectralPhase);
        float dispersionScale =
            material == 5u ? 0.18 : 0.065;
        specularLighting +=
            environmentRadiance *
            spectralColor *
            liquidDispersionStrength *
            dispersionScale *
            shadingSurfaceWeight *
            (0.24 + fresnel * 0.76);
    }

    // The GPU-derived field supplies smoothly averaged local/room density.
    float nearbySmoke = filteredHaze(worldPosition, size);
    float hazeVariation = smoothNoise(
        absoluteWorldPosition * 0.055 +
        vec2(camera.time * 0.018, -camera.time * 0.026));
    float shapedSmoke = 0.0;
    if (nearbySmoke > 0.001) {
        float smokeDetail = organicNoise(
            absoluteWorldPosition * 0.14 +
            vec2(-camera.time * 0.035, -camera.time * 0.075));
        float billowShape = smoothstep(
            0.18, 0.82, hazeVariation * 0.72 + smokeDetail * 0.28);
        shapedSmoke = clamp(
            nearbySmoke * mix(0.42, 1.22, billowShape), 0.0, 1.0);
    }
    float hazeDensity =
        (camera.atmosphere.y +
         shapedSmoke * camera.atmosphere.z) *
        (0.72 + hazeVariation * 0.46);
    vec3 scatteredLight =
        playerIllumination * tracedLighting.r *
            vec3(0.34, 0.42, 0.52) +
        min(fireIllumination, 1.35) * tracedLighting.g *
            vec3(1.00, 0.30, 0.055) +
        bouncedLighting * 0.28;
    scatteredLight +=
        sunVisibility * camera.celestial.y *
        sunColor * 0.62;
    scatteredLight +=
        skyExposure * camera.celestial.w *
        skyColor * 0.38;

    float vignette =
        1.0 - smoothstep(0.38, 0.82, distance(screenPosition, vec2(0.5))) *
                  0.22;
    illumination *= vignette;
    if (camera.debugMode != 0) {
        illumination = vec3(1.0);
    }

    float diffuseWeight =
        1.0 - surface.metallic * 0.78;
    if (isLiquidMaterial(material)) {
        // In this 2-D world an empty room is what lies behind a liquid cell.
        // Begin with exactly that empty-space shading and attenuate it by the
        // wavelength-dependent transmission. Water adds no opaque blue
        // diffuse body; its blue appearance emerges only as red and green
        // light are absorbed with increasing depth.
        float refractiveVariation =
            mix(0.96, 1.04,
                smoothNoise(
                    absoluteWorldPosition *
                        vec2(0.065, 0.043) +
                    vec2(camera.time * 0.012,
                         -camera.time * 0.009)));
        vec3 emptyRoomColor =
            palette(0u, worldCell) * illumination;
        vec3 refractedRoom =
            emptyRoomColor *
            liquidTransmission *
            refractiveVariation;
        vec3 liquidBody = refractedRoom;
        if (material == 5u) {
            // Oil contains suspended pigment and remains visibly denser than
            // clear water, but even it approaches opacity gradually.
            float oilDepthResponse =
                1.0 - exp(-liquidDepth * 0.085);
            float oilOpacity =
                clamp(oilDepthResponse * 0.64, 0.0, 0.76);
            liquidBody = mix(
                refractedRoom,
                baseColor * illumination * 0.30,
                oilOpacity);
        }
        color =
            liquidBody +
            specularLighting * camera.materialSurface.x +
            opticalScattering;
    } else {
        color =
            baseColor * illumination * diffuseWeight +
            specularLighting * camera.materialSurface.x +
            opticalScattering;
    }
    if (isSolidMaterial(material) &&
        liquidCausticStrength > 0.0001) {
        vec2 liquidOverhead =
            liquidOverheadAt(cell, size);
        float overheadAmount =
            max(liquidOverhead.x, liquidOverhead.y);
        if (overheadAmount > 0.001) {
            float waveA =
                sin(absoluteWorldPosition.x * 0.52 +
                    absoluteWorldPosition.y * 0.16 +
                    camera.time * 0.31);
            float waveB =
                sin(absoluteWorldPosition.x * -0.37 +
                    absoluteWorldPosition.y * 0.41 -
                    camera.time * 0.23);
            float focusedCaustic =
                pow(clamp(
                        0.50 + waveA * waveB * 0.50,
                        0.0, 1.0),
                    6.0);
            vec3 causticTint =
                liquidOverhead.x >= liquidOverhead.y
                    ? vec3(0.50, 0.86, 1.0)
                    : vec3(1.0, 0.50, 0.16);
            color +=
                (shadowedDirectRadiance +
                 skyExposure * camera.celestial.w *
                     skyColor * 0.42) *
                causticTint *
                focusedCaustic *
                overheadAmount *
                liquidCausticStrength * 0.26;
        }
    }
    if (camera.debugMode == 0) {
        if (originalMaterial == 0u) {
            float uncoveredSky =
                visualSkyMask *
                (1.0 - metaballCoverage);
            color = mix(
                color, proceduralSky(screenPosition, worldCell),
                uncoveredSky);
        }
        color *= 1.0 - hazeDensity * camera.atmosphere.w;
        float scatteringDensity =
            hazeDensity * exp(-shapedSmoke * 1.35);
        vec3 visibleHaze =
            scatteredLight * scatteringDensity * vignette;
        if (isLiquidMaterial(material)) {
            // Treat room haze as the background seen through the liquid so
            // shallow water matches the surrounding room and deep water
            // becomes blue only through selective absorption.
            visibleHaze *= liquidTransmission;
        }
        color += visibleHaze;
        float smokeOpacity =
            1.0 - exp(-shapedSmoke * 1.55);
        vec3 visibleSmoke =
            vec3(0.030, 0.036, 0.047) +
            scatteredLight * (0.09 + hazeVariation * 0.07);
        color = mix(color, visibleSmoke,
                    smokeOpacity * 0.70);
    }
    if (material == 6) {
        color += palette(6, worldCell) * 0.72;
    } else if (material == 9) {
        float hotWood = smoothstep(0.58, 0.95, float(data.b) / 255.0) *
                        smoothstep(0.18, 0.85, float(data.g) / 255.0);
        color += vec3(0.72, 0.105, 0.008) * hotWood;
    }
    outColor = vec4(color, 1.0);
}
