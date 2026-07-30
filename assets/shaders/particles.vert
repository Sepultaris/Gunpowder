#version 450

struct Particle {
    vec2 position;
    vec2 velocity;
    vec4 colorAndLife;
    vec2 maximumLifeAndSize;
    uint slot;
    uint padding;
};

layout(std430, binding = 5) readonly buffer ParticleBuffer {
    Particle particles[];
};

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 worldPosition;
layout(location = 2) out float lightingStrength;

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

const vec2 corners[6] = vec2[](
    vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(1.0, -1.0)
);

void main() {
    Particle particle = particles[gl_InstanceIndex];
    float maximumLife = particle.maximumLifeAndSize.x;
    float life = particle.colorAndLife.w;
    if (life <= 0.0 || maximumLife <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        fragColor = vec4(0.0);
        worldPosition = particle.position;
        lightingStrength = 0.0;
        return;
    }

    vec2 world =
        particle.position + corners[gl_VertexIndex] *
        particle.maximumLifeAndSize.y;
    vec2 normalized = (world - camera.view.xy) / camera.view.zw;
    gl_Position = vec4(normalized * 2.0 - 1.0, 0.0, 1.0);
    fragColor = vec4(
        particle.colorAndLife.rgb,
        clamp(life / maximumLife, 0.0, 1.0));
    worldPosition = world;
    // Debris and droplets are lit by the same traced visibility field.
    lightingStrength = 1.0;
}
