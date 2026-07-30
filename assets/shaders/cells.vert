#version 450

layout(location = 0) out vec2 textureCoordinate;

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

const vec2 positions[6] = vec2[](
    vec2(-1.0, -1.0),
    vec2(-1.0,  1.0),
    vec2( 1.0,  1.0),
    vec2(-1.0, -1.0),
    vec2( 1.0,  1.0),
    vec2( 1.0, -1.0)
);

const vec2 localCoordinates[6] = vec2[](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 1.0),
    vec2(0.0, 0.0),
    vec2(1.0, 1.0),
    vec2(1.0, 0.0)
);

void main() {
    vec2 local = localCoordinates[gl_VertexIndex];
    textureCoordinate = fract(camera.view.xy) + local * camera.view.zw;
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
