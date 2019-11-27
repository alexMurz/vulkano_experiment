#version 450
#extension GL_ARB_separate_shader_objects : enable
layout (push_constant) uniform PushConsts {
    mat4 mvp;
} push;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    int id = gl_InstanceIndex;
    v_uv = a_uv;
    gl_Position = push.mvp * vec4(a_pos + vec2(id * 1.0, 0.0), 0.0, 1.0);
}