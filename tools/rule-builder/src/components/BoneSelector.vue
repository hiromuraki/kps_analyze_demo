<script setup lang="ts">
import { BONES, DEFAULT_BONE_COLOR, JOINT_LAYOUT, JOINT_NAMES } from '../constants'
import { boneKey } from '../lib/geometry'

const props = defineProps<{
  selected: Record<string, string>
}>()

const emit = defineEmits<{
  (e: 'toggle', a: number, b: number): void
}>()

/** 与某关节相连的已被选中的骨骼颜色（第一个匹配项）。 */
function jointColor(index: number): string {
  for (const [a, b] of BONES) {
    if (a !== index && b !== index) continue
    const color = props.selected[boneKey(a, b)]
    if (color) return color
  }
  return DEFAULT_BONE_COLOR
}

function isJointSelected(index: number): boolean {
  return BONES.some(
    ([a, b]) => (a === index || b === index) && !!props.selected[boneKey(a, b)],
  )
}
</script>

<template>
  <svg id="joint-svg" viewBox="0 0 260 380" height="280">
    <!-- 加宽的透明命中区，便于点选 -->
    <line
      v-for="(bone, i) in BONES"
      :key="`hit-${i}`"
      :x1="JOINT_LAYOUT[bone[0]].x"
      :y1="JOINT_LAYOUT[bone[0]].y"
      :x2="JOINT_LAYOUT[bone[1]].x"
      :y2="JOINT_LAYOUT[bone[1]].y"
      class="bone-hit"
      @click="emit('toggle', bone[0], bone[1])"
    />

    <line
      v-for="(bone, i) in BONES"
      :key="`line-${i}`"
      :x1="JOINT_LAYOUT[bone[0]].x"
      :y1="JOINT_LAYOUT[bone[0]].y"
      :x2="JOINT_LAYOUT[bone[1]].x"
      :y2="JOINT_LAYOUT[bone[1]].y"
      :stroke="selected[boneKey(bone[0], bone[1])] || DEFAULT_BONE_COLOR"
      class="bone-line"
      :class="{ selected: !!selected[boneKey(bone[0], bone[1])] }"
      @click="emit('toggle', bone[0], bone[1])"
    />

    <circle
      v-for="(point, i) in JOINT_LAYOUT"
      :key="`joint-${i}`"
      :cx="point.x"
      :cy="point.y"
      :r="isJointSelected(i) ? 6 : 5"
      :fill="jointColor(i)"
      class="joint-circle"
    >
      <title>{{ JOINT_NAMES[i] }}</title>
    </circle>
  </svg>
</template>
