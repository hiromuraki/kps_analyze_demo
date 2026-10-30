<script setup lang="ts">
import { KEYPOINT_MODES, PHASES } from '../constants'
import type { RuleEntry } from '../types'

defineProps<{
  rules: RuleEntry[]
  activeRuleId: number | null
  canCapture: boolean
}>()

const emit = defineEmits<{
  (e: 'capture'): void
  (e: 'select', rule: RuleEntry): void
  (e: 'remove', id: number): void
  (e: 'update-phase', id: number, value: string): void
  (e: 'update-mode', id: number, value: string): void
  (e: 'update-part', id: number, index: number, field: 'min_value' | 'max_value', value: number): void
}>()

function onPhase(e: Event, rule: RuleEntry): void {
  emit('update-phase', rule.id, (e.target as HTMLSelectElement).value)
}

function onMode(e: Event, rule: RuleEntry): void {
  emit('update-mode', rule.id, (e.target as HTMLSelectElement).value)
}

function onPart(
  e: Event,
  rule: RuleEntry,
  index: number,
  field: 'min_value' | 'max_value',
): void {
  const value = Number((e.target as HTMLInputElement).value)
  if (!Number.isFinite(value)) return
  emit('update-part', rule.id, index, field, value)
}
</script>

<template>
  <div class="rule-list">
    <div v-if="rules.length === 0" class="empty-hint">
      选择骨骼 → 定位帧 → 点「保存当前规则」
    </div>

    <div
      v-for="rule in rules"
      :key="rule.id"
      class="rule-entry"
      :class="{ active: rule.id === activeRuleId }"
      @click="emit('select', rule)"
    >
      <div class="r-header">
        <span class="r-id">{{ rule.rule_no }}</span>
        <span class="r-frame" title="录制该规则时的帧">帧 {{ rule.frame }}</span>
        <button class="r-delete" title="删除规则" @click.stop="emit('remove', rule.id)">✕</button>
      </div>

      <div class="r-row" @click.stop>
        <label>Phase</label>
        <select :value="rule.phase" @change="onPhase($event, rule)">
          <option v-for="phase in PHASES" :key="phase" :value="phase">{{ phase }}</option>
        </select>
        <label>Mode</label>
        <select :value="rule.keypoints_mode" @change="onMode($event, rule)">
          <option v-for="mode in KEYPOINT_MODES" :key="mode" :value="mode">{{ mode }}</option>
        </select>
      </div>

      <div v-for="(part, i) in rule.parts" :key="i" class="r-row" @click.stop>
        <span class="part-label">∠{{ part.p1 }}-{{ part.p2 }}-{{ part.p4 }}</span>
        <input
          class="num"
          type="number"
          :value="part.min_value"
          @change="onPart($event, rule, i, 'min_value')"
        />
        <span class="tilde">~</span>
        <input
          class="num"
          type="number"
          :value="part.max_value"
          @change="onPart($event, rule, i, 'max_value')"
        />
        <span class="tilde">°</span>
      </div>

      <div class="r-foot">点击规则跳转到该帧并还原骨骼选择</div>
    </div>

    <div class="btn-row">
      <button class="btn-capture" :disabled="!canCapture" @click="emit('capture')">
        保存当前规则
      </button>
    </div>
  </div>
</template>
