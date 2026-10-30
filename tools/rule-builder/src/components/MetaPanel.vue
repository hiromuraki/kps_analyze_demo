<script setup lang="ts">
defineProps<{
  actionName: string
  calories: number
  ruleCount: number
}>()

const emit = defineEmits<{
  (e: 'update:name', value: string): void
  (e: 'update:calories', value: number): void
  (e: 'preview'): void
  (e: 'export'): void
}>()

function onName(e: Event): void {
  emit('update:name', (e.target as HTMLInputElement).value)
}

function onCalories(e: Event): void {
  const value = Number((e.target as HTMLInputElement).value)
  emit('update:calories', Number.isFinite(value) && value >= 0 ? value : 0)
}
</script>

<template>
  <div class="meta-panel">
    <div class="meta-row">
      <label>动作名称</label>
      <input type="text" :value="actionName" placeholder="如：哈克深蹲" @input="onName" />
    </div>
    <div class="meta-row">
      <label>卡路里/次</label>
      <input
        type="number"
        :value="calories"
        step="0.01"
        min="0"
        placeholder="0.5"
        @input="onCalories"
      />
    </div>

    <div class="meta-hint">已录入 {{ ruleCount }} 条规则</div>

    <button class="btn-preview" @click="emit('preview')">预览 JSON</button>
    <button class="btn-export" :disabled="ruleCount === 0" @click="emit('export')">
      导出规则 JSON
    </button>
  </div>
</template>
