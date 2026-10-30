<script setup lang="ts">
import { onBeforeUnmount, onMounted, ref } from 'vue'

const props = defineProps<{
  open: boolean
  json: string
}>()

const emit = defineEmits<{
  (e: 'close'): void
}>()

const copied = ref(false)
let copyTimer: number | undefined

async function copyJson(): Promise<void> {
  try {
    await navigator.clipboard.writeText(props.json)
    copied.value = true
    window.clearTimeout(copyTimer)
    copyTimer = window.setTimeout(() => {
      copied.value = false
    }, 1500)
  } catch {
    copied.value = false
  }
}

function onKeydown(e: KeyboardEvent): void {
  if (e.key === 'Escape' && props.open) emit('close')
}

onMounted(() => window.addEventListener('keydown', onKeydown))
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  window.clearTimeout(copyTimer)
})
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="modal-mask" @click.self="emit('close')">
      <div class="modal">
        <div class="modal-header">
          <h3>JSON 预览</h3>
          <div class="modal-actions">
            <button class="btn-copy" @click="copyJson">{{ copied ? '已复制 ✓' : '复制' }}</button>
            <button class="btn-close" title="关闭" @click="emit('close')">✕</button>
          </div>
        </div>
        <div class="modal-body">
          <pre>{{ json }}</pre>
        </div>
      </div>
    </div>
  </Teleport>
</template>
