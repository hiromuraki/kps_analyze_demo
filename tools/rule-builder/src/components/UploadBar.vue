<script setup lang="ts">
import type { UploadStatus } from '../types'

const props = defineProps<{
  videoName: string
  status2d: UploadStatus
  status3d: UploadStatus
}>()

const emit = defineEmits<{
  (e: 'video', file: File): void
  (e: 'kp2d', file: File): void
  (e: 'kp3d', file: File): void
}>()

function truncate(name: string): string {
  return name.length > 20 ? `${name.slice(0, 18)}…` : name
}

function label(status: UploadStatus, idle: string): string {
  if (status.state === 'loading') return `⏳ ${status.label}`
  if (status.state === 'done') return `✓ ${truncate(status.label)}`
  if (status.state === 'error') return `✕ ${status.label}`
  return idle
}

function pickVideo(e: Event): void {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('video', file)
  input.value = ''
}

function pick2d(e: Event): void {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('kp2d', file)
  input.value = ''
}

function pick3d(e: Event): void {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('kp3d', file)
  input.value = ''
}
</script>

<template>
  <div class="upload-bar">
    <span class="label">数据来源</span>
    <label class="up-btn" :class="{ done: !!props.videoName }">
      <span>{{ props.videoName ? `✓ ${truncate(props.videoName)}` : '选择视频文件' }}</span>
      <input type="file" accept="video/*" @change="pickVideo" />
    </label>
    <label
      class="up-btn"
      :class="{ done: props.status2d.state === 'done', error: props.status2d.state === 'error' }"
    >
      <span>{{ label(props.status2d, '选择 2D 骨骼') }}</span>
      <input type="file" accept=".npz" @change="pick2d" />
    </label>
    <label
      class="up-btn"
      :class="{ done: props.status3d.state === 'done', error: props.status3d.state === 'error' }"
    >
      <span>{{ label(props.status3d, '选择 3D 骨骼') }}</span>
      <input type="file" accept=".npz" @change="pick3d" />
    </label>
  </div>
</template>
