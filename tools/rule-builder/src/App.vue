<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import BoneSelector from './components/BoneSelector.vue'
import JsonPreviewModal from './components/JsonPreviewModal.vue'
import MetaPanel from './components/MetaPanel.vue'
import RuleList from './components/RuleList.vue'
import UploadBar from './components/UploadBar.vue'
import { useThreeView } from './composables/useThreeView'
import { JOINT_NAMES, PALETTE } from './constants'
import { boneKey, selectedAngles } from './lib/geometry'
import { drawOverlay } from './lib/overlay'
import { buildRuleParts, toRuleSetJson } from './lib/rules'
import type {
  KeypointMode,
  NpzData,
  Phase,
  RuleEntry,
  RuleSetJson,
  UploadNpzResponse,
  UploadStatus,
} from './types'

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const videoEl = ref<HTMLVideoElement | null>(null)
const overlayEl = ref<HTMLCanvasElement | null>(null)
const viewAreaEl = ref<HTMLElement | null>(null)
const mainEl = ref<HTMLElement | null>(null)
const threeContainerEl = ref<HTMLElement | null>(null)

const three = useThreeView(threeContainerEl)

// ---------------------------------------------------------------------------
// Data / UI state
// ---------------------------------------------------------------------------
const videoUrl = ref('')
const videoName = ref('')
const status2d = ref<UploadStatus>({ state: 'idle', label: '选择 2D 骨骼' })
const status3d = ref<UploadStatus>({ state: 'idle', label: '选择 3D 骨骼' })
const d2 = ref<NpzData | null>(null)
const d3 = ref<NpzData | null>(null)

const has2d = computed(() => d2.value !== null)
const has3d = computed(() => d3.value !== null)
const hasData = computed(() => has2d.value || has3d.value)
const totalFrames = computed(() => d2.value?.num_frames ?? d3.value?.num_frames ?? 0)

const currentFrame = ref(0)
const isPlaying = ref(false)
const selectedBones = ref<Record<string, string>>({})

const rules = ref<RuleEntry[]>([])
const activeRuleId = ref<number | null>(null)
let nextRuleId = 1
let paletteIndex = 0

const tab = ref<'rules' | 'meta'>('rules')
const metaName = ref('')
const calories = ref(0)
const previewOpen = ref(false)

const orientation = ref<'vertical' | 'horizontal'>('vertical')
const videoFlex = ref(1)
const threeFlex = ref(1)

// 右侧骨骼/规则面板宽度（占主区域宽度的比例），可拖拽分隔条调节。
const MIN_RIGHT_WIDTH = 220
const MAX_RIGHT_WIDTH = 560
const DEFAULT_RIGHT_RATIO = 0.22
const rightRatio = ref(DEFAULT_RIGHT_RATIO)
const rightPanelStyle = computed(() => ({ width: `${rightRatio.value * 100}%` }))

function generateActionNo(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID().slice(0, 8)
  }
  return Math.random().toString(16).slice(2, 10)
}
const actionNo = ref(generateActionNo())

// ---------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------
function onVideoFile(file: File): void {
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
  stopFallback()
  isPlaying.value = false
  videoName.value = file.name
  videoUrl.value = URL.createObjectURL(file)
}

async function onNpzFile(file: File, mode: KeypointMode): Promise<void> {
  const status = mode === '2d' ? status2d : status3d
  status.value = { state: 'loading', label: '解析中...' }

  const form = new FormData()
  form.append('file', file)

  try {
    const res = await fetch('/api/upload_npz', { method: 'POST', body: form }).then(
      (r) => r.json() as Promise<UploadNpzResponse>,
    )
    if (res.status !== 'success' || !res.frames || !res.num_frames) {
      status.value = { state: 'error', label: res.message ?? '解析失败' }
      return
    }
    const data: NpzData = { frames: res.frames, num_frames: res.num_frames }
    if (mode === '2d') d2.value = data
    else d3.value = data
    status.value = { state: 'done', label: file.name }
    onDataReady()
  } catch {
    status.value = { state: 'error', label: '上传失败' }
  }
}

function on2dFile(file: File): void {
  void onNpzFile(file, '2d')
}

function on3dFile(file: File): void {
  void onNpzFile(file, '3d')
}

function onDataReady(): void {
  const maxFrame = Math.max(0, totalFrames.value - 1)
  if (currentFrame.value > maxFrame) currentFrame.value = maxFrame
  nextTick(() => {
    seekVideoToFrame(currentFrame.value)
    redrawOverlay()
    updateThree()
    three.resize()
  })
}

// ---------------------------------------------------------------------------
// Video / frame synchronisation
// ---------------------------------------------------------------------------
function onVideoMetadata(): void {
  const video = videoEl.value
  if (!video) return

  const canvas = overlayEl.value
  if (canvas) {
    canvas.width = video.videoWidth || 640
    canvas.height = video.videoHeight || 480
  }

  const nextOrientation = (video.videoWidth || 0) > (video.videoHeight || 0) ? 'vertical' : 'horizontal'
  if (nextOrientation !== orientation.value) {
    orientation.value = nextOrientation
    videoFlex.value = 1
    threeFlex.value = 1
  }

  seekVideoToFrame(currentFrame.value)
  nextTick(() => {
    redrawOverlay()
    updateThree()
    three.resize()
  })
}

function seekVideoToFrame(frame: number): void {
  const video = videoEl.value
  if (!video || !video.src || !Number.isFinite(video.duration) || video.duration <= 0) return
  const last = totalFrames.value - 1
  const time = last > 0 ? (frame / last) * video.duration : 0
  if (Math.abs(video.currentTime - time) > 0.001) video.currentTime = time
}

function setFrame(frame: number, seek = true): void {
  if (!hasData.value) return
  const maxFrame = Math.max(0, totalFrames.value - 1)
  currentFrame.value = Math.max(0, Math.min(frame, maxFrame))
  lastSyncedFrame = currentFrame.value
  if (seek) seekVideoToFrame(currentFrame.value)
}

function step(delta: number): void {
  setFrame(currentFrame.value + delta)
}

// rAF: video time -> current frame
let rafId = 0
let lastSyncedFrame = -1
function syncLoop(): void {
  rafId = requestAnimationFrame(syncLoop)
  const video = videoEl.value
  if (!video || !hasData.value || !video.src || video.seeking) return

  const last = totalFrames.value - 1
  if (last <= 0) return
  if (!Number.isFinite(video.duration) || video.duration <= 0 || video.paused) return

  const frame = Math.max(0, Math.min(Math.round((video.currentTime / video.duration) * last), last))
  if (frame !== lastSyncedFrame) {
    lastSyncedFrame = frame
    if (frame !== currentFrame.value) currentFrame.value = frame
  }
}

// ---------------------------------------------------------------------------
// Playback (video-backed, or timer fallback when only .npz is loaded)
// ---------------------------------------------------------------------------
let fallbackTimer: number | null = null

function stopFallback(): void {
  if (fallbackTimer !== null) {
    window.clearInterval(fallbackTimer)
    fallbackTimer = null
  }
  isPlaying.value = false
}

function togglePlay(): void {
  if (!hasData.value) return

  const video = videoEl.value
  if (video && video.src && Number.isFinite(video.duration) && video.readyState >= 2) {
    if (video.paused) void video.play().catch(() => undefined)
    else video.pause()
    return
  }

  if (fallbackTimer !== null) {
    stopFallback()
    return
  }

  if (currentFrame.value >= totalFrames.value - 1) setFrame(0, false)
  isPlaying.value = true
  fallbackTimer = window.setInterval(() => {
    if (!hasData.value || currentFrame.value >= totalFrames.value - 1) {
      stopFallback()
      return
    }
    setFrame(currentFrame.value + 1, false)
  }, 1000 / 30)
}

// ---------------------------------------------------------------------------
// Frame controls
// ---------------------------------------------------------------------------
function onFrameInput(e: Event): void {
  const input = e.target as HTMLInputElement
  const value = Number.parseInt(input.value, 10)
  if (!Number.isFinite(value)) {
    input.value = String(currentFrame.value)
    return
  }
  setFrame(value)
}

function onProgressInput(e: Event): void {
  setFrame(Number((e.target as HTMLInputElement).value))
}

let holdTimer: number | null = null

function stopHold(): void {
  if (holdTimer !== null) {
    window.clearInterval(holdTimer)
    holdTimer = null
  }
}

function startStep(delta: number): void {
  step(delta)
  stopHold()
  holdTimer = window.setInterval(() => step(delta), 80)
}

// ---------------------------------------------------------------------------
// Bone selection & angles
// ---------------------------------------------------------------------------
function toggleBone(a: number, b: number): void {
  const key = boneKey(a, b)
  const next = { ...selectedBones.value }
  if (key in next) {
    delete next[key]
  } else {
    next[key] = PALETTE[paletteIndex % PALETTE.length]
    paletteIndex += 1
  }
  selectedBones.value = next
}

/** Angle / capture always use 2D when available, otherwise 3D. */
const dataSource = computed(() => d2.value ?? d3.value)
const currentFrameData = computed(() => dataSource.value?.frames[currentFrame.value])
const angleEntries = computed(() => {
  const frame = currentFrameData.value
  return frame ? selectedAngles(frame, selectedBones.value) : []
})
const captureParts = computed(() => {
  const frame = currentFrameData.value
  return frame ? buildRuleParts(frame, selectedBones.value) : []
})
const canCapture = computed(() => captureParts.value.length > 0)

function redrawOverlay(): void {
  const canvas = overlayEl.value
  if (!canvas) return
  drawOverlay(canvas, d2.value?.frames[currentFrame.value], selectedBones.value)
}

function updateThree(): void {
  if (!has3d.value) return
  three.update(d3.value?.frames[currentFrame.value], selectedBones.value)
}

// ---------------------------------------------------------------------------
// Rules
// ---------------------------------------------------------------------------
function nextRuleNo(): string {
  let max = 0
  for (const rule of rules.value) {
    const value = Number.parseInt(rule.rule_no.replace(/^R/, ''), 10)
    if (Number.isFinite(value)) max = Math.max(max, value)
  }
  return `R${max + 1}`
}

function captureRule(): void {
  const parts = captureParts.value
  if (!parts.length) return

  const bones = Object.entries(selectedBones.value).map(([key, color]) => {
    const [a, b] = key.split('-').map(Number)
    return { a, b, color }
  })

  const rule: RuleEntry = {
    id: nextRuleId++,
    rule_no: nextRuleNo(),
    phase: 'all',
    keypoints_mode: has2d.value ? '2d' : '3d',
    parts,
    frame: currentFrame.value,
    bones,
  }
  rules.value = [...rules.value, rule]
  activeRuleId.value = rule.id
}

/** 点击规则：跳到录制该规则时的帧，并还原当时的骨骼选择。 */
function selectRule(rule: RuleEntry): void {
  activeRuleId.value = rule.id

  const selection: Record<string, string> = {}
  for (const bone of rule.bones) {
    selection[boneKey(bone.a, bone.b)] = bone.color
  }
  selectedBones.value = selection

  if (videoEl.value) videoEl.value.pause()
  stopFallback()
  setFrame(rule.frame)
}

function removeRule(id: number): void {
  rules.value = rules.value.filter((rule) => rule.id !== id)
  if (activeRuleId.value === id) activeRuleId.value = null
}

function updateRule(id: number, field: 'phase' | 'keypoints_mode', value: string): void {
  const rule = rules.value.find((item) => item.id === id)
  if (!rule) return
  if (field === 'phase') rule.phase = value as Phase
  else rule.keypoints_mode = value as KeypointMode
}

function updatePart(
  id: number,
  index: number,
  field: 'min_value' | 'max_value',
  value: number,
): void {
  const rule = rules.value.find((item) => item.id === id)
  const part = rule?.parts[index]
  if (!part || !Number.isFinite(value)) return
  part[field] = value
}

// ---------------------------------------------------------------------------
// Meta / JSON export
// ---------------------------------------------------------------------------
function setMetaName(value: string): void {
  metaName.value = value
}

function setCalories(value: number): void {
  calories.value = value
}

const ruleSetJson = computed<RuleSetJson>(() =>
  toRuleSetJson(actionNo.value, metaName.value, calories.value, rules.value),
)
const jsonPreview = computed(() => JSON.stringify(ruleSetJson.value, null, 2))

function exportJson(): void {
  if (!rules.value.length) {
    window.alert('没有可导出的规则')
    return
  }
  const name = metaName.value.trim() || 'untitled'
  const blob = new Blob([jsonPreview.value], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${name}.json`
  anchor.click()
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// Split-pane drag (video/3D) & right panel width drag
// ---------------------------------------------------------------------------
let dragging = false
let resizingRight = false

function startDrag(e: MouseEvent): void {
  dragging = true
  e.preventDefault()
}

function startRightDrag(e: MouseEvent): void {
  resizingRight = true
  document.body.style.cursor = 'col-resize'
  document.body.style.userSelect = 'none'
  e.preventDefault()
}

function resetRightWidth(): void {
  rightRatio.value = DEFAULT_RIGHT_RATIO
}

function onMouseMove(e: MouseEvent): void {
  if (resizingRight) {
    const main = mainEl.value
    if (main) {
      const rect = main.getBoundingClientRect()
      if (rect.width > 0) {
        const maxAllowed = Math.min(MAX_RIGHT_WIDTH, rect.width * 0.6)
        const width = Math.max(MIN_RIGHT_WIDTH, Math.min(maxAllowed, rect.right - e.clientX))
        rightRatio.value = width / rect.width
      }
    }
    return
  }

  const area = viewAreaEl.value
  if (!dragging || !area) return

  const rect = area.getBoundingClientRect()
  const gripSize = 4

  if (orientation.value === 'horizontal') {
    const total = rect.width - gripSize
    if (total <= 120) return
    const left = Math.max(60, Math.min(total - 60, e.clientX - rect.left - gripSize / 2))
    videoFlex.value = left
    threeFlex.value = total - left
  } else {
    const total = rect.height - gripSize
    if (total <= 120) return
    const top = Math.max(60, Math.min(total - 60, e.clientY - rect.top - gripSize / 2))
    videoFlex.value = top
    threeFlex.value = total - top
  }
}

function endDrag(): void {
  if (resizingRight) {
    resizingRight = false
    document.body.style.cursor = ''
    document.body.style.userSelect = ''
    return
  }
  if (!dragging) return
  dragging = false
  window.setTimeout(() => three.resize(), 30)
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------
function onKeydown(e: KeyboardEvent): void {
  const tag = (e.target as HTMLElement | null)?.tagName
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return
  if (previewOpen.value) return

  if (e.key === 'ArrowLeft') {
    e.preventDefault()
    step(-1)
  } else if (e.key === 'ArrowRight') {
    e.preventDefault()
    step(1)
  } else if (e.key === ' ') {
    e.preventDefault()
    togglePlay()
  }
}

// ---------------------------------------------------------------------------
// Watchers / lifecycle
// ---------------------------------------------------------------------------
watch([currentFrame, selectedBones, d2, d3], () => {
  redrawOverlay()
  updateThree()
})

watch(has3d, (value) => {
  if (!value) return
  nextTick(() => {
    three.resize()
    updateThree()
  })
})

watch(videoUrl, async () => {
  await nextTick()
  videoEl.value?.load()
})

onMounted(() => {
  const canvas = overlayEl.value
  if (canvas) {
    canvas.width = 640
    canvas.height = 480
  }
  window.addEventListener('mousemove', onMouseMove)
  window.addEventListener('mouseup', endDrag)
  window.addEventListener('keydown', onKeydown)
  rafId = requestAnimationFrame(syncLoop)
})

onBeforeUnmount(() => {
  cancelAnimationFrame(rafId)
  stopHold()
  stopFallback()
  window.removeEventListener('mousemove', onMouseMove)
  window.removeEventListener('mouseup', endDrag)
  window.removeEventListener('keydown', onKeydown)
  if (videoUrl.value) URL.revokeObjectURL(videoUrl.value)
})
</script>

<template>
  <div class="app">
    <header>
      <h1>Rule Builder</h1>
      <div class="header-info">
        <span v-if="videoName" class="chip">🎬 {{ videoName }}</span>
        <span v-if="hasData" class="chip">帧 {{ currentFrame }} / {{ Math.max(0, totalFrames - 1) }}</span>
      </div>
    </header>

    <div ref="mainEl" class="main">
      <div class="left">
        <UploadBar
          :video-name="videoName"
          :status2d="status2d"
          :status3d="status3d"
          @video="onVideoFile"
          @kp2d="on2dFile"
          @kp3d="on3dFile"
        />

        <div ref="viewAreaEl" class="view-area" :class="orientation">
          <div class="video-box" :style="{ flexGrow: videoFlex }">
            <video
              ref="videoEl"
              :src="videoUrl"
              preload="auto"
              playsinline
              muted
              @loadedmetadata="onVideoMetadata"
              @play="isPlaying = true"
              @pause="isPlaying = false"
              @ended="isPlaying = false"
            ></video>
            <canvas ref="overlayEl"></canvas>
            <div v-if="!hasData" class="drop-hint">
              <span class="icon">🎬</span>
              上传视频 + 骨骼数据以开始
            </div>
            <div v-else-if="!has2d && has3d" class="no2d-hint">仅 3D 数据：骨骼预览在右侧</div>

            <div v-if="angleEntries.length" class="angle-float">
              <h4>📐 相邻夹角</h4>
              <div v-for="(entry, i) in angleEntries" :key="i" class="angle-entry">
                <span class="angle-dot" :style="{ background: entry.colorA }"></span>
                <span class="angle-dot" :style="{ background: entry.colorB }"></span>
                <span>∠{{ JOINT_NAMES[entry.joint] }}: {{ entry.angle.toFixed(1) }}°</span>
              </div>
            </div>
          </div>

          <div class="grip" :class="orientation" @mousedown.prevent="startDrag"></div>

          <div v-show="has3d" class="three-box" :style="{ flexGrow: threeFlex }">
            <div ref="threeContainerEl" class="three-container"></div>
          </div>
        </div>

        <div class="ctrl-bar">
          <button
            :disabled="!hasData"
            @mousedown.prevent="startStep(-1)"
            @mouseup="stopHold"
            @mouseleave="stopHold"
            @touchstart.prevent="startStep(-1)"
            @touchend="stopHold"
            @touchcancel="stopHold"
          >
            ◀◀
          </button>
          <button :disabled="!hasData" @click="togglePlay">{{ isPlaying ? '⏸' : '▶' }}</button>
          <button
            :disabled="!hasData"
            @mousedown.prevent="startStep(1)"
            @mouseup="stopHold"
            @mouseleave="stopHold"
            @touchstart.prevent="startStep(1)"
            @touchend="stopHold"
            @touchcancel="stopHold"
          >
            ▶▶
          </button>
          <input
            class="frame-input"
            type="number"
            :min="0"
            :max="Math.max(0, totalFrames - 1)"
            :value="currentFrame"
            :disabled="!hasData"
            @change="onFrameInput"
          />
          <span class="frame-total">/ <b>{{ Math.max(0, totalFrames - 1) }}</b></span>
          <input
            class="progress"
            type="range"
            :min="0"
            :max="Math.max(0, totalFrames - 1)"
            :value="currentFrame"
            :disabled="!hasData"
            @input="onProgressInput"
          />
        </div>
      </div>

      <div
        class="right-grip"
        title="拖拽调整面板宽度，双击恢复默认"
        @mousedown.prevent="startRightDrag"
        @dblclick="resetRightWidth"
      ></div>

      <aside class="right" :style="rightPanelStyle">
        <div class="section">
          <h3>骨骼选择</h3>
          <BoneSelector :selected="selectedBones" @toggle="toggleBone" />
        </div>

        <div class="section">
          <div class="tabs">
            <button :class="{ active: tab === 'rules' }" @click="tab = 'rules'">规则集</button>
            <button :class="{ active: tab === 'meta' }" @click="tab = 'meta'">元数据</button>
          </div>

          <RuleList
            v-if="tab === 'rules'"
            :rules="rules"
            :active-rule-id="activeRuleId"
            :can-capture="canCapture"
            @capture="captureRule"
            @select="selectRule"
            @remove="removeRule"
            @update-phase="(id, value) => updateRule(id, 'phase', value)"
            @update-mode="(id, value) => updateRule(id, 'keypoints_mode', value)"
            @update-part="updatePart"
          />

          <MetaPanel
            v-else
            :action-name="metaName"
            :calories="calories"
            :rule-count="rules.length"
            @update:name="setMetaName"
            @update:calories="setCalories"
            @preview="previewOpen = true"
            @export="exportJson"
          />
        </div>
      </aside>
    </div>

    <JsonPreviewModal :open="previewOpen" :json="jsonPreview" @close="previewOpen = false" />
  </div>
</template>
