import { BONES, DEFAULT_BONE_COLOR } from '../constants'
import { boneKey } from './geometry'

/** 在视频上叠加 2D 骨骼；选中的骨骼加粗并着色。 */
export function drawOverlay(
  canvas: HTMLCanvasElement,
  frame: number[][] | undefined,
  selected: Record<string, string>,
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)
  if (!frame) return

  // 选中骨骼的端点也着色，让“选中的关键点”更直观。
  const jointColors = new Map<number, string>()
  for (const [key, color] of Object.entries(selected)) {
    const [a, b] = key.split('-').map(Number)
    if (!jointColors.has(a)) jointColors.set(a, color)
    if (!jointColors.has(b)) jointColors.set(b, color)
  }

  ctx.lineCap = 'round'
  for (const [a, b] of BONES) {
    const p = frame[a]
    const q = frame[b]
    if (!p || !q) continue

    const color = selected[boneKey(a, b)]
    ctx.strokeStyle = color ?? DEFAULT_BONE_COLOR
    ctx.lineWidth = color ? 3 : 2
    ctx.globalAlpha = color ? 1 : 0.5
    ctx.beginPath()
    ctx.moveTo(p[0], p[1])
    ctx.lineTo(q[0], q[1])
    ctx.stroke()
  }

  for (let i = 0; i < frame.length; i++) {
    const point = frame[i]
    const color = jointColors.get(i)
    ctx.fillStyle = color ?? DEFAULT_BONE_COLOR
    ctx.globalAlpha = color ? 1 : 0.8
    ctx.beginPath()
    ctx.arc(point[0], point[1], color ? 4 : 3, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.globalAlpha = 1
}
