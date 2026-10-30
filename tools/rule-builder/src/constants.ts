import type { KeypointMode, Phase } from './types'

/** H36M 骨链拓扑：每项为一条骨骼的两个端点索引。 */
export const BONES: ReadonlyArray<readonly [number, number]> = [
  [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
  [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
  [12, 13], [8, 14], [14, 15], [15, 16],
]

/** H36M 17 关节名称，索引与 BONES 一致。 */
export const JOINT_NAMES: readonly string[] = [
  '骨盆', '右髋', '右膝', '右脚踝',
  '左髋', '左膝', '左脚踝',
  '脊柱', '胸腔', '鼻子', '头顶',
  '左肩', '左肘', '左手腕',
  '右肩', '右肘', '右手腕',
]

/** SVG 骨骼示意图中的关节坐标。 */
export const JOINT_LAYOUT: ReadonlyArray<{ x: number; y: number }> = [
  { x: 130, y: 210 }, { x: 148, y: 235 }, { x: 158, y: 285 }, { x: 163, y: 340 },
  { x: 112, y: 235 }, { x: 102, y: 285 }, { x: 97, y: 340 },
  { x: 130, y: 178 }, { x: 130, y: 145 }, { x: 130, y: 112 }, { x: 130, y: 85 },
  { x: 100, y: 158 }, { x: 72, y: 180 }, { x: 48, y: 200 },
  { x: 160, y: 158 }, { x: 188, y: 180 }, { x: 212, y: 200 },
]

export const DEFAULT_BONE_COLOR = '#3b82f6'

export const PALETTE: readonly string[] = [
  '#f87171', '#fb923c', '#fbbf24', '#a3e635', '#34d399',
  '#22d3ee', '#8b5cf6', '#f472b6', '#e879f9', '#f59e0b',
  '#10b981', '#ef4444', '#6366f1', '#ec4899', '#14b8a6',
]

export const PHASES: readonly Phase[] = ['all', 'static_up', 'static_down', 'descent', 'ascent']

export const KEYPOINT_MODES: readonly KeypointMode[] = ['2d', '3d']
