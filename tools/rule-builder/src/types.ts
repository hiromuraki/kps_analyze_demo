export type KeypointMode = '2d' | '3d'

export type Phase = 'all' | 'static_up' | 'static_down' | 'descent' | 'ascent'

export type UploadState = 'idle' | 'loading' | 'done' | 'error'

export interface UploadStatus {
  state: UploadState
  label: string
}

export interface NpzData {
  /** [frame][joint] -> [x, y, z] */
  frames: number[][][]
  num_frames: number
}

export interface Bone {
  a: number
  b: number
}

export interface RuleBone extends Bone {
  color: string
}

export interface RulePart {
  rule_type: '夹角'
  p1: string
  p2: string
  p3: string
  p4: string
  min_value: number
  max_value: number
}

export interface RuleEntry {
  id: number
  rule_no: string
  phase: Phase
  keypoints_mode: KeypointMode
  parts: RulePart[]
  /** Frame at which the rule was captured — used to jump back to it. */
  frame: number
  /** Bones that were selected when the rule was captured — restored on click. */
  bones: RuleBone[]
}

export interface ExportedRule {
  rule_no: string
  phase: Phase[]
  keypoints_mode: KeypointMode
  parts: RulePart[]
}

export interface RuleSetJson {
  action_no: string
  action_name: string
  rule_set: ExportedRule[]
  calories_per_rep: number
  balance_pairs: string[][]
}

export interface UploadNpzResponse {
  status: 'success' | 'error'
  frames?: number[][][]
  num_frames?: number
  message?: string
}
