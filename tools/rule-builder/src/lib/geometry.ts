import { BONES } from '../constants'

export interface AngleEntry {
  joint: number
  a: number
  b: number
  angle: number
  colorA: string
  colorB: string
}

/** 生成与顺序无关的骨骼键。 */
export function boneKey(a: number, b: number): string {
  return a < b ? `${a}-${b}` : `${b}-${a}`
}

/** 三维向量夹角（度），第三维缺省为 0。 */
export function vectorAngle(v1: number[], v2: number[]): number {
  const z1 = v1[2] || 0
  const z2 = v2[2] || 0
  const l1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2 + z1 ** 2)
  const l2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2 + z2 ** 2)
  if (l1 < 1e-9 || l2 < 1e-9) return 0
  const cos = (v1[0] * v2[0] + v1[1] * v2[1] + z1 * z2) / (l1 * l2)
  return (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI
}

/** 关节 -> 相邻关节列表，顺序与原始骨骼定义一致。 */
export const ADJACENT_JOINTS: ReadonlyMap<number, number[]> = (() => {
  const map = new Map<number, number[]>()
  const add = (from: number, to: number) => {
    const list = map.get(from)
    if (list) list.push(to)
    else map.set(from, [to])
  }
  for (const [a, b] of BONES) {
    add(a, b)
    add(b, a)
  }
  return map
})()

/**
 * 找出所有「同一个关节上选了两条骨骼」形成的夹角。
 * 规则录入与实时角度面板共用这段逻辑，保证两者完全一致。
 */
export function selectedAngles(frame: number[][], selected: Record<string, string>): AngleEntry[] {
  const entries: AngleEntry[] = []
  for (let joint = 0; joint < frame.length; joint++) {
    const neighbors = (ADJACENT_JOINTS.get(joint) ?? []).filter((n) => boneKey(joint, n) in selected)
    if (neighbors.length < 2) continue

    for (let i = 0; i < neighbors.length; i++) {
      for (let k = i + 1; k < neighbors.length; k++) {
        const a = neighbors[i]
        const b = neighbors[k]
        const pJoint = frame[joint]
        const pA = frame[a]
        const pB = frame[b]
        if (!pJoint || !pA || !pB) continue

        entries.push({
          joint,
          a,
          b,
          angle: vectorAngle(
            [pA[0] - pJoint[0], pA[1] - pJoint[1], (pA[2] || 0) - (pJoint[2] || 0)],
            [pB[0] - pJoint[0], pB[1] - pJoint[1], (pB[2] || 0) - (pJoint[2] || 0)],
          ),
          colorA: selected[boneKey(joint, a)],
          colorB: selected[boneKey(joint, b)],
        })
      }
    }
  }
  return entries
}
