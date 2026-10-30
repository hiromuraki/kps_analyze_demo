import { JOINT_NAMES } from '../constants'
import type { RuleEntry, RulePart, RuleSetJson } from '../types'
import { selectedAngles } from './geometry'

/** 根据当前帧与已选骨骼生成规则 parts（角度 ±5° 容差）。 */
export function buildRuleParts(frame: number[][], selected: Record<string, string>): RulePart[] {
  return selectedAngles(frame, selected).map(({ joint, a, b, angle }) => ({
    rule_type: '夹角',
    p1: JOINT_NAMES[a],
    p2: JOINT_NAMES[joint],
    p3: JOINT_NAMES[joint],
    p4: JOINT_NAMES[b],
    min_value: Math.max(0, Math.round(angle - 5)),
    max_value: Math.min(180, Math.round(angle + 5)),
  }))
}

/**
 * 生成导出用 JSON。内部字段（frame / bones）不会写入规则文件。
 * 预览与导出共用此函数，保证看到的内容与实际导出一致。
 */
export function toRuleSetJson(
  actionNo: string,
  actionName: string,
  calories: number,
  rules: RuleEntry[],
): RuleSetJson {
  return {
    action_no: actionNo,
    action_name: actionName.trim() || 'untitled',
    rule_set: rules.map((rule) => ({
      rule_no: rule.rule_no,
      phase: rule.phase === 'all' ? ['all'] : [rule.phase],
      keypoints_mode: rule.keypoints_mode,
      parts: rule.parts.map((part) => ({
        rule_type: part.rule_type,
        p1: part.p1,
        p2: part.p2,
        p3: part.p3,
        p4: part.p4,
        min_value: part.min_value,
        max_value: part.max_value,
      })),
    })),
    calories_per_rep: calories,
    balance_pairs: [],
  }
}
