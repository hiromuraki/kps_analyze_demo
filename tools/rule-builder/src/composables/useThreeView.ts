import { onBeforeUnmount, onMounted, type Ref } from 'vue'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { BONES, DEFAULT_BONE_COLOR, JOINT_NAMES } from '../constants'
import { boneKey } from '../lib/geometry'

interface BoneLine {
  line: THREE.Line
  material: THREE.LineBasicMaterial
}

/**
 * Three.js 3D 骨架预览。场景只构建一次，逐帧更新关节与骨骼线段。
 */
export function useThreeView(container: Ref<HTMLElement | null>) {
  let renderer: THREE.WebGLRenderer | null = null
  let scene: THREE.Scene | null = null
  let camera: THREE.PerspectiveCamera | null = null
  let controls: OrbitControls | null = null
  let resizeObserver: ResizeObserver | null = null
  let animationId = 0
  const jointMeshes: THREE.Mesh[] = []
  const boneLines = new Map<string, BoneLine>()

  function buildScene(el: HTMLElement): boolean {
    try {
      const s = new THREE.Scene()
      s.fog = new THREE.Fog(0x0a0a14, 3, 10)

      const cam = new THREE.PerspectiveCamera(45, (el.clientWidth || 1) / (el.clientHeight || 1), 0.1, 20)
      cam.position.set(1.5, 1.2, 2.5)
      cam.lookAt(0, 0.6, 0)

      const rnd = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      rnd.setPixelRatio(window.devicePixelRatio)
      rnd.setSize(el.clientWidth || 1, el.clientHeight || 1)
      el.appendChild(rnd.domElement)

      const ctl = new OrbitControls(cam, rnd.domElement)
      ctl.target.set(0, 0.6, 0)
      ctl.enableDamping = true
      ctl.dampingFactor = 0.1
      ctl.update()

      s.add(new THREE.AmbientLight(0xffffff, 0.8))
      s.add(new THREE.DirectionalLight(0xffffff, 0.4))
      s.add(new THREE.GridHelper(3, 20, 0x333355, 0x1a1a2e))

      const jointGroup = new THREE.Group()
      s.add(jointGroup)
      const jointGeometry = new THREE.SphereGeometry(0.025, 12, 12)
      for (let i = 0; i < JOINT_NAMES.length; i++) {
        // 每个关节独立材质，便于随骨骼选择着色。
        const material = new THREE.MeshBasicMaterial({ color: 0x60a5fa })
        const mesh = new THREE.Mesh(jointGeometry, material)
        jointGroup.add(mesh)
        jointMeshes.push(mesh)
      }

      const lineGroup = new THREE.Group()
      s.add(lineGroup)
      for (const [a, b] of BONES) {
        const material = new THREE.LineBasicMaterial({
          color: DEFAULT_BONE_COLOR,
          transparent: true,
          opacity: 0.45,
        })
        const geometry = new THREE.BufferGeometry()
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3))
        const line = new THREE.Line(geometry, material)
        lineGroup.add(line)
        boneLines.set(boneKey(a, b), { line, material })
      }

      scene = s
      camera = cam
      renderer = rnd
      controls = ctl

      const animate = () => {
        animationId = requestAnimationFrame(animate)
        ctl.update()
        rnd.render(s, cam)
      }
      animate()
      return true
    } catch (err) {
      console.warn('3D 视图初始化失败', err)
      return false
    }
  }

  function update(frame: number[][] | undefined, selected: Record<string, string>): void {
    if (!scene || !frame) return

    const jointColors = new Map<number, string>()
    for (const [key, color] of Object.entries(selected)) {
      const [a, b] = key.split('-').map(Number)
      if (!jointColors.has(a)) jointColors.set(a, color)
      if (!jointColors.has(b)) jointColors.set(b, color)
    }

    for (let i = 0; i < jointMeshes.length; i++) {
      const point = frame[i]
      const mesh = jointMeshes[i]
      if (point) mesh.position.set(point[0], -point[1], point[2] || 0)
      const material = mesh.material as THREE.MeshBasicMaterial
      material.color.set(jointColors.get(i) ?? 0x60a5fa)
    }

    for (const [a, b] of BONES) {
      const entry = boneLines.get(boneKey(a, b))
      const p = frame[a]
      const q = frame[b]
      if (!entry || !p || !q) continue

      const attr = entry.line.geometry.getAttribute('position') as THREE.BufferAttribute
      const arr = attr.array as Float32Array
      arr[0] = p[0]
      arr[1] = -p[1]
      arr[2] = p[2] || 0
      arr[3] = q[0]
      arr[4] = -q[1]
      arr[5] = q[2] || 0
      attr.needsUpdate = true

      const color = selected[boneKey(a, b)]
      entry.material.color.set(color ?? DEFAULT_BONE_COLOR)
      entry.material.opacity = color ? 1.0 : 0.45
    }
  }

  function resize(): void {
    const el = container.value
    if (!el || !renderer || !camera) return
    const width = el.clientWidth
    const height = el.clientHeight
    if (!width || !height) return
    camera.aspect = width / height
    camera.updateProjectionMatrix()
    renderer.setSize(width, height)
  }

  onMounted(() => {
    const el = container.value
    if (!el) return
    if (!buildScene(el)) return
    resizeObserver = new ResizeObserver(() => resize())
    resizeObserver.observe(el)
  })

  onBeforeUnmount(() => {
    cancelAnimationFrame(animationId)
    resizeObserver?.disconnect()
    controls?.dispose()
    for (const mesh of jointMeshes) {
      mesh.geometry.dispose()
      ;(mesh.material as THREE.Material).dispose()
    }
    for (const { line, material } of boneLines.values()) {
      line.geometry.dispose()
      material.dispose()
    }
    renderer?.dispose()
    renderer?.domElement.remove()
    scene = null
    camera = null
    renderer = null
    controls = null
  })

  return { update, resize }
}
