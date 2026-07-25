'use client';

import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';

import { buildCortex, placeNode } from '@/lib/cortex';
import type { WorkspaceMap, WorkspaceNode } from '@/lib/types';

interface Props {
  map: WorkspaceMap;
  /** Full height for the dedicated page; a short band for the hero. */
  height?: number | string;
  /** Whether a click selects a neuron and shows what it is. */
  interactive?: boolean;
}

/** A bound on animated dots. Past this the map is busy rather than alive. */
const MAX_SPARKS = 120;

/** One colour per kind, matched to the canvas so the two screens agree. */
const COLOURS: Record<string, number> = {
  agent: 0x6f86e8,
  skill: 0x4fb286,
  workflow: 0xd9a441,
  server: 0xc0587e,
  provider: 0x8f7bd6,
};

/**
 * The workspace as a living map, inside a brain.
 *
 * The reason it is a brain rather than a graph on a grid: what this product
 * accumulates *is* a nervous system. Agents hold skills, skills carry lessons,
 * workflows fire the same paths repeatedly and those paths get better. A box
 * diagram says "here are your objects"; this says "here is the thing you have
 * been growing", which is the honest description of a workspace whose parts
 * improve every time they run.
 *
 * Three.js directly rather than a React renderer: the scene is built once from a
 * payload and then animates itself, so a reconciler between React and WebGL
 * would be a dependency doing nothing but adding a version to keep up with.
 */
export function Brain({ map, height = '100%', interactive = true }: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<WorkspaceNode | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const element = mount.current;
    if (!element) return undefined;

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      // A machine without WebGL should get the summary underneath, not a blank
      // rectangle and a console error.
      setFailed(true);
      return undefined;
    }

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(0, 3.5, 26);
    camera.lookAt(0, -0.5, 0);

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    element.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0x8899ff, 1.1));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(12, 16, 18);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x6f86e8, 0.8);
    rim.position.set(-14, -6, -12);
    scene.add(rim);

    const radius = 10;
    const cortex = buildCortex(radius);
    const brain = new THREE.Group();
    brain.add(cortex.group);
    // Turned to three-quarters rather than square-on. Head-on, the cerebellum
    // hides behind the cortex and reads as a disc floating in the middle; from
    // this angle it sits at the back of the underside, where the silhouette
    // becomes recognisably a brain.
    brain.rotation.y = -0.85;
    scene.add(brain);

    // ---- neurons ---------------------------------------------------------
    const byKind = new Map<string, WorkspaceNode[]>();
    map.nodes.forEach((node) => {
      byKind.set(node.kind, [...(byKind.get(node.kind) ?? []), node]);
    });

    const positions = new Map<string, THREE.Vector3>();
    const pickable: THREE.Mesh[] = [];
    const geometry = new THREE.SphereGeometry(1, 16, 12);
    const disposables: Array<{ dispose: () => void }> = [geometry, cortex];

    byKind.forEach((nodes, kind) => {
      nodes.forEach((node, index) => {
        const at = placeNode(kind, index, nodes.length, radius);
        positions.set(node.id, at);

        const colour = COLOURS[kind] ?? 0x9aa4b2;
        const material = new THREE.MeshStandardMaterial({
          color: colour,
          emissive: colour,
          // What it has learned, made visible: a neuron with memory glows.
          emissiveIntensity: 0.4 + Math.min(node.memories, 12) * 0.05,
          roughness: 0.3,
        });
        disposables.push(material);

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.copy(at);
        mesh.scale.setScalar(0.22 + node.size * 0.09);
        mesh.userData.node = node;
        brain.add(mesh);
        pickable.push(mesh);

        const halo = new THREE.Mesh(
          geometry,
          new THREE.MeshBasicMaterial({
            color: colour,
            transparent: true,
            opacity: 0.12,
            depthWrite: false,
          }),
        );
        disposables.push(halo.material as THREE.Material);
        halo.position.copy(at);
        halo.scale.setScalar(0.5 + node.size * 0.16);
        halo.userData.node = node;
        brain.add(halo);
        // The halo is part of the target: a 4px sphere is not something anyone
        // can reliably click, and the glow is what the eye is aiming at anyway.
        pickable.push(halo);
      });
    });

    // ---- synapses --------------------------------------------------------
    const curves: THREE.QuadraticBezierCurve3[] = [];
    map.links.forEach((link) => {
      const from = positions.get(link.source);
      const to = positions.get(link.target);
      if (!from || !to) return;

      // Bowed outward through the midpoint, so two links between nearby
      // neurons stay distinguishable instead of overlapping into one line.
      const middle = from
        .clone()
        .add(to)
        .multiplyScalar(0.5)
        .multiplyScalar(1.18);
      const curve = new THREE.QuadraticBezierCurve3(from, middle, to);
      curves.push(curve);

      const line = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(curve.getPoints(24)),
        new THREE.LineBasicMaterial({
          color: 0x5b74d8,
          transparent: true,
          opacity: 0.45,
        }),
      );
      disposables.push(line.geometry, line.material as THREE.Material);
      brain.add(line);
    });

    // ---- firing ----------------------------------------------------------
    // One dot per synapse, travelling along it. This is the part that makes the
    // map read as alive rather than as a diagram, and it is also the cheapest
    // possible animation: no per-frame geometry, just positions.
    //
    // Tiny meshes rather than THREE.Points: a point sprite is a square unless a
    // texture rounds it off, and a generated texture is one more thing that can
    // fail to reach the GPU — which it did, leaving white squares scattered
    // through the map. A sphere is round because it is round.
    const sparkMaterial = new THREE.MeshBasicMaterial({
      color: 0x3b5bdb,
      transparent: true,
      opacity: 0.95,
      depthWrite: false,
    });
    disposables.push(sparkMaterial);

    // Added to `brain`, not the scene: the curves are in the brain's own space,
    // so a dot parented anywhere else drifts away as the map turns.
    const sparks = curves.slice(0, MAX_SPARKS).map(() => {
      const dot = new THREE.Mesh(geometry, sparkMaterial);
      dot.scale.setScalar(0.11);
      brain.add(dot);
      return dot;
    });
    const offsets = sparks.map(() => Math.random());

    // ---- interaction -----------------------------------------------------
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    let dragging = false;
    let lastX = 0;
    let spin = 0.0015;

    function onPointerDown(event: PointerEvent) {
      dragging = true;
      lastX = event.clientX;
    }
    function onPointerUp() {
      dragging = false;
    }
    function onPointerMove(event: PointerEvent) {
      if (dragging) {
        brain.rotation.y += (event.clientX - lastX) * 0.005;
        lastX = event.clientX;
        spin = 0;
      }
    }
    function onClick(event: MouseEvent) {
      if (!interactive) return;
      const bounds = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
      pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster.intersectObjects(pickable)[0];
      setSelected(hit ? (hit.object.userData.node as WorkspaceNode) : null);
    }

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointermove', onPointerMove);
    renderer.domElement.addEventListener('click', onClick);

    function resize() {
      const width = element!.clientWidth;
      const tall = element!.clientHeight || 420;
      renderer.setSize(width, tall, false);
      camera.aspect = width / tall;
      camera.updateProjectionMatrix();
    }
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(element);

    let frame = 0;
    let clock = 0;
    function animate() {
      frame = requestAnimationFrame(animate);
      clock += 0.006;
      brain.rotation.y += spin;

      for (let index = 0; index < sparks.length; index += 1) {
        const t = (clock * 0.35 + offsets[index]) % 1;
        sparks[index].position.copy(curves[index].getPoint(t));
      }

      renderer.render(scene, camera);
    }
    animate();

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('click', onClick);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointermove', onPointerMove);
      disposables.forEach((item) => item.dispose());
      renderer.dispose();
      element.removeChild(renderer.domElement);
    };
  }, [map, interactive]);

  return (
    <div className="brain" style={{ height }}>
      <div ref={mount} className="brain__canvas" />
      {failed ? (
        <div className="brain__fallback muted small">
          This browser cannot draw the map — {map.counts.agents} agent(s),{' '}
          {map.counts.skills} skill(s), {map.counts.workflows} workflow(s).
        </div>
      ) : null}
      {selected ? (
        <div className="brain__detail">
          <span className="chip">{selected.kind}</span>
          <strong>{selected.label}</strong>
          <div className="muted small">{selected.detail}</div>
          {selected.memories ? (
            <div className="muted small">
              {selected.memories} thing(s) learned
            </div>
          ) : null}
        </div>
      ) : null}
      <div className="brain__legend">
        {Object.keys(COLOURS).map((kind) => (
          <span key={kind} className="brain__key">
            <i style={{ background: `#${COLOURS[kind].toString(16).padStart(6, '0')}` }} />
            {kind}
          </span>
        ))}
      </div>
    </div>
  );
}
