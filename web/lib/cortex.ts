import * as THREE from 'three';

/**
 * A brain-shaped shell, generated rather than modelled.
 *
 * A downloaded anatomical mesh would look better and would be the wrong trade:
 * several megabytes shipped to draw a container, a licence to track, and a
 * loading state on the first screen a new user sees. This is a sphere pushed
 * around by layered noise — the ridges and sulci are the noise, the flattened
 * underside and the tapered front are the deformation, and the whole thing costs
 * a few hundred lines and no network.
 *
 * It is a *likeness*, not anatomy. Two hemispheres with a fissure between them,
 * a cerebellum tucked under the back, a stem below. Anyone who knows what a
 * brain looks like will recognise it; nobody should use it to study one.
 */

/** Deterministic value noise. The same workspace draws the same brain. */
function hash3(x: number, y: number, z: number): number {
  const n = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return n - Math.floor(n);
}

function smooth(t: number): number {
  return t * t * (3 - 2 * t);
}

/** Trilinear value noise — cheap, and continuous enough to look organic. */
function noise3(x: number, y: number, z: number): number {
  const xi = Math.floor(x);
  const yi = Math.floor(y);
  const zi = Math.floor(z);
  const xf = smooth(x - xi);
  const yf = smooth(y - yi);
  const zf = smooth(z - zi);

  let value = 0;
  for (let dx = 0; dx <= 1; dx += 1) {
    for (let dy = 0; dy <= 1; dy += 1) {
      for (let dz = 0; dz <= 1; dz += 1) {
        const weight =
          (dx ? xf : 1 - xf) * (dy ? yf : 1 - yf) * (dz ? zf : 1 - zf);
        value += weight * hash3(xi + dx, yi + dy, zi + dz);
      }
    }
  }
  return value * 2 - 1;
}

/** Several octaves of it, which is what turns bumps into folds. */
function folds(x: number, y: number, z: number): number {
  return (
    noise3(x * 2.1, y * 2.1, z * 2.1) * 0.5 +
    noise3(x * 4.7, y * 4.7, z * 4.7) * 0.3 +
    noise3(x * 9.3, y * 9.3, z * 9.3) * 0.18
  );
}

/**
 * The cerebrum: one closed shell, folded, with a fissure cut down the middle.
 *
 * Two separate hemispheres was the obvious approach and the wrong one — two
 * ellipsoids intersect in a lens that reads as a mistake, and neither half looks
 * like half a brain on its own. One surface with a *groove* is both simpler and
 * more convincing: the fissure is a deformation, not a gap, so there is no seam
 * and nothing to line up.
 */
function cerebrum(radius: number, segments: number): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(radius, segments, Math.round(segments * 0.75));
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const vertex = new THREE.Vector3();

  for (let index = 0; index < position.count; index += 1) {
    vertex.fromBufferAttribute(position, index);
    const unit = vertex.clone().normalize();

    // Sampled from |x| so the two halves mirror each other. A brain whose sides
    // do not match reads as a mistake even when nobody can say why.
    const wrinkle = folds(Math.abs(unit.x) * 1.6, unit.y * 1.6, unit.z * 1.6);

    // The longitudinal fissure: a groove along the midline, deepest at the top
    // and closing towards the underside, where the hemispheres are joined.
    const midline = Math.exp(-((unit.x / 0.14) ** 2));
    const above = Math.max(0, unit.y + 0.15);
    // Shallow: a groove, not a cleft. Cut deeper and the front view stops
    // looking like a fissure and starts looking like two things stuck together.
    const fissure = midline * above * 0.17;

    // Narrower at the front, flatter underneath — the two deformations that
    // stop an ellipsoid from looking like an egg.
    const taper = 1 - Math.max(0, unit.z) * 0.14;
    const flatten = unit.y < 0 ? 1 + unit.y * 0.18 : 1;

    const scale = radius * (1 + wrinkle * 0.055 - fissure);
    vertex
      .copy(unit)
      .multiplyScalar(scale)
      .multiply(new THREE.Vector3(0.74 * taper, 0.72 * flatten, 1.0));
    position.setXYZ(index, vertex.x, vertex.y, vertex.z);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/** The cerebellum: a smaller, tighter-folded lobe tucked under the back. */
function cerebellum(radius: number, segments: number): THREE.BufferGeometry {
  const geometry = new THREE.SphereGeometry(
    radius * 0.42,
    segments,
    Math.round(segments * 0.7),
  );
  const position = geometry.attributes.position as THREE.BufferAttribute;
  const vertex = new THREE.Vector3();

  for (let index = 0; index < position.count; index += 1) {
    vertex.fromBufferAttribute(position, index);
    const unit = vertex.clone().normalize();
    // Tighter and more regular than the cortex above it. That difference in
    // texture is most of what makes a cerebellum recognisable as one.
    const ridge = Math.sin(unit.y * 30) * 0.025 + folds(unit.x, unit.y * 2, unit.z) * 0.02;
    vertex
      .copy(unit)
      .multiplyScalar(radius * 0.42 + ridge * radius)
      .multiply(new THREE.Vector3(1.15, 0.6, 0.8));
    position.setXYZ(index, vertex.x, vertex.y - radius * 0.5, vertex.z - radius * 0.62);
  }

  geometry.computeVertexNormals();
  return geometry;
}

/** The stem, so the shape sits on something rather than floating. */
function stem(radius: number): THREE.BufferGeometry {
  const geometry = new THREE.CylinderGeometry(
    radius * 0.15,
    radius * 0.09,
    radius * 0.6,
    20,
    1,
    true,
  );
  geometry.translate(0, -radius * 0.78, -radius * 0.3);
  geometry.rotateX(0.2);
  return geometry;
}

export interface Cortex {
  group: THREE.Group;
  dispose: () => void;
}

/**
 * The whole shell, as a translucent group that the neurons live inside.
 *
 * Drawn as a wireframe *and* a glassy surface: the wireframe is what makes the
 * folds legible from outside, and the surface is what stops the interior from
 * looking like a wire sculpture. Both are transparent, because the point of the
 * container is the things inside it.
 */
export function buildCortex(radius = 10): Cortex {
  const group = new THREE.Group();
  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];

  const surface = new THREE.MeshStandardMaterial({
    color: 0x93a6ff,
    transparent: true,
    opacity: 0.1,
    roughness: 0.55,
    metalness: 0.1,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  // The wireframe is drawn on a *coarser* copy of the same shape. At the
  // density the surface needs to look smooth, a wireframe reads as fabric and
  // hides the folds it is supposed to show.
  const lines = new THREE.MeshBasicMaterial({
    color: 0x7d93f5,
    wireframe: true,
    transparent: true,
    opacity: 0.2,
    depthWrite: false,
  });
  materials.push(surface, lines);

  const smoothParts = [cerebrum(radius, 128), cerebellum(radius, 72), stem(radius)];
  const coarseParts = [cerebrum(radius, 40), cerebellum(radius, 26)];

  for (const geometry of smoothParts) {
    geometries.push(geometry);
    group.add(new THREE.Mesh(geometry, surface));
  }
  for (const geometry of coarseParts) {
    geometries.push(geometry);
    group.add(new THREE.Mesh(geometry, lines));
  }

  return {
    group,
    dispose: () => {
      geometries.forEach((geometry) => geometry.dispose());
      materials.forEach((material) => material.dispose());
    },
  };
}

/**
 * Where a node sits inside the shell.
 *
 * Laid out by kind rather than by force simulation: each kind gets its own
 * shell radius and its own band, so the map reads the same way every time it is
 * opened. A physics layout looks livelier and makes the same workspace look
 * different on every visit, which is the opposite of what a map is for.
 */
export function placeNode(
  kind: string,
  index: number,
  total: number,
  radius: number,
): THREE.Vector3 {
  const bands: Record<string, { shell: number; tilt: number }> = {
    workflow: { shell: 0.32, tilt: 0.1 },
    agent: { shell: 0.56, tilt: -0.35 },
    skill: { shell: 0.78, tilt: 0.3 },
    server: { shell: 0.9, tilt: -0.75 },
    provider: { shell: 0.9, tilt: 0.85 },
  };
  const band = bands[kind] ?? { shell: 0.7, tilt: 0 };

  // A golden-angle spiral spreads any count evenly without clumping, which a
  // naive ring does not once a workspace has more than a handful of anything.
  const golden = Math.PI * (3 - Math.sqrt(5));
  const spread = total > 1 ? index / (total - 1) : 0.5;
  const y = (spread - 0.5) * 1.4 + band.tilt;
  const ring = Math.sqrt(Math.max(0.05, 1 - Math.min(1, y * y)));
  const angle = index * golden;

  return new THREE.Vector3(
    Math.cos(angle) * ring * radius * band.shell,
    y * radius * 0.55,
    Math.sin(angle) * ring * radius * band.shell,
  );
}
