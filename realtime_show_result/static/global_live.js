import * as THREE from "./libs/three.module.js";
import { OrbitControls } from "./libs/OrbitControls.js";
import { PLYLoader } from "./libs/PLYLoader.js";
import { GLTFLoader } from "./libs/GLTFLoader.js";

const viewerEl = document.getElementById("viewer");
const statusEl = document.getElementById("status");
const tracksEl = document.getElementById("tracks");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.up.set(0, 0, 1);
camera.position.set(60, -60, 40);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewerEl.clientWidth || 1, viewerEl.clientHeight || 1);
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(30, -25, 40);
scene.add(dirLight);

const grid = new THREE.GridHelper(200, 80, 0x303030, 0x202020);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const vehicleRoot = new THREE.Group();
scene.add(vehicleRoot);
const fusedRoot = new THREE.Group();
scene.add(fusedRoot);

const state = {
    baseVehicle: null,
    trackPool: [],
    fusedMeshes: [],
    overlayTimer: null,
    pollingHandle: null,
    cameras: [],
};

const tmpEuler = new THREE.Euler(0, 0, 0, "ZYX");

window.addEventListener("resize", handleResize);
handleResize();
animate();
init();

async function init() {
    try {
        await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
        await fetchSiteInfo();
        startPolling();
    } catch (err) {
        console.error(err);
        statusEl.textContent = `Init error: ${err.message}`;
    }
}

async function fetchSiteInfo() {
    try {
        const site = await fetchJson("/api/site");
        if (site?.cameras) {
            state.cameras = site.cameras;
        }
    } catch (err) {
        console.warn("site info error", err);
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function handleResize() {
    const w = viewerEl.clientWidth || 1;
    const h = viewerEl.clientHeight || 1;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

async function loadPointCloud() {
    const loader = new PLYLoader();
    return new Promise((resolve, reject) => {
        loader.load(
            "/assets/global.ply",
            (geometry) => {
                geometry.computeBoundingBox();
                const hasColor = Boolean(geometry.getAttribute("color"));
                const material = new THREE.PointsMaterial({
                    size: 0.04,
                    vertexColors: hasColor,
                    color: hasColor ? 0xffffff : 0x8f8f8f,
                });
                const points = new THREE.Points(geometry, material);
                scene.add(points);
                if (geometry.boundingBox) {
                    const center = new THREE.Vector3();
                    geometry.boundingBox.getCenter(center);
                    controls.target.copy(center);
                }
                resolve();
            },
            undefined,
            reject
        );
    });
}

async function loadVehiclePrototype() {
    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
        loader.load(
            "/assets/vehicle.glb",
            (gltf) => {
                state.baseVehicle = (gltf.scene || gltf.scenes?.[0])?.clone(true);
                if (!state.baseVehicle) {
                    reject(new Error("vehicle.glb missing meshes"));
                    return;
                }
                state.baseVehicle.traverse((child) => {
                    if (child.isMesh) {
                        child.castShadow = false;
                        child.receiveShadow = false;
                    }
                });
                resolve();
            },
            undefined,
            reject
        );
    });
}

function startPolling() {
    const poll = async () => {
        try {
            const [tracks, fused] = await Promise.all([
                fetchJson("/api/tracks"),
                fetchJson("/api/fused"),
            ]);
            applyTracks(tracks?.items || [], tracks?.timestamp);
            applyFused(fused?.items || []);
        } catch (err) {
            console.error(err);
            statusEl.textContent = `Poll error: ${err.message}`;
        } finally {
            state.pollingHandle = setTimeout(poll, 400);
        }
    };
    poll();
}

function ensureTrackPool(size) {
    if (!state.baseVehicle) return;
    while (state.trackPool.length < size) {
        const clone = state.baseVehicle.clone(true);
        clone.visible = false;
        vehicleRoot.add(clone);
        state.trackPool.push(clone);
    }
    state.trackPool.forEach((mesh) => (mesh.visible = false));
}

function applyTracks(items, ts) {
    ensureTrackPool(items.length);
    items.forEach((item, idx) => {
        const mesh = state.trackPool[idx];
        if (!mesh) return;
        mesh.visible = true;
        const length = item.length || 4.5;
        const width = item.width || 1.9;
        const height = width * 0.5;
        const cz = item.cz ?? 0.0;
        mesh.scale.set(length, width, height);
        mesh.position.set(item.cx || 0, item.cy || 0, cz + height * 0.5);
        const roll = THREE.MathUtils.degToRad(item.roll || 0);
        const pitch = THREE.MathUtils.degToRad(item.pitch || 0);
        const yaw = THREE.MathUtils.degToRad(item.yaw || 0);
        tmpEuler.set(roll, pitch, yaw, "ZYX");
        mesh.setRotationFromEuler(tmpEuler);
    });
    updateTrackList(items, ts);
}

function applyFused(items) {
    fusedRoot.clear();
    state.fusedMeshes = [];
    const material = new THREE.LineBasicMaterial({ color: 0x4cc9f0, linewidth: 1 });
    items.forEach((det) => {
        const length = det.length || 4.5;
        const width = det.width || 1.9;
        const height = width * 0.5;
        const cz = det.cz ?? 0.0;
        const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(length, width, height));
        const mesh = new THREE.LineSegments(geometry, material);
        mesh.position.set(det.cx || 0, det.cy || 0, cz + height * 0.5);
        const yaw = THREE.MathUtils.degToRad(det.yaw || 0);
        mesh.rotation.set(0, 0, yaw);
        fusedRoot.add(mesh);
        state.fusedMeshes.push(mesh);
    });
}

function updateTrackList(items, ts) {
    tracksEl.innerHTML = "";
    const tsLabel = ts ? new Date(ts * 1000).toLocaleTimeString() : "—";
    statusEl.textContent = `Tracks: ${items.length} | Fused: ${state.fusedMeshes.length} | Timestamp: ${tsLabel}`;
    items.forEach((item) => {
        const div = document.createElement("div");
        div.className = "track-item";
        const sources = (item.sources || []).join(", ") || "n/a";
        div.innerHTML = `
            <div class="track-id">ID ${item.id}</div>
            <div>Pos: (${item.cx.toFixed(2)}, ${item.cy.toFixed(2)})</div>
            <div>Yaw: ${item.yaw.toFixed(1)}° | Score: ${item.score?.toFixed(2) ?? "0.00"}</div>
            <div>Cams: ${sources}</div>
        `;
        tracksEl.appendChild(div);
    });
}

async function fetchJson(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) {
        throw new Error(`${resp.status} ${resp.statusText}`);
    }
    return resp.json();
}
