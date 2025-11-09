import * as THREE from "./libs/three.module.js";
import { OrbitControls } from "./libs/OrbitControls.js";
import { PLYLoader } from "./libs/PLYLoader.js";
import { GLTFLoader } from "./libs/GLTFLoader.js";

const viewerEl = document.getElementById("viewer");
const cameraSelectEl = document.getElementById("camera-select");
const statusEl = document.getElementById("status");
const overlayImgEl = document.getElementById("overlay-img");

const state = {
    cameras: [],
    cameraId: null,
    pollingHandle: null,
    overlayUrl: null,
    baseVehicle: null,
    vehiclePool: [],
    vehicleRoot: new THREE.Group(),
    overlayToken: 0,
    visibleMeshes: new Map(),
    globalCloud: null,
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050505);
scene.add(state.vehicleRoot);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.up.set(0, 0, 1);
camera.position.set(30, -30, 25);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewerEl.clientWidth || 1, viewerEl.clientHeight || 1);
viewerEl.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(15, -18, 25);
scene.add(dirLight);

const grid = new THREE.GridHelper(100, 50, 0x333333, 0x1f1f1f);
grid.rotation.x = Math.PI / 2;
// scene.add(grid);
const axes = new THREE.AxesHelper(4);
scene.add(axes);

const tmpEuler = new THREE.Euler(0, 0, 0, "ZYX");
const tmpCenter = new THREE.Vector3();
const tmpSize = new THREE.Vector3();

window.addEventListener("resize", handleResize);
animate();
init();

async function init() {
    try {
        await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
    } catch (err) {
        console.error(err);
        alert("Failed to load base assets. Check /assets endpoints.");
        return;
    }

    try {
        state.cameras = await fetchJson("/api/cameras");
    } catch (err) {
        console.error(err);
        alert("Failed to load camera list. Edge bridge running?");
        return;
    }

    populateCameraSelect();
    if (state.cameraId) {
        loadVisibleCloudForCamera(state.cameraId);
        startPolling();
    }
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function handleResize() {
    const width = viewerEl.clientWidth || 1;
    const height = viewerEl.clientHeight || 1;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

function populateCameraSelect() {
    cameraSelectEl.innerHTML = "";
    if (!state.cameras.length) {
        const opt = document.createElement("option");
        opt.textContent = "No cameras";
        cameraSelectEl.appendChild(opt);
        cameraSelectEl.disabled = true;
        return;
    }
    state.cameras.forEach((cam) => {
        const opt = document.createElement("option");
        opt.value = cam.camera_id;
        opt.textContent = `${cam.camera_id} — ${cam.name || "camera"}`;
        cameraSelectEl.appendChild(opt);
    });
    cameraSelectEl.disabled = false;
    state.cameraId = Number(cameraSelectEl.value);
    cameraSelectEl.addEventListener("change", () => {
        state.cameraId = Number(cameraSelectEl.value);
        loadVisibleCloudForCamera(state.cameraId);
        startPolling();
    });
}

function startPolling() {
    if (!state.cameraId) {
        statusEl.textContent = "No camera selected";
        return;
    }
    if (state.pollingHandle) {
        clearTimeout(state.pollingHandle);
        state.pollingHandle = null;
    }
    const poll = async () => {
        try {
            await fetchDetections();
        } catch (err) {
            console.error(err);
            statusEl.textContent = `Error: ${err.message}`;
        } finally {
            state.pollingHandle = setTimeout(poll, 400);
        }
    };
    poll();
}

async function fetchDetections() {
    const camId = state.cameraId;
    if (!camId) {
        return;
    }
    const data = await fetchJson(`/api/cameras/${camId}/detections`);
    applyDetections(data?.detections || []);
    statusEl.textContent = `cam${camId} | detections ${data?.detections?.length ?? 0} | ts ${(data?.capture_ts ?? 0).toFixed ? data.capture_ts.toFixed(3) : data?.capture_ts}`;
    updateOverlayImage(camId);
}

async function updateOverlayImage(camId) {
    state.overlayToken += 1;
    const token = state.overlayToken;
    try {
        const resp = await fetch(`/api/cameras/${camId}/overlay.jpg?cacheBust=${Date.now()}`, { cache: "no-store" });
        if (!resp.ok) {
            return;
        }
        const blob = await resp.blob();
        if (token !== state.overlayToken) {
            return;
        }
        if (state.overlayUrl) {
            URL.revokeObjectURL(state.overlayUrl);
        }
        state.overlayUrl = URL.createObjectURL(blob);
        overlayImgEl.src = state.overlayUrl;
    } catch (err) {
        console.warn("Overlay fetch failed", err);
    }
}

function applyDetections(detections) {
    ensureVehiclePool(detections.length);
    state.vehiclePool.forEach((mesh) => {
        mesh.visible = false;
    });
    detections.forEach((det, idx) => {
        const mesh = state.vehiclePool[idx];
        if (!mesh) {
            return;
        }
        mesh.visible = true;
        const [cx, cy, cz] = det.center || [0, 0, 0];
        mesh.position.set(cx || 0, cy || 0, (cz || 0));

        const length = det.length || 4.5;
        const width = det.width || 1.9;
        const height = det.height || width * 0.5;
        mesh.scale.set(length, width, height);

        const roll = THREE.MathUtils.degToRad(det.roll || 0);
        const pitch = THREE.MathUtils.degToRad(det.pitch || 0);
        const yaw = THREE.MathUtils.degToRad(det.yaw || 0);
        tmpEuler.set(roll, pitch, yaw, "ZYX");
        mesh.setRotationFromEuler(tmpEuler);
    });
}

function ensureVehiclePool(size) {
    if (!state.baseVehicle) {
        return;
    }
    while (state.vehiclePool.length < size) {
        const clone = state.baseVehicle.clone(true);
        clone.visible = false;
        state.vehicleRoot.add(clone);
        state.vehiclePool.push(clone);
    }
}

function loadPointCloud() {
    return Promise.resolve();
}

function loadVehiclePrototype() {
    return new Promise((resolve, reject) => {
        const loader = new GLTFLoader();
        loader.load(
            "/assets/vehicle.glb",
            (gltf) => {
                state.baseVehicle = gltf.scene || gltf.scenes?.[0];
                if (!state.baseVehicle) {
                    reject(new Error("vehicle.glb has no scene"));
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
            (err) => reject(err)
        );
    });
}

async function fetchJson(url) {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) {
        throw new Error(`${resp.status} ${resp.statusText}`);
    }
    return resp.json();
}

async function loadVisibleCloudForCamera(camId) {
    if (!camId) {
        return;
    }
    try {
        const meta = await fetchJson(`/api/cameras/${camId}/visible-meta`);
        if (!meta || !meta.count) {
            setVisibleMeshesVisibility(false, camId);
            if (state.globalCloud) state.globalCloud.visible = true;
            return;
        }
        const resp = await fetch(`/api/cameras/${camId}/visible.bin?cacheBust=${Date.now()}`, { cache: "no-store" });
        if (!resp.ok) {
            throw new Error(`${resp.status} ${resp.statusText}`);
        }
        const buffer = await resp.arrayBuffer();
        const data = new Float32Array(buffer);
        if (!data.length) {
            setVisibleMeshesVisibility(false, camId);
            if (state.globalCloud) state.globalCloud.visible = true;
            return;
        }
        const stride = meta.stride || 3;
        const count = meta.count || Math.floor(data.length / stride);
        const hasColor = Boolean(meta.has_rgb) && stride >= 6;
        const positionArray = new Float32Array(count * 3);
        for (let i = 0; i < count; i++) {
            const base = i * stride;
            positionArray[i * 3 + 0] = data[base + 0] || 0;
            positionArray[i * 3 + 1] = data[base + 1] || 0;
            positionArray[i * 3 + 2] = data[base + 2] || 0;
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(positionArray, 3));
        if (hasColor) {
            const colorArray = new Float32Array(count * 3);
            for (let i = 0; i < count; i++) {
                const base = i * stride;
                colorArray[i * 3 + 0] = data[base + 3] ?? 0.5;
                colorArray[i * 3 + 1] = data[base + 4] ?? 0.5;
                colorArray[i * 3 + 2] = data[base + 5] ?? 0.5;
            }
            geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
        }
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();

        let mesh = state.visibleMeshes.get(camId);
        if (!mesh) {
            const material = new THREE.PointsMaterial({
                size: 0.06,
                transparent: true,
                opacity: 0.95,
            });
            mesh = new THREE.Points(geometry, material);
            mesh.name = `VisibleCloud-${camId}`;
            scene.add(mesh);
            state.visibleMeshes.set(camId, mesh);
        } else {
            mesh.geometry.dispose();
            mesh.geometry = geometry;
        }
        mesh.material.vertexColors = hasColor;
        mesh.material.color.set(hasColor ? 0xffffff : 0x4cc9f0);

        setVisibleMeshesVisibility(true, camId);
        if (state.globalCloud) {
            state.globalCloud.visible = false;
        }
        focusOnGeometry(geometry);
    } catch (err) {
        console.warn("visible cloud load failed", err);
        statusEl.textContent = `cam${camId} | visible cloud missing`;
        setVisibleMeshesVisibility(false, camId);
        if (state.globalCloud) {
            state.globalCloud.visible = true;
        }
    }
}

function setVisibleMeshesVisibility(active, camId) {
    state.visibleMeshes.forEach((mesh, id) => {
        mesh.visible = active && id === camId;
    });
}

function focusOnGeometry(geometry) {
    if (!geometry.boundingBox) {
        return;
    }
    geometry.boundingBox.getCenter(tmpCenter);
    geometry.boundingBox.getSize(tmpSize);
    controls.target.copy(tmpCenter);
    const span = Math.max(tmpSize.x, tmpSize.y, tmpSize.z, 1.0);
    camera.position.set(
        tmpCenter.x + span,
        tmpCenter.y - span,
        tmpCenter.z + span * 0.8
    );
}
