import * as THREE from "./libs/three.module.js";
import { OrbitControls } from "./libs/OrbitControls.js";
import { PLYLoader } from "./libs/PLYLoader.js";
import { GLTFLoader } from "./libs/GLTFLoader.js";

const mode = window.VIEW_MODE || "playback";
const isLive = mode === "live";
const isPlayback = mode === "playback";
const isFusion = mode.startsWith("fusion");
const fusionSource = isFusion ? mode.replace("fusion_", "") : null;

const viewerEl = document.getElementById("viewer");
const detectionListEl = document.getElementById("detection-list");
const frameIndicatorEl = document.getElementById("frame-indicator");
const frameSliderEl = document.getElementById("frame-slider");
const playBtn = document.getElementById("play-btn");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const cameraSelectEl = document.getElementById("camera-select");
const statusEl = document.getElementById("status");
const overlayImgEl = document.getElementById("overlay-img");
const overlayPlaceholderEl = document.getElementById("overlay-placeholder");

const state = {
    mode,
    fusionSource,
    config: null,
    totalFrames: 0,
    frameIndex: 0,
    playing: false,
    timer: null,
    baseVehicle: null,
    vehicleTemplates: [],
    instancedMeshes: [],
    vehicleCapacity: 0,
    showDebugMarker: false,
    frameRequestId: 0,
    vehicleCorrection: null,
    site: null,
    cameras: [],
    cameraId: null,
    pollingHandle: null,
    overlayToken: 0,
    overlayUrl: null,
    overlayPollingHandle: null,
    visibleMeshes: new Map(),
    globalCloud: null,
    cameraMarkers: [],
    markerLookup: new Map(),
    markerObjects: [],
    markerGroup: null,
    activeMarkerKey: null,
    localClouds: new Map(),
    localLoadToken: 0,
    globalColorLookup: null,
    viewLockBackup: null,
    viewLockAnimation: null,
    liveCameraSwitchToken: 0,
    layoutMode: "global",
    followTarget: null,
    followPanBackup: null,
};

setLayoutMode("global");

function setLayoutMode(mode) {
    if (!(isLive || isFusion)) {
        return;
    }
    const body = document.body;
    if (!body) {
        return;
    }
    const nextMode = mode === "local" ? "local" : "global";
    if (state.layoutMode === nextMode) {
        return;
    }
    state.layoutMode = nextMode;
    const isLocal = nextMode === "local";
    body.classList.toggle("local-mode", isLocal);
    body.classList.toggle("global-mode", !isLocal);
    handleResize();
}

// const COLOR_PALETTE = {
//     red: "#ff4d4f",
//     pink: "#ff85c0",
//     green: "#73d13d",
//     white: "#f0f0f0",
//     yellow: "#fadb14",
//     purple: "#9254de",
// };

const fusionEndpointMap = {
    raw: "/api/raw",
    fused: "/api/fused",
    tracks: "/api/tracks",
};

const defaultLocalColor = new THREE.Color(0xf7b801);
const defaultLocalColorArray = [defaultLocalColor.r, defaultLocalColor.g, defaultLocalColor.b];

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b0b);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.up.set(0, 0, 1);
camera.position.set(0, -20, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
renderer.outputEncoding = THREE.sRGBEncoding;
viewerEl.appendChild(renderer.domElement);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const markerPointerState = { isDown: false, x: 0, y: 0 };

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(15, -20, 25);
dirLight.castShadow = false;
scene.add(dirLight);

const grid = new THREE.GridHelper(100, 50, 0x333333, 0x222222);
grid.rotation.x = Math.PI / 2; // Align grid with XY plane (Z up)

const axes = new THREE.AxesHelper(2);

const vehicleRoot = new THREE.Group();
scene.add(vehicleRoot);
const tmpMatrix = new THREE.Matrix4();
const tmpMatrix2 = new THREE.Matrix4();
const tmpPosition = new THREE.Vector3();
const tmpQuaternion = new THREE.Quaternion();
const tmpScale = new THREE.Vector3();
const tmpEuler = new THREE.Euler(0, 0, 0, "ZYX");
const tmpViewPos = new THREE.Vector3();
const tmpViewTarget = new THREE.Vector3();
const tmpColor = new THREE.Color();
const defaultVehicleColor = new THREE.Color(0xffffff);
const debugMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.4, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0xff5555 })
);
debugMarker.visible = false;
scene.add(debugMarker);
scene.add(grid);
scene.add(axes);
const markerGroup = new THREE.Group();
scene.add(markerGroup);
state.markerGroup = markerGroup;

setupMarkerPointerHandlers();
window.addEventListener("keydown", handleGlobalKeydown);

window.addEventListener("resize", handleResize);
handleResize();
animate();

async function init() {
    if (isFusion) {
        await initFusionMode();
    } else if (isLive) {
        await initLiveMode();
    } else {
        await initPlaybackMode();
    }
}

async function initPlaybackMode() {
    try {
        state.config = await fetchJson("/api/config");
        state.showDebugMarker = Boolean(state.config?.showDebugMarker);
        const showSceneAxes = Boolean(state.config?.showSceneAxes);
        grid.visible = showSceneAxes;
        axes.visible = showSceneAxes;
    } catch (err) {
        console.error(err);
        alert("Failed to load configuration. Check if the backend server is running.");
        return;
    }
    try {
        state.site = await fetchJson("/api/site");
    } catch (err) {
        console.warn("Site info not available", err);
    }

    state.totalFrames = state.config.totalFrames ?? 0;
    if (frameSliderEl) {
        frameSliderEl.max = Math.max(state.totalFrames - 1, 0);
        frameSliderEl.disabled = state.totalFrames === 0;
    }
    if (prevBtn) prevBtn.disabled = state.totalFrames === 0;
    if (nextBtn) nextBtn.disabled = state.totalFrames === 0;
    if (playBtn) playBtn.disabled = state.totalFrames === 0;
    updateFrameIndicator();

    await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
    setupCameraMarkers(state.site?.camera_positions || []);

    if (state.totalFrames > 0) {
        await goToFrame(0);
    }

    wireUi();
}

async function initFusionMode() {
    try {
        state.config = await fetchConfigWithFallback();
    } catch (err) {
        console.warn("Failed to load fusion config", err);
        state.config = {};
    }
    try {
        state.site = await fetchJson("/api/site");
    } catch (err) {
        console.warn("site info error", err);
    }
    state.cameras = deriveCamerasFromSite(state.site);
    await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
    setupCameraMarkers(state.site?.camera_positions || []);
    setupFusionCameraSelect();
    startFusionPolling();
}

async function initLiveMode() {
    try {
        state.site = await fetchJson("/api/site");
        state.config = state.site?.config || {};
        grid.visible = false;
        axes.visible = false;
    } catch (err) {
        console.error(err);
        alert("Failed to load site configuration. Check if the backend server is running.");
        return;
    }

    try {
        const cameraPayload = await fetchJson("/api/cameras");
        state.cameras = Array.isArray(cameraPayload) ? cameraPayload : [];
    } catch (err) {
        console.error(err);
        alert("Failed to load camera list. Edge bridge running?");
        return;
    }

    const hasCameras = state.cameras.length > 0;
    enterLiveGlobalView({
        placeholder: hasCameras ? "Click a camera marker to view" : "No camera available",
        status: hasCameras ? "Global view" : "No camera available",
    });

    await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
    setupCameraMarkers(state.site?.camera_positions || []);

    if (!state.cameraMarkers.length && hasCameras) {
        const firstId = Number(state.cameras[0].camera_id);
        if (Number.isFinite(firstId)) {
            await selectLiveCamera(firstId, { loadingMessage: `cam${firstId} loading... (no markers)` });
        }
    }
}

function animate() {
    requestAnimationFrame(animate);
    updateViewLockAnimation();
    updateFollowOffset();
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

async function loadPointCloud() {
    const loader = new PLYLoader();
    return new Promise((resolve, reject) => {
        loader.load(
            "/assets/global.ply",
            (geometry) => {
                const flipPlyY = Boolean(state.config?.flipPlyY);
                if (flipPlyY) {
                    geometry.scale(1, -1, 1);
                }

                geometry.computeBoundingBox();
                geometry.computeVertexNormals();

                const hasColor = Boolean(geometry.getAttribute("color"));
                const material = new THREE.PointsMaterial({
                    size: 0.05,
                    vertexColors: hasColor,
                    color: hasColor ? 0xffffff : 0x8f8f8f,
                });

                const points = new THREE.Points(geometry, material);
                points.name = "GlobalCloud";
                scene.add(points);
                state.globalCloud = points;
                buildGlobalColorLookup(geometry);

                if (geometry.boundingBox) {
                    const center = new THREE.Vector3();
                    geometry.boundingBox.getCenter(center);
                    controls.target.copy(center);

                    const size = new THREE.Vector3();
                    geometry.boundingBox.getSize(size);
                    const maxDim = Math.max(size.x, size.y, size.z, 1.0);
                    const distance = maxDim * 1.6;
                    camera.position.set(
                        center.x + distance,
                        center.y - distance,
                        center.z + distance * 0.6
                    );
                }
                resolve();
            },
            undefined,
            (error) => {
                console.error("Failed to load PLY:", error);
                reject(error);
            }
        );
    });
}

async function loadVehiclePrototype() {
    const loader = new GLTFLoader();
    return new Promise((resolve, reject) => {
        loader.load(
            "/assets/vehicle.glb",
            (gltf) => {
                const content = gltf.scene || (gltf.scenes && gltf.scenes[0]);
                if (!content) {
                    reject(new Error("Vehicle GLB has no scene"));
                    return;
                }
                

                content.updateMatrixWorld(true);
                const bbox = new THREE.Box3().setFromObject(content);
                const center = new THREE.Vector3();
                bbox.getCenter(center);
                const size = new THREE.Vector3();
                bbox.getSize(size);
                const maxDim = Math.max(size.x, size.y, size.z, 1e-6);

                const unitGroup = new THREE.Group();
                unitGroup.name = "VehiclePrototype";
                unitGroup.add(content);
                content.position.sub(center);

                const normalizeVehicle =
                    state.config?.normalizeVehicle !== undefined
                        ? Boolean(state.config.normalizeVehicle)
                        : true;
                if (normalizeVehicle && maxDim > 0) {
                    unitGroup.scale.setScalar(1 / maxDim);
                }

                const vehicleYAxisUp =
                    state.config?.vehicleYAxisUp !== undefined
                        ? Boolean(state.config.vehicleYAxisUp)
                        : true;
                if (vehicleYAxisUp) {
                    unitGroup.rotation.x = Math.PI / 2; // convert Y-up to Z-up
                }

                unitGroup.traverse((obj) => {
                    if (obj.isMesh) {
                        obj.castShadow = false;
                        obj.receiveShadow = false;
                    }
                    obj.matrixAutoUpdate = true;
                    obj.updateMatrix();
                });

                unitGroup.updateMatrixWorld(true);
                unitGroup.updateMatrixWorld(true);

                state.baseVehicle = unitGroup;
                state.vehicleCorrection = unitGroup.matrixWorld.clone();
                state.vehicleTemplates = [];
                unitGroup.traverse((obj) => {
                    if (obj.isMesh) {
                        const srcMaterial = obj.material;
                        const templateMaterial = new THREE.MeshBasicMaterial({
                            color: 0xffffff,
                            transparent: Boolean(srcMaterial?.transparent),
                            opacity: srcMaterial?.opacity !== undefined ? srcMaterial.opacity : 1.0,
                            side: srcMaterial?.side ?? THREE.FrontSide,
                            depthWrite: srcMaterial?.depthWrite !== undefined ? srcMaterial.depthWrite : true,
                            depthTest: srcMaterial?.depthTest !== undefined ? srcMaterial.depthTest : true,
                        });
                        const vertexCount = obj.geometry.attributes.position.count;
                        const colorAttr = new Float32Array(vertexCount * 3).fill(1);
                        obj.geometry.setAttribute("color", new THREE.BufferAttribute(colorAttr, 3));
                        templateMaterial.vertexColors = true;
                        templateMaterial.needsUpdate = true;
                        const baseColor = defaultVehicleColor.clone();
                        state.vehicleTemplates.push({
                            geometry: obj.geometry.clone(),
                            material: templateMaterial,
                            localMatrix: obj.matrix.clone(),
                            baseColor,
                        });
                    }
                });
                state.vehicleCapacity = 0;
                state.instancedMeshes = [];
                resolve();
            },
            undefined,
            (error) => {
                console.error("Failed to load vehicle GLB:", error);
                reject(error);
            }
        );
    });
}

function wireUi() {
    playBtn.addEventListener("click", () => {
        setPlaying(!state.playing);
    });

    prevBtn.addEventListener("click", () => {
        if (state.totalFrames === 0) {
            return;
        }
        setPlaying(false);
        const prev = (state.frameIndex - 1 + state.totalFrames) % state.totalFrames;
        goToFrame(prev);
    });

    nextBtn.addEventListener("click", () => {
        if (state.totalFrames === 0) {
            return;
        }
        setPlaying(false);
        const next = (state.frameIndex + 1) % state.totalFrames;
        goToFrame(next);
    });

    frameSliderEl.addEventListener("input", (evt) => {
        if (state.totalFrames === 0) {
            return;
        }
        const target = Number(evt.target.value);
        setPlaying(false);
        goToFrame(target);
    });
}

function setPlaying(shouldPlay) {
    if (state.totalFrames <= 1) {
        shouldPlay = false;
    }

    if (state.playing === shouldPlay) {
        return;
    }

    state.playing = shouldPlay;
    playBtn.textContent = shouldPlay ? "Pause" : "Play";

    if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
    }

    if (shouldPlay) {
        const fps = Math.max(1, Number(state.config?.fps) || 10);
        const interval = 1000 / fps;
        state.timer = setInterval(() => {
            if (state.totalFrames === 0) {
                return;
            }
            const next = (state.frameIndex + 1) % state.totalFrames;
            goToFrame(next, { preservePlayback: true });
        }, interval);
    }
}

async function goToFrame(index, options = { preservePlayback: false }) {
    if (state.totalFrames === 0) {
        state.frameIndex = 0;
        updateFrameIndicator();
        updateSlider();
        renderDetections([]);
        return;
    }

    const clamped = Math.max(0, Math.min(index, Math.max(state.totalFrames - 1, 0)));
    state.frameRequestId += 1;
    const requestId = state.frameRequestId;

    if (!options.preservePlayback) {
        setPlaying(false);
    }

    try {
        const frameData = await fetchJson(`/api/frames/${clamped}`);
        if (requestId !== state.frameRequestId) {
            return;
        }
        state.frameIndex = clamped;
        updateFrameIndicator();
        updateSlider();
        renderDetections(frameData.detections ?? []);
    } catch (err) {
        console.error("Failed to load frame:", err);
    }
}

function updateFrameIndicator() {
    const total = Math.max(state.totalFrames, 0);
    const current = total > 0 ? state.frameIndex + 1 : 0;
    frameIndicatorEl.textContent = `Frame ${current} / ${total}`;
}

function updateSlider() {
    if (!frameSliderEl.disabled && state.totalFrames > 0) {
        frameSliderEl.value = state.frameIndex;
    }
}

function disposeInstancedMeshes() {
    if (!state.instancedMeshes) {
        return;
    }
    state.instancedMeshes.forEach((mesh) => {
        vehicleRoot.remove(mesh);
        if (mesh.geometry) {
            mesh.geometry.dispose();
        }
        if (mesh.material) {
            if (Array.isArray(mesh.material)) {
                mesh.material.forEach((mat) => {
                    if (mat && typeof mat.dispose === "function") {
                        mat.dispose();
                    }
                });
            } else if (typeof mesh.material.dispose === "function") {
                mesh.material.dispose();
            }
        }
    });
    state.instancedMeshes = [];
    state.vehicleCapacity = 0;
}

function rebuildInstancedMeshes(capacity) {
    disposeInstancedMeshes();
    state.vehicleCapacity = capacity;
    if (!state.vehicleTemplates || state.vehicleTemplates.length === 0 || capacity <= 0) {
        return;
    }

    state.instancedMeshes = state.vehicleTemplates.map((tpl) => {
        const geometry = tpl.geometry.clone();
        const material = tpl.material && tpl.material.clone ? tpl.material.clone() : tpl.material;
        if (material) {
            material.vertexColors = true;
            material.needsUpdate = true;
        }
        const inst = new THREE.InstancedMesh(geometry, material, capacity);
        inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        inst.count = 0;
        inst.frustumCulled = false;
        inst.matrixAutoUpdate = false;
        const colorArray = new Float32Array(capacity * 3);
        for (let i = 0; i < capacity; i += 1) {
            colorArray[i * 3 + 0] = 1.0;
            colorArray[i * 3 + 1] = 1.0;
            colorArray[i * 3 + 2] = 1.0;
        }
        inst.instanceColor = new THREE.InstancedBufferAttribute(colorArray, 3);
        vehicleRoot.add(inst);
        return inst;
    });
}

function ensureInstancedCapacity(requiredCount) {
    if (!state.vehicleTemplates || state.vehicleTemplates.length === 0) {
        return;
    }
    if (requiredCount <= 0) {
        return;
    }

    const needed = Math.max(requiredCount, 1);
    if (state.vehicleCapacity >= needed) {
        return;
    }

    const newCapacity = Math.max(needed, state.vehicleCapacity > 0 ? state.vehicleCapacity * 2 : 1);
    rebuildInstancedMeshes(newCapacity);
}

function getDetectionColor(det) {
    if (!det) {
        return null;
    }
    let raw = typeof det.color_hex === "string" && det.color_hex.length
        ? det.color_hex.trim()
        : (typeof det.colorHex === "string" ? det.colorHex.trim() : "");
    // if (!raw) {
    //     const label = typeof det.color === "string" ? det.color.trim().toLowerCase() : "";
    //     if (label && COLOR_PALETTE[label]) {
    //         raw = COLOR_PALETTE[label];
    //     }
    // }
    if (!raw) {
        return null;
    }
    const normalized = raw.startsWith("#") ? raw : `#${raw}`;
    if (!/^#([0-9a-fA-F]{6})$/.test(normalized)) {
        return null;
    }
    try {
        tmpColor.set(normalized);
        return tmpColor;
    } catch (err) {
        return null;
    }
}

function renderDetections(detections) {
    const count = Array.isArray(detections) ? detections.length : 0;
    ensureInstancedCapacity(count);

    if (count === 0 || state.instancedMeshes.length === 0) {
        debugMarker.visible = false;
        state.instancedMeshes.forEach((inst) => {
            inst.count = 0;
        });
        vehicleRoot.visible = false;
    } else {
        state.instancedMeshes.forEach((inst) => {
            inst.count = count;
        });

        const correction = state.vehicleCorrection || tmpMatrix.identity();

        for (let i = 0; i < count; i += 1) {
            const det = detections[i];
            const transformArray = Array.isArray(det.transform) ? det.transform : null;
            const detColor = getDetectionColor(det);

            if (transformArray && transformArray.length === 16) {
                tmpMatrix.fromArray(transformArray);
            } else {
                const scaleArray = Array.isArray(det.scale) && det.scale.length === 3
                    ? det.scale
                    : [det.length ?? 1, det.width ?? 1, det.height ?? 1];
                const center = Array.isArray(det.center) && det.center.length === 3
                    ? det.center
                    : [0, 0, 0];
                const rollRad = THREE.MathUtils.degToRad(det.roll_deg || 0);
                const pitchRad = THREE.MathUtils.degToRad(det.pitch_deg || 0);
                const yawRad = THREE.MathUtils.degToRad(det.yaw_deg || 0);
                tmpEuler.set(rollRad, pitchRad, yawRad, "ZYX");
                tmpQuaternion.setFromEuler(tmpEuler);
                tmpPosition.set(center[0], center[1], center[2]);
                tmpScale.set(scaleArray[0], scaleArray[1], scaleArray[2]);
                tmpMatrix.compose(tmpPosition, tmpQuaternion, tmpScale);
                if (state.showDebugMarker && i === 0) {
                    console.log("det0 fallback", det.center, tmpPosition.toArray());
                }
            }

            if (state.vehicleTemplates.length > 0) {
                for (let m = 0; m < state.vehicleTemplates.length; m += 1) {
                    const tpl = state.vehicleTemplates[m];
                    const inst = state.instancedMeshes[m];
                    tmpMatrix2.multiplyMatrices(tmpMatrix, correction);
                    tmpMatrix2.multiply(tpl.localMatrix);
                    inst.setMatrixAt(i, tmpMatrix2);
                    const colorToApply = detColor || tpl.baseColor || defaultVehicleColor;
                    if (inst.setColorAt) {
                        inst.setColorAt(i, colorToApply);
                    }
                }
            }

            if (state.showDebugMarker && i === 0) {
                tmpPosition.setFromMatrixPosition(tmpMatrix);
                debugMarker.position.copy(tmpPosition);
                debugMarker.visible = true;
                console.log("det0 transform", det.center, tmpPosition.toArray());
            }
        }

        state.instancedMeshes.forEach((inst) => {
            inst.instanceMatrix.needsUpdate = true;
            if (inst.instanceColor) {
                inst.instanceColor.needsUpdate = true;
            }
        });
        vehicleRoot.visible = true;
        if (!state.showDebugMarker) {
            debugMarker.visible = false;
        }
    }

    if (detectionListEl) {
        detectionListEl.innerHTML = "";
        detections.forEach((det, idx) => {
            const li = document.createElement("li");
            const btn = document.createElement("button");
            btn.type = "button";
            btn.textContent = formatDetectionListEntry(det, idx);
            btn.addEventListener("click", () => {
                startDetectionFollow(det, idx);
            });
            li.appendChild(btn);
            detectionListEl.appendChild(li);
        });
    }

    updateDetectionFollowPose(detections);
}

async function fetchJson(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Request failed: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

async function fetchConfigWithFallback() {
    try {
        return await fetchJson("/api/config");
    } catch (err) {
        try {
            const site = await fetchJson("/api/site");
            return site?.config || {};
        } catch {
            throw err;
        }
    }
}

function deriveCamerasFromSite(site) {
    const result = [];
    if (!site) {
        return result;
    }
    const seen = new Set();
    const markers = Array.isArray(site.camera_positions) ? site.camera_positions : [];
    markers.forEach((marker) => {
        if (!marker) {
            return;
        }
        let camId = marker.camera_id ?? marker.id;
        if (!Number.isFinite(camId)) {
            const maybeId = parseInt(`${camId}`, 10);
            if (Number.isFinite(maybeId)) {
                camId = maybeId;
            }
        }
        if (!Number.isFinite(camId)) {
            if (marker.name) {
                const digits = `${marker.name}`.match(/\d+/g);
                if (digits && digits.length) {
                    const parsed = parseInt(digits[digits.length - 1], 10);
                    if (Number.isFinite(parsed)) {
                        camId = parsed;
                    }
                }
            }
        }
        if (!Number.isFinite(camId)) {
            return;
        }
        const numericId = Number(camId);
        if (seen.has(numericId)) {
            return;
        }
        seen.add(numericId);
        result.push({
            camera_id: numericId,
            name: marker.name || marker.key || `cam${numericId}`,
        });
    });
    if (!result.length && Array.isArray(site.cameras)) {
        site.cameras.forEach((entry, idx) => {
            let label = entry;
            let numericId = null;
            if (typeof entry === "object" && entry !== null) {
                label = entry.name || entry.label || entry.id || entry.camera_id;
            }
            const text = `${label ?? ""}`;
            const digits = text.match(/\d+/g);
            if (digits && digits.length) {
                numericId = parseInt(digits[digits.length - 1], 10);
            }
            if (!Number.isFinite(numericId)) {
                numericId = idx + 1;
            }
            if (seen.has(numericId)) {
                return;
            }
            seen.add(numericId);
            result.push({
                camera_id: numericId,
                name: text || `cam${numericId}`,
            });
        });
    }
    result.sort((a, b) => a.camera_id - b.camera_id);
    return result;
}

function setupFusionCameraSelect() {
    if (!isFusion) {
        return;
    }
    const cameras = Array.isArray(state.cameras) ? state.cameras : [];
    stopFusionOverlayPolling();
    if (!cameraSelectEl) {
        const hasCameras = cameras.length > 0;
        enterFusionGlobalView({
            placeholder: hasCameras ? "Click a camera marker to view" : "No camera available",
        });
        return;
    }
    cameraSelectEl.innerHTML = "";
    if (!cameras.length) {
        const opt = document.createElement("option");
        opt.textContent = "No cameras";
        cameraSelectEl.appendChild(opt);
        cameraSelectEl.disabled = true;
        state.cameraId = null;
        enterFusionGlobalView({ placeholder: "No camera available" });
        return;
    }
    cameras.forEach((cam, idx) => {
        const opt = document.createElement("option");
        opt.value = cam.camera_id;
        opt.textContent = `${cam.camera_id} — ${cam.name || `cam${cam.camera_id}`}`;
        if (idx === 0) {
            opt.selected = true;
        }
        cameraSelectEl.appendChild(opt);
    });
    cameraSelectEl.disabled = false;
    const initial = Number(cameraSelectEl.value);
    const initialId = Number.isFinite(initial) ? initial : null;
    cameraSelectEl.onchange = () => {
        const next = Number(cameraSelectEl.value);
        if (Number.isFinite(next)) {
            selectFusionCamera(next);
        } else {
            enterFusionGlobalView();
        }
    };
    if (initialId == null) {
        enterFusionGlobalView();
    } else {
        selectFusionCamera(initialId);
    }
}

function enterFusionGlobalView(options = {}) {
    if (!isFusion) {
        return;
    }
    setLayoutMode("global");
    state.cameraId = null;
    stopFusionOverlayPolling();
    const hasCameras = Array.isArray(state.cameras) && state.cameras.length > 0;
    const placeholder = options.placeholder || (hasCameras ? "Click a camera marker to view" : "No camera available");
    showOverlayPlaceholder(placeholder);
}

function selectFusionCamera(camId) {
    if (!isFusion) {
        return;
    }
    const numericId = Number(camId);
    if (!Number.isFinite(numericId)) {
        return;
    }
    if (state.cameraId === numericId && state.overlayPollingHandle) {
        return;
    }
    state.cameraId = numericId;
    showOverlayPlaceholder(`cam${numericId} loading...`);
    startFusionOverlayPolling();
}

function startFusionOverlayPolling() {
    if (!isFusion) {
        return;
    }
    setLayoutMode("local");
    stopFusionOverlayPolling();
    if (state.cameraId == null || !overlayImgEl) {
        enterFusionGlobalView();
        return;
    }
    const overlayConfigured = typeof state.config?.overlayBaseUrl === "string"
        && state.config.overlayBaseUrl.trim().length > 0;
    if (!overlayConfigured) {
        showOverlayPlaceholder("Camera overlay unavailable", { skipToken: true });
        return;
    }
    const tick = () => {
        updateOverlayImage(state.cameraId);
    };
    tick();
    state.overlayPollingHandle = window.setInterval(tick, 900);
}

function stopFusionOverlayPolling() {
    if (state.overlayPollingHandle) {
        clearInterval(state.overlayPollingHandle);
        state.overlayPollingHandle = null;
    }
}

function enterLiveGlobalView(options = {}) {
    if (!isLive) {
        return;
    }
    setLayoutMode("global");
    state.liveCameraSwitchToken += 1;
    stopLivePolling();
    state.cameraId = null;
    setVisibleMeshesVisibility(false, null);
    if (state.globalCloud) {
        state.globalCloud.visible = true;
    }
    renderDetections([]);
    const hasCameras = Array.isArray(state.cameras) && state.cameras.length > 0;
    const placeholderText = options.placeholder || (hasCameras ? "Click a camera marker to view" : "No camera available");
    if (statusEl) {
        statusEl.textContent = options.status || (hasCameras ? "Global view" : "No camera available");
    }
    showOverlayPlaceholder(placeholderText);
}

async function selectLiveCamera(camId, options = {}) {
    if (!isLive) {
        return;
    }
    const numericId = Number(camId);
    if (!Number.isFinite(numericId)) {
        return;
    }
    if (state.cameraId === numericId && state.pollingHandle) {
        return;
    }
    stopLivePolling();
    setLayoutMode("local");
    state.cameraId = numericId;
    state.liveCameraSwitchToken += 1;
    const token = state.liveCameraSwitchToken;
    setVisibleMeshesVisibility(false, null);
    if (state.globalCloud) {
        state.globalCloud.visible = true;
    }
    renderDetections([]);
    const loadingMessage = options.loadingMessage || `cam${numericId} loading...`;
    showOverlayPlaceholder(loadingMessage);
    if (statusEl) {
        statusEl.textContent = `cam${numericId} | loading...`;
    }
    try {
        await loadVisibleCloudForCamera(numericId);
    } catch (err) {
        console.warn(`visible cloud load failed for cam${numericId}`, err);
    }
    if (token !== state.liveCameraSwitchToken) {
        return;
    }
    startLivePolling();
    updateOverlayImage(numericId);
}

function clearOverlayImage() {
    if (state.overlayUrl) {
        URL.revokeObjectURL(state.overlayUrl);
        state.overlayUrl = null;
    }
    if (overlayImgEl) {
        overlayImgEl.removeAttribute("src");
    }
}

function showOverlayPlaceholder(message = "No camera selected", options = {}) {
    if (!overlayImgEl) {
        return;
    }
    const { skipToken = false } = options;
    if (!skipToken) {
        state.overlayToken += 1;
    }
    clearOverlayImage();
    overlayImgEl.style.display = "none";
    if (overlayPlaceholderEl) {
        overlayPlaceholderEl.textContent = message;
        overlayPlaceholderEl.style.display = "flex";
    } else {
        overlayImgEl.alt = message;
    }
}

function showOverlayImage() {
    if (!overlayImgEl) {
        return;
    }
    overlayImgEl.style.display = "block";
    if (overlayPlaceholderEl) {
        overlayPlaceholderEl.style.display = "none";
    }
}

function startLivePolling() {
    stopLivePolling();
    if (state.cameraId == null) {
        if (statusEl) statusEl.textContent = "No camera selected";
        return;
    }
    const poll = async () => {
        try {
            await fetchLiveDetections();
        } catch (err) {
            console.error(err);
            if (statusEl) statusEl.textContent = `Error: ${err.message}`;
        } finally {
            state.pollingHandle = window.setTimeout(poll, 400);
        }
    };
    poll();
}

function stopLivePolling() {
    if (state.pollingHandle) {
        clearTimeout(state.pollingHandle);
        state.pollingHandle = null;
    }
}

async function fetchLiveDetections() {
    const camId = state.cameraId;
    if (camId == null) {
        return;
    }
    const data = await fetchJson(`/api/cameras/${camId}/detections`);
    renderDetections(data?.detections || []);
    if (statusEl) {
        const detCount = data?.detections?.length ?? 0;
        const ts = data?.capture_ts;
        const tsText = typeof ts === "number" ? ts.toFixed(3) : ts ?? "";
        statusEl.textContent = `cam${camId} | detections ${detCount} | ts ${tsText}`;
    }
    updateOverlayImage(camId);
}

function startFusionPolling() {
    stopLivePolling();
    if (!state.fusionSource) {
        if (statusEl) statusEl.textContent = "Fusion source not set";
        return;
    }
    const poll = async () => {
        try {
            await fetchFusionDetections();
        } catch (err) {
            console.error(err);
            if (statusEl) statusEl.textContent = `Fusion error: ${err.message}`;
        } finally {
            state.pollingHandle = window.setTimeout(poll, 400);
        }
    };
    poll();
}

async function fetchFusionDetections() {
    const source = state.fusionSource || "raw";
    const endpoint = fusionEndpointMap[source] || "/api/raw";
    const data = await fetchJson(endpoint);
    const detections = data?.items || [];
    renderDetections(detections);
    if (statusEl) {
        const ts = data?.timestamp;
        const tsText = typeof ts === "number" ? ts.toFixed(3) : ts ?? "";
        statusEl.textContent = `${source.toUpperCase()} | count ${detections.length} | ts ${tsText}`;
    }
}

function buildOverlayRequestUrl(camId) {
    if (camId == null) {
        return null;
    }
    const rawBase = typeof state.config?.overlayBaseUrl === "string" ? state.config.overlayBaseUrl.trim() : "";
    if (isFusion && !rawBase) {
        return null;
    }
    const normalizedBase = rawBase ? rawBase.replace(/\/+$/, "") : "";
    const path = `/api/cameras/${camId}/overlay.jpg`;
    const url = normalizedBase ? `${normalizedBase}${path}` : path;
    const separator = url.includes("?") ? "&" : "?";
    return `${url}${separator}cacheBust=${Date.now()}`;
}

function updateOverlayImage(camId = state.cameraId) {
    if (!overlayImgEl) {
        return;
    }
    if (camId == null) {
        clearOverlayImage();
        return;
    }
    state.overlayToken += 1;
    const token = state.overlayToken;
    const requestUrl = buildOverlayRequestUrl(camId);
    if (!requestUrl) {
        showOverlayPlaceholder("Camera overlay unavailable", { skipToken: true });
        return;
    }
    fetch(requestUrl, { cache: "no-store" })
        .then((resp) => {
            if (!resp.ok) {
                throw new Error("overlay not ready");
            }
            return resp.blob();
        })
        .then((blob) => {
            if (token !== state.overlayToken) {
                return;
            }
            if (state.overlayUrl) {
                URL.revokeObjectURL(state.overlayUrl);
            }
            state.overlayUrl = URL.createObjectURL(blob);
            overlayImgEl.src = state.overlayUrl;
            showOverlayImage();
        })
        .catch(() => {
            /* ignore */ 
        });
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
        if (statusEl) statusEl.textContent = `cam${camId} | visible cloud missing`;
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
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    controls.target.copy(center);
    const size = new THREE.Vector3();
    geometry.boundingBox.getSize(size);
    const span = Math.max(size.x, size.y, size.z, 1.0);
    camera.position.set(
        center.x + span,
        center.y - span,
        center.z + span * 0.8
    );
}

function focusOnPosition(position, span = 12) {
    if (!position) {
        return;
    }
    const cx = Number(position.x ?? position[0]) || 0;
    const cy = Number(position.y ?? position[1]) || 0;
    const cz = Number(position.z ?? position[2]) || 0;
    controls.target.set(cx, cy, cz);
    camera.position.set(
        cx + span,
        cy - span,
        cz + span * 0.7
    );
}

function shouldFlipMarkerX() {
    return Boolean(state.config?.flipMarkerX);
}

function shouldFlipMarkerY() {
    return Boolean(state.config?.flipMarkerY);
}

function buildMarkerDisplayPosition(rawPosition) {
    const pos = rawPosition || {};
    let x = Number(pos.x ?? pos[0]);
    let y = Number(pos.y ?? pos[1]);
    let z = Number(pos.z ?? pos[2]);
    if (!Number.isFinite(x)) x = 0;
    if (!Number.isFinite(y)) y = 0;
    if (!Number.isFinite(z)) z = 0;
    if (shouldFlipMarkerX()) {
        x = -x;
    }
    if (shouldFlipMarkerY()) {
        y = -y;
    }
    return { x, y, z };
}

function getMarkerFocusPosition(marker) {
    if (!marker) {
        return null;
    }
    return marker.displayPosition || marker.position || { x: 0, y: 0, z: 0 };
}

function computeMarkerViewPose(marker) {
    if (!marker) {
        return null;
    }
    const focusPos = getMarkerFocusPosition(marker);
    const eye = new THREE.Vector3(focusPos.x, focusPos.y, focusPos.z);
    const rot = marker.rotation || {};
    const yaw = THREE.MathUtils.degToRad(rot.yaw || 0);
    const pitch = THREE.MathUtils.degToRad(rot.pitch || 0);
    const dir = new THREE.Vector3(
        Math.cos(yaw) * Math.cos(pitch),
        Math.sin(yaw) * Math.cos(pitch),
        Math.sin(pitch)
    );
    if (shouldFlipMarkerX()) {
        dir.x *= -1;
    }
    if (shouldFlipMarkerY()) {
        dir.y *= -1;
    }
    if (dir.lengthSq() < 1e-6) {
        dir.set(0, 1, 0);
    } else {
        dir.normalize();
    }
    const viewDistance = Math.max(5, Number(marker.view_distance || state.config?.markerViewDistance || 20));
    const target = eye.clone().addScaledVector(dir, Math.max(5, viewDistance * 0.5));
    return { position: eye, target };
}

function lockViewToMarker(marker, options = {}) {
    const pose = computeMarkerViewPose(marker);
    if (!pose) {
        return;
    }
    if (!state.viewLockBackup) {
        state.viewLockBackup = {
            cameraPos: camera.position.clone(),
            target: controls.target.clone(),
            controlsEnabled: controls.enabled,
            enableRotate: controls.enableRotate,
            enableZoom: controls.enableZoom,
            enablePan: controls.enablePan,
        };
    }
    const duration = Math.max(0, Number(state.config?.markerViewDurationMs ?? 800));
    const instant = Boolean(options.instant) || duration === 0;
    if (instant) {
        state.viewLockAnimation = null;
        camera.position.copy(pose.position);
        controls.target.copy(pose.target);
        camera.up.set(0, 0, 1);
        camera.lookAt(pose.target);
        return;
    }
    state.viewLockAnimation = {
        start: performance.now(),
        duration,
        fromPos: camera.position.clone(),
        fromTarget: controls.target.clone(),
        toPos: pose.position.clone(),
        toTarget: pose.target.clone(),
    };
}

function updateViewLockAnimation() {
    const anim = state.viewLockAnimation;
    if (!anim) {
        return;
    }
    const now = performance.now();
    const progress = Math.min(1, (now - anim.start) / Math.max(1, anim.duration));
    const eased = progress * progress * (3 - 2 * progress);
    tmpViewPos.copy(anim.fromPos).lerp(anim.toPos, eased);
    tmpViewTarget.copy(anim.fromTarget).lerp(anim.toTarget, eased);
    camera.position.copy(tmpViewPos);
    controls.target.copy(tmpViewTarget);
    camera.up.set(0, 0, 1);
    camera.lookAt(tmpViewTarget);
    if (progress >= 1) {
        const onComplete = anim.onComplete;
        state.viewLockAnimation = null;
        if (typeof onComplete === "function") {
            onComplete();
        }
    }
}

function releaseViewLock(options = {}) {
    const animate = Boolean(options.animate);
    const backup = state.viewLockBackup;
    state.viewLockAnimation = null;
    if (!backup) {
        return;
    }
    const restoreView = () => {
        controls.enabled = backup.controlsEnabled;
        controls.enableRotate = backup.enableRotate;
        controls.enableZoom = backup.enableZoom;
        controls.enablePan = backup.enablePan;
        camera.position.copy(backup.cameraPos);
        controls.target.copy(backup.target);
        camera.up.set(0, 0, 1);
        camera.lookAt(backup.target);
        state.viewLockBackup = null;
    };
    const duration = Math.max(0, Number(state.config?.markerViewDurationMs ?? 800));
    if (!animate || duration === 0) {
        restoreView();
        return;
    }
    state.viewLockAnimation = {
        start: performance.now(),
        duration,
        fromPos: camera.position.clone(),
        fromTarget: controls.target.clone(),
        toPos: backup.cameraPos.clone(),
        toTarget: backup.target.clone(),
        onComplete: restoreView,
    };
}

function buildGlobalColorLookup(geometry) {
    const colorAttr = geometry.getAttribute("color");
    const posAttr = geometry.getAttribute("position");
    if (!colorAttr || !posAttr) {
        state.globalColorLookup = null;
        return;
    }
    const scale = Number(state.config?.markerColorVoxelScale) || 10;
    const map = new Map();
    const posArray = posAttr.array;
    const count = posAttr.count;
    for (let i = 0; i < count; i += 1) {
        const base = i * 3;
        const px = posArray[base + 0];
        const py = posArray[base + 1];
        const pz = posArray[base + 2];
        const key = `${Math.round(px * scale)}|${Math.round(py * scale)}|${Math.round(pz * scale)}`;
        if (!map.has(key)) {
            map.set(key, i);
        }
    }
    state.globalColorLookup = {
        map,
        colors: colorAttr.array,
        scale,
        neighborOffsets: [-1, 0, 1],
    };
}

function lookupGlobalColor(x, y, z) {
    const lookup = state.globalColorLookup;
    if (!lookup) {
        return null;
    }
    const scale = lookup.scale;
    const qx = Math.round(x * scale);
    const qy = Math.round(y * scale);
    const qz = Math.round(z * scale);
    let idx = lookup.map.get(`${qx}|${qy}|${qz}`);
    if (idx === undefined) {
        const offsets = lookup.neighborOffsets || [0];
        for (let ix = 0; ix < offsets.length; ix += 1) {
            for (let iy = 0; iy < offsets.length; iy += 1) {
                for (let iz = 0; iz < offsets.length; iz += 1) {
                    const key = `${qx + offsets[ix]}|${qy + offsets[iy]}|${qz + offsets[iz]}`;
                    idx = lookup.map.get(key);
                    if (idx !== undefined) {
                        break;
                    }
                }
                if (idx !== undefined) break;
            }
            if (idx !== undefined) break;
        }
    }
    if (idx === undefined) {
        return null;
    }
    const base = idx * 3;
    const colors = lookup.colors;
    return [
        colors[base + 0] ?? defaultLocalColorArray[0],
        colors[base + 1] ?? defaultLocalColorArray[1],
        colors[base + 2] ?? defaultLocalColorArray[2],
    ];
}

function createColorArrayForPositions(positionArray) {
    const count = positionArray.length / 3;
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
        const base = i * 3;
        const px = positionArray[base + 0];
        const py = positionArray[base + 1];
        const pz = positionArray[base + 2];
        const color = lookupGlobalColor(px, py, pz) || defaultLocalColorArray;
        colors[base + 0] = color[0];
        colors[base + 1] = color[1];
        colors[base + 2] = color[2];
    }
    return colors;
}

function applyGlobalColorsToGeometry(geometry) {
    if (!state.globalColorLookup || !geometry || !geometry.getAttribute("position")) {
        return false;
    }
    const positions = geometry.getAttribute("position").array;
    const colors = createColorArrayForPositions(positions);
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return true;
}

function setupCameraMarkers(markers) {
    clearCameraMarkers();
    resetToGlobalView();
    if (!Array.isArray(markers) || !markers.length || !state.markerGroup) {
        state.cameraMarkers = [];
        return;
    }
    state.cameraMarkers = markers.map((marker) => {
        const key = marker.key || marker.name || (marker.camera_id != null ? `cam${marker.camera_id}` : `marker-${Date.now()}`);
        const displayPosition = buildMarkerDisplayPosition(marker.position);
        return { ...marker, key, displayPosition };
    });
    state.markerLookup = new Map();
    state.markerObjects = [];
    state.cameraMarkers.forEach((marker) => {
        const sprite = createMarkerSprite(marker);
        if (!sprite) {
            return;
        }
        state.markerGroup.add(sprite);
        state.markerObjects.push(sprite);
        state.markerLookup.set(marker.key, marker);
    });
}

function clearCameraMarkers() {
    if (!state.markerGroup) {
        return;
    }
    const children = [...state.markerGroup.children];
    children.forEach((child) => {
        state.markerGroup.remove(child);
        if (child.material) {
            if (child.material.map && typeof child.material.map.dispose === "function") {
                child.material.map.dispose();
            }
            if (typeof child.material.dispose === "function") {
                child.material.dispose();
            }
        }
    });
    state.markerObjects = [];
    if (state.markerLookup && typeof state.markerLookup.clear === "function") {
        state.markerLookup.clear();
    }
    state.markerLookup = new Map();
    state.cameraMarkers = [];
}

function createMarkerSprite(marker) {
    if (!state.markerGroup) {
        return null;
    }
    const label = (marker.name || marker.key || (marker.camera_id != null ? `cam${marker.camera_id}` : "?")).toString();
    const textures = createMarkerTextures(label);
    const material = new THREE.SpriteMaterial({
        map: textures.base,
        transparent: true,
        depthTest: false,
        depthWrite: false,
    });
    const sprite = new THREE.Sprite(material);
    const baseScale = 3.0;
    const activeScale = 3.8;
    sprite.scale.set(baseScale, baseScale, 1);
    const pos = marker.displayPosition || marker.position || {};
    const x = Number(pos.x ?? pos[0]) || 0;
    const y = Number(pos.y ?? pos[1]) || 0;
    const z = Number(pos.z ?? pos[2]) || 0;
    sprite.position.set(x, y, z + 2.5);
    sprite.renderOrder = 2;
    sprite.userData.marker = marker;
    sprite.userData.baseTexture = textures.base;
    sprite.userData.highlightTexture = textures.highlight;
    sprite.userData.baseScale = baseScale;
    sprite.userData.activeScale = activeScale;
    marker.sprite = sprite;
    return sprite;
}

function createMarkerTextures(label) {
    return {
        base: drawMarkerTexture(label, "#ff9f1c"),
        highlight: drawMarkerTexture(label, "#2ec4b6"),
    };
}

function drawMarkerTexture(label, color) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(128, 128, 110, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.lineWidth = 8;
    ctx.strokeStyle = "rgba(0, 0, 0, 0.35)";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    const len = label.length;
    const fontSize = len <= 3 ? 110 : len <= 6 ? 80 : 60;
    ctx.font = `bold ${fontSize}px 'Segoe UI', Arial, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, 128, 138);
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    return texture;
}

function handleMarkerClick(marker) {
    if (!marker) {
        return;
    }
    const markerCamId = marker.camera_id ?? marker.id ?? marker.cameraId;
    const numericCamId = Number(markerCamId);
    if (state.activeMarkerKey === marker.key) {
        resetToGlobalView();
        if (isLive) {
            enterLiveGlobalView();
        } else if (isFusion) {
            enterFusionGlobalView();
        }
        return;
    }
    state.activeMarkerKey = marker.key;
    updateMarkerHighlight(marker.key);
    if (isLive || isFusion) {
        setLayoutMode("local");
    }
    if (Number.isFinite(numericCamId)) {
        if (isLive) {
            selectLiveCamera(numericCamId);
        } else if (isFusion) {
            selectFusionCamera(numericCamId);
        }
    }
    if (!marker.local_ply_url && !marker.local_visible_url) {
        lockViewToMarker(marker);
        if (statusEl) statusEl.textContent = `${marker.name || marker.key} | local cloud unavailable`;
        setLocalCloudVisibility(null);
        if (state.globalCloud) {
            state.globalCloud.visible = true;
        }
        return;
    }
    showLocalCloudForMarker(marker);
}

async function showLocalCloudForMarker(marker) {
    if (!marker.local_ply_url && !marker.local_visible_url) {
        return;
    }
    state.localLoadToken += 1;
    const token = state.localLoadToken;
    if (statusEl) statusEl.textContent = `Loading ${marker.name || marker.key}…`;
    try {
        let mesh = state.localClouds.get(marker.key);
        if (!mesh) {
            const flipX = shouldFlipMarkerX();
            const flipY = shouldFlipMarkerY();
            if (marker.local_visible_url) {
                const stride = Number(marker.local_visible_stride) || 3;
                mesh = await loadVisiblePointsFromBinary(marker.local_visible_url, stride, { flipX, flipY });
            } else {
                mesh = await loadPointCloudFromPly(marker.local_ply_url, { flipX, flipY });
            }
            mesh.name = `LocalCloud-${marker.key}`;
            mesh.userData.markerKey = marker.key;
            scene.add(mesh);
            state.localClouds.set(marker.key, mesh);
        }
        if (token !== state.localLoadToken) {
            return;
        }
        if (state.globalCloud) {
            state.globalCloud.visible = false;
        }
        setLocalCloudVisibility(marker.key);
        lockViewToMarker(marker);
        if (statusEl) statusEl.textContent = `${marker.name || marker.key} | local cloud`;
    } catch (err) {
        console.warn("local cloud load failed", err);
        if (statusEl) statusEl.textContent = `Local cloud error: ${err?.message || err}`;
        resetToGlobalView();
        if (isLive) {
            enterLiveGlobalView();
        } else if (isFusion) {
            enterFusionGlobalView();
        }
    }
}

function setLocalCloudVisibility(activeKey) {
    state.localClouds.forEach((mesh, key) => {
        mesh.visible = Boolean(activeKey && key === activeKey);
    });
    if (!activeKey && state.globalCloud) {
        state.globalCloud.visible = true;
    }
}

function resetToGlobalView(options = {}) {
    const animate = Boolean(options.animate);
    if (isLive || isFusion) {
        setLayoutMode("global");
    }
    state.activeMarkerKey = null;
    clearDetectionFollow();
    state.localLoadToken += 1;
    updateMarkerHighlight(null);
    setLocalCloudVisibility(null);
    if (state.globalCloud) {
        state.globalCloud.visible = true;
    }
    releaseViewLock({ animate });
}

function updateMarkerHighlight(activeKey) {
    state.markerLookup.forEach((marker) => {
        const sprite = marker.sprite;
        if (!sprite) {
            return;
        }
        const isActive = marker.key === activeKey;
        const targetTexture = isActive ? sprite.userData.highlightTexture : sprite.userData.baseTexture;
        if (targetTexture && sprite.material.map !== targetTexture) {
            sprite.material.map = targetTexture;
            sprite.material.needsUpdate = true;
        }
        const scale = isActive ? sprite.userData.activeScale : sprite.userData.baseScale;
        if (scale) {
            sprite.scale.set(scale, scale, 1);
        }
    });
}

function setupMarkerPointerHandlers() {
    const canvas = renderer.domElement;
    canvas.addEventListener("pointerdown", handleViewerPointerDown);
    canvas.addEventListener("pointerup", handleViewerPointerUp);
}

function handleViewerPointerDown(evt) {
    if (evt.button !== 0) {
        return;
    }
    markerPointerState.isDown = true;
    markerPointerState.x = evt.clientX;
    markerPointerState.y = evt.clientY;
}

function handleViewerPointerUp(evt) {
    if (!markerPointerState.isDown || evt.button !== 0) {
        return;
    }
    markerPointerState.isDown = false;
    const dx = Math.abs(evt.clientX - markerPointerState.x);
    const dy = Math.abs(evt.clientY - markerPointerState.y);
    if (dx > 3 || dy > 3) {
        return;
    }
    performMarkerHitTest(evt.clientX, evt.clientY);
}

function performMarkerHitTest(clientX, clientY) {
    if (!state.markerObjects.length) {
        return;
    }
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObjects(state.markerObjects, false);
    if (!hits.length) {
        return;
    }
    const hit = hits[0];
    const marker = hit.object?.userData?.marker;
    if (marker) {
        handleMarkerClick(marker);
    }
}

function handleGlobalKeydown(evt) {
    if (evt.key === "Escape") {
        const animate = Boolean(state.viewLockBackup);
        clearDetectionFollow();
        resetToGlobalView({ animate });
        if (isLive) {
            enterLiveGlobalView();
        } else if (isFusion) {
            enterFusionGlobalView();
        }
    }
}

function loadPointCloudFromPly(url, options = {}) {
    const { flipX = false, flipY = false } = options;
    return new Promise((resolve, reject) => {
        const loader = new PLYLoader();
        loader.load(
            url,
            (geometry) => {
                const flipPlyY = Boolean(state.config?.flipPlyY);
                if (flipPlyY) {
                    geometry.scale(1, -1, 1);
                }
                if (flipX) {
                    geometry.scale(-1, 1, 1);
                }
                if (flipY) {
                    geometry.scale(1, -1, 1);
                }
                geometry.computeBoundingBox();
                geometry.computeBoundingSphere();
                let hasColor = Boolean(geometry.getAttribute("color"));
                if (!hasColor && state.globalColorLookup) {
                    hasColor = applyGlobalColorsToGeometry(geometry);
                }
                const material = new THREE.PointsMaterial({
                    size: 0.05,
                    vertexColors: hasColor,
                    color: hasColor ? 0xffffff : 0x4cc9f0,
                    transparent: true,
                    opacity: 0.95,
                });
                const mesh = new THREE.Points(geometry, material);
                mesh.visible = false;
                resolve(mesh);
            },
            undefined,
            (error) => reject(error)
        );
    });
}

async function loadVisiblePointsFromBinary(url, stride = 3, options = {}) {
    const { flipX = false, flipY = false } = options;
    const response = await fetch(`${url}?cacheBust=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    const data = new Float32Array(buffer);
    if (!data.length) {
        throw new Error("empty visible cloud");
    }
    const step = Math.max(1, stride);
    const count = Math.floor(data.length / step);
    if (!count) {
        throw new Error("invalid visible stride");
    }
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
        const base = i * step;
        positions[i * 3 + 0] = data[base + 0] ?? 0;
        positions[i * 3 + 1] = data[base + 1] ?? 0;
        positions[i * 3 + 2] = data[base + 2] ?? 0;
    }
    if (flipX || flipY) {
        for (let i = 0; i < count; i += 1) {
            if (flipX) {
                positions[i * 3 + 0] = -positions[i * 3 + 0];
            }
            if (flipY) {
                positions[i * 3 + 1] = -positions[i * 3 + 1];
            }
        }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    let hasColor = false;
    if (state.globalColorLookup) {
        const colorArray = createColorArrayForPositions(positions);
        geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
        hasColor = true;
    }
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({
        size: 0.05,
        color: hasColor ? 0xffffff : 0xf7b801,
        vertexColors: hasColor,
        transparent: true,
        opacity: 0.95,
    });
    const mesh = new THREE.Points(geometry, material);
    mesh.visible = false;
    return mesh;
}

function getDetectionColorText(det) {
    const rawLabel = typeof det?.color === "string"
        ? det.color.trim()
        : (typeof det?.color_label === "string"
            ? det.color_label.trim()
            : (typeof det?.colorLabel === "string" ? det.colorLabel.trim() : ""));
    if (rawLabel) {
        return rawLabel;
    }
    const rawHex = typeof det?.color_hex === "string"
        ? det.color_hex.trim()
        : (typeof det?.colorHex === "string" ? det.colorHex.trim() : "");
    if (rawHex) {
        return rawHex.startsWith("#") ? rawHex : `#${rawHex}`;
    }
    return "-";
}

function getDetectionCenter(det) {
    const centerArr = Array.isArray(det?.center) && det.center.length === 3 ? det.center : [0, 0, 0];
    return {
        x: Number(centerArr[0]) || 0,
        y: Number(centerArr[1]) || 0,
        z: Number(centerArr[2]) || 0,
    };
}

function getDetectionSpan(det) {
    if (Array.isArray(det?.scale) && det.scale.length === 3) {
        return Math.max(Number(det.scale[0]) || 0, Number(det.scale[1]) || 0, Number(det.scale[2]) || 0, 1) * 2;
    }
    const dims = [
        Number(det?.length) || 0,
        Number(det?.width) || 0,
        Number(det?.height) || 0,
    ];
    const span = Math.max(...dims, 6);
    return Number.isFinite(span) ? span : 12;
}

function formatDetectionListEntry(det, idx) { // 여기서 라벨에 띄우고 싶은 거 수정
    const center = Array.isArray(det.center) && det.center.length === 3
        ? det.center
        : [0, 0, 0];
    const trackText = det.track_id != null
        ? `track ${det.track_id}`
        : (det.cam != null ? `cam ${det.cam}` : `class ${det.class_id ?? "-"}`);
    const colorText = getDetectionColorText(det);
    return `#${idx + 1} | ${trackText} | x ${center[0].toFixed(2)} | y ${center[1].toFixed(2)} | color ${colorText}`;
}

function findFollowDetection(detections, target) {
    if (!Array.isArray(detections) || !detections.length || !target) {
        return null;
    }
    if (target.trackId != null) {
        const byTrack = detections.find((d) => d && d.track_id === target.trackId);
        if (byTrack) return byTrack;
    }
    if (target.cam != null) {
        const byCam = detections.find((d) => d && d.cam === target.cam);
        if (byCam) return byCam;
    }
    if (typeof target.index === "number" && detections[target.index]) {
        return detections[target.index];
    }
    return null;
}

function startDetectionFollow(det, index = null) {
    if (!det) {
        return;
    }
    const center = getDetectionCenter(det);
    const span = getDetectionSpan(det);
    let baseOffset = camera.position.clone().sub(controls.target);
    const desiredLen = Math.max(span * 1.2, 3);
    if (!Number.isFinite(baseOffset.length()) || baseOffset.length() < 0.5 || baseOffset.length() > desiredLen * 2.5) {
        baseOffset = new THREE.Vector3(span * 0.8, -span * 0.8, span * 0.6);
    } else {
        baseOffset.normalize().multiplyScalar(desiredLen);
        baseOffset.z = Math.max(baseOffset.z, span * 0.5);
    }
    if (state.followPanBackup === null) {
        state.followPanBackup = controls.enablePan;
    }
    controls.enablePan = false;
    state.followTarget = {
        trackId: det.track_id ?? null,
        cam: det.cam ?? null,
        index,
        offset: baseOffset.clone(),
        lastCenter: center,
    };
    controls.target.set(center.x, center.y, center.z);
    camera.position.copy(new THREE.Vector3(center.x, center.y, center.z).add(baseOffset));
    camera.up.set(0, 0, 1);
    camera.lookAt(controls.target);
}

function clearDetectionFollow() {
    state.followTarget = null;
    if (state.followPanBackup !== null) {
        controls.enablePan = state.followPanBackup;
        state.followPanBackup = null;
    }
}

function updateFollowOffset() {
    const follow = state.followTarget;
    if (!follow) {
        return;
    }
    follow.offset = camera.position.clone().sub(controls.target);
}

function updateDetectionFollowPose(detections) {
    const follow = state.followTarget;
    if (!follow) {
        return;
    }
    const match = findFollowDetection(detections, follow);
    if (!match) {
        clearDetectionFollow();
        return;
    }
    const center = getDetectionCenter(match);
    const offset = follow.offset || camera.position.clone().sub(controls.target);
    controls.target.set(center.x, center.y, center.z);
    camera.position.set(center.x + offset.x, center.y + offset.y, center.z + offset.z);
    camera.up.set(0, 0, 1);
    camera.lookAt(controls.target);
    follow.lastCenter = center;
}

init().catch((err) => console.error("Initialization error:", err));
