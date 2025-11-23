import * as THREE from "./libs/three.module.js";
import { OrbitControls } from "./libs/OrbitControls.js";
import { PLYLoader } from "./libs/PLYLoader.js";
import { GLTFLoader } from "./libs/GLTFLoader.js";

const mode = window.VIEW_MODE || "playback";
const isLive = mode === "live";
const isPlayback = mode === "playback";
const isFusion = mode.startsWith("fusion");
const fusionSource = isFusion ? mode.replace("fusion_", "") : null;
const adminModeEnabled = (() => {
    if (typeof window.ADMIN_MODE === "boolean") {
        return Boolean(window.ADMIN_MODE);
    }
    return mode === "fusion_admin";
})();
const adminAutoOpen = Boolean(window.ADMIN_AUTO_OPEN);

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
const adminToggleBtnEl = document.getElementById("admin-toggle-btn");

const ADMIN_COLORS = ["red", "pink", "green", "white", "yellow", "purple", "none"];
const ADMIN_HOTKEY = { key: "a", ctrl: true, shift: true };
const adminState = {
    enabled: adminModeEnabled,
    initialized: false,
    open: false,
    refreshHandle: null,
    elements: {
        panel: null,
        toggle: null,
        list: null,
        status: null,
        trackInput: null,
        colorSelect: null,
        yawInput: null,
    },
};

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
    markerModel: null,
    markerModelPromise: null,
    globalHeightLookup: null,
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
    initialViewApplied: false,
    markerPulseStart: performance.now(),
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
const markerModelUrl = "/static/assets/street_lamp_hanging.glb";
const markerTargetHeight = 3.2;          // 가로등 높이 목표 (GLB 스케일링용)
const markerActiveScaleMult = 1.06;      // 선택 시 약간만 확대
const markerDefaultScale = 3;          // 기본 가로등 스케일 (전체 그룹)
const markerLabelBaseScale = 0.75;       // 라벨 기본 크기
const markerLabelActiveScale = 0.9;      // 라벨 활성 크기
const markerLabelOffset = 2.0;           // 라벨을 올려놓을 기본 높이(마커 z 기준)
const markerLabelBaseColor = "#ff9f1c";
const markerLabelHighlightColor = "#2ec4b6";
const markerBulbOffsetLocal = { x: 0.85, y: 0.0 }; // streetLightMaker 기준 전구 중심 오프셋(+X 방향이 전면)
const markerBackOffset = 5.0;               // 로컬 시점에서 가로등을 뒤로 살짝 빼는 거리
const markerRingBaseRadius = 1.5;
const markerRingColor = 0xffffff;
const markerRingBaseOpacity = 0.32;
const markerRingZOffset = 0.05;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0b0b);

const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 1000);
camera.up.set(0, 0, 1);
camera.position.set(0, -20, 12);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(viewerEl.clientWidth, viewerEl.clientHeight);
renderer.outputEncoding = THREE.sRGBEncoding;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.9;
viewerEl.appendChild(renderer.domElement);

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const markerPointerState = { isDown: false, x: 0, y: 0 };

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.85);
scene.add(ambient);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
dirLight.position.set(15, -20, 25);
dirLight.castShadow = false;
scene.add(dirLight);

const hemiLight = new THREE.HemisphereLight(0xeef7ff, 0x252525, 1.0);
scene.add(hemiLight);

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
const paletteColor = new THREE.Color();
const gradientLowColor = new THREE.Color(0x55607b);
const gradientHighColor = new THREE.Color(0xffffff);
const gradientMixColor = new THREE.Color();
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
window.addEventListener("touchend", handleGlobalTouchEnd, { passive: true });

let lastTouchEndTs = 0;

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
    await setupCameraMarkers(state.site?.camera_positions || []);

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
    const showSceneAxes = Boolean(state.config?.showSceneAxes);
    grid.visible = showSceneAxes;
    axes.visible = showSceneAxes;
    try {
        state.site = await fetchJson("/api/site");
    } catch (err) {
        console.warn("site info error", err);
    }
    state.cameras = deriveCamerasFromSite(state.site);
    await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
    await setupCameraMarkers(state.site?.camera_positions || []);
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
    await setupCameraMarkers(state.site?.camera_positions || []);

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
    updateMarkerGlow();
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
                buildGlobalHeightLookup(geometry);

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
                applyInitialViewOverride();
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
                        const srcMaterial = Array.isArray(obj.material)
                            ? (obj.material[0] || {})
                            : (obj.material || {});
                        const templateMaterial = new THREE.MeshBasicMaterial({
                            color: 0xffffff,
                            transparent: Boolean(srcMaterial.transparent),
                            opacity: srcMaterial.opacity !== undefined ? srcMaterial.opacity : 1.0,
                            side: srcMaterial.side ?? THREE.FrontSide,
                            depthWrite: srcMaterial.depthWrite !== undefined ? srcMaterial.depthWrite : true,
                            depthTest: srcMaterial.depthTest !== undefined ? srcMaterial.depthTest : true,
                        });
                        if (srcMaterial.map) templateMaterial.map = srcMaterial.map;
                        const positions = obj.geometry.getAttribute("position");
                        const vertexCount = positions.count;
                        const colorAttr = new Float32Array(vertexCount * 3);
                        obj.geometry.computeBoundingBox();
                        const bbox = obj.geometry.boundingBox;
                        const minZ = bbox ? bbox.min.z : -0.5;
                        const maxZ = bbox ? bbox.max.z : 0.5;
                        const heightRange = Math.max(maxZ - minZ, 1e-6);
                        const stride = positions.itemSize || 3;
                        const posArray = positions.array;
                        for (let i = 0; i < vertexCount; i += 1) {
                            const base = i * stride;
                            const zVal = posArray[base + 2] ?? 0;
                            const tRaw = (zVal - minZ) / heightRange;
                            const t = Math.min(Math.max(tRaw, 0), 1);
                            const smooth = Math.pow(t, 0.7);
                            gradientMixColor.copy(gradientLowColor).lerp(gradientHighColor, smooth);
                            colorAttr[i * 3 + 0] = gradientMixColor.r;
                            colorAttr[i * 3 + 1] = gradientMixColor.g;
                            colorAttr[i * 3 + 2] = gradientMixColor.b;
                        }
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

function getDetectionColorInfo(det) {
    if (!det) {
        return null;
    }
    const label = typeof det.color === "string" ? det.color.trim() : "";
    let hex = typeof det.color_hex === "string"
        ? det.color_hex.trim()
        : (typeof det.colorHex === "string" ? det.colorHex.trim() : "");
    if (hex && !hex.startsWith("#")) {
        hex = `#${hex}`;
    }
    if (hex && !/^#([0-9a-fA-F]{6})$/.test(hex)) {
        hex = "";
    }
    if (!label && !hex) {
        return null;
    }
    return { label, hex };
}

function getDetectionColor(det) {
    const info = getDetectionColorInfo(det);
    if (!info || !info.hex) {
        return null;
    }
    try {
        tmpColor.set(info.hex);
        return tmpColor;
    } catch (err) {
        return null;
    }
}

function hashString(value) {
    if (value == null) {
        return 0;
    }
    const str = String(value);
    let hash = 0;
    for (let i = 0; i < str.length; i += 1) {
        hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
    }
    return hash;
}

function getFallbackVehicleColor(det, idx) {
    const candidates = [
        det?.track_id,
        det?.id,
        det?.cam,
        det?.class_id,
        det?.camera_id,
    ];
    let seed = idx * 131;
    for (let i = 0; i < candidates.length; i += 1) {
        const val = candidates[i];
        if (Number.isFinite(val)) {
            seed = Number(val);
            break;
        }
        if (typeof val === "string" && val.length) {
            seed = hashString(val);
            break;
        }
    }
    const hue = ((seed % 360) + 360) % 360 / 360;
    paletteColor.setHSL(hue, 0.55, 0.55);
    return paletteColor;
}

function renderDetections(detections) {
    detections = filterDetectionsForLocalView(detections);
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
                    const colorToApply = detColor || getFallbackVehicleColor(det, i) || tpl.baseColor || defaultVehicleColor;
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
                prefillAdminFromDetection(det);
            });
            li.appendChild(btn);

            const colorInfo = getDetectionColorInfo(det);
            if (colorInfo) {
                const wrapper = document.createElement("span");
                wrapper.style.display = "inline-flex";
                wrapper.style.alignItems = "center";
                wrapper.style.gap = "4px";
                wrapper.style.marginLeft = "6px";
                const swatch = document.createElement("span");
                swatch.style.display = "inline-block";
                swatch.style.width = "12px";
                swatch.style.height = "12px";
                swatch.style.borderRadius = "2px";
                swatch.style.border = "1px solid rgba(255,255,255,0.4)";
                swatch.style.backgroundColor = colorInfo.hex || "#000000";
                swatch.title = colorInfo.hex || colorInfo.label;
                wrapper.appendChild(swatch);
                const labelSpan = document.createElement("span");
                labelSpan.textContent = colorInfo.label || colorInfo.hex;
                wrapper.appendChild(labelSpan);
                li.appendChild(wrapper);
            }

            detectionListEl.appendChild(li);
        });
    }

    updateDetectionFollowPose(detections);
}

function parseDetectionCamId(det) {
    if (!det) return null;
    if (det.camera_id != null && Number.isFinite(Number(det.camera_id))) {
        return Number(det.camera_id);
    }
    if (det.cam != null) {
        const text = `${det.cam}`;
        const m = text.match(/\d+/);
        if (m) {
            const n = Number(m[0]);
            if (Number.isFinite(n)) return n;
        }
    }
    const sources = det.source_cams || det.sources;
    if (Array.isArray(sources) && sources.length) {
        for (const s of sources) {
            const m = `${s}`.match(/\d+/);
            if (m) {
                const n = Number(m[0]);
                if (Number.isFinite(n)) return n;
            }
        }
    }
    return null;
}

function filterDetectionsForLocalView(detections) {
    if (!state.activeMarkerKey || state.layoutMode !== "local") {
        return detections;
    }
    const marker = state.markerLookup.get(state.activeMarkerKey);
    if (!marker) {
        return detections;
    }
    const markerCamId = marker.camera_id ?? marker.id;
    if (!Number.isFinite(markerCamId)) {
        return detections;
    }
    if (!Array.isArray(detections)) {
        return [];
    }
    return detections.filter((det) => {
        const camId = parseDetectionCamId(det);
        return camId === markerCamId;
    });
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

function getPollingIntervalMs(defaultMs = 400) {
    const override = state.config?.fusionPollIntervalMs ?? state.config?.pollIntervalMs;
    const explicit = Number(override);
    if (Number.isFinite(explicit) && explicit > 0) {
        return explicit;
    }
    const fps = Number(state.config?.fps);
    if (Number.isFinite(fps) && fps > 0) {
        return Math.max(50, 1000 / fps);
    }
    return defaultMs;
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
            state.pollingHandle = window.setTimeout(poll, getPollingIntervalMs());
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

function formatLocalTimestamp(ts) {
    if (typeof ts !== "number" || Number.isNaN(ts)) {
        return "";
    }
    try {
        const date = new Date(ts * 1000);
        const datePart = date.toLocaleDateString();
        const timePart = date.toLocaleTimeString(undefined, { hour12: false });
        const ms = date.getMilliseconds().toString().padStart(3, "0");
        return `${datePart} ${timePart}.${ms}`;
    } catch (err) {
        console.warn("formatLocalTimestamp failed", err);
        return "";
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
        const tsRaw = typeof ts === "number" ? ts.toFixed(3) : ts ?? "";
        const tsLocal = formatLocalTimestamp(typeof ts === "number" ? ts : Number(ts));
        const localPart = tsLocal ? ` (${tsLocal})` : "";
        statusEl.textContent = `cam${camId} | detections ${detCount} | ts ${tsRaw}${localPart}`;
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
            state.pollingHandle = window.setTimeout(poll, getPollingIntervalMs());
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
        const tsRaw = typeof ts === "number" ? ts.toFixed(3) : ts ?? "";
        const tsLocal = formatLocalTimestamp(typeof ts === "number" ? ts : Number(ts));
        const localPart = tsLocal ? ` (${tsLocal})` : "";
        statusEl.textContent = `${source.toUpperCase()} | count ${detections.length} | ts ${tsRaw}${localPart}`;
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

function parseViewVector(vec) {
    if (Array.isArray(vec) && vec.length === 3) {
        const [x, y, z] = vec.map((v) => Number(v));
        if ([x, y, z].every((v) => Number.isFinite(v))) {
            return { x, y, z };
        }
    } else if (vec && typeof vec === "object") {
        const x = Number(vec.x);
        const y = Number(vec.y);
        const z = Number(vec.z);
        if ([x, y, z].every((v) => Number.isFinite(v))) {
            return { x, y, z };
        }
    }
    return null;
}

function computeInitialViewPose() {
    const target = parseViewVector(state.config?.initialViewTarget);
    const offset = parseViewVector(state.config?.initialViewOffset);
    if (!target && !offset) {
        return null;
    }
    const tgt = target || { x: controls.target.x, y: controls.target.y, z: controls.target.z };
    const off = offset || { x: 18, y: -18, z: 12 };
    const eye = {
        x: tgt.x + off.x,
        y: tgt.y + off.y,
        z: tgt.z + off.z,
    };
    const minAboveTarget = Math.max(1.0, Math.abs(off.z) * 0.5, 3.0);
    if (eye.z <= tgt.z + 0.1) {
        eye.z = tgt.z + minAboveTarget;
    }
    return {
        target: new THREE.Vector3(tgt.x, tgt.y, tgt.z),
        eye: new THREE.Vector3(eye.x, eye.y, eye.z),
    };
}

function applyInitialViewOverride(force = false, options = {}) {
    if (state.initialViewApplied && !force) {
        return;
    }
    const pose = computeInitialViewPose();
    if (!pose) {
        return;
    }
    const { target, eye } = pose;
    const setPose = () => {
        controls.target.copy(target);
        camera.position.copy(eye);
        camera.up.set(0, 0, 1);
        camera.lookAt(target);
        state.initialViewApplied = true;
    };
    const animate = Boolean(options.animate);
    const duration = Math.max(0, Number(options.duration ?? state.config?.globalResetDurationMs ?? 900));
    if (!animate || duration === 0) {
        setPose();
        return;
    }
    state.viewLockAnimation = {
        start: performance.now(),
        duration,
        fromPos: camera.position.clone(),
        fromTarget: controls.target.clone(),
        toPos: eye.clone(),
        toTarget: target.clone(),
        onComplete: setPose,
    };
}

function shouldFlipMarkerX() {
    return Boolean(state.config?.flipMarkerX);
}

function shouldFlipMarkerY() {
    return Boolean(state.config?.flipMarkerY);
}

function computeMarkerYawRad(marker) {
    const rot = marker?.rotation || {};
    let yaw = THREE.MathUtils.degToRad(rot.yaw || 0);
    let dirX = Math.cos(yaw);
    let dirY = Math.sin(yaw);
    if (shouldFlipMarkerX()) {
        dirX *= -1;
    }
    if (shouldFlipMarkerY()) {
        dirY *= -1;
    }
    if (!Number.isFinite(dirX) || !Number.isFinite(dirY) || (dirX === 0 && dirY === 0)) {
        return 0;
    }
    return Math.atan2(dirY, dirX);
}

function rotateOffset(offset, yawRad) {
    const c = Math.cos(yawRad);
    const s = Math.sin(yawRad);
    return {
        x: offset.x * c - offset.y * s,
        y: offset.x * s + offset.y * c,
    };
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
    return { x, y, z};
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
    const yawFromDir = Math.atan2(dir.y, dir.x);
    if (dir.lengthSq() < 1e-6) {
        dir.set(0, 1, 0);
    } else {
        dir.normalize();
    }
    const viewDistance = Math.max(5, Number(marker.view_distance || state.config?.markerViewDistance || 20));
    const target = eye.clone().addScaledVector(dir, Math.max(5, viewDistance * 0.5));
    return { position: eye, target, yawRad: yawFromDir };
}

function lockViewToMarker(marker, options = {}) {
    const pose = computeMarkerViewPose(marker);
    if (!pose) {
        return;
    }
    if (pose.yawRad != null && marker && marker.object) {
        marker.object.rotation.set(0, 0, pose.yawRad);
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
    const preservePose = Boolean(options.preservePose);
    const backup = state.viewLockBackup;
    state.viewLockAnimation = null;
    if (!backup) {
        return;
    }
    const restoreView = (applyPose = true) => {
        controls.enabled = backup.controlsEnabled;
        controls.enableRotate = backup.enableRotate;
        controls.enableZoom = backup.enableZoom;
        controls.enablePan = backup.enablePan;
        if (applyPose) {
            camera.position.copy(backup.cameraPos);
            controls.target.copy(backup.target);
            camera.up.set(0, 0, 1);
            camera.lookAt(backup.target);
        }
        state.viewLockBackup = null;
    };
    if (preservePose) {
        restoreView(false);
        return;
    }
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

function buildGlobalHeightLookup(geometry) {
    const posAttr = geometry.getAttribute("position");
    if (!posAttr) {
        state.globalHeightLookup = null;
        return;
    }
    const scale = Number(state.config?.markerHeightVoxelScale) || Number(state.config?.markerColorVoxelScale) || 10;
    const map = new Map();
    const posArray = posAttr.array;
    const count = posAttr.count;
    for (let i = 0; i < count; i += 1) {
        const base = i * 3;
        const px = posArray[base + 0];
        const py = posArray[base + 1];
        const pz = posArray[base + 2];
        const key = `${Math.round(px * scale)}|${Math.round(py * scale)}`;
        const existing = map.get(key);
        if (existing === undefined || pz < existing) {
            map.set(key, pz);
        }
    }
    state.globalHeightLookup = {
        map,
        scale,
        neighborOffsets: [-2, -1, 0, 1, 2],
    };
}

function lookupGlobalHeight(x, y) {
    const lookup = state.globalHeightLookup;
    if (!lookup) {
        return null;
    }
    const qx = Math.round(x * lookup.scale);
    const qy = Math.round(y * lookup.scale);
    const offsets = lookup.neighborOffsets || [0];
    let bestZ = null;
    for (let ix = 0; ix < offsets.length; ix += 1) {
        for (let iy = 0; iy < offsets.length; iy += 1) {
            const key = `${qx + offsets[ix]}|${qy + offsets[iy]}`;
            if (lookup.map.has(key)) {
                const z = lookup.map.get(key);
                if (bestZ === null || z < bestZ) {
                    bestZ = z;
                }
            }
        }
    }
    return bestZ;
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

async function ensureMarkerModel() {
    if (state.markerModel) {
        return state.markerModel;
    }
    if (state.markerModelPromise) {
        return state.markerModelPromise;
    }
    const loader = new GLTFLoader();
    state.markerModelPromise = new Promise((resolve) => {
        loader.load(
            markerModelUrl,
            (gltf) => {
                const content = gltf.scene || (gltf.scenes && gltf.scenes[0]);
                if (!content) {
                    console.warn("Marker model missing content");
                    state.markerModel = null;
                    resolve(null);
                    return;
                }
                const model = content.clone(true);
                const bbox = new THREE.Box3().setFromObject(model);
                const size = new THREE.Vector3();
                bbox.getSize(size);
                const center = new THREE.Vector3();
                bbox.getCenter(center);
    model.traverse((obj) => {
        if (obj.isMesh) {
            if ("frustumCulled" in obj) {
                obj.frustumCulled = false;
            }
            obj.castShadow = false;
            obj.receiveShadow = false;
        }
    });
                model.position.sub(new THREE.Vector3(center.x, center.y, bbox.min.z));
                const height = Math.max(size.z, 1e-3);
                const targetHeight = markerTargetHeight;
                const scale = targetHeight / height;
                const holder = new THREE.Group();
                holder.name = "MarkerModelTemplate";
                holder.add(model);
                holder.scale.setScalar(scale);
                holder.updateMatrixWorld(true);
                state.markerModelScale = scale;
                state.markerModel = holder;
                resolve(holder);
            },
            undefined,
            (err) => {
                console.warn("Marker model load failed", err);
                state.markerModel = null;
                resolve(null);
            }
        );
    });
    return state.markerModelPromise;
}

function cloneMarkerModel() {
    if (!state.markerModel) {
        return null;
    }
    const clone = state.markerModel.clone(true);
    clone.traverse((obj) => {
        if (obj.isMesh && obj.material && obj.material.clone) {
            obj.material = obj.material.clone();
        }
    });
    return clone;
}

function createMarkerLabelSprite(textures, marker) {
    const material = new THREE.SpriteMaterial({
        map: textures.base,
        transparent: true,
        depthTest: false,
        depthWrite: false,
        sizeAttenuation: true,
        opacity: 0.85,
    });
    const sprite = new THREE.Sprite(material);
    sprite.userData.baseTexture = textures.base;
    sprite.userData.highlightTexture = textures.highlight;
    sprite.userData.baseScale = markerLabelBaseScale;
    sprite.userData.activeScale = markerLabelActiveScale;
    sprite.userData.baseOpacity = material.opacity;
    sprite.scale.set(markerLabelBaseScale, markerLabelBaseScale, 1);
    const offsetRaw = marker?.label_offset ?? marker?.labelHeight ?? marker?.label_height;
    const offset = Number(offsetRaw);
    const finalOffset = Number.isFinite(offset) ? offset : markerLabelOffset;
    sprite.position.set(0, 0, finalOffset);
    sprite.renderOrder = 2;
    return sprite;
}

function createMarkerFloorRing() {
    const inner = Math.max(0.1, markerRingBaseRadius * 0.65);
    const geometry = new THREE.RingGeometry(inner, markerRingBaseRadius, 48);
    const material = new THREE.MeshBasicMaterial({
        color: markerRingColor,
        transparent: true,
        opacity: markerRingBaseOpacity,
        side: THREE.DoubleSide,
        depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = 0; // keep the ring flat on the ground plane
    mesh.position.set(0, 0, markerRingZOffset);
    mesh.renderOrder = 1;
    mesh.userData.baseOpacity = markerRingBaseOpacity;
    mesh.userData.baseScale = 1;
    return mesh;
}

async function createMarkerSprite(marker) {
    const label = (marker.name || marker.key || (marker.camera_id != null ? `cam${marker.camera_id}` : "?")).toString();
    const textures = createMarkerTextures(label);
    const labelSprite = createMarkerLabelSprite(textures, marker);
    let modelGroup = null;
    try {
        await ensureMarkerModel();
        modelGroup = cloneMarkerModel();
    } catch (err) {
        console.warn("Marker model clone failed", err);
    }
    const group = new THREE.Group();
    group.name = `Marker-${label}`;
    if (modelGroup) {
        group.add(modelGroup);
    }
    const ring = createMarkerFloorRing();
    group.add(ring);
    group.add(labelSprite);
    const configScale = Number(state.config?.markerModelScale);
    const baseScale = Number.isFinite(configScale) ? configScale : markerDefaultScale;
    const activeScale = baseScale * markerActiveScaleMult;
    group.scale.setScalar(baseScale);
    marker.baseScale = baseScale;
    marker.activeScale = activeScale;
    marker.labelSprite = labelSprite;
    marker.baseTexture = textures.base;
    marker.highlightTexture = textures.highlight;
    marker.sprite = labelSprite;
    marker.ringMesh = ring;
    marker.object = group;
    labelSprite.userData.marker = marker;
    group.userData.marker = marker;
    return group;
}

async function setupCameraMarkers(markers) {
    clearCameraMarkers();
    resetToGlobalView();
    if (!Array.isArray(markers) || !markers.length || !state.markerGroup) {
        state.cameraMarkers = [];
        return;
    }
    state.cameraMarkers = markers.map((marker) => {
        const key = marker.key || marker.name || (marker.camera_id != null ? `cam${marker.camera_id}` : `marker-${Date.now()}`);
        const displayPosition = buildMarkerDisplayPosition(marker.position); // 이게 시점 절대 고치지 말기
        return { ...marker, key, displayPosition };
    });
    state.markerLookup = new Map();
    state.markerObjects = [];
    for (const marker of state.cameraMarkers) {
        const obj = await createMarkerSprite(marker);
        if (!obj) {
            continue;
        }
        const yawRad = computeMarkerYawRad(marker);
        const templateScale = Number(state.markerModelScale) || 1;
        const groupScale = Number(marker.baseScale) || 1;
        const offsetWorld = rotateOffset({
            x: markerBulbOffsetLocal.x * templateScale * groupScale,
            y: markerBulbOffsetLocal.y * templateScale * groupScale,
        }, yawRad);
        const backWorld = rotateOffset({ x: -markerBackOffset, y: 0 }, yawRad);
        const groundZ = lookupGlobalHeight(marker.displayPosition.x, marker.displayPosition.y);
        const baseZ = Number.isFinite(groundZ) ? groundZ : marker.displayPosition.z;
        const posX = marker.displayPosition.x + offsetWorld.x + backWorld.x;
        const posY = marker.displayPosition.y + offsetWorld.y + backWorld.y;
        obj.position.set(posX, posY, baseZ);
        obj.rotation.set(0, 0, yawRad);
        marker.yawRad = yawRad;
        state.markerGroup.add(obj);
        state.markerObjects.push(obj);
        state.markerLookup.set(marker.key, marker);
    }
    updateMarkerVisibility(null);
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

function createMarkerTextures(label) {
    return {
        base: drawMarkerTexture(label, markerLabelBaseColor),
        highlight: drawMarkerTexture(label, markerLabelHighlightColor),
    };
}

function drawMarkerTexture(label, color) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const len = label.length;
    const fontSize = len <= 3 ? 120 : len <= 6 ? 90 : 70;
    ctx.font = `bold ${fontSize}px 'Segoe UI', Arial, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const metrics = ctx.measureText(label);
    const padding = 12;
    const textWidth = metrics.width;
    const textHeight = fontSize;
    const boxWidth = textWidth + padding * 2;
    const boxHeight = textHeight + padding * 1.4;
    const boxX = 128 - boxWidth / 2;
    const boxY = 128 - boxHeight / 2;
    ctx.fillStyle = "rgba(0, 0, 0, 0.35)";
    ctx.roundRect(boxX, boxY, boxWidth, boxHeight, 14);
    ctx.fill();
    ctx.fillStyle = color;
    ctx.strokeStyle = "rgba(0,0,0,0.65)";
    ctx.lineWidth = 6;
    ctx.strokeText(label, 128, 132);
    ctx.fillText(label, 128, 132);
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
        resetToGlobalView({ initialView: true, animate: true });
        if (isLive) {
            enterLiveGlobalView();
        } else if (isFusion) {
            enterFusionGlobalView();
        }
        return;
    }
    state.activeMarkerKey = marker.key;
    updateMarkerHighlight(marker.key);
    updateMarkerVisibility(marker.key);
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
            const preferPly = Boolean(state.config?.preferPlyForColor);
            if (preferPly) {
                mesh = await loadLocalCloudWithColorFallback(marker, { flipX, flipY });
            } else {
                if (marker.local_visible_url) {
                    const stride = Number(marker.local_visible_stride) || 3;
                    const hasColor = Boolean(marker.local_visible_has_rgb);
                    mesh = await loadVisiblePointsFromBinary(marker.local_visible_url, stride, { flipX, flipY, hasColor });
                } else {
                    mesh = await loadPointCloudFromPly(marker.local_ply_url, { flipX, flipY });
                }
            }
            if (mesh) {
                mesh.name = `LocalCloud-${marker.key}`;
                mesh.userData.markerKey = marker.key;
                scene.add(mesh);
                state.localClouds.set(marker.key, mesh);
            }
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

async function loadLocalCloudWithColorFallback(marker, options = {}) {
    const { flipX = false, flipY = false } = options;
    const visibleHasRgb = Boolean(marker.local_visible_has_rgb);
    const canUseVisible = Boolean(marker.local_visible_url);
    const canUsePly = Boolean(marker.local_ply_url);
    const preferPlyForColor = canUsePly && (!canUseVisible || !visibleHasRgb);
    if (preferPlyForColor) {
        return loadPointCloudFromPly(marker.local_ply_url, { flipX, flipY });
    }
    if (canUseVisible) {
        const stride = Number(marker.local_visible_stride) || 3;
        return loadVisiblePointsFromBinary(marker.local_visible_url, stride, { flipX, flipY, hasColor: visibleHasRgb });
    }
    if (canUsePly) {
        return loadPointCloudFromPly(marker.local_ply_url, { flipX, flipY });
    }
    return null;
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
    const initialView = Boolean(options.initialView);
    const animateToInitial = initialView && animate;
    const animateRelease = initialView ? false : animate;
    if (isLive || isFusion) {
        setLayoutMode("global");
    }
    state.activeMarkerKey = null;
    clearDetectionFollow();
    state.localLoadToken += 1;
    updateMarkerHighlight(null);
    updateMarkerVisibility(null);
    setLocalCloudVisibility(null);
    if (state.globalCloud) {
        state.globalCloud.visible = true;
    }
    if (animateToInitial) {
        releaseViewLock({ preservePose: true });
    } else {
        releaseViewLock({ animate: animateRelease });
    }
    if (initialView) {
        const durationOverride = options.duration ?? options.durationMs;
        applyInitialViewOverride(true, { animate: animateToInitial, duration: durationOverride });
    }
}

function updateMarkerHighlight(activeKey) {
    state.markerLookup.forEach((marker) => {
        const labelSprite = marker.labelSprite || marker.sprite;
        const isActive = marker.key === activeKey;
        const targetTexture = isActive
            ? (marker.highlightTexture || labelSprite?.userData?.highlightTexture)
            : (marker.baseTexture || labelSprite?.userData?.baseTexture);
        if (labelSprite && targetTexture && labelSprite.material.map !== targetTexture) {
            labelSprite.material.map = targetTexture;
            labelSprite.material.needsUpdate = true;
        }
        const labelScale = isActive ? (labelSprite?.userData?.activeScale || markerLabelActiveScale) : (labelSprite?.userData?.baseScale || markerLabelBaseScale);
        if (labelSprite && labelScale) {
            labelSprite.scale.set(labelScale, labelScale, 1);
        }
        const targetScale = isActive ? marker.activeScale : marker.baseScale;
        if (marker.object && targetScale) {
            marker.object.scale.setScalar(targetScale);
        }
        const ring = marker.ringMesh;
        if (ring) {
            ring.visible = !isActive;
        }
    });
}

function updateMarkerVisibility(activeKey) {
    const hasActive = Boolean(activeKey);
    state.markerLookup.forEach((marker) => {
        const targetVisible = !hasActive || marker.key === activeKey;
        if (marker.object) {
            marker.object.visible = targetVisible;
        } else if (marker.sprite) {
            marker.sprite.visible = targetVisible;
        }
    });
}

function updateMarkerGlow() {
    if (!state.markerObjects.length) {
        return;
    }
    const t = (performance.now() - state.markerPulseStart) * 0.001;
    state.markerObjects.forEach((obj) => {
        const marker = obj.userData?.marker;
        const sprite = marker?.labelSprite || marker?.sprite;
        if (sprite && sprite.material) {
            const baseOpacity = sprite.userData?.baseOpacity ?? 0.9;
            const flicker = 0.05 * Math.sin(t * 3.1);
            sprite.material.opacity = Math.min(1, Math.max(0.2, baseOpacity + flicker));
            sprite.material.needsUpdate = true;
        }
        const ring = marker?.ringMesh;
        if (ring && ring.material) {
            const baseOpacity = ring.userData?.baseOpacity ?? markerRingBaseOpacity;
            const baseScale = ring.userData?.baseScale ?? 1;
            const scalePulse = 1.0 + 0.14 * Math.sin(t * 2.6 + 0.4);
            const opacityPulse = baseOpacity + 0.18 * Math.sin(t * 2.2 + 0.2);
            ring.scale.setScalar(baseScale * scalePulse);
            ring.material.opacity = Math.min(1, Math.max(0, opacityPulse));
            ring.material.needsUpdate = true;
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
    const hits = raycaster.intersectObjects(state.markerObjects, true);
    if (!hits.length) {
        return;
    }
    const hit = hits[0];
    let obj = hit.object;
    let marker = obj?.userData?.marker;
    while (!marker && obj && obj.parent) {
        obj = obj.parent;
        marker = obj.userData?.marker;
    }
    if (marker) {
        handleMarkerClick(marker);
    }
}

function handleGlobalKeydown(evt) {
    if (adminState.enabled && matchesAdminHotkey(evt)) {
        evt.preventDefault();
        setupAdminPanel();
        toggleAdminPanel();
        return;
    }
    if (evt.key === "Escape") {
        triggerEscapeAction();
    }
}

function handleGlobalTouchEnd(evt) {
    const now = Date.now();
    if (now - lastTouchEndTs <= 300) {
        triggerEscapeAction();
        lastTouchEndTs = 0;
        return;
    }
    lastTouchEndTs = now;
}

function triggerEscapeAction() {
    clearDetectionFollow();
    resetToGlobalView({ initialView: true, animate: true });
    if (isLive) {
        enterLiveGlobalView();
    } else if (isFusion) {
        enterFusionGlobalView();
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
    const { flipX = false, flipY = false, hasColor = false } = options;
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
    const useEmbeddedColor = Boolean(hasColor) && step >= 6;
    const colors = useEmbeddedColor ? new Float32Array(count * 3) : null;
    for (let i = 0; i < count; i += 1) {
        const base = i * step;
        positions[i * 3 + 0] = data[base + 0] ?? 0;
        positions[i * 3 + 1] = data[base + 1] ?? 0;
        positions[i * 3 + 2] = data[base + 2] ?? 0;
        if (colors) {
            colors[i * 3 + 0] = data[base + 3] ?? 0.5;
            colors[i * 3 + 1] = data[base + 4] ?? 0.5;
            colors[i * 3 + 2] = data[base + 5] ?? 0.5;
        }
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
    let hasColorAttr = false;
    if (colors) {
        let maxVal = 0;
        for (let i = 0; i < colors.length; i += 1) {
            if (colors[i] > maxVal) maxVal = colors[i];
        }
        if (maxVal > 1.01) {
            for (let i = 0; i < colors.length; i += 1) {
                colors[i] = colors[i] / 255;
            }
        }
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
        hasColorAttr = true;
    } else if (state.globalColorLookup) {
        const colorArray = createColorArrayForPositions(positions);
        geometry.setAttribute("color", new THREE.BufferAttribute(colorArray, 3));
        hasColorAttr = true;
    }
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({
        size: 0.05,
        color: hasColorAttr ? 0xffffff : 0xf7b801,
        vertexColors: hasColorAttr,
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
    // 충돌코드?
    const colorInfo = getDetectionColorInfo(det);
    if (colorInfo) {
        const { label, hex } = colorInfo;
        const colorText = label && hex
            ? `${label} (${hex})`
            : (label || hex);
        if (colorText) {
            tags.push(`color ${colorText}`);
        }
    }
    const tagText = tags.join(" | ") || `class ${det.class_id ?? "-"}`;
    return `#${idx + 1} | ${tagText} | x ${center[0].toFixed(2)} | y ${center[1].toFixed(2)} | z ${center[2].toFixed(2)} | yaw ${Number(yaw).toFixed(1)}°`;
}

function matchesAdminHotkey(evt) {
    if (!evt || !ADMIN_HOTKEY || !ADMIN_HOTKEY.key) {
        return false;
    }
    const keyMatch = typeof evt.key === "string" && evt.key.toLowerCase() === ADMIN_HOTKEY.key;
    const needCtrl = Boolean(ADMIN_HOTKEY.ctrl);
    const needShift = Boolean(ADMIN_HOTKEY.shift);
    return keyMatch && (!needCtrl || evt.ctrlKey) && (!needShift || evt.shiftKey);
}

function injectAdminStyles() {
    if (!adminState.enabled) {
        return;
    }
    const style = document.createElement("style");
    style.textContent = `
#admin-panel {
  position: fixed;
  left: 16px;
  top: 16px;
  width: min(400px, 90vw);
  max-height: 80vh;
  overflow: hidden;
  display: none;
  flex-direction: column;
  gap: 10px;
  padding: 14px;
  background: rgba(10, 12, 18, 0.96);
  border: 1px solid #1f2937;
  border-radius: 14px;
  box-shadow: 0 18px 50px rgba(0,0,0,0.55);
  z-index: 50;
  color: #e5e7eb;
  backdrop-filter: blur(6px);
}
#admin-panel.open {
  display: flex;
}
#admin-panel .admin-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}
#admin-panel .admin-title {
  font-weight: 800;
  font-size: 1rem;
  letter-spacing: 0.2px;
}
#admin-panel .admin-hotkey {
  font-size: 0.8rem;
  color: #9ca3af;
}
#admin-panel .admin-status {
  font-size: 0.85rem;
  color: #9ce0ff;
}
#admin-panel .admin-row {
  display: flex;
  gap: 8px;
  align-items: center;
}
#admin-panel input, #admin-panel select {
  flex: 1;
  padding: 8px 10px;
  background: #0f172a;
  color: #f8fafc;
  border: 1px solid #1f2937;
  border-radius: 8px;
}
#admin-panel button {
  background: #2563eb;
  color: #f8fafc;
  border: 1px solid #1d4ed8;
  border-radius: 8px;
  padding: 8px 10px;
  cursor: pointer;
}
#admin-panel button.secondary {
  background: #111827;
  border-color: #1f2937;
}
#admin-panel button.danger {
  background: #dc2626;
  border-color: #b91c1c;
}
#admin-panel #admin-track-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
  max-height: 42vh;
  overflow-y: auto;
  padding-right: 2px;
}
.admin-track {
  width: 100%;
  text-align: left;
  background: #0f172a;
  border: 1px solid #1f2937;
  border-radius: 10px;
  padding: 8px 10px;
  color: #e5e7eb;
}
.admin-track:hover {
  border-color: #2563eb;
}
.admin-track .admin-track-title {
  font-weight: 700;
}
.admin-track .admin-track-meta {
  font-size: 0.85rem;
  color: #94a3b8;
}
`;
    document.head.appendChild(style);
}

function setupAdminPanel() {
    if (!adminState.enabled || adminState.initialized) {
        return;
    }
    adminState.initialized = true;
    injectAdminStyles();

    const panel = document.createElement("div");
    panel.id = "admin-panel";
    adminState.elements.panel = panel;

    const header = document.createElement("div");
    header.className = "admin-header";
    const title = document.createElement("div");
    title.className = "admin-title";
    title.textContent = "Track Admin";
    const hotkey = document.createElement("div");
    hotkey.className = "admin-hotkey";
    hotkey.textContent = "Ctrl+Shift+A";
    const closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "secondary";
    closeBtn.textContent = "Close";
    closeBtn.addEventListener("click", () => toggleAdminPanel(false));
    header.appendChild(title);
    header.appendChild(hotkey);
    header.appendChild(closeBtn);
    panel.appendChild(header);

    const status = document.createElement("div");
    status.className = "admin-status";
    status.textContent = "Ready";
    adminState.elements.status = status;
    panel.appendChild(status);

    const row1 = document.createElement("div");
    row1.className = "admin-row";
    const trackInput = document.createElement("input");
    trackInput.type = "number";
    trackInput.placeholder = "Track id";
    adminState.elements.trackInput = trackInput;
    const refreshBtn = document.createElement("button");
    refreshBtn.type = "button";
    refreshBtn.className = "secondary";
    refreshBtn.textContent = "Refresh";
    refreshBtn.addEventListener("click", () => refreshAdminTracks(true));
    row1.appendChild(trackInput);
    row1.appendChild(refreshBtn);
    panel.appendChild(row1);

    const row2 = document.createElement("div");
    row2.className = "admin-row";
    const colorSelect = document.createElement("select");
    ADMIN_COLORS.forEach((c) => {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        colorSelect.appendChild(opt);
    });
    adminState.elements.colorSelect = colorSelect;
    const colorBtn = document.createElement("button");
    colorBtn.type = "button";
    colorBtn.textContent = "Set color";
    colorBtn.addEventListener("click", () => applyAdminColor());
    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "secondary";
    clearBtn.textContent = "Clear";
    clearBtn.addEventListener("click", () => {
        if (adminState.elements.colorSelect) {
            adminState.elements.colorSelect.value = "none";
        }
        applyAdminColor();
    });
    row2.appendChild(colorSelect);
    row2.appendChild(colorBtn);
    row2.appendChild(clearBtn);
    panel.appendChild(row2);

    const row3 = document.createElement("div");
    row3.className = "admin-row";
    const yawInput = document.createElement("input");
    yawInput.type = "number";
    yawInput.placeholder = "Yaw °";
    yawInput.step = "1";
    adminState.elements.yawInput = yawInput;
    const yawBtn = document.createElement("button");
    yawBtn.type = "button";
    yawBtn.textContent = "Set yaw";
    yawBtn.addEventListener("click", () => applyAdminYaw());
    const flipBtn = document.createElement("button");
    flipBtn.type = "button";
    flipBtn.className = "secondary";
    flipBtn.textContent = "Flip 180°";
    flipBtn.addEventListener("click", () => applyAdminFlip());
    row3.appendChild(yawInput);
    row3.appendChild(yawBtn);
    row3.appendChild(flipBtn);
    panel.appendChild(row3);

    const list = document.createElement("div");
    list.id = "admin-track-list";
    adminState.elements.list = list;
    panel.appendChild(list);

    document.body.appendChild(panel);
}

function toggleAdminPanel(forceOpen = null) {
    if (!adminState.enabled || !adminState.initialized) {
        return;
    }
    const shouldOpen = forceOpen === null ? !adminState.open : Boolean(forceOpen);
    adminState.open = shouldOpen;
    if (adminState.elements.panel) {
        adminState.elements.panel.style.display = shouldOpen ? "flex" : "none";
        adminState.elements.panel.classList.toggle("open", shouldOpen);
    }
    if (adminState.elements.toggle) {
        adminState.elements.toggle.classList.toggle("active", shouldOpen);
    }
    if (shouldOpen) {
        refreshAdminTracks(true);
        startAdminRefresh();
    } else {
        stopAdminRefresh();
    }
}

function startAdminRefresh() {
    stopAdminRefresh();
    adminState.refreshHandle = window.setInterval(() => refreshAdminTracks(false), 1500);
}

function stopAdminRefresh() {
    if (adminState.refreshHandle) {
        clearInterval(adminState.refreshHandle);
        adminState.refreshHandle = null;
    }
}

function setAdminStatus(message, isError = false) {
    if (adminState.elements.status) {
        adminState.elements.status.textContent = message;
        adminState.elements.status.style.color = isError ? "#fca5a5" : "#9ce0ff";
    }
}

function getSelectedTrackId() {
    const input = adminState.elements.trackInput;
    if (!input) {
        return null;
    }
    const val = Number(input.value);
    return Number.isFinite(val) ? val : null;
}

function prefillAdminFromTrack(track) {
    if (!adminState.enabled || !track) {
        return;
    }
    const tid = track.track_id ?? track.id;
    if (adminState.elements.trackInput && tid != null) {
        adminState.elements.trackInput.value = tid;
    }
    const yawVal = track.yaw ?? track.yaw_deg;
    if (adminState.elements.yawInput && Number.isFinite(Number(yawVal))) {
        adminState.elements.yawInput.value = Number(yawVal).toFixed(1);
    }
    const colorInfo = getDetectionColorInfo(track);
    if (adminState.elements.colorSelect && colorInfo?.label) {
        const normalized = colorInfo.label.toLowerCase();
        if (ADMIN_COLORS.includes(normalized)) {
            adminState.elements.colorSelect.value = normalized;
        }
    }
}

function prefillAdminFromDetection(det) {
    if (!adminState.enabled) {
        return;
    }
    setupAdminPanel();
    prefillAdminFromTrack(det);
}

async function fetchAdminTracks() {
    const resp = await fetch("/api/admin/tracks", { cache: "no-store" });
    if (!resp.ok) {
        throw new Error(`admin list failed (${resp.status})`);
    }
    return resp.json();
}

async function adminSetColor(trackId, color) {
    const resp = await fetch(`/api/admin/tracks/${trackId}/color`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ color }),
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "color update failed");
    }
    return resp.json();
}

async function adminSetYaw(trackId, yaw) {
    const resp = await fetch(`/api/admin/tracks/${trackId}/yaw`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ yaw }),
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "yaw update failed");
    }
    return resp.json();
}

async function adminFlipYaw(trackId, delta = 180.0) {
    const resp = await fetch(`/api/admin/tracks/${trackId}/flip`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ delta }),
    });
    if (!resp.ok) {
        const text = await resp.text();
        throw new Error(text || "flip failed");
    }
    return resp.json();
}

async function applyAdminColor() {
    const trackId = getSelectedTrackId();
    if (trackId == null) {
        setAdminStatus("Track id is required", true);
        return;
    }
    const select = adminState.elements.colorSelect;
    const raw = (select?.value || "").trim().toLowerCase();
    const color = !raw || raw === "none" ? null : raw;
    try {
        await adminSetColor(trackId, color);
        setAdminStatus(`Track ${trackId} color -> ${color ?? "cleared"}`);
        refreshAdminTracks(true);
        refreshFusionViewAfterAdmin();
    } catch (err) {
        setAdminStatus(err?.message || "Color update failed", true);
    }
}

async function applyAdminYaw() {
    const trackId = getSelectedTrackId();
    if (trackId == null) {
        setAdminStatus("Track id is required", true);
        return;
    }
    const yawInput = adminState.elements.yawInput;
    const yawVal = Number(yawInput?.value);
    if (!Number.isFinite(yawVal)) {
        setAdminStatus("Enter yaw in degrees", true);
        return;
    }
    try {
        await adminSetYaw(trackId, yawVal);
        setAdminStatus(`Track ${trackId} yaw -> ${yawVal.toFixed(1)}°`);
        refreshAdminTracks(true);
        refreshFusionViewAfterAdmin();
    } catch (err) {
        setAdminStatus(err?.message || "Yaw update failed", true);
    }
}

async function applyAdminFlip(delta = 180.0) {
    const trackId = getSelectedTrackId();
    if (trackId == null) {
        setAdminStatus("Track id is required", true);
        return;
    }
    try {
        await adminFlipYaw(trackId, delta);
        setAdminStatus(`Track ${trackId} flipped by ${delta.toFixed(1)}°`);
        refreshAdminTracks(true);
        refreshFusionViewAfterAdmin();
    } catch (err) {
        setAdminStatus(err?.message || "Flip failed", true);
    }
}

async function refreshAdminTracks(force = false) {
    if (!adminState.open && !force) {
        return;
    }
    try {
        const data = await fetchAdminTracks();
        const tracks = Array.isArray(data?.tracks) ? data.tracks : [];
        renderAdminTracks(tracks);
        setAdminStatus(`Tracks: ${tracks.length}`);
    } catch (err) {
        setAdminStatus(err?.message || "Admin refresh failed", true);
    }
}

function renderAdminTracks(tracks) {
    const listEl = adminState.elements.list;
    if (!listEl) {
        return;
    }
    listEl.innerHTML = "";
    const rows = Array.isArray(tracks) ? tracks.slice() : [];
    if (!rows.length) {
        const empty = document.createElement("div");
        empty.textContent = "No active tracks";
        empty.style.color = "#9ca3af";
        listEl.appendChild(empty);
        return;
    }
    rows.sort((a, b) => (a.id ?? 0) - (b.id ?? 0));
    rows.forEach((t) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "admin-track";
        const yawText = Number.isFinite(Number(t.yaw)) ? Number(t.yaw).toFixed(1) : "-";
        const colorLabel = t.color || "-";
        btn.innerHTML = `
            <div class="admin-track-title">#${t.id} | ${t.state || "-"}</div>
            <div class="admin-track-meta">yaw ${yawText}° | color ${colorLabel}</div>
        `;
        btn.addEventListener("click", () => {
            prefillAdminFromTrack(t);
        });
        listEl.appendChild(btn);
    });
}

function refreshFusionViewAfterAdmin() {
    if (!isFusion || state.fusionSource !== "tracks") {
        return;
    }
    fetchFusionDetections().catch((err) => {
        console.warn("fusion refresh failed", err);
    });
}

function setupAdminToggleButton() {
    if (!adminState.enabled || !adminToggleBtnEl) {
        return;
    }
    adminState.elements.toggle = adminToggleBtnEl;
    adminToggleBtnEl.style.display = "block";
    adminToggleBtnEl.addEventListener("click", () => {
        setupAdminPanel();
        toggleAdminPanel();
    });
}

if (adminState.enabled) {
    setupAdminToggleButton();
    setupAdminPanel();
    if (adminAutoOpen) {
        toggleAdminPanel(true);
    }
}

init().catch((err) => console.error("Initialization error:", err));
