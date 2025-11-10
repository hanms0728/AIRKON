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
    visibleMeshes: new Map(),
    globalCloud: null,
};

const fusionEndpointMap = {
    raw: "/api/raw",
    fused: "/api/fused",
    tracks: "/api/tracks",
};

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
const debugMarker = new THREE.Mesh(
    new THREE.SphereGeometry(0.4, 12, 12),
    new THREE.MeshBasicMaterial({ color: 0xff5555 })
);
debugMarker.visible = false;
scene.add(debugMarker);
scene.add(grid);
scene.add(axes);

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
    await Promise.all([loadPointCloud(), loadVehiclePrototype()]);
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
        state.cameras = await fetchJson("/api/cameras");
    } catch (err) {
        console.error(err);
        alert("Failed to load camera list. Edge bridge running?");
        return;
    }

    populateCameraSelect();

    await Promise.all([loadPointCloud(), loadVehiclePrototype()]);

    if (state.cameraId != null) {
        await loadVisibleCloudForCamera(state.cameraId);
        startLivePolling();
    } else if (statusEl) {
        statusEl.textContent = "No camera available";
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
                        const templateMaterial = obj.material && obj.material.clone ? obj.material.clone() : obj.material;
                        state.vehicleTemplates.push({
                            geometry: obj.geometry.clone(),
                            material: templateMaterial,
                            localMatrix: obj.matrix.clone(),
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
        const inst = new THREE.InstancedMesh(geometry, material, capacity);
        inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        inst.count = 0;
        inst.frustumCulled = false;
        inst.matrixAutoUpdate = false;
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
            li.textContent = formatDetectionListEntry(det, idx);
            detectionListEl.appendChild(li);
        });
    }
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

function populateCameraSelect() {
    if (!cameraSelectEl) {
        state.cameraId = state.cameras.length ? state.cameras[0].camera_id : null;
        return;
    }
    cameraSelectEl.innerHTML = "";
    if (!state.cameras.length) {
        const opt = document.createElement("option");
        opt.textContent = "No cameras";
        cameraSelectEl.appendChild(opt);
        cameraSelectEl.disabled = true;
        state.cameraId = null;
        return;
    }
    state.cameras.forEach((cam, idx) => {
        const opt = document.createElement("option");
        opt.value = cam.camera_id;
        opt.textContent = `${cam.camera_id} — ${cam.name || "camera"}`;
        if (idx === 0) {
            opt.selected = true;
        }
        cameraSelectEl.appendChild(opt);
    });
    cameraSelectEl.disabled = false;
    state.cameraId = Number(cameraSelectEl.value);
    cameraSelectEl.addEventListener("change", async () => {
        state.cameraId = Number(cameraSelectEl.value);
        stopLivePolling();
        await loadVisibleCloudForCamera(state.cameraId);
        startLivePolling();
    });
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

function updateOverlayImage(camId) {
    if (!overlayImgEl) return;
    state.overlayToken += 1;
    const token = state.overlayToken;
    fetch(`/api/cameras/${camId}/overlay.jpg?cacheBust=${Date.now()}`, { cache: "no-store" })
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

function formatDetectionListEntry(det, idx) {
    const center = Array.isArray(det.center) && det.center.length === 3
        ? det.center
        : [0, 0, 0];
    const yaw = det.yaw_deg ?? det.yaw ?? 0;
    const tags = [];
    if (det.track_id != null) {
        tags.push(`track ${det.track_id}`);
    } else if (det.cam) {
        tags.push(`cam ${det.cam}`);
    } else if (det.class_id != null) {
        tags.push(`class ${det.class_id}`);
    }
    if (Array.isArray(det.sources) && det.sources.length) {
        tags.push(`src ${det.sources.join(",")}`);
    }
    if (det.score != null) {
        tags.push(`score ${(Number(det.score) || 0).toFixed(2)}`);
    }
    const tagText = tags.join(" | ") || `class ${det.class_id ?? "-"}`;
    return `#${idx + 1} | ${tagText} | x ${center[0].toFixed(2)} | y ${center[1].toFixed(2)} | z ${center[2].toFixed(2)} | yaw ${Number(yaw).toFixed(1)}°`;
}

init().catch((err) => console.error("Initialization error:", err));
