#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual Free-Fly Camera (independent of server Spectator)
- Does NOT move world.get_spectator()
- Keeps its own Transform and moves an RGB sensor accordingly
- WASD move, E/Q or Space/C up/down, mouse look, Shift/Ctrl speed
- HUD shows camera position/rotation
"""

import argparse
import math
import numpy as np
import pygame
import carla
from queue import Queue, Empty

# ---------------- Utilities ---------------- #
def clamp(v, lo, hi): return lo if v < lo else hi if v > hi else v
def deg_to_rad(d): return d * math.pi / 180.0

# ---------------- Input ---------------- #
class InputState:
    def __init__(self, mouse_sens=0.15, invert_y=False):
        self.mouse_sens = mouse_sens
        # mouse up => look up (pitch decreases). invert_y flips.
        self.y_sign = 1.0 if not invert_y else -1.0
        self.capture_mouse = True
        self.reset()

    def reset(self):
        self.fwd = self.right = self.up = 0.0
        self.boost = 1.0
        self.dyaw = self.dpitch = 0.0
        self.quit = False
        self.toggle_capture = False

    def poll(self):
        self.reset()
        for e in pygame.event.get():
            if e.type == pygame.QUIT: self.quit = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: self.quit = True
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                self.toggle_capture = True

        k = pygame.key.get_pressed()
        # WASD
        self.fwd   += 1.0 if k[pygame.K_w] else 0.0
        self.fwd   -= 1.0 if k[pygame.K_s] else 0.0
        self.right += 1.0 if k[pygame.K_d] else 0.0
        self.right -= 1.0 if k[pygame.K_a] else 0.0
        # Up/Down: E/Q + Space/C
        self.up += 1.0 if (k[pygame.K_e] or k[pygame.K_SPACE]) else 0.0
        self.up -= 1.0 if (k[pygame.K_q] or k[pygame.K_c]) else 0.0
        # Speed
        if k[pygame.K_LSHIFT] or k[pygame.K_RSHIFT]: self.boost *= 3.0
        if k[pygame.K_LCTRL]  or k[pygame.K_RCTRL] : self.boost *= 0.33
        # Mouse
        if self.capture_mouse:
            mx, my = pygame.mouse.get_rel()
            self.dyaw   = mx * self.mouse_sens
            self.dpitch = (-my * self.mouse_sens) * self.y_sign
        else:
            pygame.mouse.get_rel()

# ---------------- Virtual Free-Fly Camera ---------------- #
class VirtualFreeFlyCamera:
    """Holds its own Transform; does NOT touch world.get_spectator()."""
    def __init__(self, world, start_tf, speed=10.0, mouse_sens=0.15, invert_y=False):
        self.world = world
        self.tf = start_tf  # carla.Transform independently managed
        self.speed = speed
        self.pitch_min, self.pitch_max = -89.0, 89.0
        self.input = InputState(mouse_sens=mouse_sens, invert_y=invert_y)

    def update(self, dt):
        """Process input and update self.tf; return True if user wants to quit."""
        self.input.poll()
        if self.input.toggle_capture:
            self.input.capture_mouse = not self.input.capture_mouse
            pygame.event.set_grab(self.input.capture_mouse)
            pygame.mouse.set_visible(not self.input.capture_mouse)

        loc, rot = self.tf.location, self.tf.rotation

        # Rotate
        rot.yaw   = (rot.yaw + self.input.dyaw) % 360.0
        rot.pitch = clamp(rot.pitch + self.input.dpitch, self.pitch_min, self.pitch_max)

        # Direction vectors
        yaw = deg_to_rad(rot.yaw); pitch = deg_to_rad(rot.pitch)
        forward = (math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), -math.sin(pitch))
        right   = (-math.sin(yaw), math.cos(yaw), 0.0)
        up      = (0.0, 0.0, 1.0)

        ax = self.input.right*right[0] + self.input.fwd*forward[0] + self.input.up*up[0]
        ay = self.input.right*right[1] + self.input.fwd*forward[1] + self.input.up*up[1]
        az = self.input.right*right[2] + self.input.fwd*forward[2] + self.input.up*up[2]
        norm = math.sqrt(ax*ax + ay*ay + az*az)
        if norm > 0: ax/=norm; ay/=norm; az/=norm

        v = self.speed * self.input.boost
        loc.x += ax * v * dt
        loc.y += ay * v * dt
        loc.z += az * v * dt

        # write back
        self.tf = carla.Transform(loc, rot)
        return self.input.quit

# ---------------- Main ---------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--gamma", type=float, default=2.2)
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--sens", type=float, default=0.15)
    ap.add_argument("--invert-y", action="store_true")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port); client.set_timeout(5.0)
    world = client.get_world()

    # Sync mode
    prev_settings = world.get_settings()
    if args.sync:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = 1.0/60.0
        world.apply_settings(s)
        print("[INFO] Sync mode ON")

    # pygame
    pygame.init()
    pygame.display.set_caption("Virtual Free-Fly Camera (Client)")
    screen = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    clock = pygame.time.Clock()
    pygame.event.set_grab(True); pygame.mouse.set_visible(False)
    font = pygame.font.SysFont(None, 22)

    # Start transform: use a map spawn point (not spectator)
    spawn_pts = world.get_map().get_spawn_points()
    base_tf = spawn_pts[0] if spawn_pts else carla.Transform(carla.Location(0,0,0), carla.Rotation())
    start_tf = carla.Transform(
        carla.Location(base_tf.location.x, base_tf.location.y, base_tf.location.z + 50.0),
        carla.Rotation(pitch=-15.0, yaw=base_tf.rotation.yaw, roll=0.0)
    )

    controller = VirtualFreeFlyCamera(
        world, start_tf, speed=args.speed, mouse_sens=args.sens, invert_y=args.invert_y
    )

    # Spawn RGB camera (independent sensor we move ourselves)
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(args.width))
    bp.set_attribute("image_size_y", str(args.height))
    bp.set_attribute("fov", str(args.fov))
    bp.set_attribute("gamma", str(args.gamma))
    bp.set_attribute("sensor_tick", "0")  # sync: every tick; async: as fast as possible

    cam = world.try_spawn_actor(bp, controller.tf)
    if cam is None: raise RuntimeError("Failed to spawn camera sensor")

    q = Queue(maxsize=8)
    cam.listen(lambda img: q.put(img))

    try:
        running = True
        last = None
        while running:
            dt = clock.tick(120)/1000.0
            if args.sync: world.tick()
            else: world.wait_for_tick()

            # Update only our virtual camera transform
            if controller.update(dt):
                running = False

            # Apply transform to our sensor (this is the only thing we move)
            cam.set_transform(controller.tf)

            # Drain to most recent image
            try:
                while True: last = q.get_nowait()
            except Empty:
                pass

            if last is not None:
                arr = np.frombuffer(last.raw_data, dtype=np.uint8).reshape((last.height,last.width,4))
                frame = arr[:,:,:3][:,:,::-1]
                surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))
                screen.blit(surf, (0,0))

            # HUD: virtual cam pose
            loc, rot = controller.tf.location, controller.tf.rotation
            line1 = f"Mode=VirtualCam (server Spectator untouched) | Sync={'ON' if args.sync else 'OFF'} | FPS={clock.get_fps():.0f}"
            line2 = f"Pos x={loc.x:8.2f}  y={loc.y:8.2f}  z={loc.z:7.2f}"
            line3 = f"Rot yaw={rot.yaw:7.2f}  pitch={rot.pitch:7.2f}  roll={rot.roll:7.2f}"
            line4 = "WASD move | E/Q & Space/C up/down | Shift/Ctrl speed | RMB mouse | ESC quit"

            screen.blit(font.render(line1, True, (255,255,255)), (10,10))
            screen.blit(font.render(line2, True, (255,255,255)), (10,32))
            screen.blit(font.render(line3, True, (255,255,255)), (10,54))
            screen.blit(font.render(line4, True, (230,230,230)), (10,76))
            pygame.display.flip()

    finally:
        if cam: cam.stop(); cam.destroy()
        if args.sync: world.apply_settings(prev_settings); print("[INFO] Sync mode OFF (restored)")
        pygame.quit()

if __name__ == "__main__":
    main()