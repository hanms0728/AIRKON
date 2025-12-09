#!/usr/bin/env python3

import glob
import math
import os
import sys

import pygame
from pygame.locals import K_ESCAPE, K_q, K_w, K_a, K_s, K_d


def _append_carla_egg():
    try:
        egg_path = glob.glob(
            '../carla/dist/carla-*%d.%d-%s.egg' % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64'
            )
        )[0]
        sys.path.append(egg_path)
    except IndexError:
        pass


_append_carla_egg()

import carla  # noqa: E402


TRAFFIC_CONE_BLUEPRINT = 'static.prop.trafficcone01'
TRAFFIC_CONE_SPEED = 4.0  # meters per second
TRAFFIC_CONE_START = carla.Transform(
    carla.Location(x=0.0, y=0.0, z=0.05),
    carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
)


class TrafficConeController:
    def __init__(self, host='localhost', port=2000):
        pygame.init()
        self.display = pygame.display.set_mode((420, 200))
        pygame.display.set_caption('Traffic Cone Controller')
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font(None, 24)
        except Exception:
            self.font = None

        self.client = carla.Client(host, port)
        self.client.set_timeout(5.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        self.cone_actor = self._spawn_cone()
        self.key_state = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False,
        }

    def _spawn_cone(self):
        try:
            bp = self.blueprint_library.find(TRAFFIC_CONE_BLUEPRINT)
        except RuntimeError as exc:
            print(f'[Cone] Blueprint lookup failed: {exc}')
            return None
        try:
            cone = self.world.spawn_actor(bp, TRAFFIC_CONE_START)
            print('[Cone] Spawned controllable traffic cone.')
            return cone
        except RuntimeError as exc:
            print(f'[Cone] Failed to spawn cone: {exc}')
            return None

    def run(self):
        if self.cone_actor is None:
            print('No cone to control; exiting.')
            self._cleanup()
            return
        try:
            running = True
            while running:
                dt = self.clock.tick(60) / 1000.0
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    if event.type == pygame.KEYDOWN:
                        if event.key in (K_ESCAPE, K_q):
                            running = False
                            break
                        self._handle_key(event.key, True)
                    elif event.type == pygame.KEYUP:
                        self._handle_key(event.key, False)
                self._update_cone(dt)
                self._draw_status()
                pygame.display.flip()
        finally:
            self._cleanup()

    def _handle_key(self, key, pressed):
        if key == K_w:
            self.key_state['forward'] = pressed
        elif key == K_s:
            self.key_state['backward'] = pressed
        elif key == K_a:
            self.key_state['left'] = pressed
        elif key == K_d:
            self.key_state['right'] = pressed

    def _update_cone(self, dt):
        if self.cone_actor is None or dt <= 0.0:
            return
        dx = 0.0
        dy = 0.0
        if self.key_state['forward']:
            dx += 1.0
        if self.key_state['backward']:
            dx -= 1.0
        if self.key_state['right']:
            dy += 1.0
        if self.key_state['left']:
            dy -= 1.0
        if dx == 0.0 and dy == 0.0:
            return
        length = math.hypot(dx, dy)
        if length == 0.0:
            return
        dx /= length
        dy /= length
        distance = TRAFFIC_CONE_SPEED * dt
        transform = self.cone_actor.get_transform()
        transform.location.x += dx * distance
        transform.location.y += dy * distance
        transform.location.z = TRAFFIC_CONE_START.location.z
        try:
            self.cone_actor.set_transform(transform)
        except RuntimeError as exc:
            print(f'[Cone] Failed to move cone: {exc}')

    def _draw_status(self):
        if self.display is None:
            return
        self.display.fill((16, 16, 16))
        if self.font is None:
            return
        lines = [
            'Traffic cone controller',
            'W/A/S/D: move, ESC/Q: quit'
        ]
        if self.cone_actor is not None:
            loc = self.cone_actor.get_transform().location
            lines.append(f'Pos: x={loc.x:.1f} y={loc.y:.1f} z={loc.z:.2f}')
        for idx, text in enumerate(lines):
            surf = self.font.render(text, True, (220, 220, 220))
            self.display.blit(surf, (10, 20 + idx * 26))

    def _cleanup(self):
        if self.cone_actor is not None:
            try:
                self.cone_actor.destroy()
                print('[Cone] Destroyed traffic cone.')
            except RuntimeError:
                pass
        pygame.quit()


def main():
    controller = TrafficConeController()
    controller.run()


if __name__ == '__main__':
    main()
