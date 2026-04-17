from dataclasses import dataclass
import math
import time

import pygame

from collector import CollectionState, current_target, set_target_from_pixels
from constants import COLLECTION_RECORD_SECONDS


@dataclass
class CollectionEvents:
    quit_requested: bool = False
    begin_recording: bool = False
    toggle_preview: bool = False
    toggle_model: bool = False
    clear_history: bool = False
    toggle_collection: bool = False


class CollectionWindow:
    def __init__(self) -> None:
        pygame.init()
        self.visible = True
        self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Data Collection")
        pygame.mouse.set_visible(False)

        self.screen_size = self.surface.get_size()

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def toggle_visibility(self) -> None:
        self.visible = not self.visible
        if self.visible:
            self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        else:
            self.surface = pygame.display.set_mode((1, 1), pygame.HIDDEN)
            pygame.mouse.set_visible(True)

    def poll_events(self, collection: CollectionState) -> CollectionEvents:
        polled = CollectionEvents()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                polled.quit_requested = True
            elif event.type == pygame.MOUSEMOTION and not collection.recording:
                set_target_from_pixels(collection, event.pos[0], event.pos[1])
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    polled.quit_requested = True
                elif event.key == pygame.K_SPACE:
                    polled.begin_recording = True
                elif event.key == pygame.K_g:
                    polled.toggle_preview = True
                elif event.key == pygame.K_v:
                    polled.toggle_model = True
                elif event.key == pygame.K_c:
                    polled.clear_history = True
                elif event.key == pygame.K_m:
                    polled.toggle_collection = True
        return polled

    def render(self, collection: CollectionState) -> None:
        if not self.visible:
            return

        width, height = self.screen_size
        self.surface.fill((0, 0, 0))

        target = current_target(collection)
        target_x = int(target[0] * (width - 1))
        target_y = int(target[1] * (height - 1))
        target_color = (120, 255, 90)
        recording_color = (0, 245, 255)
        crosshair_color = recording_color if collection.recording else target_color

        outer_radius = 28
        inner_radius = 7
        arm_length = 42

        if collection.recording:
            progress = min(
                1.0,
                (time.monotonic() - collection.record_started_at) / COLLECTION_RECORD_SECONDS,
            )
            pygame.draw.circle(self.surface, (55, 55, 55), (target_x, target_y), outer_radius + 8, 2)
            if progress > 0.0:
                arc_rect = pygame.Rect(
                    target_x - (outer_radius + 8),
                    target_y - (outer_radius + 8),
                    (outer_radius + 8) * 2,
                    (outer_radius + 8) * 2,
                )
                pygame.draw.arc(
                    self.surface,
                    recording_color,
                    arc_rect,
                    -math.pi / 2,
                    -math.pi / 2 + progress * math.tau,
                    4,
                )

        pygame.draw.circle(self.surface, crosshair_color, (target_x, target_y), outer_radius, 2)
        pygame.draw.circle(self.surface, crosshair_color, (target_x, target_y), inner_radius)
        pygame.draw.line(
            self.surface,
            crosshair_color,
            (target_x - arm_length, target_y),
            (target_x + arm_length, target_y),
            2,
        )
        pygame.draw.line(
            self.surface,
            crosshair_color,
            (target_x, target_y - arm_length),
            (target_x, target_y + arm_length),
            2,
        )

        pygame.display.flip()
