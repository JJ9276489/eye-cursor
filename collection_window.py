from dataclasses import dataclass
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
        pygame.font.init()
        self.visible = True
        self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        pygame.display.set_caption("Data Collection")
        pygame.mouse.set_visible(True)

        self.screen_size = self.surface.get_size()
        self.title_font = pygame.font.SysFont(None, 56)
        self.body_font = pygame.font.SysFont(None, 38)
        self.small_font = pygame.font.SysFont(None, 28)

    def close(self) -> None:
        pygame.display.quit()
        pygame.quit()

    def toggle_visibility(self) -> None:
        self.visible = not self.visible
        if self.visible:
            self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            pygame.mouse.set_visible(True)
        else:
            self.surface = pygame.display.set_mode((1, 1), pygame.HIDDEN)

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
        split = collection.record_split if collection.recording else collection.next_split
        split_color = (0, 170, 255) if split == "eval" else (80, 255, 0)
        if collection.recording:
            split_color = (0, 110, 255) if split == "eval" else (0, 255, 255)

        pygame.draw.circle(self.surface, split_color, (target_x, target_y), 30, 2)
        pygame.draw.circle(self.surface, split_color, (target_x, target_y), 8)
        pygame.draw.line(self.surface, split_color, (target_x - 44, target_y), (target_x + 44, target_y), 2)
        pygame.draw.line(self.surface, split_color, (target_x, target_y - 44), (target_x, target_y + 44), 2)

        title = "Recording" if collection.recording else "Manual Collection"
        self._blit_text(self.title_font, title, (60, 56), (255, 255, 255))
        self._blit_text(self.body_font, f"split {split.upper()}", (60, 112), split_color)
        self._blit_text(
            self.small_font,
            f"random split {int(collection.eval_ratio * 100)}% eval | train {collection.split_counts['train']} eval {collection.split_counts['eval']}",
            (60, 142),
            (190, 190, 190),
        )

        if collection.recording:
            progress = min(
                1.0,
                (time.monotonic() - collection.record_started_at) / COLLECTION_RECORD_SECONDS,
            )
            bar_left = 60
            bar_top = 164
            bar_width = min(520, width - 120)
            bar_height = 20
            pygame.draw.rect(
                self.surface,
                (80, 80, 80),
                pygame.Rect(bar_left, bar_top, bar_width, bar_height),
                1,
            )
            pygame.draw.rect(
                self.surface,
                split_color,
                pygame.Rect(bar_left, bar_top, int(progress * bar_width), bar_height),
            )
            self._blit_text(
                self.body_font,
                f"capturing {len(collection.pending_samples)} samples",
                (60, 202),
                (255, 255, 255),
            )
        else:
            self._blit_text(
                self.body_font,
                "Move the target with the mouse, fix your gaze, then press space.",
                (60, 184),
                (255, 255, 255),
            )
            self._blit_text(
                self.body_font,
                "The target freezes immediately and records for 1 second.",
                (60, 246),
                (205, 205, 205),
            )

        self._blit_text(
            self.small_font,
            f"target {target_x}, {target_y}",
            (40, height - 82),
            (220, 220, 220),
        )
        self._blit_text(
            self.small_font,
            f"captures saved {collection.capture_count}",
            (40, height - 46),
            (220, 220, 220),
        )
        self._blit_text(
            self.small_font,
            "m hide collection  g preview  v model  q quit",
            (40, height - 118),
            (180, 180, 180),
        )

        pygame.display.flip()

    def _blit_text(
        self,
        font: pygame.font.Font,
        text: str,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        surface = font.render(text, True, color)
        self.surface.blit(surface, position)
