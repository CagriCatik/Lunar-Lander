"""Optional Pygame overlay; core MPC remains independent of rendering."""

import numpy as np


class FlightOverlay:
    def __init__(self, env):
        import pygame

        self.pygame = pygame
        self.env = env.unwrapped
        pygame.font.init()
        self.font = pygame.font.Font(None, 20)
        self.history = []
        pygame.display.set_caption("Lunar Lander | MPC | cyan: actual · yellow: prediction")

    def pixels(self, states):
        points = np.asarray(states)
        x = (points[:, 0] + 10) * 30
        y = 400 - (points[:, 1] + self.env.helipad_y + 18 / 30) * 30
        return np.clip(np.column_stack([x, y]), -10000, 10000).astype(int).tolist()

    def draw(self, state, prediction, reward, latency, action):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                return False
        screen = self.env.screen
        self.history.append(state.copy())
        if len(self.history) > 1:
            pygame.draw.lines(screen, (99, 216, 255), False, self.pixels(self.history), 2)
        if len(prediction) > 1:
            pygame.draw.lines(screen, (255, 226, 119), False, self.pixels(prediction), 2)
        lines = [
            f"Reward {reward:7.1f}  |  Control {latency:5.1f} ms",
            f"Main {action[0]:+.2f}  |  Lateral {action[1]:+.2f}",
            "Cyan: actual flight   Yellow: nominal MPC prediction",
        ]
        for row, text in enumerate(lines):
            surface = self.font.render(text, True, (220, 230, 245), (12, 16, 28))
            screen.blit(surface, (10, 10 + row * 20))
        pygame.display.flip()
        return True
