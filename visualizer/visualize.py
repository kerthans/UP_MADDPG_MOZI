# visualizer/visualize.py

import pygame
import sys
import numpy as np

# 可视化常量
WIDTH, HEIGHT = 800, 800
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
DEAD_COLOR = (128, 128, 128)

class Visualizer:
    def __init__(self, env, speed=10):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("多无人机对抗仿真")
        self.env = env
        self.clock = pygame.time.Clock()
        self.speed = speed  # 控制运动速度的参数

    def draw(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.screen.fill(WHITE)
        # 绘制红方无人机
        for i in range(self.env.num_red):
            pos = self.env.red_positions[i]
            alive = self.env.red_alive[i]
            color = RED if alive > 0 else DEAD_COLOR
            x, y = self._to_screen(pos)
            pygame.draw.circle(self.screen, color, (x, y), 10)
        # 绘制蓝方无人机
        for i in range(self.env.num_blue):
            pos = self.env.blue_positions[i]
            alive = self.env.blue_alive[i]
            color = BLUE if alive > 0 else DEAD_COLOR
            x, y = self._to_screen(pos)
            pygame.draw.circle(self.screen, color, (x, y), 10)
        pygame.display.flip()
        self.clock.tick(self.speed)  # 控制帧率

    def _to_screen(self, pos):
        # 将环境坐标转换为屏幕坐标
        x = int(WIDTH / 2 + pos[0] * 10)
        y = int(HEIGHT / 2 - pos[1] * 10)
        return x, y
        
    def close(self):
        pygame.quit()
        sys.exit()
