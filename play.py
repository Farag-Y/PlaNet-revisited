import pygame
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from env_wrapper import Env

DISPLAY_SIZE = 480
FOOTER_HEIGHT = 130

# Key bindings per action dimension (positive key, negative key, label)
KEY_BINDINGS = [
    (pygame.K_RIGHT, pygame.K_LEFT,  "← / →"),
    (pygame.K_UP,    pygame.K_DOWN,  "↑ / ↓"),
    (pygame.K_d,     pygame.K_a,     "A / D"),
    (pygame.K_w,     pygame.K_s,     "W / S"),
]


def build_action(keys, action_size, action_min, action_max):
    action = np.zeros(action_size, dtype=np.float32)
    for i, (pos_key, neg_key, _) in enumerate(KEY_BINDINGS[:action_size]):
        if keys[pos_key]:
            action[i] = action_max
        elif keys[neg_key]:
            action[i] = action_min
    return torch.tensor(action).unsqueeze(0)


def draw_overlay(surface, font, small_font, cfg_env, action, action_size, episode_reward, step):
    overlay_rect = pygame.Rect(0, DISPLAY_SIZE, DISPLAY_SIZE, FOOTER_HEIGHT)
    pygame.draw.rect(surface, (20, 20, 20), overlay_rect)
    pygame.draw.line(surface, (60, 60, 60), (0, DISPLAY_SIZE), (DISPLAY_SIZE, DISPLAY_SIZE), 1)

    x, y = 12, DISPLAY_SIZE + 8

    title = font.render(cfg_env, True, (220, 220, 220))
    surface.blit(title, (x, y))

    stats = small_font.render(f"step {step}   reward {episode_reward:.2f}", True, (160, 160, 160))
    surface.blit(stats, (DISPLAY_SIZE - stats.get_width() - 12, y + 4))

    y += 30
    pygame.draw.line(surface, (50, 50, 50), (x, y), (DISPLAY_SIZE - x, y), 1)
    y += 8

    for i in range(min(action_size, len(KEY_BINDINGS))):
        _, _, label = KEY_BINDINGS[i]
        val = action[0, i].item()
        bar_color = (80, 180, 80) if val > 0 else (180, 80, 80) if val < 0 else (80, 80, 80)
        binding_text = small_font.render(f"action[{i}]  {label}  {val:+.2f}", True, bar_color)
        surface.blit(binding_text, (x, y))
        y += 20

    if action_size > len(KEY_BINDINGS):
        note = small_font.render(f"(actions [{len(KEY_BINDINGS)}..{action_size-1}] = 0, no keys mapped)", True, (100, 100, 100))
        surface.blit(note, (x, y))
        y += 18

    quit_text = small_font.render("Q quit   R reset", True, (100, 100, 100))
    surface.blit(quit_text, (x, DISPLAY_SIZE + FOOTER_HEIGHT - 18))


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    env = Env(cfg.env, seed=cfg.seed, max_episode_length=cfg.max_episode_length,
              action_repeat=1, bit_depth=cfg.bit_depth)

    action_size = env.action_size
    action_min, action_max = env.action_range

    print(f"\nPlaying: {cfg.env}")
    print(f"Action dims: {action_size}  range: [{action_min:.2f}, {action_max:.2f}]")
    for i in range(min(action_size, len(KEY_BINDINGS))):
        _, _, label = KEY_BINDINGS[i]
        print(f"  action[{i}]: {label}")
    if action_size > len(KEY_BINDINGS):
        print(f"  action[{len(KEY_BINDINGS)}..{action_size-1}]: unmapped (always 0)")
    print("  Q: quit   R: reset\n")

    pygame.init()
    pygame.display.set_caption(cfg.env)
    screen = pygame.display.set_mode((DISPLAY_SIZE, DISPLAY_SIZE + FOOTER_HEIGHT))
    font       = pygame.font.SysFont("monospace", 15, bold=True)
    small_font = pygame.font.SysFont("monospace", 13)
    clock = pygame.time.Clock()

    obs = env.reset()
    action = torch.zeros(1, action_size)
    episode_reward = 0.0
    step = 0
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                if event.key == pygame.K_r:
                    print(f"Episode ended — reward: {episode_reward:.2f}  steps: {step}")
                    obs = env.reset()
                    episode_reward = 0.0
                    step = 0

        keys = pygame.key.get_pressed()
        action = build_action(keys, action_size, action_min, action_max)

        _, reward, done = env.step(action)
        episode_reward += reward
        step += 1

        if done:
            print(f"Episode ended — reward: {episode_reward:.2f}  steps: {step}")
            obs = env.reset()
            episode_reward = 0.0
            step = 0

        frame = env.render_frame(height=DISPLAY_SIZE, width=DISPLAY_SIZE)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))
        draw_overlay(screen, font, small_font, cfg.env, action, action_size, episode_reward, step)
        pygame.display.flip()
        clock.tick(30)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
