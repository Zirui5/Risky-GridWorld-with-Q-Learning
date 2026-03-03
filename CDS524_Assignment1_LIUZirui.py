"""
Risky GridWorld (Human / AI) — interactive demo (pygame)

MENU:
  ENTER   start
  ESC     quit

IN GAME:
  H       human mode (arrow keys)
  A       AI mode (auto-play)
  SPACE   AI single-step (AI mode)
  R       reset episode (same map)
  N       new map (regenerate map)
  M       back to menu
  ESC     quit

Notes
-----
- This is a playable UI for the assignment (the "interactive_demo" UI part).
- It optionally loads a trained Q-table from `qtable.pkl` (same folder as this script).
  If missing, AI still runs using a simple fallback heuristic instead of crashing.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pygame


# ============================
# Layout / UI config
# ============================
GRID = 8
CELL = 62
PADDING = 22

HUD_H = 160
LEGEND_W = 220

W = GRID * CELL + 2 * PADDING
H = GRID * CELL + 2 * PADDING + HUD_H

FPS = 12

# Colors (RGB)
BG     = (24, 24, 34)
EMPTY  = (44, 40, 60)
WALL   = (135, 135, 135)
TRAP   = (210, 55, 55)
START  = (70, 130, 230)
GOAL   = (60, 185, 95)
ENEMY  = (255, 140, 0)
AGENT  = (255, 215, 0)
TEXT   = (240, 240, 240)
SUBTXT = (200, 200, 200)

# Actions (keep consistent with your notebook)
A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY = 0, 1, 2, 3, 4
ACTION_TO_DELTA = {
    A_UP: (-1, 0),
    A_DOWN: (1, 0),
    A_LEFT: (0, -1),
    A_RIGHT: (0, 1),
    A_STAY: (0, 0),
}
ACTION_NAME = {A_UP: "UP", A_DOWN: "DOWN", A_LEFT: "LEFT", A_RIGHT: "RIGHT", A_STAY: "STAY"}


@dataclass
class StepResult:
    next_state: Tuple[int, int, int, int]
    reward: float
    done: bool
    info: dict


# ============================
# Environment
# ============================
class RiskyGrid:
    """
    8x8 gridworld with:
      - walls (obstacles)
      - traps (terminal, negative reward)
      - a moving enemy (random move after the agent)
    State: (agent_r, agent_c, enemy_r, enemy_c)
    """

    def __init__(self, seed: int = 1, obstacle_prob: float = 0.2, trap_prob: float = 0.2):
        self.start = (0, 0)
        self.goal = (GRID - 1, GRID - 1)

        # Rewards (match what you used in training / demo)
        self.step_reward = -0.1
        self.goal_reward = 20.0
        self.invalid_penalty = -1.0
        self.trap_penalty = -25.0
        self.enemy_penalty = -25.0

        self.actions = [A_UP, A_DOWN, A_LEFT, A_RIGHT, A_STAY]

        self.obstacle_prob = obstacle_prob
        self.trap_prob = trap_prob

        self.rng = np.random.default_rng(seed)
        self.obstacles: set[Tuple[int, int]] = set()
        self.traps: set[Tuple[int, int]] = set()

        self._generate_map()

        # Agent always starts at start
        self.agent = self.start

        # Important: keep enemy start consistent *within the same map*.
        # That makes behavior more “stable” (and easier to explain).
        self.enemy0 = self._spawn_enemy()
        self.enemy = self.enemy0

    def _in_bounds(self, rc: Tuple[int, int]) -> bool:
        r, c = rc
        return 0 <= r < GRID and 0 <= c < GRID

    def _neighbors(self, rc: Tuple[int, int]):
        r, c = rc
        for a in (A_UP, A_DOWN, A_LEFT, A_RIGHT):
            dr, dc = ACTION_TO_DELTA[a]
            nxt = (r + dr, c + dc)
            if self._in_bounds(nxt) and (nxt not in self.obstacles):
                yield nxt

    def _has_path(self) -> bool:
        """BFS check to ensure start can reach goal (so we don't generate impossible maps)."""
        from collections import deque

        q = deque([self.start])
        seen = {self.start}
        while q:
            cur = q.popleft()
            if cur == self.goal:
                return True
            for nxt in self._neighbors(cur):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return False

    def _generate_obstacles(self):
        all_cells = [(r, c) for r in range(GRID) for c in range(GRID)]
        candidates = [x for x in all_cells if x not in (self.start, self.goal)]

        # Try multiple times until we get a solvable map
        for _ in range(2000):
            self.obstacles.clear()
            for cell in candidates:
                if self.rng.random() < self.obstacle_prob:
                    self.obstacles.add(cell)
            if self._has_path():
                return

        # Fallback: if we somehow fail, no walls.
        self.obstacles.clear()

    def _generate_traps(self):
        self.traps.clear()
        for r in range(GRID):
            for c in range(GRID):
                rc = (r, c)
                if rc in self.obstacles or rc in (self.start, self.goal):
                    continue
                if self.rng.random() < self.trap_prob:
                    self.traps.add(rc)

    def _spawn_enemy(self) -> Tuple[int, int]:
        free = [
            (r, c)
            for r in range(GRID)
            for c in range(GRID)
            if (r, c) not in self.obstacles
            and (r, c) not in self.traps
            and (r, c) not in (self.start, self.goal)
        ]
        return free[int(self.rng.integers(0, len(free)))] if free else (GRID - 1, 0)

    def _generate_map(self):
        self._generate_obstacles()
        self._generate_traps()

    def reset(self):
        """Reset episode on the same map."""
        self.agent = self.start
        self.enemy = self.enemy0
        return self.state()

    def new_map(self):
        """Generate a new map, and also refresh enemy0."""
        self._generate_map()
        self.enemy0 = self._spawn_enemy()
        return self.reset()

    def state(self) -> Tuple[int, int, int, int]:
        ar, ac = self.agent
        er, ec = self.enemy
        return (ar, ac, er, ec)

    def enemy_move(self):
        """Enemy moves randomly (can also stay). Enemy never goes into start/goal/walls."""
        er, ec = self.enemy
        moves: List[Tuple[int, int]] = []
        for a in self.actions:
            dr, dc = ACTION_TO_DELTA[a]
            nxt = (er + dr, ec + dc)
            if not self._in_bounds(nxt):
                continue
            if nxt in self.obstacles or nxt in (self.start, self.goal):
                continue
            moves.append(nxt)
        if moves:
            self.enemy = moves[int(self.rng.integers(0, len(moves)))]

    def step(self, action: int) -> StepResult:
        ar, ac = self.agent
        dr, dc = ACTION_TO_DELTA[action]
        nxt = (ar + dr, ac + dc)

        # Invalid move: agent stays, but still pays penalty, and enemy moves.
        if (not self._in_bounds(nxt)) or (nxt in self.obstacles):
            self.enemy_move()
            if self.enemy == self.agent:
                return StepResult(self.state(), self.enemy_penalty, True, {"enemy": True})
            return StepResult(self.state(), self.invalid_penalty, False, {"invalid": True})

        # Valid move
        self.agent = nxt

        if self.agent in self.traps:
            return StepResult(self.state(), self.trap_penalty, True, {"trap": True})

        if self.agent == self.goal:
            return StepResult(self.state(), self.goal_reward, True, {"goal": True})

        # Enemy acts after agent
        self.enemy_move()
        if self.enemy == self.agent:
            return StepResult(self.state(), self.enemy_penalty, True, {"enemy": True})

        return StepResult(self.state(), self.step_reward, False, {})


# ============================
# Policy (Q-table + fallback)
# ============================
class QTablePolicy:
    """
    Q-table format: dict[state_tuple] -> np.array(|A|)
    If state not in Q, we fall back to a simple heuristic (safer than pure random).
    """

    def __init__(self, actions: List[int], rng: Optional[np.random.Generator] = None):
        self.actions = actions
        self.Q: Dict[Tuple[int, int, int, int], np.ndarray] = {}
        self.rng = rng or np.random.default_rng(0)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
        return True

    def act(self, state: Tuple[int, int, int, int], env: Optional[RiskyGrid] = None) -> int:
        # If we have Q-values, go greedy.
        if state in self.Q:
            q = self.Q[state]
            return int(self.actions[int(np.argmax(q))])

        # Otherwise, use a small heuristic so AI doesn't look “drunk”.
        if env is None:
            return int(self.rng.choice(self.actions))

        ar, ac, er, ec = state
        gr, gc = env.goal

        best_a: Optional[int] = None
        best_score = -1e9

        for a in self.actions:
            dr, dc = ACTION_TO_DELTA[a]
            nr, nc = ar + dr, ac + dc

            if not env._in_bounds((nr, nc)) or (nr, nc) in env.obstacles:
                continue

            score = 0.0

            # Prefer moving closer to goal (Manhattan distance)
            old_d = abs(ar - gr) + abs(ac - gc)
            new_d = abs(nr - gr) + abs(nc - gc)
            score += (old_d - new_d) * 2.0

            # Strongly avoid stepping into traps
            if (nr, nc) in env.traps:
                score -= 100.0

            # Avoid being adjacent to enemy (simple safety margin)
            if abs(nr - er) + abs(nc - ec) <= 1:
                score -= 10.0

            # Slight dislike for staying
            if a == A_STAY:
                score -= 0.5

            if score > best_score:
                best_score = score
                best_a = a

        return int(best_a) if best_a is not None else int(self.rng.choice(self.actions))


# ============================
# UI helpers
# ============================
def wrap_text(text: str, font: pygame.font.Font, max_w: int) -> List[str]:
    """Lightweight word-wrap for HUD/menu text."""
    if not text:
        return [""]

    words = text.split()
    lines: List[str] = []
    cur: List[str] = []

    for w in words:
        test = " ".join(cur + [w])
        if font.size(test)[0] <= max_w:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
                cur = [w]
            else:
                # One very long word; just push it to avoid infinite loop.
                lines.append(w)
                cur = []

    if cur:
        lines.append(" ".join(cur))
    return lines


def draw_grid(screen: pygame.Surface, env: RiskyGrid):
    """Draw the 8x8 grid with rounded tiles."""
    for r in range(GRID):
        for c in range(GRID):
            x = PADDING + c * CELL
            y = PADDING + r * CELL
            rc = (r, c)

            color = EMPTY
            if rc in env.obstacles:
                color = WALL
            if rc in env.traps:
                color = TRAP
            if rc == env.start:
                color = START
            if rc == env.goal:
                color = GOAL
            if rc == env.enemy:
                color = ENEMY
            if rc == env.agent:
                color = AGENT

            pygame.draw.rect(
                screen,
                color,
                (x, y, CELL - 6, CELL - 6),
                border_radius=14,
            )


def draw_legend(screen: pygame.Surface, font: pygame.font.Font, x: int, y: int):
    items = [
        ("Empty", EMPTY),
        ("Wall", WALL),
        ("Trap", TRAP),
        ("Start", START),
        ("Goal", GOAL),
        ("Enemy", ENEMY),
        ("Agent", AGENT),
    ]
    box = 16
    pad = 8
    line_h = 22

    for i, (name, color) in enumerate(items):
        yy = y + i * line_h
        pygame.draw.rect(screen, color, (x, yy, box, box), border_radius=4)
        screen.blit(font.render(name, True, SUBTXT), (x + box + pad, yy - 2))


def draw_hud(
    screen: pygame.Surface,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
    hud1: str,
    hud2: str,
):
    """HUD strip: status line + detail line(s) + legend on the right."""
    hud_y = PADDING + GRID * CELL + 10

    pygame.draw.rect(screen, (18, 18, 26), (0, hud_y - 8, W, HUD_H))

    # Leave room for legend
    text_area_w = W - 2 * PADDING - LEGEND_W
    text_area_w = max(text_area_w, 240)

    # Line 1: wrap if needed (up to 2 lines)
    hud1_lines = wrap_text(hud1, title_font, text_area_w)[:2]
    x0 = PADDING
    y0 = hud_y + 8
    title_gap = 26

    for i, line in enumerate(hud1_lines):
        screen.blit(title_font.render(line, True, TEXT), (x0, y0 + i * title_gap))

    # Line 2: start below line 1
    detail_y0 = y0 + len(hud1_lines) * title_gap + 10
    line_gap = 22
    bottom_margin = 12
    available_h = (hud_y - 8 + HUD_H) - (detail_y0 + bottom_margin)
    max_lines = max(1, available_h // line_gap)

    hud2_lines = wrap_text(hud2, small_font, text_area_w)[:max_lines]
    for i, line in enumerate(hud2_lines):
        screen.blit(small_font.render(line, True, SUBTXT), (x0, detail_y0 + i * line_gap))

    # Legend on the right
    legend_x = W - PADDING - LEGEND_W + 10
    legend_y = hud_y + 10
    draw_legend(screen, small_font, legend_x, legend_y)


def draw_menu(screen: pygame.Surface, big_font: pygame.font.Font, font: pygame.font.Font):
    screen.fill(BG)

    title = big_font.render("Risky GridWorld", True, TEXT)
    subtitle = font.render("Interactive Demo (Human / AI)", True, SUBTXT)
    screen.blit(title, (PADDING, PADDING))
    screen.blit(subtitle, (PADDING, PADDING + 60))

    goal = "Goal: reach the green cell without stepping on traps (red) or colliding with the enemy (orange)."
    y = PADDING + 120

    max_w = W - 2 * PADDING
    for line in wrap_text(goal, font, max_w):
        screen.blit(font.render(line, True, TEXT), (PADDING, y))
        y += 28

    y += 14

    controls = [
        "Controls:",
        "  ENTER  start",
        "  H      human mode (arrow keys)",
        "  A      AI mode (auto-play)",
        "  SPACE  AI single-step",
        "  R      reset episode",
        "  N      new map",
        "  M      back to menu",
        "  ESC    quit",
    ]
    for line in controls:
        screen.blit(font.render(line, True, TEXT), (PADDING, y))
        y += 28


def draw_game_over_overlay(
    screen: pygame.Surface,
    big_font: pygame.font.Font,
    font: pygame.font.Font,
    msg: str,
):
    overlay = pygame.Surface((W, H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 170))
    screen.blit(overlay, (0, 0))

    title = big_font.render("Game Over", True, TEXT)
    screen.blit(title, (PADDING, PADDING))

    wrapped = wrap_text(msg, font, W - 2 * PADDING)
    for i, line in enumerate(wrapped[:6]):
        screen.blit(font.render(line, True, TEXT), (PADDING, PADDING + 80 + i * 30))

    hint = font.render("Press R to retry, M for menu, ESC to quit.", True, SUBTXT)
    screen.blit(hint, (PADDING, H - 60))


# ============================
# Main loop
# ============================
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Risky GridWorld (Human / AI)")
    clock = pygame.time.Clock()

    big_font = pygame.font.SysFont("consolas", 46)
    title_font = pygame.font.SysFont("consolas", 24)
    font = pygame.font.SysFont("consolas", 20)
    small = pygame.font.SysFont("consolas", 18)

    env = RiskyGrid(seed=1, obstacle_prob=0.2, trap_prob=0.2)

    policy = QTablePolicy(actions=env.actions, rng=np.random.default_rng(123))

    # Default: look for qtable.pkl next to this script
    here = os.path.dirname(os.path.abspath(__file__))
    q_path = os.path.join(here, "qtable.pkl")

    # If you want, you can override by setting an env var:
    #   set QTABLE_PATH=D:\Downloads\qtable.pkl
    q_path = os.environ.get("QTABLE_PATH", q_path)

    q_loaded = policy.load(q_path)

    game_state = "MENU"   # MENU, PLAY, GAME_OVER
    mode = "HUMAN"        # HUMAN or AI
    auto_ai = False

    total_return = 0.0
    done = False
    end_reason = ""
    last_msg = "Ready."

    env.reset()

    running = True
    while running:
        clock.tick(FPS)
        action: Optional[int] = None

        # ---- input handling ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # MENU
                if game_state == "MENU":
                    if event.key == pygame.K_RETURN:
                        game_state = "PLAY"
                        env.reset()
                        total_return = 0.0
                        done = False
                        end_reason = ""
                        last_msg = "Started. Use arrows (HUMAN) or press A for AI."
                    continue

                # PLAY / GAME_OVER common keys
                if event.key == pygame.K_m:
                    game_state = "MENU"
                    continue

                if event.key == pygame.K_r:
                    env.reset()
                    total_return = 0.0
                    done = False
                    end_reason = ""
                    last_msg = "Reset episode."
                    if game_state == "GAME_OVER":
                        game_state = "PLAY"

                if event.key == pygame.K_n:
                    env.new_map()
                    total_return = 0.0
                    done = False
                    end_reason = ""
                    last_msg = "New map generated."
                    if game_state == "GAME_OVER":
                        game_state = "PLAY"

                if event.key == pygame.K_h:
                    mode = "HUMAN"
                    auto_ai = False
                    last_msg = "Mode: HUMAN (arrow keys)."

                if event.key == pygame.K_a:
                    mode = "AI"
                    auto_ai = True
                    last_msg = "Mode: AI (auto-play)."

                if event.key == pygame.K_SPACE and mode == "AI" and (not done) and game_state == "PLAY":
                    auto_ai = False
                    action = policy.act(env.state(), env=env)

                # Human move only during PLAY
                if mode == "HUMAN" and (not done) and game_state == "PLAY":
                    if event.key == pygame.K_UP:
                        action = A_UP
                    elif event.key == pygame.K_DOWN:
                        action = A_DOWN
                    elif event.key == pygame.K_LEFT:
                        action = A_LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = A_RIGHT
                    elif event.key == pygame.K_RETURN:
                        action = A_STAY

        # ---- AI auto-step ----
        if game_state == "PLAY" and mode == "AI" and auto_ai and (not done) and action is None:
            action = policy.act(env.state(), env=env)

        # ---- environment step ----
        if game_state == "PLAY" and action is not None and (not done):
            res = env.step(action)
            total_return += res.reward

            if res.done:
                done = True
                if res.info.get("goal"):
                    end_reason = "GOAL"
                elif res.info.get("trap"):
                    end_reason = "TRAP"
                elif res.info.get("enemy"):
                    end_reason = "ENEMY"
                else:
                    end_reason = "DONE"

                last_msg = f"{end_reason}. total_return={total_return:.2f}"
                game_state = "GAME_OVER"
            else:
                last_msg = f"mode={mode} a={ACTION_NAME[action]} r={res.reward:.2f} total={total_return:.2f}"

        # ---- render ----
        if game_state == "MENU":
            draw_menu(screen, big_font, font)
            pygame.display.flip()
            continue

        screen.fill(BG)
        draw_grid(screen, env)

        ar, ac, er, ec = env.state()
        hud1 = (
            f"Mode: {mode} | Return: {total_return:.2f} | "
            f"Agent=({ar},{ac}) Enemy=({er},{ec}) | Q: {'loaded' if q_loaded else 'missing'}"
        )
        hud2 = last_msg
        draw_hud(screen, title_font, small, hud1, hud2)

        if game_state == "GAME_OVER":
            draw_game_over_overlay(screen, big_font, font, f"{end_reason}. {last_msg}")

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()