import os
import random
from enum import Enum
from getkey import getkey
import numpy as np

keys = ["w", "s", "a", "d"]


class ObjectsInGame(Enum):
    WALL = 0
    SPACE = 1
    GOAL = 2


class maze:
    def __init__(self, game_size, max_frame, extra_paths) -> None:
        # game_size + walls
        self.play_ground_size = game_size + 1
        self.player_location = (1, 1)
        self.goal_location = (game_size - 1, game_size - 1)
        self.play_ground = {}
        self.wall_range = self.play_ground_size - 1
        # this should use carefully , it may not allow user to have minimum moves
        self.max_frame = max_frame
        self.extra_paths = extra_paths
        self.actions = [0, 1, 2, 3]
        self.action_map = {
            0: [(0, -1), "UP"],
            1: [(0, 1), "DOWN"],
            2: [(-1, 0), "LEFT"],
            3: [(1, 0), "RIGHT"],
        }
        self.huristic = 0
        self.step_counter = 0
        # 0 for hiting the wall , 0.1 for space and 1 for goal
        self.reward_map = {0: 0, 1: 0.1, 2: 1}

    # ----------------------------------
    # map generation codes :
    # ----------------------------------
    def generate_playGround(self):
        for row in range(self.play_ground_size):
            for col in range(self.play_ground_size):
                # if col % self.wall_range == 0 or row % self.wall_range == 0:
                # everythin is wall , rather then playser position nad goal
                self.play_ground[(row, col)] = ObjectsInGame.WALL

        self._carve_the_path(self.player_location[0], self.player_location[1])
        self.play_ground[self.goal_location] = ObjectsInGame.GOAL

    def generate_visual_pattern(self):
        pattern = []
        for row in range(self.play_ground_size):
            pattern_row = []
            for col in range(self.play_ground_size):
                pos = (row, col)
                if (
                    pos in self.play_ground
                    and self.play_ground[pos] == ObjectsInGame.WALL
                ):
                    pattern_row.append("#")
                elif pos == self.goal_location:
                    pattern_row.append("G")
                elif pos == self.player_location:
                    pattern_row.append("P")
                else:
                    pattern_row.append(" ")
            pattern.append(pattern_row)
        return pattern

    def generate_playGround_from_pattern(self, pattern):
        pattern = []
        for row in range(self.play_ground_size):
            pattern_row = []
            for col in range(self.play_ground_size):
                pos = (row, col)
                if pos in self.play_ground and pattern[row][col] == "#":
                    self.play_ground[pos] = ObjectsInGame.WALL
                elif pos == self.goal_location:
                    self.play_ground[pos] = ObjectsInGame.GOAL
                elif pos == self.player_location:
                    self.play_ground[pos] = ObjectsInGame.SPACE
                else:
                    self.play_ground[pos] = ObjectsInGame.SPACE
            pattern.append(pattern_row)
        return pattern

    # ----------------------------------
    # creating a path to goal  :
    # ----------------------------------
    def _carve_the_path(self, x, y):
        self.play_ground[(x, y)] = ObjectsInGame.SPACE

        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 < nx < self.wall_range and 0 < ny < self.wall_range:
                if self.play_ground.get((nx, ny)) == ObjectsInGame.WALL:
                    mid_x = (x + nx) // 2
                    mid_y = (y + ny) // 2
                    self.play_ground[(mid_x, mid_y)] = ObjectsInGame.SPACE
                    self._carve_the_path(nx, ny)

    def add_extra_paths(self, extra_count=10):
        self.extra_paths = extra_count
        attempts = 0
        added = 0

        while added < extra_count and attempts < extra_count * 5:
            x = random.randint(1, self.wall_range - 1)
            y = random.randint(1, self.wall_range - 1)

            if x % 2 == 1 or y % 2 == 1:  # only consider wall positions
                neighbors = [
                    ((x - 1, y), (x + 1, y)),
                    ((x, y - 1), (x, y + 1)),
                ]
                for a, b in neighbors:
                    if (
                        a in self.play_ground
                        and b in self.play_ground
                        and self.play_ground[a] == ObjectsInGame.SPACE
                        and self.play_ground[b] == ObjectsInGame.SPACE
                    ):
                        self.play_ground[(x, y)] = ObjectsInGame.SPACE
                        added += 1
                        break
            attempts += 1

    # ----------------------------------
    # render the screan :
    # ----------------------------------
    def clear_screan(self):
        os.system("clear")

    def print_maze(self, maze_pattern):
        self.clear_screan()
        for row in maze_pattern:
            print(" ".join(row))

    # ----------------------------------
    # reset the game:
    # hard_reset will change the map !!
    # ----------------------------------
    def soft_reset(self):
        self.player_location = (1, 1)
        self.huristic = self._manhattan_distance(
            self.player_location, self.goal_location
        )
        return self.get_state()

    def hard_reset(self):
        self.play_ground = {}
        self.generate_playGround()
        self.add_extra_paths(extra_count=self.extra_paths)
        self.player_location = (1, 1)
        self.huristic = self._manhattan_distance(
            self.player_location, self.goal_location
        )
        return self.get_state()

    def reset(self, hard_reset=False):
        if hard_reset:
            return self.hard_reset()
        return self.soft_reset()

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # ----------------------------------
    # interactions with agent :
    # ----------------------------------
    def get_state(self):
        return np.array([self.player_location], dtype=np.float32)

    def get_actions(self):
        return self.actions

    def get_actions_discription(self):
        discriptions = []
        for action in self.action_map.values():
            discriptions.append(action[1])

    def act(self, action: int):
        self.step_counter += 1
        isSuccess = False
        info = "message"
        reward = 0
        next_state = 0
        game_done = False
        if action not in self.actions:
            return isSuccess, "action is out of range , ", reward, next_state, game_done

        directions = self.action_map.get(action, [(0, 0)])
        dy, dx = directions[0]
        position = self.player_location[0] + dy, self.player_location[1] + dx
        if self.play_ground[position] == ObjectsInGame.WALL:
            isSuccess, info, reward, next_state, game_done = (
                True,
                "hit the wall ",
                self.reward_map.get(ObjectsInGame.WALL.value),
                self.get_state(),
                False,
            )
            self.player_location = position

        elif self.play_ground[position] == ObjectsInGame.SPACE:
            isSuccess, info, reward, next_state, game_done = (
                True,
                "ok",
                self.reward_map.get(ObjectsInGame.SPACE.value),
                self.get_state(),
                False,
            )
            self.player_location = position
        else:
            isSuccess, info, reward, next_state, game_done = (
                True,
                "you win",
                self.reward_map.get(ObjectsInGame.GOAL.value),
                self.get_state(),
                True,
            )
            self.player_location = position
        return isSuccess, info, reward, next_state, game_done
