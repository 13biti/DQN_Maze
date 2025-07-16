import os
import random
from enum import Enum
from getkey import getkey

keys = ["w", "s", "a", "d"]


class objects_in_game(Enum):
    wall = -1
    space = 0
    goal = 1


class maze:
    def __init__(self, game_size) -> None:
        # game_size + walls
        self.play_ground_size = game_size + 1
        self.player_location = (1, 1)
        self.goal_location = (game_size - 1, game_size - 1)
        self.play_ground = {}
        self.wall_range = self.play_ground_size - 1

    def generate_playGround(self):
        for row in range(self.play_ground_size):
            for col in range(self.play_ground_size):
                # if col % self.wall_range == 0 or row % self.wall_range == 0:
                # everythin is wall , rather then playser position nad goal
                self.play_ground[(row, col)] = objects_in_game.wall

        self._carve_the_path(self.player_location[0], self.player_location[1])
        self.play_ground[self.goal_location] = objects_in_game.goal

    def generate_playGround_pattern(self):
        pattern = []
        for row in range(self.play_ground_size):
            pattern_row = []
            for col in range(self.play_ground_size):
                pos = (row, col)
                if (
                    pos in self.play_ground
                    and self.play_ground[pos] == objects_in_game.wall
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

    def _carve_the_path(self, x, y):
        self.play_ground[(x, y)] = objects_in_game.space

        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 < nx < self.wall_range and 0 < ny < self.wall_range:
                if self.play_ground.get((nx, ny)) == objects_in_game.wall:
                    mid_x = (x + nx) // 2
                    mid_y = (y + ny) // 2
                    self.play_ground[(mid_x, mid_y)] = objects_in_game.space
                    self._carve_the_path(nx, ny)

    def add_extra_paths(self, extra_count=10):
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
                        and self.play_ground[a] == objects_in_game.space
                        and self.play_ground[b] == objects_in_game.space
                    ):
                        self.play_ground[(x, y)] = objects_in_game.space
                        added += 1
                        break
            attempts += 1

    def clear_screan(self):
        os.system("clear")

    def print_maze(self, maze_pattern):
        self.clear_screan()
        for row in maze_pattern:
            print(" ".join(row))

    def move_player(self, dx, dy):
        new_x, new_y = self.player_location[0] + dx, self.player_location[1] + dy
        if 0 < new_x < self.wall_range and 0 < new_y < self.wall_range:
            if self.play_ground[(new_x, new_y)] != objects_in_game.wall:
                self.player_location = (new_x, new_y)  # Update player position
                return (
                    True
                    if self.play_ground[(new_x, new_y)] == objects_in_game.goal
                    else False
                )
        return False


game = maze(16)
game.generate_playGround()
game.add_extra_paths(extra_count=15)
while True:
    maze_pattern = game.generate_playGround_pattern()
    game.print_maze(maze_pattern)
    key = getkey()
    if key.lower() == "w":
        game.move_player(-1, 0)  # up
    elif key.lower() == "s":
        game.move_player(1, 0)  # down
    elif key.lower() == "a":
        game.move_player(0, -1)  # left
    elif key.lower() == "d":
        game.move_player(0, 1)  # right
    elif key.lower() == "q":
        print("\nGame quit!")
        break
    else:
        continue

    if game.player_location == game.goal_location:
        game.print_maze(game.generate_playGround_pattern())
        print("\n🎉 You reached the goal! 🎉")
        break
