import os
import random
from enum import Enum
from types import new_class, prepare_class
from typing import Self


class objects_in_game(Enum):
    wall = -1
    space = 0
    goal = 1


class maze:
    def __init__(self, game_size) -> None:
        # game_size + walls
        self.play_ground_size = game_size + 2
        self.keys = ["w", "s", "a", "d"]
        self.goal_location = (1, 1)
        self.player_location = (game_size, game_size)
        self.play_ground = {}
        self.wall_range = self.play_ground_size - 1

    def generate_playGround(self):
        for row in range(self.play_ground_size):
            for col in range(self.play_ground_size):
                # if col % self.wall_range == 0 or row % self.wall_range == 0:
                # everythin is wall , rather then playser position nad goal
                self.play_ground[(row, col)] = objects_in_game.wall

        self.play_ground[self.goal_location] = objects_in_game.goal
        self._cerve_the_path(self.player_location[0], self.player_location[1])

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
                else:
                    pattern_row.append(" ")
            pattern.append(pattern_row)
        return pattern

    def _cerve_the_path(self, x, y):
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
                    self._cerve_the_path(nx, ny)

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
                return True
        else:
            return False


""" my own method is not worked 
    def _cerve_the_path(self, init_x, init_y): # In here, I try to randomly create a path instead of choosing walls.
        # Also, to avoid looping around, I am choosing the next 2 or 3 positions.
        # For example, if I am at (0, 0), I can choose any house within 2 or 3 (not decided yet) directions around it.
        # Then simply just carve one position, and then in that new direction, I can decide again.

        jump_size = 2
        self.play_ground[(init_x, init_y)] = objects_in_game.space

        # Choose a direction to move
        decision = random.choice(
            [(0, jump_size), (0, -jump_size), (jump_size, 0), (-jump_size, 0)]
        )

        next_x = init_x + decision[0]
        next_y = init_y + decision[1]

        # Check if new position is within bounds
        if 0 < next_x < self.wall_range and 0 < next_y < self.wall_range:
            if self.play_ground.get((next_x, next_y)) != objects_in_game.goal:
                # if the current position is (0,0) and new direction is (0,2) , then i should cerve the position between them, which is (0,1)
                mid_x = (init_x + next_x) // 2
                mid_y = (init_y + next_y) // 2
                self.play_ground[(mid_x, mid_y)] = objects_in_game.space

                self._cerve_the_path(next_x, next_y)
"""


game = maze(16)
game.generate_playGround()
game.add_extra_paths(extra_count=15)
maze_pattern = game.generate_playGround_pattern()
game.print_maze(maze_pattern)
