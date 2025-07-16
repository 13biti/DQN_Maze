import os
from enum import Enum


class objects_in_game(Enum):
    wall = -1
    space = 0
    goal = 1


class maze:
    def __init__(self, game_size) -> None:
        # game_size + walls
        self.play_ground_size = game_size + 2
        self.game_size = game_size
        self.keys = ["w", "s", "a", "d"]
        self.goal_location = (0, 4)
        self.player_location = (0, 0)
        self.play_ground = {}

    def generate_playGround(self):
        wall_range = self.play_ground_size - 1
        for row in range(self.play_ground_size):
            for col in range(self.play_ground_size):
                if col % wall_range == 0 or row % wall_range == 0:
                    self.play_ground[(row, col)] = objects_in_game.wall

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

    def clear_screan(self):
        os.system("clear")

    def print_maze(self, maze_pattern):
        self.clear_screan()
        for row in maze_pattern:
            print(" ".join(row))


game = maze(8)
game.generate_playGround()
maze_pattern = game.generate_playGround_pattern()
game.print_maze(maze_pattern)
