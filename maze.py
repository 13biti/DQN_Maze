import os


class maze:
    def __init__(self) -> None:
        self.play_ground_size = 4
        self.keys = ["w", "s", "a", "d"]
        self.goal_location = (0, 4)
        self.player_location = (0, 0)

    def clear_screan(self):
        os.system("clear")

    def print_maze(self, maze_pattern):
        self.clear_screan()
        for row in maze_pattern:
            print(" ".join(row))
