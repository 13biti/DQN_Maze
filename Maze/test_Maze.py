import time
from MDP_maze import maze
import random


def test_actions():
    game = maze(8, 0, 10)
    actions = game.get_actions()
    action_name = game.get_actions_discription()
    print("maze game test the actions \n")
    total_test_steps = 10
    for step in range(total_test_steps):
        random_action = random.randint(0, len(actions) - 1)
        random_action = actions[random_action]
        print(random_action)
        print(action_name)
        isSuccess, info, reward, next_state, game_done = game.act(random_action)
        if isSuccess:
            print(
                f"try number: {step} \n action : {action_name[random_action]} info: {info} reward: {reward} next_state: {next_state} is game done: {game_done} "
            )
        else:
            print("action not in range ")
            return False
    return True


def test_playGround():
    game = maze(8, 0, 10)
    print("maze game test the playGround \n ")
    map_pattern = game.generate_visual_pattern()
    map = game.get_playGround()
    map2 = game.generate_playGround_from_pattern(map_pattern)
    for i in map.keys():
        if map.get(i) != map2.get(i):
            print(
                f"differension found : \n location {i} , map: {map.get(i)} , map2: {map2.get(i)}"
            )
            return False

    print("test was successfull ")
    return True


def main():
    test_playGround()
    test_actions()


if __name__ == "__main__":
    main()
