import gymnasium as gym
import numpy as np
from pynput import keyboard


ENV_ID = "CartPole-v1"

env = gym.make(ENV_ID, render_mode="human")

# Number of episodes
num_episodes = 5

START = False
DONE = True
CURR_ACTION = None
EXIT = False
def on_press(key):
    global START, DONE, CURR_ACTION, EXIT
    if key == keyboard.Key.esc:
        print('Restarting the game')
        DONE = True
    elif key == keyboard.Key.backspace:
        print('Exiting')
        EXIT = True
    elif key in [keyboard.Key.left, keyboard.KeyCode.from_char('a')]:
        DONE = False
        CURR_ACTION = 0
    elif key in [keyboard.Key.right, keyboard.KeyCode.from_char('d')]:
        DONE = False
        CURR_ACTION = 1
    else:
        print("Invalid button press")
        return False
    return True

def print_controls():
    print("====================================================")
    print("    To start, move left or right")
    print("    Press 'a' or <- to move left")
    print("    Press 'd' or -> to move right")
    print("    Press 'esc' to stop the game")
    print("    Press 'backspace' to exit the game")
    print("====================================================")
    print()

with keyboard.Events() as events:
    print_controls()
    episode = 0
    while not EXIT:
        observation = env.reset()  # Reset the environment
        episode += 1
        total_reward = 0
        truncation = 0

        event = events.get()
        if event is not None and isinstance(event, keyboard.Events.Press):
            on_press(event.key)

        while not DONE and not EXIT:
            event = events.get(1/30)
            if event is not None and isinstance(event, keyboard.Events.Press):
                on_press(event.key)
            
            observation, reward, term, trunc, info = env.step(CURR_ACTION)
            total_reward += reward

            if reward % 500 == 0:
                truncation += 1
                print(f"You've done {truncation} truncations! Keep it up!")

            DONE = DONE or term

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
    print("Exiting the game")

env.close()