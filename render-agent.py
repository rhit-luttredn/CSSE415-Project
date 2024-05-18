import gymnasium as gym
import numpy as np
import torch
from tkinter import filedialog
from pynput import keyboard

ENV_ID = "CartPole-v1"
AGENT_PATH = None
EXIT_KEYS = [keyboard.Key.backspace, keyboard.Key.esc]
PAUSE_KEYS = [keyboard.Key.space]


def on_press(event):
    exit = False
    pause = False
    if not isinstance(event, keyboard.Events.Press):
        return exit, pause
    
    key = event.key
    if key in EXIT_KEYS:
        exit = True
    elif key in PAUSE_KEYS:
        pause = True
    return exit, pause


def load_model(model_path):
    model = torch.jit.load(model_path)
    model.eval()
    return model


filename = filedialog.askopenfilename(
    initialdir = "/work/cssema415/202430/11/models",
    # initialdir = "/home/luttredn/machine-learning/CSSE415-Project/models",
    title = "Select a Model",
    filetypes = (("Pytorch Model", "*.pt*"), ("all files", "*.*"))
)
print("Loading model from: ", filename)

model = load_model(filename)
env = gym.make(ENV_ID, render_mode="human")

obses = []
with keyboard.Events() as events:
    episode = 0
    exit = False
    while not exit:
        observation, info = env.reset()
        episode += 1
        total_reward = 0
        truncation = 0
        done = False

        while not done and not exit:
            event = events.get(1/30)
            if event is not None:
                exit, pause = on_press(event)
                if pause:
                    print("Pausing...")
                    while not exit and pause:
                        event = events.get()
                        exit, pause = on_press(event)
                        pause = not pause
                        if not pause:
                            print("Resuming...")
                if exit:
                    print("Exiting...")
                    break
            
            action = model(torch.from_numpy(observation).view(1, -1)).item()
            observation, reward, term, trunc, info = env.step(action)
            total_reward += reward

            obses.append(observation)

            if reward % 500 == 0:
                truncation += 1
                print(f"You've done {truncation} truncations! Keep it up!")

            done = done or term

        if obses:
            obses = np.array(obses)
            # Get the max and min values for each feature
            max_vals = np.max(obses, axis=0)
            min_vals = np.min(obses, axis=0)
            print(f"Maximum values: {max_vals}")
            print(f"Minimum values: {min_vals}")

        obses = []

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
        
    print("Exiting the game")


env.close()