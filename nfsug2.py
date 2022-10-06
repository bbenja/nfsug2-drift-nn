from PIL import Image, ImageGrab, ImageFilter, ImageOps, ImageEnhance
from keys import PressKey, ReleaseKey, W, A, S, D, SHIFT, SPACE, ESC, ENTER, DOWN, RIGHT, LEFT
from time import sleep, perf_counter
import pygetwindow
import pyautogui
import pytesseract
import cv2
import numpy as np
from re import sub
from mss import mss
from pymem import *
from pymem.process import *
from pymem.ptypes import RemotePointer
import keyboard
import win32gui
from random import choice
import gym
from gym.spaces import Discrete, Box

class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = Discrete(3)     #A, SPACE, D
        self.observation_space = Box(0.0, 50.0, shape=(2,))

        # self.states = 2
        # self.actions = 3


    def step(self, action):
        # Apply action
        control_car(action)

        # Calc reward
        reward = get_score()

        # Get state
        state = get_car_diag()

        # Calculate if done
        done = False

        info = {}

        return state, reward, done, info

    def reset(self):
        state = get_car_diag()
        reset_race()
        return state

    def render(self):
        pass

    def close(self):
        pass


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=100, target_model_update=1e-2)
    return dqn


config = r"--psm 8 --oem 1"
path = "C:\\Users\\bbenja\\Desktop\\nfsug.jpg"
# W = 0x11; A = 0x1E; S = 0x1F;D = 0x20


def get_window_image():
    # window = pygetwindow.getWindowsWithTitle("NFS Underground 2")[0]
    # left, top = window.topleft
    # right, bottom = window.bottomright
    # pyautogui.screenshot(path)
    # im = Image.open(path)
    # im = im.crop((left, top, right, bottom))

    # im.save(path)
    # im.show(path)

    # im = ImageGrab.grab(bbox=(-7, 0, 807, 867))
    sct = mss()
    im = np.array(sct.grab((-7, 0, 807, 867)))
    cv2.imshow("frame", im)

    return im


def get_data_from_image(img):
    # img = Image.open(path)
    img = img.convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3)
    img = img.filter(ImageFilter.MedianFilter())
    cv2.imshow("image", cv2.resize(np.array(img), (500, 500)))


    crop_rectangle = (160, 230, 245, 260)
    total = img.crop(crop_rectangle)   # (left, upper, right, lower)
    # total = total.point(lambda x: 0 if x < 110 else 255)
    cv2.imshow("total", np.array(total))
    # total.show()

    crop_rectangle = (470, 785, 510, 825)
    angle = img.crop(crop_rectangle)
    # angle = angle.point(lambda x: 0 if x < 125 else 255)
    cv2.imshow("angle", np.array(angle))
    # angle.show()

    detected_total = pytesseract.image_to_string(total, config=config)
    detected_angle = pytesseract.image_to_string(angle, config=config)
    detected_total = "".join(i for i in detected_total if i in "0123456789")
    detected_angle = sub("\D", "", detected_angle)
    return detected_total, detected_angle

actions_array = [A, SPACE, D]
actions_history = []
keypress_pause = 0.3
highscore, iteration, episode = 0, 0, 0
states = 17
active_state = 0
Q = 1000 * np.random.rand(states, len(actions_array))
lr = 0.5
y = 0.5

def control_car(chosen_action):
    for action in actions_array:
        ReleaseKey(action)
    PressKey(W)
    PressKey(actions_array[chosen_action])
    print(actions_array[chosen_action])


def run_q_algorithm():
    global active_state, highscore, episode, actions_history
    chosen_action = np.argmax(Q[active_state, :])
    actions_history.append(chosen_action)
    print(f"state: {active_state}\tvalue: {Q[active_state, chosen_action]}\taction: {chosen_action}")

    control_car(chosen_action)
    reward = update_reward()
    Q[active_state, chosen_action] = (1 - lr) * Q[active_state, chosen_action] + lr * (
                highscore + y * np.max(Q[(active_state + 1), :]))
    # print(Q)
    if active_state + 2 != states:
        active_state += 1
    else:
        reset_race()
        episode += 1
        print(f"EP: {episode}")
        np.savetxt("qmatrix" + str(highscore) + ".out", Q, delimiter=',')
        np.savetxt("actions_history" + str(highscore) + ".out", actions_history, delimiter=',')
        active_state = 0
        actions_history = []
        highscore = 0

def update_reward():
    global highscore
    highscore = mem.read_float((mem.base_address + 0x464650))
    for i in range(len(actions_history)):
        Q[i][actions_history[i]] += highscore

def reset_race():
    #ESC, RIGHT, ENTER, LEFT, ENTER
    PressKey(ESC)
    sleep(0.1)
    ReleaseKey(ESC)

    PressKey(RIGHT)
    sleep(0.1)
    ReleaseKey(RIGHT)

    PressKey(ENTER)
    sleep(0.1)
    ReleaseKey(ENTER)

    PressKey(LEFT)
    sleep(0.1)
    ReleaseKey(LEFT)

    PressKey(ENTER)
    sleep(0.1)
    ReleaseKey(ENTER)

    sleep(3)


def getPointerAddress(base, offsets):
    remote_pointer = RemotePointer(mem.process_handle, base)
    for offset in offsets:
        if offset != offsets[-1]:
            remote_pointer = RemotePointer(mem.process_handle, remote_pointer.value + offset)
        else:
            return remote_pointer.value + offset

speed_offsets = [0x42C]
angle_offsets = [0x214, 0x20, 0x394, 0xC8C, 0x4, 0x0, 0x6C]

def get_car_diag():
    speed = mem.read_float(getPointerAddress(mem.base_address + 0x0049CCF8, speed_offsets))
    angle = mem.read_int(getPointerAddress(mem.base_address + 0x004B4754, angle_offsets))
    return (speed, angle)

def get_score():
    total = mem.read_float((mem.base_address + 0x464650))
    return total

def print_action(action):
    bindings = {"W": 0x11, "A": 0x1E,
                "S": 0x1F, "D": 0x20,
                "ESC": 0x01, "ENTER": 0x1C,
                "DOWN": 0xD0, "ONE": 0x02,
                "CTRL": 0x1D, "SHIFT": 0x2A,
                "SPACE": 0x39}
    return list(bindings.keys())[list(bindings.values()).index(action)]

mem = Pymem("SPEED2.EXE")
# Instantiate the env
env = CustomEnv()


if __name__ == "__main__":
    # window = win32gui.FindWindow(None, "NFS Underground 2")
    # win32gui.SetForegroundWindow(window)
    print("The observation space: {}".format(env.observation_space))
    for _ in range(5):
        print(env.observation_space.sample())


    model = build_model(2, 3)
    # dqn = build_agent(model, 3)
    # dqn.compile(Adam(1e-3), metrics=["mae"])
    # dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)
    #
    # dqn.save_weights("dqn", overwrite=True)
    #
    reset_race()
    model.load_weights("dqn")
    for _ in range(50):
        speed, angle = get_car_diag()
        data = [speed, angle]
        data = np.expand_dims(data, -1)
        action = model.predict(data)
        print(action)


    # for _ in range(1000):
    #     action = env.action_space.sample()
    #     data = env.step(action)
    #     print(data)

    # while (True):
    #     PressKey(W)
    #     # get state
    #     state = get_car_diag()
    #     # predict based on the state
    #     action = choice(actions_array)
    #     # insert predicted action into the game
    #
    #     sleep(keypress_pause)
    #     ReleaseKey(action)
    #     # check the highscore and calculate the reward
    #     highscore = get_score()
    #     print(f"{round(state[0])}, {state[1]}\t-> {print_action(action)}\t-> {highscore}")

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break
