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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory


def build_model(w, h, actions):
    model = Sequential()
    model.add(Conv2D(32, (4, 4), strides=(2, 2),
                     activation="relu", padding="same",
                     input_shape=(w, h, 1)))
    model.add(Flatten())
    model.add(Dense(24, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps",
                                  value_max=1., value_min=.1,
                                  value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=1)
    dqn = DQNAgent(model=model, policy=policy, memory=memory,
                   nb_actions=actions, nb_steps_warmup=10000)
                   # enable_dueling_network=True, dueling_type="avg")
    return dqn


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = Discrete(3)     #A, SPACE, D
        self.observation_space = Box(0, 255, [800, 350, 1], dtype=np.uint8)
        print(self.observation_space.shape)
        self.state = get_car_diag()
        print(self.state)


    def step(self, action):
        control_car(action)
        self.state = get_car_diag()

        reward = 0
        if self.state[1] != 0:
            reward += int(self.state[1])
        reward += get_score()

        done = False

        info = {}

        obs = np.reshape(get_car_from_image(get_window_image()), (350, 800, 1))
        print(obs.shape)
        #return self.state, reward, done, info
        return obs, reward, done, info

    def reset(self):
        reset_race()
        obs = np.reshape(get_car_from_image(get_window_image()), (350, 800, 1))
        print(obs.shape)
        return obs

    def render(self):
        pass

    def close(self):
        pass


mem = Pymem("SPEED2.EXE")
actions_array = [A, SPACE, D]
keypress_pause = 0.3
speed_offsets = [0x42C]
angle_offsets = [0x214, 0x20, 0x394, 0xC8C, 0x4, 0x0, 0x6C]
# config = r"--psm 8 --oem 1"
# path = "C:\\Users\\bbenja\\Desktop\\nfsug.jpg"


def get_window_image():
    sct = mss()
    img = np.array(sct.grab((0, 100, 800, 820)))[:, :, :3]
    return img


def get_car_from_image(img):
    img = img[160:, :, :]   # cut top part
    img = img[:350, :, :]   # cut bottom part --> img = (350, 800, 4)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img = (350, 800)
    return img


# def get_data_from_image(img):
#     img = img.convert('L')
#     enhancer = ImageEnhance.Contrast(img)
#     img = enhancer.enhance(3)
#     img = img.filter(ImageFilter.MedianFilter())
#     cv2.imshow("image", cv2.resize(np.array(img), (500, 500)))
#
#
#     crop_rectangle = (160, 230, 245, 260)
#     total = img.crop(crop_rectangle)   # (left, upper, right, lower)
#     # total = total.point(lambda x: 0 if x < 110 else 255)
#     cv2.imshow("total", np.array(total))
#     # total.show()
#
#     crop_rectangle = (470, 785, 510, 825)
#     angle = img.crop(crop_rectangle)
#     # angle = angle.point(lambda x: 0 if x < 125 else 255)
#     cv2.imshow("angle", np.array(angle))
#     # angle.show()
#
#     detected_total = pytesseract.image_to_string(total, config=config)
#     detected_angle = pytesseract.image_to_string(angle, config=config)
#     detected_total = "".join(i for i in detected_total if i in "0123456789")
#     detected_angle = sub("\D", "", detected_angle)
#     return detected_total, detected_angle




def control_car(chosen_action):
    for action in actions_array:
        ReleaseKey(action)
    PressKey(W)
    PressKey(actions_array[chosen_action])
    print(print_action(actions_array[chosen_action]))


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


def get_car_diag():
    speed = mem.read_float(getPointerAddress(mem.base_address + 0x0049CCF8, speed_offsets))
    angle = mem.read_int(getPointerAddress(mem.base_address + 0x004B4754, angle_offsets))
    return speed, angle

def get_score():
    total = mem.read_float((mem.base_address + 0x464650))
    return total

def print_action(action):
    bindings = {"0x11": "W", "0x1e": "A",
                "0x1f": "S", "0x20": "D",
                "0x01": "ESC", "0x1c": "ENT",
                "0xd0": "DOWN", "0x1d": "CTRL",
                "0x2a": "SHIFT", "0x39": "SPACE"}
    return bindings[str(hex(action))]




if __name__ == "__main__":
    env = CustomEnv()
    obs = env.reset()
    height, width, ch = env.observation_space.shape
    actions = env.action_space.n
    # window = win32gui.FindWindow(None, "NFS Underground 2")
    # win32gui.SetForegroundWindow(window)


    model = build_model(width, height, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(1e-4))

    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
    dqn.save_weights("dqn", overwrite=True)

    model.load_weights("dqn")

    dqn.test(env, nb_max_episode_steps=3000, visualize=False)



    # while (True):
    #     get_car_from_image(get_window_image())
    #
    #     # get state
    #     state = get_car_diag()
    #     # predict based on the state
    #     action = choice(actions_array)
    #     # insert predicted action into the game
    #
    #     # check the highscore and calculate the reward
    #     highscore = get_score()
    #     print(f"{round(state[0])}, {state[1]}\t-> {print_action(action)}\t-> {highscore}")
    #
    #     key = cv2.waitKey(1)
    #     if key == 27:
    #         break
