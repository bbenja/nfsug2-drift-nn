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
from pymem import Pymem
# from pymem.process import *
from pymem.ptypes import RemotePointer
import keyboard
import win32gui
import gym
from gym.spaces import Discrete, Box
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

def build_pilotnet_model():
    model = Sequential(name="PilotNet")
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_regularizer=l2(1e-3)))
    model.add(Dense(50, activation='relu', kernel_regularizer=l2(1e-3)))
    model.add(Dense(10, activation='relu', kernel_regularizer=l2(1e-3)))
    model.add(Dense(4, activation="softmax"))
    # model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer="adam",
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


def build_model(w, h, actions):
    model = Sequential()
    model.add(Conv2D(32, 5, strides=(2, 2),
                     activation="relu", padding="same",
                     input_shape=(1, w, h)))
    model.add(Conv2D(64, 8,
                     activation="relu", padding="same",
                     input_shape=(1, w, h)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(64, activation="elu"))
    model.add(Dense(32, activation="elu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    # policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr="eps",
    #                               value_max=50., value_min=-50.,
    #                               value_test=1., nb_steps=10000)
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=10000, window_length=1)
    dqn = DQNAgent(model=model, policy=policy, memory=memory,
                   nb_actions=actions, nb_steps_warmup=10000,
                   enable_dueling_network=True, dueling_type="avg")
    return dqn


class CustomEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.img_size = (350, 350)
        self.action_space = Discrete(len(actions_array))     #A, SPACE, D
        self.observation_space = Box(np.zeros(self.img_size, dtype=np.uint8),
                                     np.full(self.img_size, 255, dtype=np.uint8),
                                     [self.img_size[0], self.img_size[1]],
                                     dtype=np.uint8)
        self.state = [0.0, 0.0]
        self.action_history = []
        self.score = get_score()
        self.done = False


    def step(self, action):
        reward = -25
        try:
            if self.action_history[-20:].count(action) > 10:
                reward -= 200
        except:
            pass

        self.action_history.append(action)
        control_car(action)
        speed, angle = get_car_diag()
        speed *= 3.6

        reward += speed * angle

        # if speed > 20.0:
        #     reward += 10 * speed
        #     if angle != 0:
        #         reward += 15 * angle
        #     else:
        #         reward -= 100
        # else:
        #     reward -= 201

        if speed < self.state[0] - 20:
            reward -= 1000
            self.done = True
        self.state = [speed, angle]

        score = get_score()
        reward += (score - self.score)
        self.score = score


        info = {}

        obs = get_car_from_image(get_window_image())

        col = obs[220:, :, :]

        print(f"\n{action}, {reward}, {self.state}")

        # mr, mg, mb = col.mean(axis=0).mean(axis=0)


        obs = np.reshape(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (350, 800))


        obs = cv2.resize(obs, (self.observation_space.shape[1],
                               self.observation_space.shape[0]))
        # cv2.imshow("obs", obs)
        # cv2.waitKey(1)
        return obs, reward, self.done, info

    def reset(self):
        reset_race()
        self.state = [0, 0]
        self.done = False
        obs = get_car_from_image(get_window_image())
        obs = np.reshape(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY), (350, 800))
        obs = cv2.resize(obs, (self.observation_space.shape[1],
                               self.observation_space.shape[0]))
        return obs

    def render(self):
        pass

    def close(self):
        pass


mem = Pymem("SPEED2.EXE")
actions_array = [A, W, D, S]
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
    img = img[:350, :, :]   # cut bottom part --> img = (350, 800, 3)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # img = (350, 800)
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
    # PressKey(W)
    PressKey(actions_array[chosen_action])
    # print(print_action(actions_array[chosen_action]))


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
    # bindings = {"0x11": "W", "0x1e": "A",
    #             "0x1f": "S", "0x20": "D",
    #             "0x2": "D",
    #             "0x01": "ESC", "0x1c": "ENT",
    #             "0xd0": "DOWN", "0x1d": "CTRL",
    #             "0x2a": "SHIFT", "0x39": "SPACE"}
    return actions_array[action]


keys = []
images_and_controls = []


if __name__ == "__main__":
    env = CustomEnv()
    height, width = env.observation_space.shape
    actions = env.action_space.n
    # env.reset()
    #
    # keyboard.hook(lambda x: keys.append(x.name))
    # for idx in range(5000):
    #     speed, angle = get_car_diag()
    #     img = cv2.resize(get_car_from_image(get_window_image())[150:, :, :], (200, 66))
    #     key = keys[-1]
    #     images_and_controls.append([idx, img, key, speed, angle])
    #     # cv2.imshow("frame", img)
    #     # cv2.waitKey(1)
    #     print(idx, key, speed, angle)
    # keyboard.unhook_all()
    # sleep(2)
    # with open("./nfsug2-drift-nn/dataset/data.txt", "w") as f:
    #     for idx, img, key, speed, angle in images_and_controls:
    #         f.write(f"{idx},{key},{speed},{angle}\n")
    #         cv2.imwrite(f"./nfsug2-drift-nn/dataset/{idx}.jpg", img)



    # window = win32gui.FindWindow(None, "NFS Underground 2")
    # win32gui.SetForegroundWindow(window)

    ### LOAD AND TRAIN
    # x, y = [], []
    # with open("./nfsug2-drift-nn/dataset/data.txt", "r") as f:
    #     for line in f:
    #         idx = line.split(",")[0]
    #         key = line.split(",")[1]
    #         img = np.expand_dims(cv2.imread(f"./nfsug2-drift-nn/dataset/{idx}.jpg"), axis=0)
    #         x.append(img)
    #         if key == "left":
    #             y.append([1, 0, 0, 0])
    #         elif key == "up":
    #             y.append([0, 1, 0, 0])
    #         elif key == "right":
    #             y.append([0, 0, 1, 0])
    #         elif key == "down":
    #             y.append([0, 0, 0, 1])
    # x = np.reshape(x, (len(x), 66, 200, 3))
    # y = np.reshape(y, (len(y), 4))
    # model = build_pilotnet_model()
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    # history = model.fit(x, y, batch_size=64, epochs=20, validation_split=0.1)
    # plt.plot(history.history["loss"], label="loss")
    # plt.plot(history.history["val_loss"], label="val_loss)
    # plt.legend()
    # plt.show()
    # print(model.evaluate(x, y, batch_size=1, verbose=1))
    # model.save("./nfsug2-drift-nn/classification")

    model = load_model("./nfsug2-drift-nn/classification")

    while True:
        img = get_car_from_image(get_window_image())
        cv2.imshow("frame", img)
        cv2.waitKey(1)
        img = cv2.resize(img, (200, 66))
        img = np.expand_dims(img, axis=0)
        action = np.argmax(model.predict(img))
        print(action)
        PressKey(actions_array[action])

    # predicts = []
    # for img in x:
    #     predicts.append(model.predict(np.expand_dims(img, axis=0)))
    #
    # predicts = np.argmax(np.reshape(predicts, (len(predicts), 4)), axis=1)
    # y = np.argmax(np.asarray(y), axis=1)
    # # plt.hist(y, bins=4)
    # # plt.show()
    # plt.plot(y, label="gt")
    # plt.plot(predicts, label="p", alpha=0.7)
    # plt.legend()
    # plt.show()




    # model = build_model(width, height, actions)
    # model.summary()
    # dqn = build_agent(model, actions)
    # dqn.compile(Adam(1e-3))
    # ### RL TRAINING
    # model.load_weights("./nfsug2-drift-nn/saves_first_lap/dqn")
    # dqn.fit(env, nb_steps=36000, visualize=False,
    #         verbose=1, nb_max_episode_steps=300)
    # dqn.save_weights("./nfsug2-drift-nn/saves/dqn", overwrite=True)
    # model.load_weights("./nfsug2-drift-nn/saves/dqn")
    # dqn.test(env, nb_max_episode_steps=3000, visualize=False, verbose=0)



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
