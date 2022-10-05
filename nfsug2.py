from PIL import Image, ImageGrab, ImageFilter, ImageOps, ImageEnhance
from keys import PressKey, ReleaseKey, W, A, S, D, SHIFT, SPACE, ESC, ENTER, DOWN
from time import sleep
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


config = r"--psm 8 --oem 1"
path = "C:\\Users\\bbenja\\Desktop\\nfsug.jpg"


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
    # detected_total = sub("\D", "", detected_total)
    detected_total = "".join(i for i in detected_total if i in "0123456789")
    detected_angle = sub("\D", "", detected_angle)
    return detected_total, detected_angle

actions = [W, A, S, D, SPACE]
actions_history = []
key_pause = 0.3


mem = Pymem("SPEED2.EXE")

def getPointerAddress(base, offsets):
    remote_pointer = RemotePointer(mem.process_handle, base)
    for offset in offsets:
        if offset != offsets[-1]:
            remote_pointer = RemotePointer(mem.process_handle, remote_pointer.value + offset)
        else:
            return remote_pointer.value + offset


if __name__ == "__main__":
    while (True):
        # img = get_window_image()
        # data = get_data_from_image(img)
        # print(data)

        speed_offsets = [0x42C]
        speed = mem.read_float(getPointerAddress(mem.base_address + 0x0049CCF8, speed_offsets))
        angle_offsets = [0x214, 0x20, 0x394, 0xC8C, 0x4, 0x0, 0x6C]
        angle = mem.read_int(getPointerAddress(mem.base_address + 0x004B4754, angle_offsets))
        total = mem.read_float((mem.base_address + 0x464650))

        print(f"total: {total}\tangle: {angle}\tspeed: {round(speed*3.6)}")



        key = cv2.waitKey(1)
        if key == 27:
            break