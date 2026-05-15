import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import random
import math
import numpy as np
import seaborn as sns
import cv2
import json


class Pose2DAnimation:
    def __init__(self, skeleton, bone_LR, img_size, ax, joints, lcolor="#3498db", rcolor="#e74c3c", ccolor="#2fb551"):
        self.ax = ax
        self.joints = joints
        self.skeleton = skeleton
        self.bone_LR = bone_LR

        self.lcolor = lcolor
        self.rcolor = rcolor
        self.ccolor = ccolor
        
        self.lines = []
        for idx in range(len(self.skeleton)):
            color = self._get_color(self.bone_LR[idx])
            line, = ax.plot([], [], lw=3, c=color)
            self.lines.append(line)

        self.ax.set_xlim(0, img_size[1])
        self.ax.set_ylim(0, img_size[0])
        ax.invert_yaxis()
        self.ax.set_aspect('equal')

    def _get_color(self, lr_code):
        if lr_code == 0:
            return self.lcolor
        elif lr_code == 2:
            return self.rcolor
        else:
            return self.ccolor

    def update(self, frame):
        joints = self.joints[frame] 
        x, y = joints[0], joints[1]
        for idx, (line, (i, j)) in enumerate(zip(self.lines, self.skeleton)):
            line.set_data([x[i], x[j]], [y[i], y[j]])
            line.set_color(self._get_color(self.bone_LR[idx]))
        return self.lines


def plot_motion_2d(joints, skeleton, bone_LR, img_size, motion_srate):
    fig, ax = plt.subplots()
    vis = Pose2DAnimation(skeleton, bone_LR, img_size, ax, joints)

    anim = FuncAnimation(
        fig,
        vis.update,
        frames=joints.shape[0],
        interval=1000/motion_srate,
        repeat=False,
        blit=True,
    )
    plt.close()
    return anim.to_jshtml()


def plot_imu_signal(imu_signal, limb_list=None):
    if len(imu_signal.shape) == 2:
        plt.figure(figsize=(10, 2))
        plt.plot(imu_signal, label=['x', 'y', 'z'])
        plt.title('IMU Signal')
        plt.legend()
    
    elif len(imu_signal.shape) == 3:
        fig, ax = plt.subplots(len(limb_list), 1, figsize=(10, 2*len(limb_list)))
        for i, limb in enumerate(limb_list):
            ax[i].plot(imu_signal[:, i, :], label=['x', 'y', 'z'])
            ax[i].set_title(limb)
            ax[i].set_ylim([-10, 10])
            ax[i].legend()

    plt.tight_layout()
    plt.show()