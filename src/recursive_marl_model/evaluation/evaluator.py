import csv
import os

import numpy as np

import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('WebAgg')



class RewardEvaluator:

    def __init__(self, file_path) -> None:
        self.path_path = os.path.join(file_path, 'reward_evaluate.csv')

    def save_reward(self, count, value):
        with open(self.path_path, 'a') as f:
            csv.writer(f).writerow((count, value))

    def plot(self):

        with open(self.path_path, 'r') as f:
            num, reward = list(zip(*csv.reader(f)))

        plt.scatter(x=num[::10], y=reward[::10])
        plt.show()

    