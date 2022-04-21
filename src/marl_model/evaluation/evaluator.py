import os
import pickle
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


class RewardEvaluator:

    names = ['timestamp', 'count', 'rewards', 'eval_reward', 'model', 'env']

    def __init__(self) -> None:
        self.data = []
        self.model = None
        self.env = None

    def save(self, count, rewards, eval_reward, model, env, file_path):
        # model_file = f'model-{count:0>8}-{rewards:08.2f}.pickle'
        # env_file = f'env-{count:0>8}-{rewards:08.2f}.pickle'

        self.model = model
        self.env = env
        model_file = f'model.pickle'
        env_file = f'env.pickle'

        with open(os.path.join(file_path, model_file), 'wb+') as model_fd:
            pickle.dump(model, model_fd)
        with open(os.path.join(file_path, env_file), 'wb+') as env_fd:
            pickle.dump(env, env_fd)

        d = [pd.Timestamp.now(), count, rewards, eval_reward, model_file, env_file]
        df = pd.DataFrame([d], columns=self.names)

        df.to_csv(os.path.join(file_path, 'episode_reward.csv'),
                  index=False,
                  float_format="%08.2f",
                  mode='a',
                  header=False)

        self.data.append(d)

    @classmethod
    def load(cls, file_path):
        df = pd.read_csv(os.path.join(file_path, 'episode_reward.csv'), names=cls.names)
        obj = cls()

        obj.data = df.values.tolist()

        model_file = f'model.pickle'
        env_file = f'env.pickle'

        with open(os.path.join(file_path, model_file), 'rb') as model_fd:
            obj.model = pickle.load(model_fd)
        with open(os.path.join(file_path, env_file), 'rb') as env_fd:
            obj.env = pickle.load(env_fd)

        return obj

    @classmethod
    def plot(cls, file_path):

        evaluators = []

        if os.path.exists(os.path.join(file_path, 'episode_reward.csv')):
            evaluators.append(cls.load(file_path))
        else:
            for directory in os.listdir(file_path):
                evaluators.append(cls.load(os.path.join(file_path, directory)))

        fig = go.Figure()
        for ev in evaluators:
            df = pd.DataFrame(ev.data, columns=cls.names)
            fig.add_scatter(x=df['count'], y=df['rewards'], mode='lines', name=f'{ev.model}')
#            fig.add_scatter(x=df['count'], y=df['eval_reward'], mode='lines')

        fig.show()
