import sys
sys.path.append('../bin')
from q_learn import QLearn

from keras.models import Sequential, Model, model_from_json
from keras.layers.core import Dense
from keras.optimizers import sgd

import numpy as np

import gym, json


# because the env returning dif sized matrices is dumb AF
# also changes up the reward value
def step_override (f):
    def step_ovrd (*args, **kwargs):
        obs_t, reward, done, info = f (*args, **kwargs)

        new_obs = obs_t.reshape (2)
    #    try:
    #        new_obs = [obs_t [0][0], obs_t [1][0]]
    #    except IndexError:
    #        print obs_t
    #        new_obs = []

        pos_factor = 0
        vel_factor = 20
        # new_obs is of form *i think* [x-pos, vel]
        new_reward = 2*reward + pos_factor * new_obs [0] + vel_factor * abs (new_obs [1])
        return new_obs, new_reward, done, info
    return step_ovrd

# QLearn sublass for acrobat environment
class MountainCarQLearn (QLearn):
    hidden_size = 64
    num_obs = 2
    num_actions = 3
    max_memory = 60000
    batch_size = 100
    unfreeze_count = 10
    discount = 0.999

    # create the environment
    def get_env (self):
        e = gym.make ('MountainCar-v0')
        e.step = step_override (e.step)
        return e

    # create and return model
    def get_model (self):
        m = Sequential ()
        m.add (Dense (self.hidden_size, input_shape=(self.num_obs,), activation='relu'))
        m.add (Dense (self.hidden_size, activation='tanh'))
        m.add (Dense (self.num_actions))
        #with open("model.json", "r") as outfile:
        #    model = model_from_json (json.load (outfile))
        #model.load_weights ("model.h5")
        m.compile(sgd(lr=.003), "mse")
        return m

    def solved (self, frame_count, score):
        return frame_count >= 2000 # timed out


mcql = MountainCarQLearn ()
mcql.setup ()
mcql.train (-100, render_func=lambda x: x % 20 == 0)
mcql.save ("model")
