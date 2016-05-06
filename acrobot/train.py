import sys
sys.path.append('../bin')
from q_learn import QLearn

from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.optimizers import sgd

import gym


# QLearn sublass for acrobat environment
class AcrobotQLearn (QLearn):
    hidden_size = 8
    num_actions = 2

    # create the environment
    def get_env (self):
        return gym.make ('Acrobot-v0')

    # create and return model
    def get_model (self):
        m = Sequential ()
        m.add (Dense (self.hidden_size, input_shape=(4,), activation='relu'))
        m.add (Dense (self.hidden_size, activation='relu'))
        m.add (Dense (self.hidden_size, activation='relu'))
        m.add (Dense (self.num_actions))
        m.compile(sgd(lr=.01), "mse")
        return m

    # NEVER SOLVED MUHAHAHA
    def solved (self, score):
        return score >= 2000


aql = AcrobotQLearn ()
aql.setup ()
aql.train (2000)
aql.save ("model")
