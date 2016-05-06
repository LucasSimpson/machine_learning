from keras.models import model_from_json
from keras.optimizers import sgd

import numpy as np

import json, gym, time

class Disturb (object):
    prob = 1.0 / (50) # 1 in 500 frames
    min_space = 2 # at least 50 frames between disturbances
    action_space = 2 # 2 options in action space

    def __init__ (self):
        self.count = 0

    def disturb (self, default):
        if self.count > 0:
            self.count -= 1
            return default

        if np.random.rand () <= self.prob:
            self.count = self.min_space
            print 'disturbed'
            #return np.random.randint (self.action_space)
            # always disturb right MUHAHAHAHA
            return 0

        return default

    def __call__ (self, default):
        return self.disturb (default)
disturb = Disturb ()

with open("model3.json", "r") as outfile:
    model = model_from_json (json.load (outfile))
model.load_weights ("model3.h5")
model.compile(sgd(lr=.01), "mse")

def get_env ():
    e = gym.make ('CartPole-v0')
    return e
env = get_env ()

def eval_iter (disturbance=False):
    frame_count = 0
    loss = 0
    done = False
    obs_t = env.reset ()

    while not done:
        # render
        env.render ()
        print "Frame: {:04d}".format (frame_count)
        #time.sleep (0.01667)

        # set previous obs (state[t-1])
        obs_tm1 = obs_t

        # select action from quality network
        q = model.predict (np.matrix ([obs_t]))
        if disturbance:
            action = disturb (np.argmax (q))
        else:
            action = np.argmax (q)

        # evaluate action
        obs_t, reward, done, info = env.step (action)
        if reward >= 0:
            frame_count += 1

    return frame_count

while (True):
    eval_iter (disturbance=False)
