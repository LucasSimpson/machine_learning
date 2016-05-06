from keras.models import Sequential
from keras.optimizers import sgd

import numpy as np


class FreezeExperienceReplay (object):
    def __init__ (self, model, max_memory, discount, unfreeze_count, num_actions):
        self.max_memory = max_memory
        self.discount = discount
        self.unfreeze_count = unfreeze_count
        self.num_actions = num_actions
        self.memory = list ()

        # TODO dont assume sequential model
        # note taining algo has no affect because frozen model is never trianed
        self.frozen_model = Sequential.from_config (model.get_config ())
        self.frozen_model.compile (sgd(lr=.01), "mse")


    # memory [i] = [[state_t, action_t, reward_t, state_t + 1], done]
    def remember (self, state_t, action, reward, state_tp1, done):
        self.memory.append ([[state_t, action, reward, state_tp1], done])
        if len (self.memory) > self.max_memory:
            del self.memory [0]

    # return batch pairs of inputs/outputs
    def get_batch (self, model, epoch, batch_size_=32):
        batch_size = min (len (self.memory), batch_size_)
        state_dim = len (self.memory [0][0][0])

        inputs = np.zeros ((batch_size, state_dim))
        targets = np.zeros ((batch_size, self.num_actions))

        # freeze/update model
        if epoch % self.unfreeze_count == 0:
            self.frozen_model.set_weights (model.get_weights ())

        # create batch
        for i, r_id in enumerate (np.random.randint (0, len (self.memory), size=batch_size)):

            s_t, a_t, r_t, s_tp1 = self.memory [r_id] [0]
            done = self.memory [r_id] [1]

            inputs [i] = s_t
            targets [i] = self.frozen_model.predict (np.matrix ([s_t]))[0]

            if done:
                targets [i, a_t] = r_t
            else:
                Q_sa = np.max (self.frozen_model.predict (np.matrix ([s_tp1]))[0])
                targets [i, a_t] = r_t + self.discount * Q_sa

        return inputs, targets


    def __str__ (self):
        r = ''
        for mem in self.memory:
            r += str (mem) + '\n'
        return r
