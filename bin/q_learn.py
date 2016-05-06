import numpy as np

import json

from freeze_experience_replay import FreezeExperienceReplay


class QLearn (object):
    unfreeze_count = 5
    epsilon_init = 20
    epsilon_decay = 0.5e-3
    discount = 0.9
    max_memory = 10000
    batch_size = 5

    def __init__ (self):
        pass

    # initializes dependencies
    def setup (self):
        self.env = self.get_env ()
        self.model = self.get_model ()
        self.expReplay = FreezeExperienceReplay (self.model, self.max_memory, self.discount, self.unfreeze_count, self.num_actions)

    # do one epoch of training
    def train_step (self, epoch, render=False):
        frame_count = 0
        score = 0
        loss = 0
        done = False
        obs_t = self.env.reset ()

        while not done:
            # render
            if render:
                self.env.render ()

            # set previous obs (state[t-1])
            obs_tm1 = obs_t

            # select action from quality network or randomly
            epsilon_ = self.epsilon_init * np.exp (-epoch * self.epsilon_decay)
            if np.random.rand () < epsilon_:
                action = np.random.randint (0, self.num_actions, size=1)
            else:
                q = self.model.predict (np.matrix ([obs_t]))
                action = np.argmax (q)

            # evaluate action
            obs_t, reward, done, info = self.env.step (action)
            frame_count += 1

            # save in experience replay
            self.expReplay.remember (obs_tm1, action, reward, obs_t, done)

            # adapt model on experiences
            inputs, targets = self.expReplay.get_batch (self.model, epoch, self.batch_size)

            # sum reward and loss
            score += reward
            loss += self.model.train_on_batch (inputs, targets)

            # exit condition
            if self.solved (frame_count, score):
                done = True

        # return score and total loss
        return score, loss

    # train model until desired_score is achieved
    def train (self, desired_score, filename="model", render_func=lambda x: x % 100 == 0):
        i = 0
        highest = -99999999
        scores = list ()
        averages = list ()
        while highest < desired_score:
            # inc counter
            i += 1

            # do a training step
            f, loss = self.train_step (i, render=render_func (i))

            # save
            self.save (filename)

            # set highest score
            if f > highest:
                highest = f

            # append score to list
            scores.append (f)
            if len (scores) > 25:
                del scores [0]

            # get running average
            averages.append (np.mean (scores))

            # print useful info
            print "Epoch: {:05d} | Loss: {:e} | Score/High: {:04f}/{:04f} | Mean: {:2f} | Delta: {:1f}".format (i, loss, f, highest, averages [-1], averages [-1] - averages [-min (25, len (averages))])

    # save model and weights to files
    def save (self, filename="model"):
        self.model.save_weights("%s.h5" % filename, overwrite=True)
        with open("%s.json" % filename, "w") as outfile:
            json.dump(self.model.to_json(), outfile)

    # get environment
    def get_env (self):
        raise NotImplemented ()

    # get the model
    def get_model (self):
        raise NotImplemented ()

    # return whether the score is good enough
    def solved (self, score):
        raise NotImplemented ()
