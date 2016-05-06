from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.optimizers import sgd

import matplotlib.pyplot as plt

import gym, json, time, bisect

import numpy as np

# parameters
hidden_size = 8 # size of hidden layers in NN
num_actions = 2 # [left, right]
epsilon = 0.2 # chance of random action space sampling
e_decay = 0.5e-3 # decay rate of epsilon
batch_size = 10 # batch size
memory_size = 3000 # remembered states

def get_model ():
    m = Sequential ()
    m.add (Dense (hidden_size, input_shape=(4,), activation='relu'))
    m.add (Dense (hidden_size, activation='relu'))
    m.add (Dense (hidden_size, activation='relu'))
    m.add (Dense (num_actions))
    m.compile(sgd(lr=.01), "mse")
    return m
model = get_model ()
#model.load_weights ("model.h5")

def get_env ():
    e = gym.make ('CartPole-v0')
    return e
env = get_env ()


# defines a memory. wrapper necessary for bisect () method
# [[state_t, action, reward, state_tp1], done, final_reward]
class Memory (object):
    def __init__ (self, state_t, action, reward, state_tp1, done, final_reward):
        self.data = [[state_t, action, reward, state_tp1], done, final_reward]

    def __getitem__ (self, index):
        return self.data [index]

    def __cmp__ (self, other):
        return self.data [2] - other.data [2]


class ExperienceReplay (object):
    def __init__ (self, model, max_memory=2000, discount=0.7, unfreeze_count=5):
        self.max_memory = max_memory
        self.discount = discount
        self.unfreeze_count = unfreeze_count
        self.memory = []
        self.buffer = []

        self.frozen_model = Sequential.from_config (model.get_config ())
        self.frozen_model.compile (sgd(lr=.01), "mse")

    # returns True if expReplay has at least batch_size distinct entries
    def hasSufficientData (self, batch_size):
        return len (self.memory) >= batch_size

    # buffers the memories, as the final score is not yet known
    def remember_buffer (self, state_t, action, reward, state_tp1, done):
        self.buffer.append ([state_t, action, reward, state_tp1, done])

    # commits the buffered memories to experience replay, with the
    # final reward so that memories can be selected according to a
    # PDF described by final reward and sorted
    def commit_buffer (self, final_reward):
        for memory in self.buffer:
            m = memory + [final_reward]
            self.remember (*m)
        self.buffer = []

    # memory [i] = [[state_t, action_t, reward_t, state_t + 1], done, final_reward,]
    # memories are inserted in sorted order so as to bias better performing memories
    def remember (self, state_t, action, reward, state_tp1, done, final_reward):
        m = Memory (state_t, action, reward, state_tp1, done, final_reward)
        pos = bisect.bisect (self.memory, m)
        self.memory.insert (pos, m)
        if len (self.memory) > self.max_memory:
            del self.memory [0]

    # return batch pairs of inputs/outputs
    def get_batch (self, model, epoch, batch_size_=32):
        # get some meta data
        batch_size = min (len (self.memory), batch_size_)
        state_dim = len (self.memory [0][0][0])

        # initialize memory to all zero
        inputs = np.zeros ((batch_size, state_dim))
        targets = np.zeros ((batch_size, num_actions))

        # freeze/update model
        if epoch % self.unfreeze_count == 0:
            self.frozen_model.set_weights (model.get_weights ())

        # create batch
        # get set of memories according to PDF as described by final reward
        scores = [mem [2] for mem in self.memory]
        total = np.sum (scores)
        probs = map (lambda x: x / total, scores)
        indices = [i for i in range (len (self.memory))]
        sampled_memory_indices = np.random.choice (indices, size=batch_size, p=probs)
        #for i, r_id in enumerate (sampled_memory_indices):
        for i, r_id in enumerate (np.random.randint (0, len (self.memory), size=batch_size)):

            # get data from memory
            s_t, a_t, r_t, s_tp1 = self.memory [r_id] [0]
            done = self.memory [r_id] [1]

            # set input and targets
            inputs [i] = s_t
            targets [i] = self.frozen_model.predict (np.matrix ([s_t]))[0]

            # calculate desired reward
            if done:
                targets [i, a_t] = r_t
            else:
                Q_sa = np.max (self.frozen_model.predict (np.matrix ([s_tp1]))[0])
                targets [i, a_t] = r_t + self.discount * Q_sa

        # return labeled data
        return inputs, targets


    def __str__ (self):
        r = ''
        for mem in self.memory:
            r += str (mem) + '\n'
        return r

expReplay = ExperienceReplay (model, max_memory=memory_size)

def train_iter (epoch):
    frame_count = 0
    score = 0
    loss = 0
    done = False
    obs_t = env.reset ()

    while not done:
        # render
        if epoch % 100 == 0:
            pass
            #env.render ()
            #time.sleep (0.05)

        # set previous obs (state[t-1])
        obs_tm1 = obs_t

        # select action from quality network or randomly
        epsilon_ = epsilon * np.exp (-epoch * e_decay)
        if np.random.rand () < epsilon_:
            action = np.random.randint (0, num_actions, size=1)
        else:
            q = model.predict (np.matrix ([obs_t]))
            action = np.argmax (q)

        # evaluate action
        obs_t, reward, done, info = env.step (action)
        frame_count += 1

        # add score
        score += reward

        # save in experience replay
        expReplay.remember_buffer (obs_tm1, action, reward, obs_t, done)

        # adapt model on experiences if there are experiences
        if expReplay.hasSufficientData (batch_size):
            inputs, targets = expReplay.get_batch (model, epoch, batch_size)
            # add loss
            loss += model.train_on_batch (inputs, targets)

        # exit condition
        if frame_count >= 200: # bullllsshhiittt
            done = True

    # commit memories
    expReplay.commit_buffer (score)
    return frame_count, loss


def eval_iter ():
    frame_count = 0
    score = 0
    done = False
    obs_t = env.reset ()

    while not done:
        # set previous obs (state[t-1])
        obs_tm1 = obs_t

        # select action from quality network
        q = model.predict (np.matrix ([obs_t]))
        action = np.argmax (q)

        # evaluate action
        obs_t, reward, done, info = env.step (action)
        frame_count += 1

        # add score
        score += reward

        # exit condition
        if frame_count >= 200: # bullllsshhiittt
            done = True

    return score

def full_eval ():
    # 100 run evaluation
    print 'doing evaluation... '
    aves = list ()
    for a in range (100):
        aves.append (eval_iter ())
    print 'eval average over 100 runs is %s' % np.mean (aves)
    return np.mean (aves) >= 195

i = 0
highest = 0
scores = [0]
averages = list ()
SOLVED = False
#env.monitor.start ('cartpole-mon1')
while (not SOLVED):
#while (np.mean (scores) < 195):
    i += 1
    f, loss = train_iter (i)

    if f > highest:
        highest = f

    if highest >= 200:
        SOLVED = full_eval ()

    scores.append (f)
    if len (scores) > 100:
        del scores [0]

    averages.append (np.mean (scores))
    print "Epoch: {:05d} | Loss: {:e} | Steps/High: {:04d}/{:04d} | Mean: {:2f} | Delta: {:1f}".format (i, loss, f, highest, averages [-1], averages [-1] - averages [-min (25, len (averages))])



#env.monitor.close ()


model.save_weights("model3.h5", overwrite=True)
with open("model3.json", "w") as outfile:
    json.dump(model.to_json(), outfile)

print "done"
