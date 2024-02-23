from q_learning import QLearning
from agt_server.local_games.lemonade_arena import LemonadeArena
from agt_server.agents.test_agents.lemonade.stick_agent.my_agent import StickAgent
from agt_server.agents.test_agents.lemonade.always_stay.my_agent import ReserveAgent
from agt_server.agents.test_agents.lemonade.decrement_agent.my_agent import DecrementAgent
from agt_server.agents.test_agents.lemonade.increment_agent.my_agent import IncrementAgent
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf



class MyRLAgent(QLearning):
    def __init__(self, name, num_possible_states, num_possible_actions, initial_state, learning_rate, discount_factor, exploration_rate, training_mode, save_path=None) -> None:
        super().__init__(name, num_possible_states, num_possible_actions, initial_state,
                         learning_rate, discount_factor, exploration_rate, training_mode, save_path)
        self.state_size = 4
        self.action_size = num_possible_actions
        self.memory = deque(maxlen=2000)
        self.gamma = discount_factor   # discount rate
        self.epsilon = exploration_rate  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.batch_size = 32
        self.save_path = save_path
        
        # NOTE: Feel Free to edit Setup or Get Action in q_learning.py for further customization or simply even build a q-learning
        #       agent from scratch in my_agent.py

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))
    
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state, verbose=0)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self):
        self.model.load_weights(self.save_path)

    def save(self):
        self.model.save_weights(self.save_path)
    
    def get_action(self):
        if(len(self.get_action_history())>2):
            return self.act(self.state)
        else:
            return np.random.randint(0, 12)

    def update(self):
        my_action_hist = self.get_action_history()
        my_util_hist = self.get_util_history()
        opp1_action_hist = self.get_opp1_action_history()
        opp2_action_hist = self.get_opp2_action_history()
        if(len(opp1_action_hist) > 2):
            self.state = np.reshape([opp1_action_hist[-1], opp1_action_hist[-2], opp2_action_hist[-1], opp2_action_hist[-2]], [1, self.state_size])
            self.last_state = np.reshape([opp1_action_hist[-2], opp1_action_hist[-3], opp2_action_hist[-2], opp2_action_hist[-3]], [1, self.state_size])
            self.last_action = my_action_hist[-2]
            self.last_util = my_util_hist[-1]
            self.memorize(self.last_state, self.last_action, self.last_util, self.state, False)
            if(self.training_mode):
                if len(self.memory) > self.batch_size:
                    self.replay(self.batch_size)

# TODO: Give your agent a NAME 
name = "Arnie the lemonade seller"


# TODO: Determine how many states that your agent will be using
NUM_POSSIBLE_STATES = 256
INITIAL_STATE = [np.random.randint(0,12), np.random.randint(0,12), np.random.randint(0,12), np.random.randint(0,12)]


# Lemonade as 12 possible actions [0 - 11]
NUM_POSSIBLE_ACTIONS = 12
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 1.0

################### SUBMISSION #####################
rl_agent_submission = MyRLAgent(name, NUM_POSSIBLE_STATES, NUM_POSSIBLE_ACTIONS,
                                   INITIAL_STATE, LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE, False, "1000-round-dqn-256-state-space")
####################################################



if __name__ == "__main__":

    # Set if in Training Mode
    Training = False

    if(Training):
        # Train and Save weights
        rl_agent_submission.set_training_mode(True)
        if rl_agent_submission.training_mode: 
            print("TRAINING PERFORMANCE")
            arena = LemonadeArena(
                num_rounds=1000,
                timeout=10,
                players=[
                    rl_agent_submission,
                    StickAgent("Bug1"),
                    ReserveAgent("Bug2"),
                    DecrementAgent("Bug3"),
                    IncrementAgent("Bug4")
                ]
            ) # NOTE: FEEL FREE TO EDIT THE AGENTS HERE TO TRAIN AGAINST A DIFFERENT DISTRIBUTION OF AGENTS. A COUPLE OF EXAMPLE AGENTS
            # TO TRAIN AGAINST ARE IMPORTED FOR YOU. 
            arena.run()
        rl_agent_submission.save()


    else:
        # Test with saved weights
        print("TESTING PERFORMANCE")
        rl_agent_submission.load()
        rl_agent_submission.set_training_mode(False)
        arena = LemonadeArena(
            num_rounds=1000,
            timeout=1,
            players=[
                rl_agent_submission,
                StickAgent("Bug1"),
                ReserveAgent("Bug2"),
                DecrementAgent("Bug3"),
                IncrementAgent("Bug4")
            ]
        )
        # NOTE: FEEL FREE TO EDIT THE AGENTS HERE TO TEST AGAINST A DIFFERENT DISTRIBUTION OF AGENTS. A COUPLE OF EXAMPLE AGENTS
        #       TO TEST AGAINST ARE IMPORTED FOR YOU. 
        arena.run()