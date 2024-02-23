from my_rl_lemonade_agent import rl_agent_submission 
from my_rl_lemonade_agent import MyRLAgent
from my_lemonade_agent import nrl_agent_submission 
from agt_server.agents.base_agents.lemonade_agent import LemonadeAgent
from agt_server.local_games.lemonade_arena import LemonadeArena
from agt_server.agents.test_agents.lemonade.stick_agent.my_agent import StickAgent
from agt_server.agents.test_agents.lemonade.always_stay.my_agent import ReserveAgent
from agt_server.agents.test_agents.lemonade.decrement_agent.my_agent import DecrementAgent
from agt_server.agents.test_agents.lemonade.increment_agent.my_agent import IncrementAgent
import numpy as np

# NOTE: The README will contain helpful methods for implementing your agent, please take a look at it!
class MyAgent(LemonadeAgent):
    def __init__(self, name):
        self.state_size = 256
        self.action_size = 12
        self.agent = MyRLAgent("BRUH", self.state_size, self.action_size, (0, 0, 0, 0), 0.001, 0.95, 1.0, True)
        self.done = False
        self.batch_size = 32
        super().__init__(name)

    def get_action(self):
        if(len(self.get_action_history())>=3):
            return self.agent.act(self.state)
        else:
            return np.random.randint(0, 12)

    def update(self):
        my_action_hist = self.get_action_history()
        my_util_hist = self.get_util_history()
        opp1_action_hist = self.get_opp1_action_history()
        opp2_action_hist = self.get_opp2_action_history()
        if(len(opp1_action_hist) > 2):
            self.state = (opp1_action_hist[-1]//3, opp1_action_hist[-2]//3, opp2_action_hist[-1]//3, opp2_action_hist[-2]//3)
            self.last_state = (opp1_action_hist[-2]//3, opp1_action_hist[-3]//3, opp2_action_hist[-2]//3, opp2_action_hist[-3]//3)
            self.last_action = my_action_hist[-2]
            self.last_util = my_util_hist[-1]
            self.agent.memorize(self.last_state, self.last_action, self.last_util, self.state)
            if len(self.agent.memory) > self.batch_size:
                self.agent.replay(self.batch_size)

    

# TODO: Give your agent a NAME 
name = "SKELETONRUSH"

################### SUBMISSION #####################
# TODO: Set to your RL Agent by default, change it to whatever you want as long as its a agent that inherits LemonadeAgent
agent_submission = rl_agent_submission
################### SUBMISSION #####################

if __name__ == "__main__":
    print("TESTING PERFORMANCE")
    arena = LemonadeArena(
        num_rounds=1000,
        timeout=1,
        players=[
            agent_submission,
            StickAgent("Bug1"),
            ReserveAgent("Bug2"),
            DecrementAgent("Bug3"),
            IncrementAgent("Bug4")
        ]
    )
    # NOTE: FEEL FREE TO EDIT THE AGENTS HERE TO TEST AGAINST A DIFFERENT DISTRIBUTION OF AGENTS. A COUPLE OF EXAMPLE AGENTS
    #       TO TEST AGAINST ARE IMPORTED FOR YOU. 
    arena.run()