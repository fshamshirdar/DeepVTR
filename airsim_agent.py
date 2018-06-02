import gym
import gym_airsim

from agent import Agent
from AirSimClient import AirSimClientBase # needed for write_png
import constants

class AirSimAgent(Agent):
    def __init__(self):
        super(AirSimAgent, self).__init__()
        self.env = gym.make('AirSim-v1')
        self.env.reset()
        self.goal = None

    def random_walk(self):
        action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
        next_state, _, done, _ = self.env.step(action)
        return next_state, action, done

    def teach(self):
        state = self.env.reset()
        for i in range(constants.AIRSIM_AGENT_TEACH_LEN):
            next_state, action, done = random_walk()
            self.sptm.append_keyframe(state, action, done)
            state = next_state
            if done:
                break 
        self.goal = state
 
    def repeat(self):
        if (self.goal == None):
            return

        self.sptm.build_graph()
        goal, goal_index, similarity = self.sptm.find_closest(self.goal)
        sequence = deque(maxlen=constants.SEQUENCE_LENGTH)

        current_state = self.env.reset()
        previous_state = current_state
        sequence.append(state)
        while (True):
            matched_index = self.sptm.relocalize(sequence)
            path = self.sptm.find_shortest_path(matched_index, goal_index)
            print (path)
            future_state = path[1].state
            action = self.navigation.forward(previous_state, current_state, future_state).data.cpu()
            next_state, _, done, _ = self.env.step(action)

    def run(self):
        print ("Running teaching phase")
        self.teach()
        print ("Running repeating phase")
        self.repeat()
