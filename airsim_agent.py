import gym
import gym_airsim

from agent import Agent
from AirSimClient import AirSimClientBase # needed for write_png
import constants

class AirSimAgent(Agent):
    def __init__(self):
        self.agent = agent
        self.env = gym.make('AirSim-v1')
        self.env.reset()

    def random_walk(self):
        action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
        next_state, _, done, _ = self.env.step(action)
        return next_state, action, done

    def teach(self):
        state = self.env.reset()
        for i in range(constants.AIRSIM_AGENT_TEACH_LEN):
            next_state, action, done = random_walk()
            self.agent.
            if done:
                break 

    def dry_run_test(self, args):
        goal_variable = None
        source_variable = None
        source_picked = False
        goal_index = 0

        with open(os.path.join(args.datapath, "teach.txt"), 'r') as reader:
            for image_path in reader:
                print (image_path)
                image_path = image_path.strip()
                image = Image.open(os.path.join(args.datapath, image_path)).convert('RGB')
                image_tensor = self.preprocess(image)

#               plt.figure()
#               plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#               plt.show()

                image_tensor.unsqueeze_(0)
                image_variable = Variable(image_tensor).cuda()
                self.sptm.append_keyframe(image_variable)

        with open(os.path.join(args.datapath, "repeat.txt"), 'r') as reader:
            sequence = deque(maxlen=constants.SEQUENCE_LENGTH)
            for image_path in reader:
                print (image_path)
                image_path = image_path.strip()
                image = Image.open(os.path.join(args.datapath, image_path)).convert('RGB')
                image_tensor = self.preprocess(image)

#               plt.figure()
#               plt.imshow(image_tensor.cpu().numpy().transpose((1, 2, 0)))
#               plt.show()

                image_tensor.unsqueeze_(0)
                image_variable = Variable(image_tensor).cuda()
                sequence.append(image_variable)

                if (len(sequence) == constants.SEQUENCE_LENGTH):
                    self.sptm.relocalize(sequence)

#        self.sptm.build_graph()
#        goal, goal_index = self.sptm.find_closest(goal_variable)
#        source, source_index = self.sptm.find_closest(source_variable)
#        if (source != None and goal != None):
#            print (source_index, goal_index)
#            path = self.sptm.find_shortest_path(source_index, goal_index)
#            print (path)
