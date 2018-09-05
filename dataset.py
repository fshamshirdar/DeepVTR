from PIL import Image
import os
import os.path
import random
import math
import numpy as np

import constants
from vizdoom import *

import torch.utils.data
import torchvision.transforms as transforms

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class RecordedAirSimDataLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, locomotion=True, transform=None, validation=False, loader=default_image_loader):
        self.base_path = datapath
        self.is_locomotion = locomotion
        self.transform = transform
        self.loader = loader
        self.indexes = []
        self.actions = []
        self.size = 0
        if validation:
            phase = "validation.txt"
        else:
            phase = "training.txt"

        for index in open(os.path.join(self.base_path, phase)):
            index = index.strip()
            action_file = open(os.path.join(self.base_path, index, "action.txt"))
            actions = [int(val) for val in action_file.read().split('\n') if val.isdigit()]
            self.indexes.append(index)
            self.actions.append(actions)
            self.size += len(actions)-1

    def __getitem__(self, index):
        round_index, index = self.getIndex(index)
        if (self.is_locomotion):
            return self.getLocomotionItem(round_index, index)
        else:
            return self.getPlaceItem(round_index, index)

    def getIndex(self, index):
        round_index = 0
        while (index >= len(self.actions[round_index])-1):
            index -= len(self.actions[round_index])-1
            round_index += 1
        return round_index, index

    def getLocomotionItem(self, round_index, index):
        action = self.actions[round_index][index]

        future_addition_index = random.randint(1, constants.DATASET_MAX_ACTION_DISTANCE)
        #future_index = index + future_addition_index
        #if future_index >= len(self.actions[round_index]):
        #    future_index = index + 1

        permitted_actions = [i for i in range(0, constants.LOCO_NUM_CLASSES)]
        for i in range(1, future_addition_index+1):
            future_index = index + i
            if (future_index >= len(self.actions[round_index])):
                future_index -= 1
                break
            future_action = self.actions[round_index][future_index]
            if (future_action not in permitted_actions):
                future_index -= 1
                break
            if (future_action == constants.ACTION_MOVE_FORWARD and constants.ACTION_MOVE_BACKWARD in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
            elif (future_action == constants.ACTION_MOVE_BACKWARD and constants.ACTION_MOVE_FORWARD in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_FORWARD)
            elif (future_action == constants.ACTION_TURN_RIGHT and constants.ACTION_TURN_LEFT in permitted_actions):
                permitted_actions.remove(constants.ACTION_TURN_LEFT)
            elif (future_action == constants.ACTION_TURN_LEFT and constants.ACTION_TURN_RIGHT in permitted_actions):
                permitted_actions.remove(constants.ACTION_TURN_RIGHT)
            elif (future_action == constants.ACTION_MOVE_RIGHT and constants.ACTION_MOVE_LEFT in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_LEFT)
            elif (future_action == constants.ACTION_MOVE_LEFT and constants.ACTION_MOVE_RIGHT in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_RIGHT)

        previous_index = index - 1
        if previous_index < 0:
            previous_index = 0

        current_state = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(index)+".png"))
        # previous_state = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(previous_index)+".png"))
        future_state = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(future_index)+".png"))
        if self.transform is not None:
            current_state = self.transform(current_state)
            # previous_state = self.transform(previous_state)
            future_state = self.transform(future_state)

        # state = np.concatenate([previous_state, current_state, future_state], axis=0)
        state = np.concatenate([current_state, future_state], axis=0)

        return state, action

    def getPlaceItem(self, round_index, index):
        if (constants.PLACE_TOP_MODEL == constants.PLACE_TOP_TRIPLET):
            return self.getPlaceTripletItem(round_index, index)
        else: # siamese
            return self.getPlaceSiameseItem(round_index, index)

    def getPlaceTripletItem(self, round_index, index):
        positive_addition_index = random.randint(1, constants.DATASET_MAX_ACTION_DISTANCE)
        positive_index = index + positive_addition_index
        if positive_index >= len(self.actions[round_index]):
            positive_index = index + 1
        # negative_index = random.randint(1, self.size-1)
        negative_index_ahead = index + random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
        negative_index_behind = index - random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
        if negative_index_ahead >= len(self.actions[round_index]):
            negative_index = negative_index_behind
        elif negative_index_behind < 0:
            negative_index = negative_index_ahead
        elif random.random() < 0.5:
            negative_index = negative_index_behind
        else:
            negative_index = negative_index_ahead

        anchor = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(index)+".png"))
        positive = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(positive_index)+".png"))
        negative = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(negative_index)+".png"))
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def getPlaceSiameseItem(self, round_index, index):
        if (random.random() < 0.5): # positive
            class_value = 1
            positive_addition_index = random.randint(1, constants.DATASET_MAX_ACTION_DISTANCE)
            pair_index = index + positive_addition_index
            if pair_index >= len(self.actions[round_index]):
                pair_index = index + 1
        else: # negative
            class_value = 0
            # negative_index = random.randint(1, self.size-1)
            negative_index_ahead = index + random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
            negative_index_behind = index - random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
            if negative_index_ahead >= len(self.actions[round_index]):
                pair_index = negative_index_behind
            elif negative_index_behind < 0:
                pair_index = negative_index_ahead
            elif random.random() < 0.5:
                pair_index = negative_index_behind
            else:
                pair_index = negative_index_ahead

        anchor = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(index)+".png"))
        pair = self.loader(os.path.join(self.base_path, self.indexes[round_index], str(pair_index)+".png"))
        if self.transform is not None:
            anchor = self.transform(anchor)
            pair = self.transform(pair)

        # state = np.concatenate([anchor, pair], axis=0)
        return anchor, pair, class_value

    def __len__(self):
        return self.size

class OnlineVizDoomDataLoader(torch.utils.data.Dataset):
    def __init__(self, wad, locomotion=True, transform=None):
        self.is_locomotion = locomotion
        self.seed = self.new_seed()
        self.game = self.initialize_game(wad, game_args=[])
        self.transform = transform
        self.actions = []
        self.states = []
        self.size = 0
        self.collect()

    def new_seed(self):
        self.seed = random.randint(1, 1234567890)
        return self.seed

    def initialize_game(self, wad, game_args):
        game = DoomGame()
        game.load_config(constants.VIZDOOM_DEFAULT_CONFIG)
        for args in game_args:
            game.add_game_args(args)
        game.set_doom_scenario_path(wad)
        game.set_seed(self.seed)
        game.init()
        return game

    def reset_map(self):
        self.new_seed()
        selected_map = (constants.VIZDOOM_MAP_NAME_TEMPLATE % random.randint(constants.VIZDOOM_MIN_RANDOM_TEXTURE_MAP_INDEX, constants.VIZDOOM_MAX_RANDOM_TEXTURE_MAP_INDEX))
        self.game.set_doom_map(selected_map)
        return self.reset_episode()

    def reset_episode(self):
        self.game.set_seed(self.seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        return state

    def step(self, action, repeat=4):
        self.game.make_action(constants.VIZDOOM_ACTIONS_LIST[action], repeat)
        self.game.make_action(constants.VIZDOOM_STAY_IDLE, repeat * 2)
        state = self.game.get_state().screen_buffer.transpose([1, 2, 0])
        # time.sleep(0.1)
        return state

    def random_walk(self):
        action = random.randint(0, constants.LOCO_NUM_CLASSES-1)
        state = self.step(action)
        return state, action, False

    def collect(self):
        current_state = self.reset_map()
        for i in range(5): # needed for skipping first outliers
            current_state, _, _ = self.random_walk()

        for i in range(constants.DATA_COLLECTION_PLAYING_ROUNG_LENGTH):
            future_state, action, done = self.random_walk()
            if (done == True):
                break

            self.states.append(current_state)
            self.actions.append(action)

            current_state = future_state
        self.size = len(self.states)-1

    def __getitem__(self, index):
        if (self.is_locomotion):
            return self.getLocomotionItem(index)
        else:
            return self.getPlaceItem(index)

    def getLocomotionItem(self, round_index, index):
        action = self.actions[index]

        future_addition_index = random.randint(1, constants.DATASET_MAX_ACTION_DISTANCE)
        #future_index = index + future_addition_index
        #if future_index >= len(self.actions):
        #    future_index = index + 1

        permitted_actions = [i for i in range(0, constants.LOCO_NUM_CLASSES)]
        for i in range(1, future_addition_index+1):
            future_index = index + i
            if (future_index >= len(self.actions)):
                future_index -= 1
                break
            future_action = self.actions[future_index]
            if (future_action not in permitted_actions):
                future_index -= 1
                break
            if (future_action == constants.ACTION_MOVE_FORWARD and constants.ACTION_MOVE_BACKWARD in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_BACKWARD)
            elif (future_action == constants.ACTION_MOVE_BACKWARD and constants.ACTION_MOVE_FORWARD in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_FORWARD)
            elif (future_action == constants.ACTION_TURN_RIGHT and constants.ACTION_TURN_LEFT in permitted_actions):
                permitted_actions.remove(constants.ACTION_TURN_LEFT)
            elif (future_action == constants.ACTION_TURN_LEFT and constants.ACTION_TURN_RIGHT in permitted_actions):
                permitted_actions.remove(constants.ACTION_TURN_RIGHT)
            elif (future_action == constants.ACTION_MOVE_RIGHT and constants.ACTION_MOVE_LEFT in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_LEFT)
            elif (future_action == constants.ACTION_MOVE_LEFT and constants.ACTION_MOVE_RIGHT in permitted_actions):
                permitted_actions.remove(constants.ACTION_MOVE_RIGHT)

        previous_index = index - 1
        if previous_index < 0:
            previous_index = 0

        current_state = self.states[index]
        future_state = self.states[future_index]
        if self.transform is not None:
            current_state = self.transform(current_state)
            # previous_state = self.transform(previous_state)
            future_state = self.transform(future_state)

        # state = np.concatenate([previous_state, current_state, future_state], axis=0)
        state = np.concatenate([current_state, future_state], axis=0)

        return state, action

    def getPlaceItem(self, index):
        if (constants.PLACE_TOP_MODEL == constants.PLACE_TOP_TRIPLET):
            return self.getPlaceTripletItem(index)
        else: # siamese
            return self.getPlaceSiameseItem(index)

    def getPlaceTripletItem(self, index):
        positive_addition_index = random.randint(1, constants.DATASET_MAX_ACTION_DISTANCE)
        positive_index = index + positive_addition_index
        if positive_index >= len(self.actions):
            positive_index = index + 1
        # negative_index = random.randint(1, self.size-1)
        negative_index_ahead = index + random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
        negative_index_behind = index - random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
        if negative_index_ahead >= len(self.actions):
            negative_index = negative_index_behind
        elif negative_index_behind < 0:
            negative_index = negative_index_ahead
        elif random.random() < 0.5:
            negative_index = negative_index_behind
        else:
            negative_index = negative_index_ahead

        anchor = self.states[index]
        positive = self.states[positive_index]
        negative = self.states[negative_index]
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def getPlaceSiameseItem(self, index):
        if (random.random() < 0.5): # positive
            class_value = 1
            positive_addition_index = random.randint(1, constants.DATASET_MAX_ACTION_DISTANCE)
            pair_index = index + positive_addition_index
            if pair_index >= len(self.actions):
                pair_index = index + 1
        else: # negative
            class_value = 0
            # negative_index = random.randint(1, self.size-1)
            negative_index_ahead = index + random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
            negative_index_behind = index - random.randint(constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MIN_INDEX, constants.TRAINING_PLACE_NEGATIVE_SAMPLE_MAX_INDEX)
            if negative_index_ahead >= len(self.actions):
                pair_index = negative_index_behind
            elif negative_index_behind < 0:
                pair_index = negative_index_ahead
            elif random.random() < 0.5:
                pair_index = negative_index_behind
            else:
                pair_index = negative_index_ahead

        anchor = self.states[index]
        pair = self.states[pair_index]
        if self.transform is not None:
            anchor = self.transform(anchor)
            pair = self.transform(pair)

        # state = np.concatenate([anchor, pair], axis=0)
        return anchor, pair, class_value

    def __len__(self):
        return self.size

class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, datapath, size=100000, transform=None,
                 loader=default_image_loader):
        self.base_path = datapath
        self.size = size
        self.data = {}
        self.pairs = []
        for index in open(os.path.join(self.base_path, "index.txt")):
            print ("reading index: ", index)
            index = index.strip()
            data = []
            for line in open(os.path.join(self.base_path, index, "index.txt")):
                data.append({'filename': line.rstrip('\n')})

            print ("number of images: ", len(data))
            i = 0
            for line in open(os.path.join(self.base_path, index, "fGPS.txt")):
                gps_info = line.rstrip('\n').split(",")
                data[i]['gps'] = [float(gps_info[0]), float(gps_info[1])]
                i = i + 1
                if (i >= len(data)):
                    break

            print ("number of gps info: ", i)
            self.data[index] = data
#            if (len(data) < self.size):
#                self.size = len(data)

        if os.path.exists(os.path.join(self.base_path, "pairs.txt")):
            for line in open(os.path.join(self.base_path, "pairs.txt")):
                pairs = line.rstrip('\n').split(",")
                self.pairs.append(((pairs[0], pairs[1]), (pairs[2], pairs[3])))
        else:
            self.make_pairs()
            pairs_file = open(os.path.join(self.base_path, "pairs.txt"), 'w')
            for pair in self.pairs:
                pairs_file.write("{},{},{},{}\n".format(pair[0][0], pair[0][1], pair[1][0], pair[1][1]))
            pairs_file.close()

        if (len(self.pairs) < size):
            self.size = len(self.pairs)

        self.transform = transform
        self.loader = loader

    def distance(self, gps1, gps2):
        return math.hypot(gps1[0] - gps2[0], gps1[1] - gps2[1])

    def find_arbitrary_match(self, anchor_gps, positive_data_index):
        shuffled_index = list(range(len(self.data[positive_data_index])))
        random.shuffle(shuffled_index)
        for i in shuffled_index:
            if (self.distance(self.data[positive_data_index][shuffled_index[i]]['gps'], anchor_gps) < 0.00002):
                return i
        return -1

    def make_pairs(self):
        import time
        for i in list(self.data.keys()):
            positive_keys = list(self.data.keys())
            positive_keys.remove(i)
            t1 = time.time()
            for anchor in self.data[i]:
                for positive_index in positive_keys:
                    closest_sample = self.data[positive_index][0]
                    min_distance = 100.
                    for sample in self.data[positive_index]:
                        distance = self.distance(anchor['gps'], sample['gps']) 
                        if (distance < min_distance):
                            min_distance = distance
                            closest_sample = sample
                    if min_distance < 0.0002:
                        self.pairs.append(((i, anchor['filename']), (positive_index, closest_sample['filename'])))
            t2 = time.time()
            print (t2-t1)
        return self.pairs

    def __getitem__(self, index):
        ((anchor_data_index, anchor_path), (positive_data_index, positive_path)) = self.pairs[index]
        negative_data_index = random.choice(list(self.data.keys()))
        negative_dict = random.choice(self.data[negative_data_index])
        negative_path = negative_dict['filename']

        anchor = self.loader(os.path.join(self.base_path, anchor_data_index, anchor_path))
        positive = self.loader(os.path.join(self.base_path, positive_data_index, positive_path))
        negative = self.loader(os.path.join(self.base_path, negative_data_index, negative_path))
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative

    def __len__(self):
        return self.size

    """
    def __getitem__(self, index):
        keys = list(self.data.keys())
        anchor_data_index = random.choice(keys)
        negative_data_index = random.choice(keys)
        keys.remove(anchor_data_index)
        positive_data_index = random.choice(keys)

        anchor_dict = self.data[anchor_data_index][index]
        negative_dict = random.choice(self.data[negative_data_index])
        positive_dict = None

        positive_index = self.find_arbitrary_match(anchor_dict['gps'], positive_data_index)
        if (positive_index == -1):
            print ("did not find a close image")
            positive_data_index = anchor_data_index
            if (index+3 < len(self.data[positive_data_index])):
                positive_dict = self.data[positive_data_index][index+3]
            else:
                positive_dict = self.data[positive_data_index][index-3]
        else:
            positive_dict = self.data[positive_data_index][positive_index]

        anchor = self.loader(os.path.join(self.base_path, anchor_data_index, anchor_dict['filename']))
        positive = self.loader(os.path.join(self.base_path, positive_data_index, positive_dict['filename']))
        negative = self.loader(os.path.join(self.base_path, negative_data_index, negative_dict['filename']))
        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative
    """


if __name__ == "__main__":
    from tqdm import tqdm

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(RecordedAirSimDataLoader("dataset/", locomotion=True), batch_size=1, shuffle=False, **kwargs)
    for data in tqdm(train_loader):
        state, action = data
        print (state.shape)
        print (action)
