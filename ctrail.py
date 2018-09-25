import numpy as np
import math
import constants
import matplotlib.pyplot as plt
from collections import namedtuple, deque

from place_recognition import PlaceRecognition

class Waypoint:
    def __init__(self, state, rep=None, position=None, created_at=None, density=None, steps_to_goal=None):
        self.state = state
        self.rep = rep
        self.position = position
        self.created_at = created_at
        self.density = density
        self.steps_to_goal = steps_to_goal

class Path:
    def __init__(self, created_at=None, density=None):
        self.waypoints = []
        self.created_at = created_at
        self.density = density

class Trail:
    def __init__(self, placeRecognition):
        self.pathes = []
        self.memory_size = 0
        self.placeRecognition = placeRecognition
        self.sequence_similarity = deque(maxlen=constants.SEQUENCE_LENGTH)

    def append_waypoints(self, waypoints, created_at): # [{'state': state, 'position': position}]
        # steps_to_goal = len(waypoints)
        steps_to_goal = 0
        for waypoint in waypoints:
            if (waypoint['action'] not in [constants.ACTION_TURN_RIGHT, constants.ACTION_TURN_LEFT]):
                steps_to_goal += 1

        path = Path(created_at, 1.)
        for waypoint in waypoints:
            rep = self.placeRecognition.forward(waypoint['state'])
            path.waypoints.append(Waypoint(state=waypoint['state'], rep=rep.data.cpu(), position=waypoint['position'], created_at=created_at, density=1.0, steps_to_goal=steps_to_goal)) # temporary for cpu()
            # steps_to_goal -= 1
            if (waypoint['action'] not in [constants.ACTION_TURN_RIGHT, constants.ACTION_TURN_LEFT]):
                steps_to_goal -= 1
        self.memory_size += len(waypoints)
        self.pathes.append(path)
        return True

    def len(self):
        return self.memory_size

    def clear(self):
        self.pathes = []

    def update_waypoints(self):
        i = 0
        while i < len(self.pathes):
            # waypoint.density = 1.0 - (constants.TRAIL_EVAPORATION_COEFFICIENT_PER_CYCLE * (cycle - waypoint.created_at));
            self.pathes[i].density -= constants.TRAIL_EVAPORATION_COEFFICIENT_RATE
            if (self.pathes[i].density < 0):
                del self.pathes[i]
            else:
                i += 1

    def draw_waypoints(self):
        x, y, z = [], [], []
        for path in self.pathes:
            for waypoint in path.waypoints:
                x.append(waypoint.position[0])
                y.append(waypoint.position[1])
                z.append(path.density)

        x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # fig, ax = plt.subplots()
        plt.scatter(x, y, c=z, s=10, edgecolor='')
        plt.pause(0.05)

    def calculate_threshold(self, similarity_array, k):
        n = len(similarity_array)
        if (n == 0):
            return 0.
        size = min([n, k])
        threshold = np.percentile(similarity_array, (n - size) * 100 / float(n))
        return threshold

    def find_closest_waypoint(self, state, backward=False, last_matched=[]):
        results = self.relocalize(state, backward, last_matched) # results contains (index, similaity, velocity)

        best_score = 0.
        best_state = None
        best_velocity = 0

        min_steps_to_goal = 10000 
        for path_id in range(len(self.pathes)):
            # print (results[path_id])
            for item in results[path_id]:
                if (self.pathes[path_id].waypoints[item[0]].steps_to_goal <= min_steps_to_goal):
                    min_steps_to_goal = self.pathes[path_id].waypoints[item[0]].steps_to_goal
                    best_state = self.pathes[path_id].waypoints[item[0]].state
                    best_score = min_steps_to_goal
                    best_velocity = item[2]

        return best_state, best_score, best_velocity, results

    def calculate_threshold_domain(self, rep, path_id, search_domain):
        similarity_dict = {}
        similarity_array = []
        for index in search_domain:
            similarity = self.placeRecognition.compute_similarity_score(self.pathes[path_id].waypoints[index].rep, rep)
            similarity_dict[index] = similarity
            similarity_array.append(similarity)

        threshold = self.calculate_threshold(similarity_array, constants.TRAIL_K_NEAREST_NEIGHBORS)
        return threshold, similarity_dict

    def relocalize(self, state, backward=False, last_matched=[]):
        if (self.len() == 0):
            return [[]]

        rep = self.placeRecognition.forward(state).data.cpu() 

        results = []
        matched_indexes = []

        for path_id in range(len(self.pathes)):
            # Temporality
            search_domain = []
            memory_size = len(self.pathes[path_id].waypoints)
            if (path_id < len(last_matched)):
                for i in last_matched[path_id]:
                    for index in range(i[0]-int(constants.TRAIL_TEMPORALITY_BEHIND_WINDOW_SIZE), i[0]+int(constants.TRAIL_TEMPORALITY_AHEAD_WINDOW_SIZE)):
                        if (index > 0 and index < memory_size and index not in search_domain):
                            search_domain.append(index)

            threshold, similarity_dict = self.calculate_threshold_domain(rep, path_id, search_domain)
            # threshold = 0.
            if (threshold < constants.TRAIL_WEAK_SIMILARITY_THRESHOLD):
                # print ('%d has low threshold: %f %d' % (path_id, threshold, len(search_domain)))
                # print (similarity_dict)
                search_domain = range(memory_size)
                threshold, similarity_dict = self.calculate_threshold_domain(rep, path_id, search_domain)
                threshold = max([threshold, constants.TRAIL_SIMILARITY_THRESHOLD])

            path_results = []
            path_matched_indexes = []
            for index, similarity in similarity_dict.items():
                if (similarity >= threshold):
                    path_results.append((index, similarity, 0.))
                    path_matched_indexes.append(index)

            results_size = len(path_results)
            left_bound = int(results_size * constants.TRAIL_SIMILARITY_INNER_BOUND_RATE)
            right_bound = int(results_size * (1. - constants.TRAIL_SIMILARITY_INNER_BOUND_RATE))
            path_results = path_results[left_bound:right_bound]
            path_matched_indexes = path_matched_indexes[left_bound:right_bound]

            if (len(path_results) > 2):
                # adding next states if still higher than a number
                # lookahead_base_index = results[int(len(results)-1)][0]
                path_matched_indexes_copy = path_matched_indexes.copy()
                for lookahead_base_index in path_matched_indexes_copy:
                    for i in range(constants.TRAIL_LOOKAHEAD_MIN_INDEX, constants.TRAIL_LOOKAHEAD_MAX_INDEX):
                        index = lookahead_base_index + i
                        if (index not in similarity_dict):
                            # print ('---> index not in similarity dict: ', index, similarity_dict.keys())
                            break
                        if (index not in path_matched_indexes and similarity_dict[index] > constants.TRAIL_LOOKAHEAD_SIMILARITY_THRESHOLD):
                            path_results.append((index, similarity_dict[index], 0.))
                            path_matched_indexes.append(index)
                            # print ("lookahead: ", index)
            else:
                path_results = []

            # print ('results: ', threshold, path_results)
            results.append(path_results)
            matched_indexes.append(path_matched_indexes)

        return results
