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

class Trail:
    def __init__(self, placeRecognition):
        self.waypoints = []
        self.placeRecognition = placeRecognition
        self.sequence_similarity = deque(maxlen=constants.SEQUENCE_LENGTH)

    def append_waypoints(self, waypoints, created_at): # [{'state': state, 'position': position}]
        # steps_to_goal = len(waypoints)

        steps_to_goal = 0
        for waypoint in waypoints:
            if (waypoint['action'] not in [constants.ACTION_TURN_RIGHT, constants.ACTION_TURN_LEFT]):
                steps_to_goal += 1

        for waypoint in waypoints:
            if (self.placeRecognition is None): # ground-base trail
                rep = None
            else:
                rep = self.placeRecognition.forward(waypoint['state'])
                rep = rep.data.cpu() # temporary for cpu()

            self.waypoints.append(Waypoint(state=waypoint['state'], rep=rep, position=waypoint['position'], created_at=created_at, density=1.0, steps_to_goal=steps_to_goal))

            # steps_to_goal -= 1
            if (waypoint['action'] not in [constants.ACTION_TURN_RIGHT, constants.ACTION_TURN_LEFT]):
                steps_to_goal -= 1
        return True

    def len(self):
        return len(self.waypoints)

    def clear(self):
        self.waypoints = []

    def clear_sequence(self):
        self.sequence_similarity = deque(maxlen=constants.SEQUENCE_LENGTH)

    def update_waypoints(self):
        i = 0
        while i < len(self.waypoints):
            # waypoint.density = 1.0 - (constants.TRAIL_EVAPORATION_COEFFICIENT_PER_CYCLE * (cycle - waypoint.created_at));
            self.waypoints[i].density -= constants.TRAIL_EVAPORATION_COEFFICIENT_RATE
            if (self.waypoints[i].density < 0):
                del self.waypoints[i]
            else:
                i += 1

        # for index, waypoint in enumerate(self.waypoints):
        #     # waypoint.density = 1.0 - (constants.TRAIL_EVAPORATION_COEFFICIENT_PER_CYCLE * (cycle - waypoint.created_at));
        #     waypoint.density -= constants.TRAIL_EVAPORATION_COEFFICIENT_RATE
        #     if (waypoint.density < 0):
        #         del self.waypoints[index]

    def draw_waypoints(self):
        plt.clf()

        x, y, z = [], [], []
        for waypoint in self.waypoints:
            x.append(waypoint.position[0])
            y.append(waypoint.position[1])
            z.append(waypoint.density)

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

    def find_closest(self, input):
        rep = self.placeRecognition.forward(input).data.cpu()
        similarities = np.asarray([ self.placeRecognition.compute_similarity_score(rep, waypoint.rep) for waypoint in self.waypoints ])
        index = similarities.argmax()
        similarity = similarities[index]
        if (similarity > constants.GOAL_SIMILARITY_THRESHOLD):
            return self.waypoints[index], index, similarity
        else:
            return None, -1, 0.0

    def find_best_waypoint(self, state, backward=False, last_matched=[]):
        results = self.relocalize(state, backward, last_matched) # results contains (index, similaity, velocity)

        min_steps_to_goal = 10000 
        for item in results:
            if (self.waypoints[item[0]].steps_to_goal < min_steps_to_goal):
                min_steps_to_goal = self.waypoints[item[0]].steps_to_goal

        last_matched_indexes = []
        best_state = None
        best_score = 0.
        best_index = -1
        best_velocity = 0
        for item in results:
            score = (constants.TRAIL_STEP_TO_TARGET_WEIGHT * (min_steps_to_goal / self.waypoints[item[0]].steps_to_goal) +
                     constants.TRAIL_SIMILARITY_WEIGHT * (item[1])) / (constants.TRAIL_STEP_TO_TARGET_WEIGHT + constants.TRAIL_SIMILARITY_WEIGHT)
            if (score > best_score):
                best_score = score
                best_state = self.waypoints[item[0]].state
                best_index = item[0]
                best_velocity = item[2]

        return best_state, best_score, best_velocity, results

    def find_closest_waypoint(self, state, backward=False, last_matched=[]):
        results = self.relocalize(state, backward, last_matched) # results contains (index, similaity, velocity)

        best_state = None
        best_score = 0.
        best_index = -1
        best_velocity = 0

        min_steps_to_goal = 10000 
        for item in results:
            if (self.waypoints[item[0]].steps_to_goal <= min_steps_to_goal):
                min_steps_to_goal = self.waypoints[item[0]].steps_to_goal
                best_state = self.waypoints[item[0]].state
                best_score = min_steps_to_goal
                best_index = item[0]
                best_velocity = item[2]

        print ("closest match: ", best_score, best_index)
        return best_state, best_score, best_velocity, results

    def find_most_similar_waypoint(self, state, backward=False, last_matched=[]):
        results = self.relocalize(state, backward, last_matched) # results contains (index, similaity, velocity)

        best_state = None
        best_score = 0.
        best_index = -1
        best_velocity = 0

        for item in results:
            if (item[1] > best_score):
                best_state = self.waypoints[item[0]].state
                best_score = item[1]
                best_index = item[0]
                best_velocity = item[2]

        return best_state, best_score, best_velocity, results

    def calculate_threshold_domain(self, rep, search_domain):
        similarity_dict = {}
        similarity_array = []
        for index in search_domain:
            similarity = self.placeRecognition.compute_similarity_score(self.waypoints[index].rep, rep)
            similarity_dict[index] = similarity
            similarity_array.append(similarity)

        threshold = self.calculate_threshold(similarity_array, constants.TRAIL_K_NEAREST_NEIGHBORS)
        return threshold, similarity_dict

    def relocalize(self, state, backward=False, last_matched=[]):
        return self.global_relocalize(state, backward, last_matched)
        # return self.knn_relocalize(state, backward, last_matched)

    def global_relocalize(self, state, backward=False, last_matched=[]):
        if (self.len() == 0):
            return []

        rep = self.placeRecognition.forward(state).data.cpu() 
        memory_size = len(self.waypoints)

        # Temporality
        search_domain = []
        for i in last_matched:
            for index in range(i[0]-int(constants.TRAIL_TEMPORALITY_BEHIND_WINDOW_SIZE), i[0]+int(constants.TRAIL_TEMPORALITY_AHEAD_WINDOW_SIZE)):
                if (index > 0 and index < memory_size and index not in search_domain):
                    search_domain.append(index)

        # print (search_domain)

        threshold, similarity_dict = self.calculate_threshold_domain(rep, search_domain)
        # threshold = 0.
        if (threshold < constants.TRAIL_WEAK_SIMILARITY_THRESHOLD):
            print ('low threshold: ', threshold)
            # print (similarity_dict)
            search_domain = range(memory_size)
            threshold, similarity_dict = self.calculate_threshold_domain(rep, search_domain)
            # threshold = max([threshold, constants.TRAIL_SIMILARITY_THRESHOLD])
            threshold = constants.TRAIL_SIMILARITY_THRESHOLD
        else:
            threshold = constants.TRAIL_WEAK_SIMILARITY_THRESHOLD

        results = []
        matched_indexes = []
        for index, similarity in similarity_dict.items():
            if (similarity >= threshold):
                results.append((index, similarity, 0.))
                matched_indexes.append(index)

        results_size = len(results)
        left_bound = int(results_size * constants.TRAIL_SIMILARITY_INNER_BOUND_RATE)
        right_bound = int(results_size * (1. - constants.TRAIL_SIMILARITY_INNER_BOUND_RATE))
        # results = results[left_bound:right_bound]
        # matched_indexes = matched_indexes[left_bound:right_bound]

        if (len(results) > 2):
            # adding next states if still higher than a number
            # lookahead_base_index = results[int(len(results)-1)][0]
            matched_indexes_copy = matched_indexes.copy()
            for lookahead_base_index in matched_indexes_copy:
                for i in range(constants.TRAIL_LOOKAHEAD_MIN_INDEX, constants.TRAIL_LOOKAHEAD_MAX_INDEX):
                    index = lookahead_base_index + i
                    if (index < 0 or index > memory_size):
                        break
                    if (index not in similarity_dict):
                        # print ('---> index not in similarity dict: ', index, similarity_dict.keys())
                        break
                    if (index not in matched_indexes and similarity_dict[index] > constants.TRAIL_LOOKAHEAD_SIMILARITY_THRESHOLD):
                        results.append((index, similarity_dict[index], 0.))
                        matched_indexes.append(index)
                        # print ("lookahead: ", index)
        else:
            results = []

        print ('results: ', threshold, results)
        return results

    def knn_relocalize(self, state, backward=False, last_matched=[]):
        if (self.len() == 0):
            return []

        rep = self.placeRecognition.forward(state).data.cpu() 
        memory_size = len(self.waypoints)

        # Temporality
        search_domain = []
        for i in last_matched:
            for index in range(i[0]-int(constants.TRAIL_TEMPORALITY_BEHIND_WINDOW_SIZE), i[0]+int(constants.TRAIL_TEMPORALITY_AHEAD_WINDOW_SIZE)):
                if (index > 0 and index < memory_size and index not in search_domain):
                    search_domain.append(index)

        # print (search_domain)

        threshold, similarity_dict = self.calculate_threshold_domain(rep, search_domain)
        # threshold = 0.
        if (threshold < constants.TRAIL_WEAK_SIMILARITY_THRESHOLD):
            print ('low threshold: ', threshold)
            # print (similarity_dict)
            search_domain = range(memory_size)
            threshold, similarity_dict = self.calculate_threshold_domain(rep, search_domain)
            threshold = max([threshold, constants.TRAIL_SIMILARITY_THRESHOLD])

        results = []
        matched_indexes = []
        for index, similarity in similarity_dict.items():
            if (similarity >= threshold):
                results.append((index, similarity, 0.))
                matched_indexes.append(index)

        results_size = len(results)
        left_bound = int(results_size * constants.TRAIL_SIMILARITY_INNER_BOUND_RATE)
        right_bound = int(results_size * (1. - constants.TRAIL_SIMILARITY_INNER_BOUND_RATE))
        results = results[left_bound:right_bound]
        matched_indexes = matched_indexes[left_bound:right_bound]

        if (len(results) > 2):
            # adding next states if still higher than a number
            # lookahead_base_index = results[int(len(results)-1)][0]
            matched_indexes_copy = matched_indexes.copy()
            for lookahead_base_index in matched_indexes_copy:
                for i in range(constants.TRAIL_LOOKAHEAD_MIN_INDEX, constants.TRAIL_LOOKAHEAD_MAX_INDEX):
                    index = lookahead_base_index + i
                    if (index not in similarity_dict):
                        # print ('---> index not in similarity dict: ', index, similarity_dict.keys())
                        break
                    if (index not in matched_indexes and similarity_dict[index] > constants.TRAIL_LOOKAHEAD_SIMILARITY_THRESHOLD):
                        results.append((index, similarity_dict[index], 0.))
                        matched_indexes.append(index)
                        # print ("lookahead: ", index)
        else:
            results = []

        # print ('results: ', threshold, results)
        return results

    def relocalize1(self, state, backward=False):
        if (self.len() == 0):
            return []

        rep = self.placeRecognition.forward(state).data.cpu() 
        memory_size = len(self.waypoints)
        # Applying SeqSLAM
        similarity_array = []
        for index in range(memory_size): # heuristic on the search domain
            similarity_array.append(self.placeRecognition.compute_similarity_score(self.waypoints[index].rep, rep))

        self.sequence_similarity.append(similarity_array)

        results = []
        sequence_size = len(self.sequence_similarity)
        max_similarity_score = 0
        best_velocity = 0
        matched_index = -1
        for index in range(memory_size):
            iter_max_similarity_score = 0
            iter_best_velocity = 0
            for sequence_velocity in constants.SEQUENCE_VELOCITIES:
                similarity_score = 0
                for sequence_index in range(len(self.sequence_similarity)):
                    if backward:
                        calculated_index = min(int(index + (sequence_velocity * sequence_index)), memory_size-1)
                    else: # forward
                        calculated_index = max(int(index - (sequence_velocity * sequence_index)), 0)
                    similarity_score += self.sequence_similarity[sequence_size - sequence_index - 1][calculated_index]
                similarity_score /= sequence_size
                # if (similarity_score > max_similarity_score):
                #     matched_index = index
                #     max_similarity_score = similarity_score
                #     best_velocity = sequence_velocity
                if (similarity_score > iter_max_similarity_score):
                    iter_max_similarity_score = similarity_score
                    iter_best_velocity = sequence_velocity
            if (iter_max_similarity_score > constants.TRAIL_SIMILARITY_THRESHOLD):
                results.append((index, iter_max_similarity_score, iter_best_velocity))

        return results

    def find_closest_ground_waypoint(self, pose, backward=False):
        results = self.ground_relocalize(pose, backward) # results contains (index, similaity, velocity)

        best_state = None
        best_position = None
        best_score = 0.
        best_index = -1

        min_steps_to_goal = 10000 
        for item in results:
            if (self.waypoints[item[0]].steps_to_goal <= min_steps_to_goal):
                min_steps_to_goal = self.waypoints[item[0]].steps_to_goal
                best_state = self.waypoints[item[0]].state
                best_position = self.waypoints[item[0]].position
                best_score = min_steps_to_goal
                best_index = item[0]

        print ("closest match: ", best_score, best_index)
        return best_state, best_position, best_score, results

    def ground_relocalize(self, position, backward=False):
        results = []
        memory_size = len(self.waypoints)
        for index in range(memory_size):
            distance = math.sqrt((position[0] - self.waypoints[index].position[0]) ** 2 +
                                 (position[1] - self.waypoints[index].position[1]) ** 2 +
                                 (position[2] - self.waypoints[index].position[2]) ** 2)
            if (distance < constants.TRAIL_GROUND_RADIUS_THRESHOLD):
                results.append((index, distance))

        return results
        
        ### closest point
        # memory_size = len(self.waypoints)
        # min_distance = 10000.
        # matched_index = -1
        # for index in range(memory_size):
        #     distance = math.sqrt((position[0] - self.waypoints[index].position[0]) ** 2 +
        #                          (position[1] - self.waypoints[index].position[1]) ** 2 +
        #                          (position[2] - self.waypoints[index].position[2]) ** 2)
        #     if (distance < min_distance):
        #         min_distance = distance
        #         matched_index = index
        # 
        # return matched_index, min_distance, 0

    def ground_lookahead_relocalize(self, position):
        memory_size = len(self.waypoints)
        min_distance = 10000.
        matched_index = -1
        for index in reversed(range(memory_size)):
            distance = math.sqrt((position[0] - self.waypoints[index].position[0]) ** 2 +
                                 (position[1] - self.waypoints[index].position[1]) ** 2 +
                                 (position[2] - self.waypoints[index].position[2]) ** 2)
            if (distance < constants.DQN_MAX_DISTANCE_THRESHOLD):
                matched_index = index
                return matched_index, distance, 0

        return matched_index, min_distance, 0
