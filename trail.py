import numpy as np
import math
import constants
from collections import namedtuple

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

    def append_waypoint(self, input, created_at=None, steps_to_goal=None, position=None):
        rep = self.placeRecognition.forward(input)
        self.waypoints.append(Waypoint(state=input, rep=rep.data.cpu(), position=position, created_at=created_at, density=1.0, steps_to_goal=steps_to_goal)) # temporary for cpu()
        return rep, True

    def len(self):
        return len(self.waypoints)

    def get_waypoints(self):
        return self.waypoints

    def clear(self):
        self.waypoints = []

    def update_waypoints(self):
        for index, waypoint in enumerate(self.waypoints):
            # waypoint.density = 1.0 - (constants.TRAIL_EVAPORATION_COEFFICIENT_PER_CYCLE * (cycle - waypoint.created_at));
            waypoint.density -= constants.TRAIL_EVAPORATION_COEFFICIENT_RATE
            if (waypoint.density < 0):
                del self.waypoints[index]

    def find_closest(self, input):
        rep = self.placeRecognition.forward(input).data.cpu()
        similarities = np.asarray([ self.placeRecognition.compute_similarity_score(rep, waypoint.rep) for waypoint in self.waypoints ])
        index = similarities.argmax()
        similarity = similarities[index]
        if (similarity > constants.GOAL_SIMILARITY_THRESHOLD):
            return self.waypoints[index], index, similarity
        else:
            return None, -1, 0.0

    def find_best_waypoint(self, sequence, backward=False):
        results = self.relocalize(sequence, backward) # results contains (index, similaity, velocity)

        min_steps_to_goal = 10000 
        for item in results:
            if (self.waypoints[item[0]].steps_to_goal < min_steps_to_goal):
                min_steps_to_goal = self.waypoints[item[0]].steps_to_goal

        best_score = 0.
        best_index = -1
        best_velocity = 0.
        for item in results:
            score = (constants.TRAIL_STEP_TO_TARGET_WEIGHT * (min_steps_to_goal / self.waypoints[item[0]].steps_to_goal) +
                     constants.TRAIL_SIMILARITY_WEIGHT * (item[1])) / (constants.TRAIL_STEP_TO_TARGET_WEIGHT + constants.TRAIL_SIMILARITY_WEIGHT)
            if (score > best_score):
                best_score = score
                best_index = item[0]
                best_velocity = item[2]

        return best_index, best_score, best_velocity

    def find_closest_waypoint(self, sequence, backward=False):
        results = self.relocalize(sequence, backward) # results contains (index, similaity, velocity)

        best_score = 0.
        best_index = -1
        best_velocity = 0.

        min_steps_to_goal = 10000 
        for item in results:
            if (self.waypoints[item[0]].steps_to_goal < min_steps_to_goal):
                min_steps_to_goal = self.waypoints[item[0]].steps_to_goal
                best_score = min_steps_to_goal
                best_index = item[0]
                best_velocity = item[2]

        return best_index, best_score, best_velocity

    def find_most_similar_waypoint(self, sequence, backward=False):
        results = self.relocalize(sequence, backward) # results contains (index, similaity, velocity)

        best_score = 0.
        best_index = -1
        best_velocity = 0.

        for item in results:
            if (item[1] > best_score):
                best_score = item[1]
                best_index = item[0]
                best_velocity = item[2]

        return best_index, best_score, best_velocity

    def relocalize(self, sequence, backward=False):
        if (self.len() == 0 or len(sequence) == 0):
            return []

        # sequence_reps = [ self.placeRecognition.forward(frame).data.cpu() for frame in sequence ] # to cache
        sequence_reps = sequence
        memory_size = len(self.waypoints)
        sequence_size = len(sequence)
        similarity_matrix = []
        # Applying SeqSLAM
        for index in range(memory_size): # heuristic on the search domain
            similarity_array = []
            for sequence_index in range(0, sequence_size):
                similarity_array.append(self.placeRecognition.compute_similarity_score(self.waypoints[index].rep, sequence_reps[sequence_index]))
            similarity_matrix.append(similarity_array)

        results = []
        max_similarity_score = 0
        best_velocity = 0
        matched_index = -1
        for index in range(memory_size):
            iter_max_similarity_score = 0
            iter_best_velocity = 0
            for sequence_velocity in constants.SEQUENCE_VELOCITIES:
                similarity_score = 0
                for sequence_index in range(0, sequence_size):
                    if backward:
                        calculated_index = min(int(index + (sequence_velocity * sequence_index)), memory_size-1)
                    else: # forward
                        calculated_index = max(int(index - (sequence_velocity * sequence_index)), 0)
                    similarity_score += similarity_matrix[calculated_index][sequence_size - sequence_index - 1]
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

    def ground_relocalize(self, position):
        memory_size = len(self.waypoints)
        min_distance = 10000.
        matched_index = -1
        for index in range(memory_size):
            distance = math.sqrt((position[0] - self.waypoints[index].position[0]) ** 2 +
                                 (position[1] - self.waypoints[index].position[1]) ** 2 +
                                 (position[2] - self.waypoints[index].position[2]) ** 2)
            if (distance < min_distance):
                min_distance = distance
                matched_index = index

        return matched_index, min_distance, 0

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
