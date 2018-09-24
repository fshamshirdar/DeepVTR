import numpy as np
import math
import networkx
import constants
from collections import namedtuple

from place_recognition import PlaceRecognition

Keyframe = namedtuple('Keyframe', 'state, rep, action, terminal, position')
NODE_TEMPLATE = '%d_%d' # robot_id, frame_id

class MSPTM:
    def __init__(self, placeRecognition, num_robots=1):
        self.num_robots = num_robots
        self.memory = [[] for i in range(self.num_robots)]
        self.graph = networkx.Graph()
        self.placeRecognition = placeRecognition
        self.shortcuts = []

    def append_keyframe(self, robot_id, input, action=None, terminal=False, position=None):
        rep = self.placeRecognition.forward(input)
        self.memory[robot_id].append(Keyframe(state=input, rep=rep.data.cpu(), action=0, terminal=terminal, position=position)) # temporary for cpu()
        return rep, True

    def get_num_robots(self):
        return self.num_robots

    def len(self, robot_id=0):
        return len(self.memory[robot_id]);

    def get_memory(self, robot_id=0):
        return self.memory[robot_id]

    def get_graph(self):
        return self.graph

    def clear(self):
        self.memory = [[] for i in range(self.num_robots)]
        self.graph = networkx.Graph()
        self.shortcuts = []

    def build_graph(self, with_shortcuts=True):
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(range(memory_size))
        for robot_id in range(self.num_robots):
            for first_id in range(len(self.memory[robot_id]) - 1):
                first = '%d_%d' % (robot_id, first_id)
                second = '%d_%d' % (robot_id, first_id+1)
                self.graph.add_edge(first, second)
                self.graph.add_edge(second, first)

                if (with_shortcuts):
                    for shortcut_id in range(frame_id + 1 + constants.MIN_SHORTCUT_DISTANCE, len(self.memory[robot_id])):
                        values = []
                        for shift in range(-constants.SHORTCUT_WINDOW, constants.SHORTCUT_WINDOW + 1):
                            first_shifted = first_id + shift
                            second_shifted = shortcut_id + shift
                            if first_shifted < memory_size and second_shifted < memory_size and first_shifted >= 0 and second_shifted >= 0:
                                values.append(self.placeRecognition.compute_similarity_score(self.memory[first_shifted].rep, self.memory[second_shifted].rep))
                        quality = np.median(values)
                        # print (first, second, quality)
                        if (quality > constants.SHORTCUT_SIMILARITY_THRESHOLD):
                            self.shortcuts.append((quality, robot_id, first_id, robot_id, shortcut_id))
                            shortcut_str = '%d_%d' % (robot_id, shortcut_id)
                            self.graph.add_edge(first, shortcut_str)
                            self.graph.add_edge(shortcut_str, first)

                    for robot_id
                    for shortcut_id in range(frame_id + 1 + constants.MIN_SHORTCUT_DISTANCE, len(self.memory[robot_id])):
                        values = []
                        for shift in range(-constants.SHORTCUT_WINDOW, constants.SHORTCUT_WINDOW + 1):
                            first_shifted = first_id + shift
                            second_shifted = shortcut_id + shift
                            if first_shifted < memory_size and second_shifted < memory_size and first_shifted >= 0 and second_shifted >= 0:
                                values.append(self.placeRecognition.compute_similarity_score(self.memory[first_shifted].rep, self.memory[second_shifted].rep))
                        quality = np.median(values)
                        # print (first, second, quality)
                        if (quality > constants.SHORTCUT_SIMILARITY_THRESHOLD):
                            self.shortcuts.append((quality, robot_id, first_id, robot_id, shortcut_id))
                            shortcut_str = '%d_%d' % (robot_id, shortcut_id)
                            self.graph.add_edge(first, shortcut_str)
                            self.graph.add_edge(shortcut_str, first)
                     

    def add_shortcut(self, first, second, quality):
        self.shortcuts.append((quality, first, second))
        self.graph.add_edge(first, second)
        self.graph.add_edge(second, first)

    def find_shortest_path(self, source, goal):
        shortest_path = networkx.shortest_path(self.graph, source=source, target=goal, weight='weight')
        return shortest_path

    def find_closest(self, input):
        rep = self.placeRecognition.forward(input).data.cpu()
        similarities = np.asarray([ self.placeRecognition.compute_similarity_score(rep, keyframe.rep) for keyframe in self.memory ])
        index = similarities.argmax()
        similarity = similarities[index]
        if (similarity > constants.GOAL_SIMILARITY_THRESHOLD):
            return self.memory[index], index, similarity
        else:
            return None, -1, 0.0

    def relocalize(self, sequence, backward=False):
        sequence_reps = [ self.placeRecognition.forward(frame).data.cpu() for frame in sequence ]
        memory_size = len(self.memory)
        sequence_size = len(sequence)
        similarity_matrix = []
        # Applying SeqSLAM
        for index in range(memory_size): # heuristic on the search domain
            similarity_array = []
            for sequence_index in range(0, sequence_size):
                similarity_array.append(self.placeRecognition.compute_similarity_score(self.memory[index].rep, sequence_reps[sequence_index]))
            similarity_matrix.append(similarity_array)

        # print (similarity_matrix)

        max_similarity_score = 0
        best_velocity = 0
        matched_index = -1
        for index in range(memory_size):
            for sequence_velocity in constants.SEQUENCE_VELOCITIES:
                similarity_score = 0
                for sequence_index in range(0, sequence_size):
                    if backward:
                        calculated_index = min(int(index + (sequence_velocity * sequence_index)), memory_size-1)
                    else: # forward
                        calculated_index = max(int(index - (sequence_velocity * sequence_index)), 0)
                    similarity_score += similarity_matrix[calculated_index][sequence_size - sequence_index - 1]
                similarity_score /= sequence_size
                if (similarity_score > max_similarity_score):
                    matched_index = index
                    max_similarity_score = similarity_score
                    best_velocity = sequence_velocity

        return matched_index, max_similarity_score, best_velocity

    def ground_relocalize(self, position):
        memory_size = len(self.memory)
        min_distance = 10000.
        matched_index = -1
        for index in range(memory_size):
            distance = math.sqrt((position[0] - self.memory[index].position[0]) ** 2 +
                                 (position[1] - self.memory[index].position[1]) ** 2 +
                                 (position[2] - self.memory[index].position[2]) ** 2)
            if (distance < min_distance):
                min_distance = distance
                matched_index = index

        return matched_index, min_distance, 0

    def ground_lookahead_relocalize(self, position):
        memory_size = len(self.memory)
        min_distance = 10000.
        matched_index = -1
        for index in reversed(range(memory_size)):
            distance = math.sqrt((position[0] - self.memory[index].position[0]) ** 2 +
                                 (position[1] - self.memory[index].position[1]) ** 2 +
                                 (position[2] - self.memory[index].position[2]) ** 2)
            if (distance < constants.DQN_MAX_DISTANCE_THRESHOLD):
                matched_index = index
                return matched_index, distance, 0

        return matched_index, min_distance, 0

    def particle_filter_localization(self, sequence):
        return -1

if __name__ == "__main__":
    sptm = SPTM()
    sptm.append(1)
    sptm.append(2)
    sptm.append(3)
    sptm.append(4)
    sptm.build_graph()
    shortest_path = sptm.find_shortest_path(1, 3)
    print (shortest_path)
