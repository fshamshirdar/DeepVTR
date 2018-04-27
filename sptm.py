import numpy as np
import networkx
import constants
from collections import namedtuple

from place_recognition import PlaceRecognition

Keyframe = namedtuple('Keyframe', 'state, rep, action, terminal')

class SPTM:
    def __init__(self, placeRecognition):
        self.memory = []
        self.graph = networkx.Graph()
        self.placeRecognition = placeRecognition

    def append_keyframe(self, input, terminal=False):
        rep = self.placeRecognition.forward(input)
        self.memory.append(Keyframe(state=input, rep=rep.data.cpu(), action=0, terminal=terminal)) # temporary for cpu()
        return True

    def len(self):
        return len(self.memory);

    def find_shortcuts(self):
        shortcuts = []
        for first in range(len(self.memory)):
            for second in range(first + 1 + constants.MIN_SHORTCUT_DISTANCE, len(self.memory)):
                values = []
                for shift in range(-constants.SHORTCUT_WINDOW, constants.SHORTCUT_WINDOW + 1):
                    first_shifted = first + shift
                    second_shifted = second + shift
                    if first_shifted < len(shortcuts_matrix) and second_shifted < len(shortcuts_matrix) and first_shifted >= 0 and second_shifted >= 0:
                        values.append(shortcuts_matrix[first_shifted][second_shifted])
                quality = median(values)
                distance = get_distance(keyframe_coordinates[first], keyframe_coordinates[second])
                shortcuts.append((quality, first, second, distance))
        return np.array(shortcuts)

    def build_graph(self):
        memory_size = len(self.memory)
        self.graph = networkx.Graph()
        self.graph.add_nodes_from(range(memory_size))
        for first in range(memory_size - 1):
            self.graph.add_edge(first, first + 1)
            self.graph.add_edge(first + 1, first)

#        self.compute_shortcuts(keyframes, keyframe_coordinates)
#        for index in range(self.get_number_of_shortcuts()):
#            edge = self.get_shortcut(index)
#            first, second = edge
#            assert abs(first - second) > MIN_SHORTCUT_DISTANCE
#            self.add_double_sided_edge(*edge)

    def find_shortest_path(self, source, goal):
        shortest_path = networkx.shortest_path(self.graph, source=source, target=goal, weight='weight')
        return shortest_path

    def find_closest(self, input):
        rep = self.placeRecognition.forward(input).data.cpu()
        similarities = np.asarray([ self.placeRecognition.compute_similarity_score(rep, keyframe.rep) for keyframe in self.memory ])
        index = similarities.argmax()
        similarity = similarities[index]
        print (similarity)
        if (similarity > constants.GOAL_SIMILARITY_THRESHOLD):
            return self.memory[index], index
        else:
            return None, -1

if __name__ == "__main__":
    sptm = SPTM()
    sptm.append(1)
    sptm.append(2)
    sptm.append(3)
    sptm.append(4)
    sptm.build_graph()
    shortest_path = sptm.find_shortest_path(1, 3)
    print (shortest_path)
