import copy
import numpy


class Container:
    def __init__(self, list_3d):
        # 3D list which contains values in proper format
        self.__values = list()

        self.__fill_values(list_3d)
        self.__generate_parsing_buffers()

    def __fill_values(self, source_3d):
        self.__values = copy.deepcopy(source_3d)

    @classmethod
    def load_data(cls, array, number_nodes):
        items_per_node = int(len(array) / number_nodes)
        items_left = len(array) % number_nodes
        items_sizes = numpy.zeros(number_nodes, dtype=numpy.int32)
        items_sizes.fill(items_per_node)

        for i in range(items_left):
            items_sizes[i] += 1

        index_for_deletion = list()
        for i in range(items_sizes.size):
            if items_sizes[i] == 0:
                index_for_deletion.append(i)

        items_sizes = numpy.delete(items_sizes, index_for_deletion)

        values = list()
        start = 0
        for i in range(items_sizes.size):
            node = list()
            for j in range(items_sizes[i]):
                node.append(array[start])
                start += 1
            values.append(node)

        return cls(values)

    def __generate_parsing_buffers(self):
        # 1D list containing heights for values list
        self.__heights = [len(node) for node in self.__values]
        # 2D list containing widths for values list
        self.__widths = [[len(item_set) for item_set in node] for node in self.__values]
        # 1D list containing start for each node in __widths/__start_of_item_sets
        self.__start_2d_parsing_buffer = [0] * len(self.__widths)
        # 1D list containing start position for values per node
        self.__start_values_per_node = [0] * len(self.__values)
        # 1D list containing end position for values per node
        self.__end_values_per_node = [0] * len(self.__values)
        # 2D list containing start index for each item_set in __values
        self.__start_of_item_sets = [0] * len(self.__values)

        # generate start for widths
        start_poz = 0
        for node_idx in range(len(self.__widths)):
            if node_idx == 0:
                self.__start_2d_parsing_buffer[node_idx] = 0
            else:
                start_poz += len(self.__widths[node_idx - 1])
                self.__start_2d_parsing_buffer[node_idx] = start_poz

        # generate start for _values per node and starts of item_sets
        start_poz = 0
        for node_idx in range(len(self.__values)):
            self.__start_values_per_node[node_idx] = start_poz
            aux_starts_item_sets = [0] * len(self.__values[node_idx])
            for item_set_idx in range(len(self.values[node_idx])):
                aux_starts_item_sets[item_set_idx] = start_poz
                start_poz += len(self.values[node_idx][item_set_idx])
            self.__end_values_per_node[node_idx] = start_poz
            self.__start_of_item_sets[node_idx] = aux_starts_item_sets

    # generates a new container having 0 values using estimations based on parsing buffers
    def generate_write_container(self, items_heights, items_widths):
        number_nodes = len(items_heights)
        zeros_list = [None] * (number_nodes - 1)
        for i in range(number_nodes - 1):
            left = i+1
            # print('left', left)
            node = [None] * (len(self.__widths[left]) + items_heights[i]*len(self.__widths[left]))
            idx = 0
            for width in self.__widths[left]:
                node[idx] = [0] * width
                idx += 1

            for j in range(items_heights[i]):
                for width in self.__widths[left]:
                    node[idx] = [0] * (width + items_widths[i][j])
                    idx += 1
            zeros_list[i] = node
        return Container(zeros_list)

    # fill __values using it's synonymous 1d buffer
    def fill_from_1d_synonymous(self, array_1d):
        for idx_node in range(len(self.__start_of_item_sets)):
            for idx_i in range(len(self.__start_of_item_sets[idx_node])):
                start = self.__start_of_item_sets[idx_node][idx_i]
                width = self.__widths[idx_node][idx_i]
                for i in range(width):
                    self.__values[idx_node][idx_i][i] = array_1d[start + i]
        # clean for 0's
        self.clean_container()

    # remove all arrays filled with zero value
    def clean_container(self):
        new_list = [[item_set for item_set in node if not all(val == 0 for val in item_set)] for node in self.__values]

        self.__values = new_list

        self.__generate_parsing_buffers()

    # remove last node
    def remove_last_node(self):
        del self.__values[-1]
        self.__generate_parsing_buffers()

    @property
    def values(self):
        return self.__values

    @property
    def heights(self):
        return self.__heights

    @property
    def widths(self):
        return self.__widths

    @property
    def start_2d_parsing_buffer(self):
        return self.__start_2d_parsing_buffer

    @property
    def start_values_per_node(self):
        return self.__start_values_per_node

    @property
    def start_of_item_sets(self):
        return self.__start_of_item_sets

    @property
    def end_values_per_node(self):
        return self.__end_values_per_node


# if __name__ == '__main__':
#     l3d = numpy.array([32, 38, 39, 41, 48], dtype=numpy.int32)
#     items_heights = [2, 1, 1]
#     items_widths = [[1, 1], [1], [1]]
#     c = Container.load_1d_array(l3d, 256)
#     print(c.values)
#     print('heights', c.heights)
#     print('widths', c.widths)
#     print('start 2d parsing buffer', c.start_2d_parsing_buffer)
#     print('start index for values per node', c.start_values_per_node)
#     print('start of item sets', c.start_of_item_sets)
#     cw = c.generate_write_container(c.heights, c.widths)
#     print("generated writer", cw.values)
#     print("generated writer heights", cw.heights)
#     print("generated writer widths", cw.widths)
#     print("generated start2d parsing buffer writer", cw.start_2d_parsing_buffer)
#     print('generated start index for value per node writer', cw.start_values_per_node)
#     print('generated start of item sets', cw.start_of_item_sets)
#     array_1d = numpy.array([1, 2, 3, 0, 0, 6, 7, 0, 0, 0, 1, 2], dtype=numpy.int32)
#     cw.fill_from_1d_synonymous(array_1d)
#     print("generated writer", cw.values)
#     cw.clean_container()
#     print("generated writer", cw.values)
