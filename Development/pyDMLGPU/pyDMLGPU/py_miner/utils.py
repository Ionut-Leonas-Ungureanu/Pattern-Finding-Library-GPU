import pyopencl as cl
import numpy as np


def buffer_from_3d(context, source_3d):
    flat_array = make_array_from_3d(source_3d)
    return cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_array)


def make_array_from_3d(source_3d):
    flat_list = [element for node in source_3d for item_set in node for element in item_set]
    flat_array = np.array(flat_list, dtype=np.int32)
    return flat_array


def buffer_from_2d(context, source_2d):
    flat_list = [element for node in source_2d for element in node]
    flat_array = np.array(flat_list, dtype=np.int32)
    return cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_array)


def buffer_from_1d(context, source_1d):
    flat_array = np.array(source_1d, dtype=np.int32)
    return cl.Buffer(context, cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_array)


# if __name__ == '__main__':
#     platform = cl.get_platforms()[0]
#     device = platform.get_devices(cl.device_type.GPU)
#     print(device)
#     ctx = cl.Context(device)
#     print(ctx)
#
#     l3d = [[[1], [2]], [[3], [4], [5]]]
#     print('buffer from 3d', buffer_from_3d(ctx, l3d))
#
#     l2d = [[1, 1], [1, 1, 1]]
#     print('buffer from 2d', buffer_from_2d(ctx, l2d))
#
#     l1d = [2, 3]
#     print('buffer from 1d', buffer_from_1d(ctx, l1d))
