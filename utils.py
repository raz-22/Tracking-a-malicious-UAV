import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

def update_graph(num, data, tail,dot):
    if num >= 100:
        tail.set_data(data[num-50:num-1, 0], data[num-50:num-1, 1])
        tail.set_3d_properties(data[num-50:num-1, 2])
    elif num >= 1:
        tail.set_data(data[:num-1, 0], data[:num-1, 1])
        tail.set_3d_properties(data[:num-1, 2])

    dot.set_data(data[num, 0], data[num, 1])
    dot.set_3d_properties(data[num, 2])
    dot.set_color('r')

    tail.set_color('b')

    return tail,dot
