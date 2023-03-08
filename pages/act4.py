import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm 
from scipy.spatial import Delaunay

import tensorflow as tf
import streamlit as st

def _plt_basic_object_(points):
    """Plots a basic Object, assuming its convex and not too complex"""

    tri = Delaunay(points).convex_hull
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    S = ax.plot_trisurf(points[:,0], points[:,1], points[:,2],
                        triangles=tri,
                        shade=True, cmap = cm.rainbow ,lw=0.7)

    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-10, 10)

    plt.show()

def _cube_(bottom_lower=(0,0,0), side_length = 5):

    bottom_lower = np.array(bottom_lower)

    points = np.vstack([
        bottom_lower,
        bottom_lower + [0, side_length, 0],
        bottom_lower + [side_length, side_length, 0],
        bottom_lower + [side_length, 0, 0],
        bottom_lower + [0, 0, side_length],
        bottom_lower + [0, side_length, side_length],
        bottom_lower + [side_length, 0, side_length],
        bottom_lower + [side_length,0 ,side_length],
        bottom_lower + [side_length,0 ,side_length],
        bottom_lower, 
    ])

    return points


def translate_obj(points, amount):
    return tf.add(points, amount)

init_kite_ = _cube_(side_length=4)
points = tf.constant(init_kite_, dtype=tf.float32)


translation_amount = tf.constant([4, 5, 3], dtype=tf.float32)
translated_object = translate_obj(points, translation_amount)


def main():
    _plt_basic_object_(translated_object)

if __name__== '__main__':
    main()

