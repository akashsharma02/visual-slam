import pytest
import os, sys
from multiprocessing import Queue
from vslam.visualizer import InteractiveViz
import numpy as np

def test_visualizer_initialization(caplog):
    queue = Queue()
    viz = InteractiveViz(queue)
    assert "Initialized Interactive Visualizer" not in caplog.text

def test_visualizer_start(caplog):
    queue = Queue()
    viz = InteractiveViz(queue)
    viz.start()
    viz.terminate()
    assert "Started Interactive Visualizer" not in caplog.text

def test_visualizer_straight_line(caplog):
    queue = Queue(1000)
    for i in range(1,50):
        pose = np.eye(4)
        pose[2,-1] = 5*i
        pose[0,-1] = 1
        points = np.random.normal(loc=pose[:3,-1], scale=10, size=(20,3))
        colors = np.zeros((20,3))
        colors[:,0] = 255
        point_cloud = [points, colors]
        queue.put([point_cloud, pose])
    viz = InteractiveViz(queue,os.getcwd())
    viz.start()

    assert "Finished Visualizing" not in caplog.text
