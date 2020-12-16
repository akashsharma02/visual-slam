import pytest
import os, sys
from multiprocessing import Queue
from vslam.visualizer import InteractiveViz

def test_visualizer_initialization(caplog):
    queue = Queue()
    viz = InteractiveViz(queue)
    assert "Initialized Interactive Visualizer" not in caplog.text

def test_visualizer_start(caplog):
    queue = Queue()
    viz = InteractiveViz(queue)
    viz.start()
    assert "Started Interactive Visualizer" not in caplog.text

