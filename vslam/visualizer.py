import os
import time
import logging
from multiprocessing import Process, Queue

import vtk
import numpy as np


def createCameraPolyData(R: np.ndarray, t: np.ndarray, only_polys=False) -> vtk.vtkPolyData:
    """Create polygon wireframe for camera given a camera pose

    Args:
        R: rotation matrix numpy array of shape (3, 3)
        t: translation vector numpy array of shape (3, 1)
    :returns:
        vtkPolyData

    """
    camera_points = np.array([[0,   0,   0],
                             [-1,  -1, 1.5],
                             [ 1,  -1, 1.5],
                             [ 1,   1, 1.5],
                             [-1,   1, 1.5],
                             [-0.5, 1, 1.5],
                             [ 0.5, 1, 1.5],
                             [ 0, 1.2, 1.5],
                             [ 1,-0.5, 1.5],
                             [ 1, 0.5, 1.5],
                             [ 1.2, 0, 1.5]])

    # camera_points = (0.05*camera_points - t).dot(R)
    camera_points = R.dot(camera_points.T).T + t

    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(camera_points.shape[0])
    for i in range(camera_points.shape[0]):
        vpoints.SetPoint(i, camera_points[i])

    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    poly_cells = vtk.vtkCellArray()

    if not only_polys:
        line_cells = vtk.vtkCellArray()

        line_cells.InsertNextCell( 5 );
        line_cells.InsertCellPoint( 1 );
        line_cells.InsertCellPoint( 2 );
        line_cells.InsertCellPoint( 3 );
        line_cells.InsertCellPoint( 4 );
        line_cells.InsertCellPoint( 1 );

        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 1 );
        line_cells.InsertCellPoint( 0 );
        line_cells.InsertCellPoint( 2 );

        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 3 );
        line_cells.InsertCellPoint( 0 );
        line_cells.InsertCellPoint( 4 );

        # x-axis indicator
        line_cells.InsertNextCell( 3 );
        line_cells.InsertCellPoint( 8 );
        line_cells.InsertCellPoint( 10 );
        line_cells.InsertCellPoint( 9 );
        vpoly.SetLines(line_cells)
    else:
        # left
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 1 );
        poly_cells.InsertCellPoint( 4 );

        # right
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 3 );
        poly_cells.InsertCellPoint( 2 );

        # top
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 4 );
        poly_cells.InsertCellPoint( 3 );

        # bottom
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 0 );
        poly_cells.InsertCellPoint( 2 );
        poly_cells.InsertCellPoint( 1 );

        # x-axis indicator
        poly_cells.InsertNextCell( 3 );
        poly_cells.InsertCellPoint( 8 );
        poly_cells.InsertCellPoint( 10 );
        poly_cells.InsertCellPoint( 9 );

    # up vector (y-axis)
    poly_cells.InsertNextCell( 3 );
    poly_cells.InsertCellPoint( 5 );
    poly_cells.InsertCellPoint( 6 );
    poly_cells.InsertCellPoint( 7 );

    vpoly.SetPolys(poly_cells)
    return vpoly

def createCameraActor(R: np.ndarray, t: np.ndarray) -> vtk.vtkActor:
    """TODO: Docstring for createCameraActor.

    :function: TODO
    :returns: TODO

    """
    vtk_camera_poly_data = createCameraPolyData(R, t)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vtk_camera_poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetLineWidth(2)

    return actor


def createPointcloudPolyData(points: np.ndarray, colors: np.ndarray=None) ->  vtk.vtkPolyData:
    """Creates VTK polygon data for pointcloud (3D numpy array)

    Args:
        points: 3D numpy array point cloud with shape (n, 3)
        colors: optional 3D numpy array with shae (n, 3) with dtype=uint8
    :returns:
        vtkPolyData
    """
    vpoints = vtk.vtkPoints()
    vpoints.SetNumberOfPoints(points.shape[0])
    for i in range(points.shape[0]):
        vpoints.SetPoint(i, points[i])
    vpoly = vtk.vtkPolyData()
    vpoly.SetPoints(vpoints)

    if colors is not None:
        vcolors = vtk.vtkUnsignedCharArray()
        vcolors.SetNumberOfComponents(3)
        vcolors.SetName("Colors")
        vcolors.SetNumberOfTuples(points.shape[0])
        for i in range(points.shape[0]):
            vcolors.SetTuple3(i, colors[i, 0], colors[i, 1], colors[i, 2])
        vpoly.GetPointData().SetScalars(vcolors)

    vcells = vtk.vtkCellArray()

    for i in range(points.shape[0]):
        vcells.InsertNextCell(1)
        vcells.InsertCellPoint(i)

    vpoly.SetVerts(vcells)

    return vpoly

def createPointcloudActor(points: np.ndarray, colors: np.ndarray=None) -> vtk.vtkActor:
    """Creates a VTK actor for point cloud object

    Args:
        points: pointcloud with shape (n, 3)
        colors: optional colors with shape (n, 3). Expects ndarray dtype = uint8
    Returns:
        VTK actor
    """
    vtk_pointcloud_poly_data = createPointcloudPolyData(points, colors)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(vtk_pointcloud_poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)

    return actor



class InteractiveViz(Process):

    """Interactive Visualizer built on top of VTK"""

    def __init__(self, data_queue: Queue, render_path: str=None) -> None:
        """Initialize Interactive Visualizer

        Args:
            data_queue: Input data queue to be rendered
        Returns:
            None
        """
        super(InteractiveViz, self).__init__()
        self.queue = data_queue
        self.render_path = render_path

        logging.info("Initialized Interactive Visualizer")

    def run(self) -> None:
        """Loop to handle events using a vtkTimerCallback

        Args:
            None
        Returns:
            None
        """
        logging.debug("Started Interactive Visualization")

        renderer = vtk.vtkRenderer()
        # Set renderer background color to black
        renderer.SetBackground(0, 0, 0)

        camera = vtk.vtkCamera()
        camera.SetPosition((1, 0, -3));
        camera.SetViewUp((0, -1, 0));
        camera.SetFocalPoint((0, 0, 1));
        renderer.SetActiveCamera(camera)

        render_window = vtk.vtkRenderWindow()
        render_window.SetWindowName("Visual SLAM visualizer")
        render_window.SetSize(1600, 900)
        render_window.AddRenderer(renderer)

        interactor = vtk.vtkRenderWindowInteractor()
        interact_style = vtk.vtkInteractorStyleTrackballCamera()
        interactor.SetInteractorStyle(interact_style)
        interactor.SetRenderWindow(render_window)

        interactor.Initialize()

        callback = vtkTimerCallback(self.render_path)
        callback.queue = self.queue

        interactor.AddObserver("TimerEvent", callback.execute)
        # 20 FPS rendering using timer
        timer_id = interactor.CreateRepeatingTimer(50)

        logging.info("Started Interactive Visualizer")
        interactor.Start()


class vtkTimerCallback(object):

    """VTK Timer Callback, converts objects to vtk geometry objects and prepares to render on window on callback execute"""

    def __init__(self, render_path: str=None) -> None:
        """ Initialize VTK Timer callback.

        :Docstring for vtkTimerCallback.: TODO

        """
        self.timer_count = 0
        self.render_path = render_path

        self.pointcloud_actor = None
        self.camera_position = None
        self.focal_point = None

        # Used to blend for cinematic visualization
        self.alpha = None

    def execute(self, caller_object: InteractiveViz, event: vtk.vtkEvent):
        """ Executes callback one time on the timerEvent being observed

        Args:
            caller_object: InteractiveViz caller object
            event: Contains information about the event
        returns:
            None

        """
        while not self.queue.empty():
            render_window = caller_object.GetRenderWindow()
            renderer = render_window.GetRenderers().GetFirstRenderer()

            # Raise exception if queue is empty
            pointcloud, pose = self.queue.get(False)
            logging.debug("Read pointcloud and pose from callback queue")

            if pointcloud is not None:
                points, colors = pointcloud[0], pointcloud[1]
                pointcloud_actor = createPointcloudActor(points, colors)
                renderer.AddActor(pointcloud_actor)
                self.pointcloud_actor = pointcloud_actor

            if pose is not None:
                R, t = pose[:3, :3], pose[:3, 3]
                camera_actor = createCameraActor(R, t)
                camera_actor.GetProperty().SetColor((0, 255, 0))
                renderer.AddActor(camera_actor)

            render_window.Render()

            self.timer_count += 1
        logging.debug("Finished Visualizing")
