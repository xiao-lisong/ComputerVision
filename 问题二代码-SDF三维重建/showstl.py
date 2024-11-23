import pyvista as pv
import threading
import sys

if __name__ == "__main__":
    #file = "Thingi10K/raw_meshes/80363.stl"
    file = "output5.stl"
    mesh = pv.read(file)
    mesh.plot()