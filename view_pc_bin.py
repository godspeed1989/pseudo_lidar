import os
import fire
import numpy as np

from OpenGL.GL import glLineWidth
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl

class plot3d(object):
    def __init__(self, title='null'):
        #
        self.glview = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        glLineWidth(1)
        coord.setSize(1,1,1)
        #self.glview.addItem(coord)
        self.glview.setMinimumSize(QtCore.QSize(600, 500))
        self.glview.pan(1, 0, 0)
        self.glview.setCameraPosition(azimuth=180)
        self.glview.setCameraPosition(elevation=0)
        self.glview.setCameraPosition(distance=5)
        self.items = []
        #
        self.view = QtGui.QWidget()
        self.view.window().setWindowTitle(title)
        hlayout = QtGui.QHBoxLayout()
        snap_btn = QtGui.QPushButton('&Snap')
        def take_snap():
            qimg = self.glview.readQImage()
            qimg.save('1.jpg')
        snap_btn.clicked.connect(take_snap)
        hlayout.addWidget(snap_btn)
        hlayout.addStretch()
        layout = QtGui.QVBoxLayout()
        #
        layout.addLayout(hlayout)
        layout.addWidget(self.glview)
        self.view.setLayout(layout)
    def add_item(self, item):
        self.glview.addItem(item)
        self.items.append(item)
    def clear(self):
        for it in self.items:
            self.glview.removeItem(it)
        self.items.clear()
    def add_points(self, points, colors):
        points_item = gl.GLScatterPlotItem(pos=points, size=1.5, color=colors)
        self.add_item(points_item)
    def add_line(self, p1, p2, color, width=3):
        lines = np.array([[p1[0], p1[1], p1[2]],
                          [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=color, width=width, antialias=True)
        self.add_item(lines_item)
    def plot_bbox_mesh(self, gt_boxes3d, color=(0,1,0,1)):
        b = gt_boxes3d
        for k in range(0,4):
            i,j=k,(k+1)%4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)
            i,j=k+4,(k+1)%4 + 4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)
            i,j=k,k+4
            self.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]], color)

def value_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = (pc_inte - minimum + 0.1) / (maximum - minimum + 0.1)
    r = (np.maximum((1 - ratio), 0))
    b = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b]).transpose()

def view_points_cloud(pc=None):
    app = QtGui.QApplication([])
    glview = plot3d()
    if pc is None:
        pc = np.random.rand(1024, 3)
    pc_color = np.ones([pc.shape[0], 4])
    glview.add_points(pc, pc_color)
    glview.view.show()
    return app.exec()

def view_pc_qt(pc, boxes3d=None):
    app = QtGui.QApplication([])
    glview = plot3d()
    if pc.shape[1] == 3:
        points = pc[:,:3]
        pc_color = np.ones([pc.shape[0], 4])
    elif pc.shape[1] == 4:
        points = pc[:,:3]
        pc_color = value_to_rgb(pc[:,3])

    if boxes3d is not None:
        for box3d in boxes3d:
            glview.plot_bbox_mesh(box3d)
    glview.add_points(points, pc_color)
    glview.view.show()
    return app.exec()

def view(fpath):
    scan = np.fromfile(fpath, dtype=np.float32).reshape(-1, 4)
    print(scan.shape)
    view_pc_qt(scan)

if __name__ == '__main__':
    fire.Fire()
