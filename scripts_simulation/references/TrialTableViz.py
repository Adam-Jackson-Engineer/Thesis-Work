import sys          # Accessing system-specific parameters and functions
import numpy as np  
from PyQt5.QtWidgets import QApplication, QMainWindow, QSlider, QVBoxLayout, QWidget, QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(1300, 300, 1400, 1400)  # Widget location/dimensions: x position, y position, width, height
        self.setWindowTitle('6 DoF Table Animation')

        self.widget = QWidget(self)  # Creating a central widget
        self.setCentralWidget(self.widget)  # Setting the central widget of the window

        self.layout = QVBoxLayout()  # Creating a vertical layout
        self.widget.setLayout(self.layout)  # Setting the layout of the central widget

        self.sliders = []  # Initializing an empty list for sliders
        for i in range(6):  # Looping to create 6 sliders for x, y, z, theta, phi, psi
            slider = QSlider(Qt.Horizontal)  # Creating a horizontal slider
            slider.setMinimum(-100)  # Setting minimum slider value
            slider.setMaximum(100)  # Setting maximum slider value
            slider.setValue(0)  # Setting the initial slider value
            slider.valueChanged[int].connect(self.updateGL)  # Connecting the slider value change signal to the updateGL function
            self.layout.addWidget(slider)  # Adding the slider to the layout
            self.sliders.append(slider)  # Adding the slider to the sliders list

        self.glWidget = GLWidget(self)  # Creating an instance of the GLWidget class
        self.layout.addWidget(self.glWidget)  # Adding the GLWidget to the layout

    def updateGL(self, value):  # Function to update the OpenGL widget when a slider value changes
        x, y, z, theta, phi, psi = [slider.value() / 10.0 for slider in self.sliders]  # Getting values from sliders and scaling them
        self.glWidget.updatePosition(x, y, z, theta, phi, psi)  # Updating the position and orientation in the GLWidget

class GLWidget(QOpenGLWidget):  # Defining the GLWidget class, inheriting from QOpenGLWidget for OpenGL rendering
    def __init__(self, parent):
        super(GLWidget, self).__init__(parent)  # Initializing the QOpenGLWidget class
        self.x = self.y = self.z = self.theta = self.phi = self.psi = 0  # Initializing position and orientation variables

    def initializeGL(self):  # Overridden method called once before the first call to paintGL() or resizeGL()
        glEnable(GL_DEPTH_TEST)  # Enabling depth testing for 3D rendering
        glClearColor(0, 0, 0, 1)  # Setting the color used to clear the color buffer, in this case, black

    def resizeGL(self, w, h):  # Overridden method called upon widget resize
        glMatrixMode(GL_PROJECTION)  # Setting the current matrix to the projection matrix
        glLoadIdentity()  # Resetting the projection matrix
        gluPerspective(45, w / h, 0.1, 100.0)  # Setting up a perspective projection matrix
        glMatrixMode(GL_MODELVIEW)  # Switching back to the modelview matrix

    def paintGL(self):  # Overridden method to handle OpenGL painting
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # Clearing the color and depth buffers
        glLoadIdentity()  # Resetting the current matrix
        glTranslatef(0.0, 0.0, -10)  # Moving the drawing origin 10 units into the screen
        glRotatef(self.theta, 1, 0, 0)  # Rotating around the x-axis by theta degrees
        glRotatef(self.phi, 0, 1, 0)  # Rotating around the y-axis by phi degrees
        glRotatef(self.psi, 0, 0, 1)  # Rotating around the z-axis by psi degrees
        glTranslatef(self.x, self.y, self.z)  # Translating the drawing origin to the position specified by sliders
        self.drawTable()  # Calling the method to draw the table

    def drawTable(self):  # Method to draw a 3D table
        glBegin(GL_QUADS)  # Beginning to draw quadrilaterals
        glColor3f(0.5, 0.35, 0.05)  # Setting the color to brown for the table

        # Specifying each vertex of the quads to form a rectangular prism (table)
        # Front face
        glVertex3f(-1, -1, 1)
        glVertex3f(1, -1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(-1, 1, 1)

        # Back face
        glVertex3f(-1, -1, -1)
        glVertex3f(-1, 1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, -1, -1)

        # Top face
        glVertex3f(-1, 1, -1)
        glVertex3f(-1, 1, 1)
        glVertex3f(1, 1, 1)
        glVertex3f(1, 1, -1)

        # Bottom face
        glVertex3f(-1, -1, -1)
        glVertex3f(1, -1, -1)
        glVertex3f(1, -1, 1)
        glVertex3f(-1, -1, 1)

        # Right face
        glVertex3f(1, -1, -1)
        glVertex3f(1, 1, -1)
        glVertex3f(1, 1, 1)
        glVertex3f(1, -1, 1)

        # Left face
        glVertex3f(-1, -1, -1)
        glVertex3f(-1, -1, 1)
        glVertex3f(-1, 1, 1)
        glVertex3f(-1, 1, -1)

        glEnd()  # Ending the drawing of quads

    def updatePosition(self, x, y, z, theta, phi, psi):  # Method to update the position and orientation
        self.x, self.y, self.z, self.theta, self.phi, self.psi = x, y, z, theta, phi, psi  # Setting the new values
        self.update()  # Scheduling a paint event for the widget

if __name__ == '__main__':  # Checking if this script is being run directly
    app = QApplication(sys.argv)  # Creating a QApplication object
    window = MainWindow()  # Creating an instance of MainWindow
    window.show()  # Showing the main window
    sys.exit(app.exec_())  # Starting the application's main loop and exiting with the return value
