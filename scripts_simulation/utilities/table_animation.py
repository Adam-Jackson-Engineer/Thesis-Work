"""
table_animation.py

Handles the animation of the table in 3D.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QOpenGLWidget
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import utilities.table_parameters as tp

class TableAnimation(QMainWindow):
    """
    Main window class for table animation.
    """
    def __init__(self):
        super().__init__()
        self.setGeometry(1300, 300, 1400, 1400)  # Widget location/dimensions: x position, y position, width, height
        self.setWindowTitle('6 DoF Table Animation')

        self.widget = QWidget(self)
        self.setCentralWidget(self.widget) 

        self.layout = QVBoxLayout()
        self.widget.setLayout(self.layout) 

        self.gl_widget = GLWidget(self)
        self.layout.addWidget(self.gl_widget)

    def update_gl(self, states, reference_states=None, path_states=None, task=None):
        """
        Updates the OpenGL widget with new state information.

        Args:
            states (np.ndarray): Array containing the current state of the table.
            reference_states (np.ndarray or bool): Array containing reference states, if any.
            path_states (np.ndarray or bool): Array containing path states, if any.
            task (str): Task description to display in the widget.
        """
        self.gl_widget.update_position(states, reference_states, path_states, task)

class GLWidget(QOpenGLWidget):
    """
    OpenGL widget class for rendering the table.
    """
    def __init__(self, parent):
        super().__init__(parent)
        # Initialize table states
        self.x = tp.X_0
        self.y = tp.Y_0
        self.z = tp.Z_0
        self.phi = tp.PHI_0
        self.theta = tp.THETA_0
        self.psi = tp.PSI_0

        # Initialize reference states
        self.x_ref = tp.X_0
        self.y_ref = tp.Y_0
        self.z_ref = tp.Z_0
        self.phi_ref = tp.PHI_0
        self.theta_ref = tp.THETA_0
        self.psi_ref = tp.PSI_0
        self.reference = False

        # Initialize path states
        self.x_path = tp.X_0
        self.y_path = tp.Y_0
        self.z_path = tp.Z_0
        self.phi_path = tp.PHI_0
        self.theta_path = tp.THETA_0
        self.psi_path = tp.PSI_0
        self.path = False

        # Task description
        self.task = None
        self.print_task = True
        
    def initializeGL(self):
        """
        Initializes OpenGL settings.
        """
        glEnable(GL_DEPTH_TEST)
        glClearColor(0, 0, 0, 1)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        try:
            glutInit()
        except:
            print("GLUT not initialized corrent: Continuing without printing tasks")
            self.print_task = False

    def resizeGL(self, w, h):
        """
        Handles the resizing of the OpenGL widget.

        Args:
            w (int): Width of the widget.
            h (int): Height of the widget.
        """
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        """
        Renders the OpenGL scene.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        eye_distance = 7
        eye_height = 5
        gluLookAt(eye_distance * np.sqrt(2) / 2, eye_distance * np.sqrt(2) / 2, eye_height, 0, 0, 0, 0, 0, 1)

        self.draw_grid(grid_size_x=tp.ROOM_LENGTH, grid_size_y=tp.ROOM_WIDTH, step=0.5)
        self.draw_axes()

        glPushMatrix()
        glTranslatef(self.x, self.y, self.z)
        glRotatef(self.psi * 180 / np.pi, 0, 0, 1)
        glRotatef(self.theta * 180 / np.pi, 0, 1, 0)
        glRotatef(self.phi * 180 / np.pi, 1, 0, 0)
        self.draw_table(1.0)
        glPopMatrix()

        if self.reference:
            glPushMatrix()
            glTranslatef(self.x_ref, self.y_ref, self.z_ref)
            glRotatef(self.psi_ref * 180 / np.pi, 0, 0, 1)
            glRotatef(self.theta_ref * 180 / np.pi, 0, 1, 0)
            glRotatef(self.phi_ref * 180 / np.pi, 1, 0, 0)
            self.draw_table(0.5)
            glPopMatrix()

        if self.path:
            glPushMatrix()
            glTranslatef(self.x_path, self.y_path, self.z_path)
            glRotatef(self.psi_path * 180 / np.pi, 0, 0, 1)
            glRotatef(self.theta_path * 180 / np.pi, 0, 1, 0)
            glRotatef(self.phi_path * 180 / np.pi, 1, 0, 0)
            self.draw_table(0.25)
            glPopMatrix()

        if self.task and self.print_task:
            self.draw_task(self.task)

    def draw_axes(self):
        """
        Draws the reference axes.
        """
        glLineWidth(2.0)  # Set the line width to 2.0
        glBegin(GL_LINES)
        origin = [-3.0, -2.5, 0.0]
        
        # X axis (red)
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0] + 1.0, origin[1], origin[2])

        # Y axis (green)
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0], origin[1] + 1.0, origin[2])

        # Z axis (blue)
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(origin[0], origin[1], origin[2])
        glVertex3f(origin[0], origin[1], origin[2] + 1.0)

        glEnd()
        glLineWidth(1.0)  # Reset the line width to the default value


    def draw_table(self, opacity):
        """
        Draws a 3D table.

        Args:
            opacity (float): Opacity of the table.
        """
        glBegin(GL_QUADS)
        glColor4f(0.99, 0.77, 0.51, opacity)

        # Leader - Green
        glColor4f(0.0, 1.0, 0.0, opacity)
        glVertex3f(-tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)

        # Follower - Blue
        glColor4f(0.0, 0.0, 1.0, opacity)
        glVertex3f(tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)

        # Top face
        # glColor4f(0.99, 0.77, 0.51, opacity)
        glColor4f(0.99, 0., 0., opacity)
        glVertex3f(tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)

        # Bottom face
        glColor4f(0.44, 0.33, 0.20, opacity)
        glVertex3f(tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)

        # Right face
        glColor4f(0.99, 0.77, 0.51, opacity)
        glVertex3f(tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)

        # Left face
        glColor4f(0.99, 0.77, 0.51, opacity)
        glVertex3f(tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)
        glVertex3f(tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, -tp.TABLE_HEIGHT / 2)
        glVertex3f(-tp.TABLE_LENGTH / 2, -tp.TABLE_WIDTH / 2, tp.TABLE_HEIGHT / 2)

        glEnd()

    def draw_grid(self, grid_size_x=6, grid_size_y=5, step=0.5):
        """
        Draws a grid on the ground.

        Args:
            grid_size_x (float): Size of the grid in the X direction.
            grid_size_y (float): Size of the grid in the Y direction.
            step (float): Spacing between grid lines.
        """
        glColor3f(0.5, 0.5, 0.5)
        glBegin(GL_LINES)
        half_x = grid_size_x / 2
        half_y = grid_size_y / 2

        # Draw horizontal lines
        for x in np.arange(-half_x, half_x + step, step):
            glVertex3f(x, -half_y, 0)
            glVertex3f(x, half_y, 0)

        # Draw vertical lines
        for y in np.arange(-half_y, half_y + step, step):
            glVertex3f(-half_x, y, 0)
            glVertex3f(half_x, y, 0)

        glEnd()

    def draw_task(self, task):
        """
        Draws the task description in the top right corner of the OpenGL widget.

        Args:
            task (str): The task description to display.
        """

        # Set up for 2D overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width(), 0, self.height(), -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)

        glColor3f(1.0, 1.0, 1.0)  # Set text color to white

        # Position text in the bottom left corner
        x_offset = 100
        y_offset = 100
        glRasterPos2f(x_offset, y_offset)
        
        for char in task:
            glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, ord(char))

        # Restore settings
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def update_position(self, states, reference_states, path_states, task=None):
        """
        Updates the position and orientation of the table.

        Args:
            states (np.ndarray): Array containing the current state of the table.
            reference_states (np.ndarray or bool): Array containing reference states, if any.
        """
        # Modify the table states
        self.x = states[0]
        self.y = states[1]
        self.z = states[2]
        self.phi = states[3]
        self.theta = states[4]
        self.psi = states[5]

        # Modify the reference states
        if reference_states is not None:
            self.reference = True
            self.x_ref = reference_states[0]
            self.y_ref = reference_states[1]
            self.z_ref = reference_states[2]
            self.phi_ref = reference_states[3]
            self.theta_ref = reference_states[4]
            self.psi_ref = reference_states[5]

        # Modify the path states
        if path_states is not None:
            self.path = True
            self.x_path = path_states[0]
            self.y_path = path_states[1]
            self.z_path = path_states[2]
            self.phi_path = path_states[3]
            self.theta_path = path_states[4]
            self.psi_path = path_states[5]

        if task is not None:
            self.task = task

        # Update the visual
        self.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TableAnimation()
    window.show()
    sys.exit(app.exec_())  