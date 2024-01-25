import numpy as np


class KalmanFilter():
    """
    The class will be initialized with six parameters:
         dt : time for one cycle used to estimate state (sampling time)
         u_x, u_y : accelerations in the x-, and y-directions respectively
         std_acc: process noise magnitude
         x_sdt_meas, y_sdt_meas :standard deviations of the measurement in the x- and y-directions
        respectively
    """
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        dt2 = 1/2 * dt**2
        dt3 = 1/2 * dt**3
        dt4 = 1/4 * dt**4


        self.u = np.array([u_x, u_y])
        self.x = np.array([[0, 0, 0, 0]]).T
        self.A = np.identity(4) + np.diag([dt, dt], k=2)
        self.B = np.array([[dt2, 0], [0, dt2], [dt, 0], [0, dt]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = (np.diag([dt4, dt4, dt2*2, dt2*2]) + np.diag([dt3, dt3], k=2) \
                   + np.diag([dt3, dt3], k=-2)) * std_acc**2
        
        self.R = np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])
        self.P = np.identity(4)

    def predict(self):
        """
            Predict of the state estimate 𝑥 and the error prediction 𝑃.
            This task also call the time update process (u) because it projects forward
            the current state to the next time step.
        """
        self.x = (self.A @ self.x) + (self.B @ self.u.T).reshape(-1, 1)
        self.P = self.A @ self.P @ self.A.T + self.Q

        return self.x

    def update(self, z):
        """
            Takes measurements 𝑧 as input (centroid coordinates x,y of detected circles)
            and Update the predicted state estimate x_k 
        """
        if z.shape == (2,):
            z = z.reshape(-1, 1)

        # Compute Kalman Gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update the estimate via z
        self.x = self.x + K @ (z - self.H @ self.x)

        # Predict the error covariance
        self.P = (np.identity(4) - K @ self.H) @ self.P

        return self.x.T  