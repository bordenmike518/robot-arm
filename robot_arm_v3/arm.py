import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Robot_Arm:
    
    def __init__(self):
        self.P = np.array([0.0, 0.0, 0.3])          # Position Vector
        self.R = np.array([0.0, 0.0, 0.0])          # Rotation Vector (Orientation)
        self.l = np.array([0.050, 0.100, 0.100, 0.050])   # Arm link lengths in meters
        self.I = np.matrix([[1, 0, 0],           # Identity matrix
                            [0, 1, 0],
                            [0, 0, 1]])
        self.θ = np.zeros(7)

    def forward(self, θ):
        T_list = [0,0,0,0,0,0,0]

        DH = [[θ[0], -np.pi/2., 0, self.l[0]],
              [θ[1],  np.pi/2., 0,         0],
              [θ[2], -np.pi/2., 0, self.l[1]],
              [θ[3],  np.pi/2., 0,         0],
              [θ[4], -np.pi/2., 0, self.l[2]],
              [θ[5],  np.pi/2., 0,         0],
              [θ[6],         0, 0, self.l[3]]]
        
        buff = np.matrix([[1, 0, 0, 0],           # Identity matrix
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])

        for i in range(7):  
            next = [[np.cos(DH[i][0]), -np.sin(DH[i][0])*np.cos(DH[i][1]),  np.sin(DH[i][0])*np.sin(DH[i][1]), DH[i][2]*np.cos(DH[i][0])],
                    [np.sin(DH[i][0]),  np.cos(DH[i][0])*np.cos(DH[i][1]), -np.cos(DH[i][0])*np.sin(DH[i][1]), DH[i][2]*np.sin(DH[i][0])],
                    [               0,                   np.sin(DH[i][1]),                   np.cos(DH[i][1]),                  DH[i][3]],
                    [               0,                                  0,                                  0,                         1]]
            T_list[i] = np.dot(buff, next)
            buff = next

        return np.array(T_list)

    def get_xyz(self, t_list):
        X = np.zeros(8, dtype=np.int32)
        Y = np.zeros(8, dtype=np.int32)
        Z = np.zeros(8, dtype=np.int32)
        for i in range(7):
            X[i+1] = int(t_list[i,0,3]*1000)
            Y[i+1] = int(t_list[i,1,3]*1000)
            Z[i+1] = int(t_list[i,2,3]*1000)
        return X, Y, Z

    def magnitude(self, L):
        return np.sqrt(L.dot(L))

    def inverse(self, P, R, ψ):
        w = P - (self.l[3] * R)
        Lbs = self.l[0] * np.array([0.0, 0.0, 1.0])
        Lsw = w - Lbs
        Lsw_mag = self.magnitude(Lsw)
        cos_θ_prime_4 = (self.l[1]**2 + self.l[2]**2 - Lsw_mag**2) / (2 * self.l[1] * self.l[2])
        self.θ[3] = np.pi - math.acos(cos_θ_prime_4)
        if ((60/180)*np.pi > self.θ[3] or self.θ[3] > (300/180)*np.pi):
            print("ERROR :: The given pose is outside the robot worknpace.")
        else:
            Usw = Lsw / Lsw_mag
            Ksw = np.array([[    0.0, -Usw[2],  Usw[1]],
                            [ Usw[2],     0.0, -Usw[0]],
                            [-Usw[1],  Usw[0],     0.0]])
            Sψ = np.sin(ψ)
            Cψ = np.cos(ψ)
            Rψ = self.I + ((1 - Cψ) * Ksw**2) + Sψ * Ksw
            self.θ[2] = 0
            θ_prime_1 = np.arctan2(w[1], w[0])
            alpha = np.arcsin((w[2] - self.l[0]) / Lsw_mag)
            beta = math.acos((self.l[1]**2 + Lsw_mag**2 - self.l[2]**2) / (2 * self.l[1] * Lsw_mag))
            θ_prime_2 = (np.pi/2.0) - alpha - beta
            R_prime_03 = np.array([[np.cos(θ_prime_1)*np.cos(θ_prime_2), -np.cos(θ_prime_1)*np.sin(θ_prime_1), -np.sin(θ_prime_1)],
                                   [np.cos(θ_prime_2)*np.sin(θ_prime_1), -np.sin(θ_prime_1)*np.sin(θ_prime_2),  np.cos(θ_prime_1)],
                                   [                 -np.sin(θ_prime_2),                   -np.cos(θ_prime_2),               0.0]])
            Xs = Ksw * R_prime_03
            Ys = -(Ksw**2) * R_prime_03
            Zs = (self.I + Ksw**2) * R_prime_03
            self.θ[0] = np.arctan((-Sψ*Xs[1,1]-Cψ*Ys[1,1]-Zs[1,1])/(-Sψ*Xs[0,1]-Cψ*Ys[0,1]-Zs[0,1]))
            self.θ[1] = math.acos(-Sψ*Xs[2,1]-Cψ*Ys[2,1]-Zs[2,1])
            self.θ[2] = np.arctan((Sψ*Xs[2,2]+Cψ*Ys[2,2]+Zs[2,2])/(-Sψ*Xs[2,0]-Cψ*Ys[2,0]-Zs[2,0]))
            R_34 = np.array([[np.cos(self.θ[3]), 0.0, -np.sin(self.θ[3])],
                             [np.sin(self.θ[3]), 0.0, -np.cos(self.θ[3])],
                             [0.0,  1.0, 0.0]])
            Xw =  R_34.T * Ksw.T * R_prime_03.T
            Yw = -(R_34.T * (Ksw**2).T * R_prime_03.T)
            Zw =  R_34.T * (self.I + Ksw**2).T * R_prime_03.T
            self.θ[4] = np.arctan((Sψ*Xw[1,2]+Cψ*Yw[1,2]+Zw[1,2])/(Sψ*Xw[0,2]+Cψ*Yw[0,2]+Zw[0,2]))
            self.θ[5] = np.arcsin(Sψ*Xw[2,2]+Cψ*Yw[2,2]+Zw[2,2])
            self.θ[6] = np.arctan(-(Sψ*Xw[2,1]+Cψ*Yw[2,1]+Zw[2,1])/(Sψ*Xw[2,0]+Cψ*Yw[2,0]+Zw[2,0]))
            return self.θ
            
            
