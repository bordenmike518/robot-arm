import math
import numpy as np


class Robot_Arm:
    
    def __init__(self):
        self.P = np.array([0.0, 0.0, 0.3])          # Position Vector
        self.R = np.array([0.0, 0.0, 0.0])          # Rotation Vector (Orientation)
        self.d = np.array([0.050, 0.100, 0.100, 0.050])   # Arm link lengths in meters
        self.I = np.matrix([[1, 0, 0],           # Identity matrix
                            [0, 1, 0],
                            [0, 0, 1]])
        self.θ = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        self.k = np.ones(3, dtype=np.int32)
    
    def forward(self, θ):
        T_list = []

        #       θ        α      a       d
        DH = [[θ[0], -np.pi/2., 0, self.d[0]],
              [θ[1],  np.pi/2., 0,         0],
              [θ[2],  np.pi/2., 0, self.d[1]],
              [θ[3], -np.pi/2., 0,         0],
              [θ[4], -np.pi/2., 0, self.d[2]],
              [θ[5],  np.pi/2., 0,         0],
              [θ[6],         0, 0, self.d[3]]]

        for i in range(7):
            TM = np.array([[np.cos(DH[i][0]), -np.sin(DH[i][0])*np.cos(DH[i][1]),  np.sin(DH[i][0])*np.sin(DH[i][1]), DH[i][2]*np.cos(DH[i][0])],
                           [np.sin(DH[i][0]),  np.cos(DH[i][0])*np.cos(DH[i][1]), -np.cos(DH[i][0])*np.sin(DH[i][1]), DH[i][2]*np.sin(DH[i][0])],
                           [               0,                   np.sin(DH[i][1]),                   np.cos(DH[i][1]),                  DH[i][3]],
                           [               0,                                  0,                                  0,                         1]])
            if (i == 0):
                buff = TM
            else:
                buff = np.dot(buff, TM)
            T_list.append(buff)

        return np.array(T_list)
        
    def inverse(self, P07, R07):
        P02 = np.array([0.0, 0.0, self.d[0]]).reshape(3,)
        P24 = np.array([0.0, self.d[1], 0.0]).reshape(3,)
        P46 = np.array([0.0, 0.0, self.d[2]]).reshape(3,)
        P67 = np.array([0.0, 0.0, self.d[3]]).reshape(3,)
        θ_virt = np.zeros(7, dtype=np.float32)
        θ_virt[2] = 0.0 # Redundant, but helps illistrate the idea
        P26 = P07 - P02 - np.dot(R07, P67)
        print(' P26 = \n{}'.format(P26))
        P26_mag = self.magnitude(P26)
        print(' P26_mag = \n{}'.format(P26_mag))
        θ_virt[3] = self.GC(4) * math.acos((P26_mag**2 - self.d[1]**2 - self.d[2]**2) / (2 * self.d[1] * self.d[2]))
        print(' θ_virt[3] = \n{}'.format(θ_virt[3]))
        R01 = np.array([-np.sin(self.θ[0]), np.cos(self.θ[0]), 0.]).reshape(3,)
        if (self.magnitude(np.cross(P26, R01)) > 0):
            θ_virt[0] = np.arctan2(P26[1], P26[0])
        else:
            θ_virt[0] = 0.0
        print(' θ_virt[0] = \n{}'.format(θ_virt[0]))
        φ = np.arccos((self.d[1]**2 + P26_mag**2 - self.d[2]**2) / (2 * self.d[1] * P26_mag))
        print(' φ = {}'.format(φ))
        θ_virt[1] = np.arctan2(np.sqrt(P26[0]**2 + P26[1]**2), P26[2]) + self.GC(4) * φ
        print(' θ_virt[1] = \n{}'.format(θ_virt[1]))
        print(' θ_virt = \n{}'.format(θ_virt))
        P_virt_0 = self.forward(θ_virt)[:,:3,3]
        print(' P_virt_0 = \n{}'.format(P_virt_0))
        v_virt_sew = np.cross(self.cvd((P_virt_0[3].T - P_virt_0[1].T), self.magnitude(P_virt_0[3].T - P_virt_0[1].T)), self.cvd((P_virt_0[5].T - P_virt_0[1].T), self.magnitude(P_virt_0[5].T - P_virt_0[1].T)))
        print(' v_virt_sew = \n{}'.format(v_virt_sew))
        if (self.magnitude(v_virt_sew) == 0):
            v_virt_sew_hat = [0., 0., 0.]
        else:
            v_virt_sew_hat = v_virt_sew / self.magnitude(v_virt_sew)
        print(' v_virt_sew_hat = \n{}'.format(v_virt_sew_hat))
        P_0 = self.forward(self.θ)[:,:3,3]
        print(' P_0 = \n{}'.format(P_0))
        v_sew = np.cross(self.cvd((P_0[3].T - P_0[1].T), self.magnitude(P_0[3].T - P_0[1].T)), self.cvd((P_0[5].T - P_0[1].T), self.magnitude(P_0[5].T - P_0[1].T)))
        print(' v_sew = \n{}'.format(v_sew))
        if (self.magnitude(v_sew) == 0):
            v_sew_hat = [0., 0., 0.]
        else:
            v_sew_hat = v_sew / self.magnitude(v_sew)
        print(' v_sew_hat = \n{}'.format(v_sew_hat))
        sgψ = np.sign(np.dot(np.cross(v_virt_sew_hat, v_sew_hat), P26))
        print(' sgψ = \n{}'.format(sgψ))
        ψ = sgψ * math.acos(np.dot(v_virt_sew_hat, v_sew_hat))
        print(' ψ = \n{}'.format(ψ))
        I = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
        print(' I = \n{}'.format(I))
        P26_hat = P26 / P26_mag
        print(' P26_hat = \n{}'.format(P26_hat))
        P26_hat_cross = np.array([[        0.0, -P26_hat[2],  P26_hat[1]],
                                  [ P26_hat[2],         0.0, -P26_hat[0]],
                                  [-P26_hat[1],  P26_hat[0],         0.0]])
        print(' P26_hat_cross = \n{}'.format(P26_hat_cross))
        R_virt = self.forward(θ_virt)[:,:3,:3]
        print(' R_virt = \n{}'.format(R_virt))
        R0ψ = I + np.sin(ψ) * P26_hat_cross + (1 - np.cos(ψ)) * P26_hat_cross**2
        print(' R0ψ = \n{}'.format(R0ψ))
        As =  np.dot(P26_hat_cross, R_virt[2])
        print(' As = \n{}'.format(As))
        Bs = np.dot(-P26_hat_cross**2, R_virt[2])
        print(' Bs = \n{}'.format(Bs))
        Cs =  np.dot(np.dot(P26_hat, P26_hat.T), R_virt[2])
        print(' Cs = \n{}'.format(Cs))
        self.θ[0] = np.arctan2(self.GC(2)*(As[1,1] * np.sin(ψ) + Bs[1,1] * np.cos(ψ) + Cs[1,1]), 
                               self.GC(2)*(As[0,1] * np.sin(ψ) + Bs[0,1] * np.cos(ψ) + Cs[0,1]))
        print(' self.θ[0] = \n{}'.format(self.θ[0]))
        print('\n\n')
        print(As[2,1] * np.sin(ψ))
        print(Bs[2,1] * np.cos(ψ))
        print(Cs[2,1])
        print(As[2,1] * np.sin(ψ) + Bs[2,1] * np.cos(ψ) + Cs[2,1])
        print('\n\n')
        self.θ[1] = self.GC(2) * np.arccos(As[2,1] * np.sin(ψ) + Bs[2,1] * np.cos(ψ) + Cs[2,1])
        print(' self.θ[1] = \n{}'.format(self.θ[1]))
        self.θ[2] = np.arctan2(self.GC(2)*(-As[2,2] * np.sin(ψ) - Bs[2,2] * np.cos(ψ) - Cs[2,2]), 
                               self.GC(2)*(-As[2,0] * np.sin(ψ) - Bs[2,0] * np.cos(ψ) - Cs[2,0]))
        print(' self.θ[2] = \n{}'.format(self.θ[2]))
        self.θ[3] = θ_virt[3]
        print(' self.θ[3] = \n{}'.format(self.θ[3]))
        R34 = np.matrix([[np.cos(self.θ[3]), 0,  np.sin(self.θ[3])],
                         [np.sin(self.θ[3]), 0, -np.cos(self.θ[3])],
                         [                0, 1,                  0]])
        print(' R34 = \n{}'.format(R34))
        print(' R07 shape = {}'.format(R07.shape))
        Aw = R34.T * As.T * R07
        print(' Aw = \n{}'.format(Aw))
        Bw = R34.T * Bs.T * R07
        print(' Bw = \n{}'.format(Bw))
        Cw = R34.T * Cs.T * R07
        print(' Cw = \n{}'.format(Cw))
        self.θ[4] = np.arctan2(self.GC(6)*(Aw[1,2] * np.sin(ψ) + Bw[1,2] * np.cos(ψ) + Cw[1,2]), 
                               self.GC(6)*(Aw[0,2] * np.sin(ψ) + Bw[0,2] * np.cos(ψ) + Cw[0,2]))
        print(' self.θ[4] = \n{}'.format(self.θ[4]))
        self.θ[5] = self.GC(6) * np.arccos(Aw[2,2] * np.sin(ψ) + Bw[2,2] * np.cos(ψ) + Cw[2,2])
        print(' self.θ[5] = \n{}'.format(self.θ[5]))
        self.θ[6] = np.arctan2(self.GC(6)*(Aw[2,1] * np.sin(ψ) + Bw[2,1] * np.cos(ψ) + Cw[2,1]), 
                               self.GC(6)*(-Aw[2,0] * np.sin(ψ) - Bw[2,0] * np.cos(ψ) - Cw[2,0]))
        print(' self.θ[6] = \n{}'.format(self.θ[6]))
        return self.θ
    
    def magnitude(self, L):
        return np.sqrt(L.dot(L))
    
    # TODO: Figure out what this is for
    def GC(self, k):
        if (k == 2):
            return self.k[0]
        if (k == 4):
            return self.k[1]
        if (k == 6):
            return self.k[2]

    def cvd(self, X, y):
        Z = np.zeros(3)
        for i, x in enumerate(X):
            if (y == 0):
                Z[i] = 0.
            else:
                Z[i] = x / y
        return Z.reshape(3,)
    
    def get_xyz(self, t_list):
        X = np.zeros(8, dtype=np.int32)
        Y = np.zeros(8, dtype=np.int32)
        Z = np.zeros(8, dtype=np.int32)
        for i in range(7):
            X[i+1] = int(t_list[i,0,3]*1000)
            Y[i+1] = int(t_list[i,1,3]*1000)
            Z[i+1] = int(t_list[i,2,3]*1000)
        return X, Y, Z

