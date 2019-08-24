import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from jacobian import Jacobian


class JacobianInverseKinematics:
    def __init__(self):
        self.joint_count = 7
        self.theta_list = np.zeros(self.joint_count)
        self.tms = list()

    def get_transformation_matrix(self, motor_num, theta):
        link_lengths = [50, 100, 100, 50]
        ninety = np.pi/2.
        one_80 = np.pi

        PT = [[ theta, ninety, 0, link_lengths[0]],
              [ theta,-ninety, 0,             0],
              [ theta, ninety, 0, link_lengths[1]],
              [ theta,-ninety, 0,             0],
              [ theta, ninety, 0, link_lengths[2]],
              [ theta,-ninety, 0,             0],
              [ theta,      0, 0, link_lengths[3]]]
              
        return [[np.cos(PT[motor_num][0]), -np.sin(PT[motor_num][0])*np.cos(PT[motor_num][1]), np.sin(PT[motor_num][0])*np.sin(PT[motor_num][1]), PT[motor_num][2]*np.cos(PT[motor_num][0])],
                [np.sin(PT[motor_num][0]), np.cos(PT[motor_num][0])*np.cos(PT[motor_num][1]), -np.cos(PT[motor_num][0])*np.sin(PT[motor_num][1]), PT[motor_num][2]*np.sin(PT[motor_num][0])],
                [0, np.sin(PT[motor_num][1]), np.cos(PT[motor_num][1]), PT[motor_num][3]],
                [0,0,0,1]]
    
    def forward(self, theta_list):
        p_list = list()
        for i, angle in enumerate(theta_list):
            tm = self.get_transformation_matrix(i, angle)
            if(i == 0): 
                buff = tm
            else:
                buff = np.dot(buff, tm)
            p_list.append(buff)
        return np.array(p_list)
    
    def inverse(self, goal, q):
        gamma = 0.01
        fk = self.forward(q)
        end = fk[-1,:3,3].tolist()
        end.extend(fk[-1,:3,2].tolist())
        d = np.subtract(goal, end)
        #d[3:] *= 2
        J = Jacobian(q, [50,50,50,50,50,50])
        step = np.dot(np.linalg.pinv(J), d.reshape(6,))
        q = q + (step * gamma)
        return q
    
    def normalize(self, x):
        return x/np.linalg.norm(x)
    
    def radians(self, degrees):
        return (degrees/180) * np.pi

def main():
    print('New')
    plt.pause(1)
    jik = JacobianInverseKinematics()
    q = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    goal = np.array([35., -35., 191., 0.0, 0.0, 0.0])
    loop = 0
    pthreshold = 0.001
    rthreshold = 0.0000001
    perr = 100
    perr = 100
    plt.ion()
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [0., 0., 0., 0., 0., 0., 0., 0.]
    y = [0., 0., 0., 0., 0., 0., 0., 0.]
    z = [0., 0., 0., 0., 0., 0., 0., 0.]
    while((perr > pthreshold or rerr > rthreshold) and loop < 1<<14):
        ax.scatter(35., -35., 191., c='r')
        loop += 1
        q = jik.inverse(goal, q)
        coords = jik.forward(q)
        pos = coords[-1,:3,3].tolist()
        pos.extend(coords[-1,:3,2].tolist())
        for i in range(7):
            x[i+1] = coords[i,0,3]
            y[i+1] = coords[i,1,3]
            z[i+1] = coords[i,2,3]
        ax.plot([int(x_) for x_ in x], [int(y_) for y_ in y], [int(z_) for z_ in z], linewidth=4)
        plt.draw()
        plt.pause(0.000001)
        ax.cla()
        perr = np.sqrt(np.sum(np.power(np.subtract(goal[:3], pos[:3]), 2)))
        rerr = np.sqrt(np.sum(np.power(np.subtract(goal[3:], pos[3:]), 2)))
        print('{}, {}, {} '.format(pos[3], pos[4], pos[5]))
        print('{0:} :: p{1:.6} r{2:.6}'.format(loop, perr, rerr))

if __name__ == '__main__':
    main()
