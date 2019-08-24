from arm import Robot_Arm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rad(deg):
    return (deg/180)*np.pi

rob = Robot_Arm()
pos = np.array([-0.032, -0.032, 0.210])
rot = np.array([-0.653, -0.653, 0.382])
q = rob.inverse(pos, rot, 0.0)
print(q)
t_list = rob.forward(q)
X, Y, Z = rob.get_xyz(t_list)
print(X[-1])
print(Y[-1])
print(Z[-1])
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(X, Y, Z, linewidth=4)
plt.xlabel('X')
plt.ylabel('Y')
t_list = rob.forward([rad(45), rad(45), 0.0, rad(-90), 0.0, rad(-22.5), 0.0])
print('t_list')
print(t_list[-1])
X, Y, Z = rob.get_xyz(t_list)
ax = fig.add_subplot(122, projection='3d')
ax.plot(X, Y, Z, linewidth=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

