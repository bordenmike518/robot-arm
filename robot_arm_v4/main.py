from arm import Robot_Arm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def rad(deg):
    return (deg/180)*np.pi

# Initialize Rob (AKA 'Robot' - 'ot')
print('\nRunning :: Rob Initialization')
rob = Robot_Arm()

# Create Target
print('\nRunning :: Initializing Targets')
P07 = np.array([0.13266407, 0.13266407, 0.030865828]).reshape(3,)
R07 = np.array([[-0.27059805, -0.70710678,  0.65328148],
                [-0.27059805,  0.70710678,  0.65328148],
                [-0.92387953,          0., -0.38268343]])
#R07 = R07[:,2].reshape(3,)
xR07 = np.array([-0.65328148, -0.65328148, 0.38268343]).reshape(3,)
print(P07)
print(R07)
t_list = rob.forward([rad(45), rad(45), 0.0, rad(-90), 0.0, rad(-22.5), 0.0])
print(t_list[-1])

# Apply Inverse Kinematics
print('\nRunning :: Inverse Kinematics')
Q = rob.inverse(P07, R07)
print(Q)

# Apply Forward Kinematics
print('\nRunning :: Forward Kinematics')
t_list = rob.forward(Q)
print(t_list[-1])

# Get (x, y, z) position from base to end effector
print('\nRunning :: Get X, Y, Z')
X, Y, Z = rob.get_xyz(t_list)
print(X)
print(Y)
print(Z)

# Plot Inverse Kinematics result
print('\nRunning :: Plotting')
fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.plot(X, Y, Z, linewidth=4)
plt.xlabel('X')
plt.ylabel('Y')

# Apply Forward Kinematics to actual target for visualization
print('\nRunning :: True Forward Kinematics for Visualization')
t_list = rob.forward([rad(45), rad(45), 0.0, rad(-90), 0.0, rad(-22.5), 0.0])
print(t_list[-1])
X, Y, Z = rob.get_xyz(t_list)
ax = fig.add_subplot(122, projection='3d')
ax.plot(X, Y, Z, linewidth=4)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
