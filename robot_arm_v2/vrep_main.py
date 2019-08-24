import os
import sys
import time
import pickle
import numpy as np
from vrep import vrep

np.random.seed(seed=int(time.time()))

velocity=0.005
previousSimultationTime = 0
t = time.time()
first = True
err = False
data_size = 0

if (not os.path.exists('data/position_data.csv')):
    with open('data/position_data.csv', 'wb') as f:
        pickle.dump([], f)
    with open('data/joint_data.csv', 'wb') as f:
        pickle.dump([], f)
    position_data = list()
    joint_data = list()
else:
    with open('data/position_data.csv', 'rb') as f:
        position_data = pickle.load(f)
    with open('data/joint_data.csv', 'rb') as f:
        joint_data = pickle.load(f)

vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1', 19990, True, True, 5000, 5)

if (clientID != -1):
    print('Connected to remote API server')
else:
    print('Connection not successful')
    sys.exit('Could not connect')

collision_objects = [0]*9
for i in range(9):
    _, collision_objects[i] = vrep.simxGetCollisionHandle(clientID, 'Collision{}'.format(i), vrep.simx_opmode_blocking)

ret2, target = vrep.simxGetObjectHandle(clientID, 'target', vrep.simx_opmode_blocking)
ret3, dummy = vrep.simxGetObjectHandle(clientID, 'Dummy', vrep.simx_opmode_blocking)
joint_handles = list()
for i in range(6):
    handle = 'joint_{}'.format(i+1)
    _, joint_handle = vrep.simxGetObjectHandle(clientID, handle, vrep.simx_opmode_blocking)
    joint_handles.append(joint_handle)
while (True):
    for i in range(6):
        q = np.random.uniform(-2.61799387,2.61799387, 1)
        rc = vrep.simxSetJointPosition(clientID, joint_handles[i], q, vrep.simx_opmode_oneshot)
        if (rc != 0):
            print('Return Code = {}'.format(rc))
            err = True
    for co in collision_objects:
        returnCode, collisionState = vrep.simxReadCollision(clientID, co, vrep.simx_opmode_blocking)
        if (collisionState):
            print('\n*************************************')
            print('COLLISION ON {}'.format(co))
            print('*************************************\n')
            err = True
            break
    if (not err):
        joint_angles = [0.0]*6
        rcpos, pos = vrep.simxGetObjectPosition(clientID, dummy, -1, vrep.simx_opmode_blocking)
        rcori, ori = vrep.simxGetObjectOrientation(clientID, dummy, -1, vrep.simx_opmode_blocking)
        #vrep.simxSetObjectPosition(clientID, target, -1, pos, vrep.simx_opmode_oneshot)
        #vrep.simxSetObjectOrientation(clientID, target, -1, ori, vrep.simx_opmode_oneshot)
        for i in range(6):
            re, angle = vrep.simxGetJointPosition(clientID, joint_handles[i], vrep.simx_opmode_blocking)
            joint_angles[i] = angle
        print("Size = {}".format(len(position_data)))
        print("Time = {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t))))
        print("Pos =")
        print(pos)
        print("Ori =")
        print(ori)
        print("JAs =")
        print(joint_angles)
        print("#===================================================\n")
        pos.extend(ori)
        if (pos not in position_data):
            position_data.append(pos)
            joint_data.append(joint_angles)
            data_size = len(position_data)
            if (data_size % 100 == 0):
                with open('data/position_data.csv', 'wb') as f:
                    pickle.dump(position_data, f)
                with open('data/joint_data.csv', 'wb') as f:
                    pickle.dump(joint_data, f)
        #time.sleep(0.5)
    err = False



