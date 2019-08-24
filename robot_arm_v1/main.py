import os
import pybullet as p
from time import sleep
import numpy as np

os.chdir(os.path.dirname(__file__))
dirpath = os.getcwd()
p.connect(p.GUI)
p.loadURDF(dirpath + '/plane.urdf')
robotArmId = p.loadURDF(dirpath + '/robot_arm.urdf')
p.resetBasePositionAndOrientation(robotArmId,[0,0,0],[0,0,0,1])
robotArmEndEffectorIndex = 6
numJoints = p.getNumJoints(robotArmId) - 1
print('NUM JOINTS = {}'.format(numJoints))

#Joint damping coefficents. Using large values for the joints that we don't want to move.
#jd=[100.0,100.0,100.0,100.0,100.0,0.5]
jd=[0.5,0.5,0.5,0.5,0.5,0.5]

p.setGravity(0,0,-10)

pos = np.array(input('Position : ').split(), dtype=np.float64) / 1000
orn = p.getQuaternionFromEuler(np.array(input("Orientation : ").split(), dtype=np.float64))

while 1:
    p.stepSimulation()
    jointPoses = p.calculateInverseKinematics(robotArmId,robotArmEndEffectorIndex,pos,orn,jointDamping=jd)

    for i in range (numJoints):
        p.setJointMotorControl2(bodyIndex=robotArmId,jointIndex=i,controlMode=p.POSITION_CONTROL,targetPosition=jointPoses[i],targetVelocity=0,force=500,positionGain=0.03,velocityGain=0.1)
    sleep(0.05)
