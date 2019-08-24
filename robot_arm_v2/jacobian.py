import numpy as np

def Jacobian(q, a):
    cos = lambda theta: np.cos(theta)
    sin = lambda theta: np.sin(theta)
    pi = np.pi
    pi = np.pi
    l0, l1, l2, l3 = a[0], a[1]+a[2], a[3]+a[4], a[5]

    J = np.zeros((6, 7))

    # q1
    J[0][0]=l1*sin(q[0])*sin(q[1])+l2*((sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*sin(q[3])+sin(q[0])*sin(q[1])*cos(q[3]))+l3*((((sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*cos(q[3])-sin(q[0])*sin(q[1])*sin(q[3]))*cos(q[4])-(sin(q[0])*sin(q[2])*cos(q[1])-cos(q[0])*cos(q[2]))*sin(q[4]))*sin(q[5])+((sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*sin(q[3])+sin(q[0])*sin(q[1])*cos(q[3]))*cos(q[5]))
    J[1][0]=-l1*sin(q[1])*cos(q[0])+l2*((sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*sin(q[3])-sin(q[1])*cos(q[0])*cos(q[3]))+l3*((((sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*cos(q[3])+sin(q[1])*sin(q[3])*cos(q[0]))*cos(q[4])+(sin(q[0])*cos(q[2])+sin(q[2])*cos(q[0])*cos(q[1]))*sin(q[4]))*sin(q[5])+((sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*sin(q[3])-sin(q[1])*cos(q[0])*cos(q[3]))*cos(q[5]))
    J[2][0]=0
    J[3][0]=0
    J[4][0]=0
    J[5][0]=1

    # q2
    J[0][1]=(-l1*cos(q[1])+l2*(sin(q[1])*sin(q[3])*cos(q[2])-cos(q[1])*cos(q[3]))+l3*(((sin(q[1])*cos(q[2])*cos(q[3])+sin(q[3])*cos(q[1]))*cos(q[4])-sin(q[1])*sin(q[2])*sin(q[4]))*sin(q[5])-(-sin(q[1])*sin(q[3])*cos(q[2])+cos(q[1])*cos(q[3]))*cos(q[5])))*cos(q[0])
    J[1][1]=(-l1*cos(q[1])+l2*(sin(q[1])*sin(q[3])*cos(q[2])-cos(q[1])*cos(q[3]))+l3*(((sin(q[1])*cos(q[2])*cos(q[3])+sin(q[3])*cos(q[1]))*cos(q[4])-sin(q[1])*sin(q[2])*sin(q[4]))*sin(q[5])-(-sin(q[1])*sin(q[3])*cos(q[2])+cos(q[1])*cos(q[3]))*cos(q[5])))*sin(q[0])
    J[2][1]=-l1*sin(q[1])-l2*sin(q[1])*cos(q[3])-l2*sin(q[3])*cos(q[1])*cos(q[2])+l3*sin(q[1])*sin(q[3])*sin(q[5])*cos(q[4])-l3*sin(q[1])*cos(q[3])*cos(q[5])+l3*sin(q[2])*sin(q[4])*sin(q[5])*cos(q[1])-l3*sin(q[3])*cos(q[1])*cos(q[2])*cos(q[5])-l3*sin(q[5])*cos(q[1])*cos(q[2])*cos(q[3])*cos(q[4])
    J[3][1]=sin(q[0])
    J[4][1]=-cos(q[0])
    J[5][1]=0

    # q3
    J[0][2]=l2*sin(q[0])*sin(q[3])*cos(q[2])+l2*sin(q[2])*sin(q[3])*cos(q[0])*cos(q[1])-l3*sin(q[0])*sin(q[2])*sin(q[4])*sin(q[5])+l3*sin(q[0])*sin(q[3])*cos(q[2])*cos(q[5])+l3*sin(q[0])*sin(q[5])*cos(q[2])*cos(q[3])*cos(q[4])+l3*sin(q[2])*sin(q[3])*cos(q[0])*cos(q[1])*cos(q[5])+l3*sin(q[2])*sin(q[5])*cos(q[0])*cos(q[1])*cos(q[3])*cos(q[4])+l3*sin(q[4])*sin(q[5])*cos(q[0])*cos(q[1])*cos(q[2])
    J[1][2]=l2*sin(q[0])*sin(q[2])*sin(q[3])*cos(q[1])-l2*sin(q[3])*cos(q[0])*cos(q[2])+l3*sin(q[0])*sin(q[2])*sin(q[3])*cos(q[1])*cos(q[5])+l3*sin(q[0])*sin(q[2])*sin(q[5])*cos(q[1])*cos(q[3])*cos(q[4])+l3*sin(q[0])*sin(q[4])*sin(q[5])*cos(q[1])*cos(q[2])+l3*sin(q[2])*sin(q[4])*sin(q[5])*cos(q[0])-l3*sin(q[3])*cos(q[0])*cos(q[2])*cos(q[5])-l3*sin(q[5])*cos(q[0])*cos(q[2])*cos(q[3])*cos(q[4])
    J[2][2]=(l2*sin(q[2])*sin(q[3])+l3*sin(q[2])*sin(q[3])*cos(q[5])+l3*sin(q[2])*sin(q[5])*cos(q[3])*cos(q[4])+l3*sin(q[4])*sin(q[5])*cos(q[2]))*sin(q[1])
    J[3][2]=-sin(q[1])*cos(q[0])
    J[4][2]=-sin(q[0])*sin(q[1])
    J[5][2]=cos(q[1])

    # q4
    J[0][3]=l2*sin(q[0])*sin(q[2])*cos(q[3])+l2*sin(q[1])*sin(q[3])*cos(q[0])-l2*cos(q[0])*cos(q[1])*cos(q[2])*cos(q[3])-l3*sin(q[0])*sin(q[2])*sin(q[3])*sin(q[5])*cos(q[4])+l3*sin(q[0])*sin(q[2])*cos(q[3])*cos(q[5])+l3*sin(q[1])*sin(q[3])*cos(q[0])*cos(q[5])+l3*sin(q[1])*sin(q[5])*cos(q[0])*cos(q[3])*cos(q[4])+l3*sin(q[3])*sin(q[5])*cos(q[0])*cos(q[1])*cos(q[2])*cos(q[4])-l3*cos(q[0])*cos(q[1])*cos(q[2])*cos(q[3])*cos(q[5])
    J[1][3]=l2*sin(q[0])*sin(q[1])*sin(q[3])-l2*sin(q[0])*cos(q[1])*cos(q[2])*cos(q[3])-l2*sin(q[2])*cos(q[0])*cos(q[3])+l3*sin(q[0])*sin(q[1])*sin(q[3])*cos(q[5])+l3*sin(q[0])*sin(q[1])*sin(q[5])*cos(q[3])*cos(q[4])+l3*sin(q[0])*sin(q[3])*sin(q[5])*cos(q[1])*cos(q[2])*cos(q[4])-l3*sin(q[0])*cos(q[1])*cos(q[2])*cos(q[3])*cos(q[5])+l3*sin(q[2])*sin(q[3])*sin(q[5])*cos(q[0])*cos(q[4])-l3*sin(q[2])*cos(q[0])*cos(q[3])*cos(q[5])
    J[2][3]=-l2*sin(q[1])*cos(q[2])*cos(q[3])-l2*sin(q[3])*cos(q[1])+l3*sin(q[1])*sin(q[3])*sin(q[5])*cos(q[2])*cos(q[4])-l3*sin(q[1])*cos(q[2])*cos(q[3])*cos(q[5])-l3*sin(q[3])*cos(q[1])*cos(q[5])-l3*sin(q[5])*cos(q[1])*cos(q[3])*cos(q[4])
    J[3][3]=sin(q[0])*cos(q[2])+sin(q[2])*cos(q[0])*cos(q[1])
    J[4][3]=sin(q[0])*sin(q[2])*cos(q[1])-cos(q[0])*cos(q[2])
    J[5][3]=sin(q[1])*sin(q[2])

    # q5
    J[0][4]=l3*(-sin(q[0])*sin(q[2])*sin(q[4])*cos(q[3])+sin(q[0])*cos(q[2])*cos(q[4])-sin(q[1])*sin(q[3])*sin(q[4])*cos(q[0])+sin(q[2])*cos(q[0])*cos(q[1])*cos(q[4])+sin(q[4])*cos(q[0])*cos(q[1])*cos(q[2])*cos(q[3]))*sin(q[5])
    J[1][4]=l3*(-sin(q[0])*sin(q[1])*sin(q[3])*sin(q[4])+sin(q[0])*sin(q[2])*cos(q[1])*cos(q[4])+sin(q[0])*sin(q[4])*cos(q[1])*cos(q[2])*cos(q[3])+sin(q[2])*sin(q[4])*cos(q[0])*cos(q[3])-cos(q[0])*cos(q[2])*cos(q[4]))*sin(q[5])
    J[2][4]=l3*(sin(q[1])*sin(q[2])*cos(q[4])+sin(q[1])*sin(q[4])*cos(q[2])*cos(q[3])+sin(q[3])*sin(q[4])*cos(q[1]))*sin(q[5])
    J[3][4]=(sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*sin(q[3])-sin(q[1])*cos(q[0])*cos(q[3])
    J[4][4]=-(sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*sin(q[3])-sin(q[0])*sin(q[1])*cos(q[3])
    J[5][4]=-sin(q[1])*sin(q[3])*cos(q[2])+cos(q[1])*cos(q[3])

    # q6
    J[0][5]=l3*(-sin(q[0])*sin(q[2])*sin(q[3])*sin(q[5])+sin(q[0])*sin(q[2])*cos(q[3])*cos(q[4])*cos(q[5])+sin(q[0])*sin(q[4])*cos(q[2])*cos(q[5])+sin(q[1])*sin(q[3])*cos(q[0])*cos(q[4])*cos(q[5])+sin(q[1])*sin(q[5])*cos(q[0])*cos(q[3])+sin(q[2])*sin(q[4])*cos(q[0])*cos(q[1])*cos(q[5])+sin(q[3])*sin(q[5])*cos(q[0])*cos(q[1])*cos(q[2])-cos(q[0])*cos(q[1])*cos(q[2])*cos(q[3])*cos(q[4])*cos(q[5]))
    J[1][5]=l3*(sin(q[0])*sin(q[1])*sin(q[3])*cos(q[4])*cos(q[5])+sin(q[0])*sin(q[1])*sin(q[5])*cos(q[3])+sin(q[0])*sin(q[2])*sin(q[4])*cos(q[1])*cos(q[5])+sin(q[0])*sin(q[3])*sin(q[5])*cos(q[1])*cos(q[2])-sin(q[0])*cos(q[1])*cos(q[2])*cos(q[3])*cos(q[4])*cos(q[5])+sin(q[2])*sin(q[3])*sin(q[5])*cos(q[0])-sin(q[2])*cos(q[0])*cos(q[3])*cos(q[4])*cos(q[5])-sin(q[4])*cos(q[0])*cos(q[2])*cos(q[5]))
    J[2][5]=l3*(sin(q[1])*sin(q[2])*sin(q[4])*cos(q[5])+sin(q[1])*sin(q[3])*sin(q[5])*cos(q[2])-sin(q[1])*cos(q[2])*cos(q[3])*cos(q[4])*cos(q[5])-sin(q[3])*cos(q[1])*cos(q[4])*cos(q[5])-sin(q[5])*cos(q[1])*cos(q[3]))
    J[3][5]=-((sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*cos(q[3])+sin(q[1])*sin(q[3])*cos(q[0]))*sin(q[4])+(sin(q[0])*cos(q[2])+sin(q[2])*cos(q[0])*cos(q[1]))*cos(q[4])
    J[4][5]=((sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*cos(q[3])-sin(q[0])*sin(q[1])*sin(q[3]))*sin(q[4])+(sin(q[0])*sin(q[2])*cos(q[1])-cos(q[0])*cos(q[2]))*cos(q[4])
    J[5][5]=(sin(q[1])*cos(q[2])*cos(q[3])+sin(q[3])*cos(q[1]))*sin(q[4])+sin(q[1])*sin(q[2])*cos(q[4])

    # q7
    J[0][6]=0
    J[1][6]=0
    J[2][6]=0
    J[3][6]=(((sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*cos(q[3])+sin(q[1])*sin(q[3])*cos(q[0]))*cos(q[4])+(sin(q[0])*cos(q[2])+sin(q[2])*cos(q[0])*cos(q[1]))*sin(q[4]))*sin(q[5])+((sin(q[0])*sin(q[2])-cos(q[0])*cos(q[1])*cos(q[2]))*sin(q[3])-sin(q[1])*cos(q[0])*cos(q[3]))*cos(q[5])
    J[4][6]=-(((sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*cos(q[3])-sin(q[0])*sin(q[1])*sin(q[3]))*cos(q[4])-(sin(q[0])*sin(q[2])*cos(q[1])-cos(q[0])*cos(q[2]))*sin(q[4]))*sin(q[5])-((sin(q[0])*cos(q[1])*cos(q[2])+sin(q[2])*cos(q[0]))*sin(q[3])+sin(q[0])*sin(q[1])*cos(q[3]))*cos(q[5])
    J[5][6]=-((sin(q[1])*cos(q[2])*cos(q[3])+sin(q[3])*cos(q[1]))*cos(q[4])-sin(q[1])*sin(q[2])*sin(q[4]))*sin(q[5])-(sin(q[1])*sin(q[3])*cos(q[2])-cos(q[1])*cos(q[3]))*cos(q[5])

    return J
