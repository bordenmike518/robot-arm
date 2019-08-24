import numpy as np

def Jacobian(q_vector, a_vector):
    a = np.cos(q_vector[0])
    b = np.sin(q_vector[0])
    c = np.cos(q_vector[1])
    d = np.sin(q_vector[1])
    e = np.cos(q_vector[2])
    f = np.sin(q_vector[2])
    g = np.cos(q_vector[3])
    h = np.sin(q_vector[3])
    i = np.cos(q_vector[4])
    j = np.sin(q_vector[4])
    k = np.cos(q_vector[5]+(np.pi/2.0))
    l = np.sin(q_vector[5]+(np.pi/2.0))

    t = a_vector[0]
    u = a_vector[1]
    v = a_vector[2]
    w = a_vector[3]
    x = a_vector[4]
    y = a_vector[5]


    J = np.zeros((6,6))

    J[0,0] = b*d*u+b*d*v+b*d*g*w+b*c*e*h*w+a*f*h*w+b*d*g*x+b*c*e*h*x+a*f*h*x+i*-b*c*e*g*k*y-i*a*f*g*k*y-i*-b*d*h*k*y-a*e*j*k*y+b*c*f*j*k*y+b*d*g*l*y+b*c*e*h*l*y+a*f*h*l*y
    J[1,0] = -a*d*u-a*d*v-a*d*g*w-a*c*e*h*w+b*f*h*w-a*d*g*x-a*c*e*h*x+b*f*h*x+i*a*c*e*g*k*y+i*-b*f*g*k*y-i*a*d*h*k*y-b*e*j*k*y-a*c*f*j*k*y-a*d*g*l*y-a*c*e*h*l*y+b*f*h*l*y
    J[2,0] = t+c*u+c*v+c*g*w-d*e*h*w+c*g*x-d*e*h*x+i*d*e*g*k*y+i*c*h*k*y-d*f*j*k*y+c*g*l*y-d*e*h*l*y
    J[3,0] = b*d*g*k+b*c*e*h*k+a*f*h*k-i*-b*c*e*g*l+i*a*f*g*l+i*-b*d*h*l+a*e*j*l-b*c*f*j*l
    J[4,0] = -a*d*g*k-a*c*e*h*k+b*f*h*k-i*a*c*e*g*l-i*-b*f*g*l+i*a*d*h*l+b*e*j*l+a*c*f*j*l
    J[5,0] = c*g*k-d*e*h*k-i*d*e*g*l-i*c*h*l+d*f*j*l 
    J[0,1] = -a*c*u-a*c*v-a*c*g*w-a*-d*e*h*w+b*f*h*w-a*c*g*x-a*-d*e*h*x+b*f*h*x+i*a*-d*e*g*k*y-i*b*f*g*k*y-i*a*c*h*k*y-b*e*j*k*y-a*-d*f*j*k*y-a*c*g*l*y-a*-d*e*h*l*y+b*f*h*l*y
    J[1,1] = -b*c*u-b*c*v-b*c*g*w-b*-d*e*h*w-a*f*h*w-b*c*g*x-b*-d*e*h*x-a*f*h*x+i*b*-d*e*g*k*y+i*a*f*g*k*y-i*b*c*h*k*y+a*e*j*k*y-b*-d*f*j*k*y-b*c*g*l*y-b*-d*e*h*l*y-a*f*h*l*y
    J[2,1] = t-d*u-d*v-d*g*w-c*e*h*w-d*g*x-c*e*h*x+i*c*e*g*k*y+i*-d*h*k*y-c*f*j*k*y-d*g*l*y-c*e*h*l*y
    J[3,1] = -a*c*g*k-a*-d*e*h*k+b*f*h*k-i*a*-d*e*g*l+i*b*f*g*l+i*a*c*h*l+b*e*j*l+a*-d*f*j*l
    J[4,1] = -b*c*g*k-b*-d*e*h*k-a*f*h*k-i*b*-d*e*g*l-i*a*f*g*l+i*b*c*h*l-a*e*j*l+b*-d*f*j*l
    J[5,1] = -d*g*k-c*e*h*k-i*c*e*g*l-i*-d*h*l+c*f*j*l 
    J[0,2] = -a*d*u-a*d*v-a*d*g*w-a*c*-f*h*w+b*e*h*w-a*d*g*x-a*c*-f*h*x+b*e*h*x+i*a*c*-f*g*k*y-i*b*e*g*k*y-i*a*d*h*k*y-b*-f*j*k*y-a*c*e*j*k*y-a*d*g*l*y-a*c*-f*h*l*y+b*e*h*l*y
    J[1,2] = -b*d*u-b*d*v-b*d*g*w-b*c*-f*h*w-a*e*h*w-b*d*g*x-b*c*-f*h*x-a*e*h*x+i*b*c*-f*g*k*y+i*a*e*g*k*y-i*b*d*h*k*y+a*-f*j*k*y-b*c*e*j*k*y-b*d*g*l*y-b*c*-f*h*l*y-a*e*h*l*y
    J[2,2] = t+c*u+c*v+c*g*w-d*-f*h*w+c*g*x-d*-f*h*x+i*d*-f*g*k*y+i*c*h*k*y-d*e*j*k*y+c*g*l*y-d*-f*h*l*y
    J[3,2] = -a*d*g*k-a*c*-f*h*k+b*e*h*k-i*a*c*-f*g*l+i*b*e*g*l+i*a*d*h*l+b*-f*j*l+a*c*e*j*l
    J[4,2] = -b*d*g*k-b*c*-f*h*k-a*e*h*k-i*b*c*-f*g*l-i*a*e*g*l+i*b*d*h*l-a*-f*j*l+b*c*e*j*l
    J[5,2] = c*g*k-d*-f*h*k-i*d*-f*g*l-i*c*h*l+d*e*j*l 
    J[0,3] = -a*d*u-a*d*v+a*d*h*w-a*c*e*g*w+b*f*g*w+a*d*h*x-a*c*e*g*x+b*f*g*x-i*a*c*e*h*k*y+i*b*f*h*k*y-i*a*d*g*k*y-b*e*j*k*y-a*c*f*j*k*y+a*d*h*l*y-a*c*e*h*l*y+b*f*g*l*y
    J[1,3] = -b*d*u-b*d*v+b*d*h*w-b*c*e*g*w-a*f*g*w+b*d*h*x-b*c*e*g*x-a*f*g*x-i*b*c*e*h*k*y-i*a*f*h*k*y-i*b*d*g*k*y+a*e*j*k*y-b*c*f*j*k*y+b*d*h*l*y-b*c*e*g*l*y-a*f*g*l*y
    J[2,3] = t+c*u+c*v-c*h*w-d*e*h*w-c*h*x-d*e*g*x-i*d*e*h*k*y+i*c*g*k*y-d*f*j*k*y-c*h*l*y-d*e*g*l*y
    J[3,3] = a*d*h*k-a*c*e*g*k+b*f*g*k+i*a*c*e*h*l-i*b*f*h*l+i*a*d*g*l+b*e*j*l+a*c*f*j*l
    J[4,3] = b*d*h*k-b*c*e*g*k-a*f*g*k+i*b*c*e*h*l+i*a*f*h*l+i*b*d*g*l-a*e*j*l+b*c*f*j*l
    J[5,3] = -c*h*k-d*e*g*k+i*d*e*h*l-i*c*g*l+d*f*j*l 
    J[0,4] = -a*d*u-a*d*v-a*d*g*w-a*c*e*h*w+b*f*h*w-a*d*g*x-a*c*e*h*x+b*f*h*x-j*a*c*e*g*k*y+j*b*f*g*k*y+j*a*d*h*k*y-b*e*i*k*y-a*c*f*i*k*y-a*d*g*l*y-a*c*e*h*l*y+b*f*h*l*y
    J[1,4] = -b*d*u-b*d*v-b*d*g*w-b*c*e*h*w-a*f*h*w-b*d*g*x-b*c*e*h*x-a*f*h*x-j*b*c*e*g*k*y-j*a*f*g*k*y+j*b*d*h*k*y+a*e*i*k*y-b*c*f*i*k*y-b*d*g*l*y-b*c*e*h*l*y-a*f*h*l*y
    J[2,4] = t+c*u+c*v+c*g*w-d*e*h*w+c*g*x-d*e*h*x-j*d*e*g*k*y-j*c*h*k*y-d*f*i*k*y+c*g*l*y-d*e*h*l*y
    J[3,4] = -a*d*g*k-a*c*e*h*k+b*f*h*k+j*a*c*e*g*l-j*b*f*g*l-j*a*d*h*l+b*e*i*l+a*c*f*i*l
    J[4,4] = -b*d*g*k-b*c*e*h*k-a*f*h*k+j*b*c*e*g*l+j*a*f*g*l-j*b*d*h*l-a*e*i*l+b*c*f*i*l
    J[5,4] = c*g*k-d*e*h*k+j*d*e*g*l+j*c*h*l+d*f*i*l 
    J[0,5] = -a*d*u-a*d*v-a*d*g*w-a*c*e*h*w+b*f*h*w-a*d*g*x-a*c*e*h*x+b*f*h*x-i*a*c*e*g*l*y+i*b*f*g*l*y+i*a*d*h*l*y+b*e*j*l*y+a*c*f*j*l*y-a*d*g*k*y-a*c*e*h*k*y+b*f*h*k*y
    J[1,5] = -b*d*u-b*d*v-b*d*g*w-b*c*e*h*w-a*f*h*w-b*d*g*x-b*c*e*h*x-a*f*h*x-i*b*c*e*g*l*y-i*a*f*g*l*y+i*b*d*h*l*y-a*e*j*l*y+b*c*f*j*l*y-b*d*g*k*y-b*c*e*h*k*y-a*f*h*k*y
    J[2,5] = t+c*u+c*v+c*g*w-d*e*h*w+c*g*x-d*e*h*x-i*d*e*g*l*y-i*c*h*l*y+d*f*j*l*y+c*g*k*y-d*e*h*k*y
    J[3,5] = a*d*g*l+a*c*e*h*l-b*f*h*l-i*a*c*e*g*k+i*b*f*g*k+i*a*d*h*k+b*e*j*k+a*c*f*j*k
    J[4,5] = b*d*g*l+b*c*e*h*l+a*f*h*l-i*b*c*e*g*k-i*a*f*g*k+i*b*d*h*k-a*e*j*k+b*c*f*j*k
    J[5,5] = -c*g*l+d*e*h*l-i*d*e*g*k-i*c*h*k+d*f*j*k
    return J
