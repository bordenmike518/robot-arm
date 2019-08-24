def RZ(theta):
    return [[cos(θ) -sin(θ) 0 0],
            [sin(θ) cos(θ) 0 0],
            [0 0 1 0],
            [0 0 0 1]]

def RX(alpha):
    return [[1 0 0 0],
            [0 cos(alpha) -sin(alpha) 0],
            [0 sin(alpha)  cos(alpha) 0],
            [0 0 0 1]]

def TZ(d):
    return [[1 0 0 0],
            [0 1 0 0],
            [0 0 1 d],
            [0 0 0 1]]

def TX(a):
    return [[1 0 0 a],
            [0 1 0 0],
            [0 0 1 0],
            [0 0 0 1]]
            

