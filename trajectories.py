import numpy as np
from math import pi
import matplotlib.pyplot as plt

def circle2D(radius: float, start_angle: float, end_angle: float, cx=0, cy=0, num=50) -> np.ndarray:
    """Builds a 2D circular trajectory

    Params
    --
    - radius: [float] radius in meter(s)
    - start_angle [float] Angle to start at
    - end_angle [float] Angle to end at
    - cx [float] x-coordinate of the center
    - cy [float] y-coordinate of the center
    - num [int] number of points

    Returns
    --
    - xy [ndarray] array of x/y points. each row is a column
    """
    theta=np.linspace(start_angle, end_angle, num=num)
    x=cx+radius*np.cos(theta)
    y=cy+radius*np.sin(theta)

    xy=np.vstack((x,y))
    print(xy.shape)
    return(xy.transpose())

def circle3D(radius: float, start_angle: float, end_angle: float, center: list, num=50):
    pass

def helix(radius: float, theta_max: float, num=100):
    """Ref: https://scipython.com/book/chapter-7-matplotlib/examples/depicting-a-helix/
    """
    n = num
    

    # Plot a helix along the x-axis
    t_max = theta_max * np.pi
    theta = np.linspace(0, t_max, n)
    x = theta
    z =  radius*np.sin(theta)
    y =  radius*np.cos(theta)

    xyz=np.vstack((x,y,z))
    return(xyz.transpose())


    
def testCircle2D():
    # Test circle2D
    r=1
    cx, cy=1,0
    xy=circle2D(r,0,pi, cx=cx, cy=cy)
    plt.plot(xy[:,0], xy[:,1])
    plt.xlim(cx-2*r, cx+2*r)
    plt.ylim(cy-2*r, cy+2*r)
    plt.scatter(cx,cy, color='red')
    plt.grid()
    plt.show()

def testCircle3D():
    pass

def testHelix():
    r =2
    t_max = 8
    xyz = helix(r, t_max)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'b', lw=2)
    plt.show()

if __name__ == '__main__':
    # testCircle2D()
    testHelix()