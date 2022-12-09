"""
Author: Mohamed Abdelkader
Contact: mohamedashraf123@gmail.com
Copyright 2022
"""

import numpy as np
from math import comb
import matplotlib.pyplot as plt

import logging
FORMAT = ' [%(levelname)s] %(asctime)s [%(funcName)s] :\t%(message)s'
logging.basicConfig(format=FORMAT)

def bezier(cnt_pts: np.ndarray, deg: int, start=0, end=1) -> np.ndarray:
    """
    Computes the points of the bezier curve defined by contorl points cnt_pts and degree deg

    Params
    --
    cnt_pts [ndarray] Control points of shape of (deg+1) x m. Columns are the point's coordinates. Rows are the different points
    deg [int] degree of the curve
    start [double] Value of the starting time (parameter)
    end [double] Value of the end time (parameter)

    Returns
    --
    curve: [ndarray] M x n. M is the number of measurements. n is the dimension of points
    """

    # Sanity checks
    if (deg<0):
        logging.error("deg can not be <0")
        return

    if type(cnt_pts) != np.ndarray:
        logging.error("The type of cnt_pts {} is not numpy.ndarray".format(type(cnt_pts)))
        return

    if len(cnt_pts.shape) != 2:
        logging.error( "Shape of cnt_pts {} is incorrect. It should have rows and columns".format(cnt_pts.shape))
        return

    # Number of control points
    r = cnt_pts.shape[0]
    if r != (deg+1):
        logging.error("ERROR: number of control points {} are not equal to deg+1= {}".format(r,deg+1))
        return
        
    # Dimension of points
    p_dim = cnt_pts.shape[1]

    # create time array [0,1]
    t = np.linspace(start,end)

    
    curve = np.zeros((len(t),p_dim))

    # Computes binomial coefficents
    # Rows: number of coeff. Cols, length of time (parameter) vector
    c = np.zeros((deg+1,len(t)))
    for ti in range(len(t)):
        
        for di in range(deg+1):
            c[di,ti] = comb(deg,di)*(1-t[ti])**(deg-di) * t[ti]**di
            for p in range(p_dim):
                curve[ti,p] = curve[ti,p] + c[di,ti] * cnt_pts[di,p]

    return curve

def main():
    cnt_pts = np.array([[2,3],
                        [2.5,1], 
                        [5,1],
                        [7,8],
                        [0,10]
                        ])
    curve = bezier(cnt_pts, 4, start=0, end=1.5)

    plt.plot(curve[:,0], curve[:,1])
    plt.scatter(cnt_pts[:,0], cnt_pts[:,1], color='red')
    plt.show()

if __name__ == '__main__':
    main()



    