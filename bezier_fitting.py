import numpy as np
from math import pi, comb
from trajectories import circle2D
import matplotlib.pyplot as plt
from time import time

import bezier

########### Params ##################
# Prediction time
t_prediction =0.5
# Bezier curve degree
deg=3
d1=deg+1 # used for convenient calculations
# Dimension of control points. 2: 2D. 3: 3D
pm=2
###############################

# Generate  measurements; half of the trajectory (half a circle)
M = 50 # Number of measurements
# Measurement times
t=np.linspace(0,1,num=M)
# Generate trajectory points
r=1 # radius
xy=circle2D(r,0,pi,cx=1,cy=2, num=M)
# Compute signals + some noise
x=xy[:,0] + (np.random.random(size=M)/1000)
y=xy[:,1]+ (np.random.random(size=M)/1000)

# Full true trajectory
r=1 # radius
xy=circle2D(r,0,2*pi,cx=1,cy=2, num=M)
x_full=xy[:,0]
y_full=xy[:,1]


# Compute least square martices, A,b
# Coeff matrix
A=np.zeros((pm*M,pm*d1))
b=np.zeros((pm*M,1))

t_start= time()
for i in range(M):
    b[pm*i+0,0] = x[i]
    b[pm*i+1,0] = y[i]
    for j in range(pm):
        for k in range(d1):
            A[pm*i+j,pm*k+j]=comb(deg,k)*(1-t[i])**(deg-k) * t[i]**k
t_end = time()

# Least square
print("time to compute A: ", t_end-t_start)
print("Size of A: ", A.shape)

t_start = time()
sol = np.linalg.lstsq(A, b, rcond=None)[0]
t_end = time()

# print("Solution: \n", sol)
print("Solution time: ", t_end-t_start)

cnt_pts = np.reshape(sol, (d1,pm))
C_original=bezier.bezier(cnt_pts, deg, start=0, end=1)
C_predicted=bezier.bezier(cnt_pts, deg, start=1, end=1+t_prediction)

plt.plot(C_original[:,0], C_original[:,1], 'k', linewidth=2)
plt.plot(C_predicted[:,0], C_predicted[:,1], 'x',color='orange')
plt.scatter(cnt_pts[:,0], cnt_pts[:,1], color='blue')
plt.plot(x,y, 'x', color='red')
plt.plot(x_full, y_full ,color='green')
plt.legend([' Original bezier curve', 'Predicted over {} second(s)'.format(t_prediction) , 'control points', 'measurements', 'full trajectory'])
plt.title("Bezier curve with degree {}".format(deg))
plt.xlabel("x-axis, meters")
plt.ylabel("y-axis, meters")
plt.show()
