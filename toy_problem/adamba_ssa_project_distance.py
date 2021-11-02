# The experiments for numerical safe set algorithm
# This version is for purely consider the distance as safety index


import time
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from AdamBA import AdamBA




x = 1
y = -0.5
theta = np.pi / 2

# state
s = [x, y, theta]

dt = 1e-2  # second

# design the nomial control
u = [1, 0]

# simulation
framen = 1
step_num = 10000
for i in range(step_num):

    # compute the safety control
    u_new = u
    if safetyindex(s) > 0:
        # safe control
        [A, b] = AdamBA(s, u, dt)

        # set the QP objective
        H = np.eye(2, 2)
        f = [-u[0], -u[1]]
        # u_new = quadprog(H, f, A, b, [], [], [], [], u)

        # print(A)
        # print(b)
        # exit(0)
        # print(u)
        # exit(0)
        u_new, status = quadprog(H, f, A, [b], initvals=np.array(u), verbose=True)

        # u_new, status = quadprog(H, f, A, [b], initvals=None, verbose=True)
        # print(u)
        # print(u_new)
        # print(status)
        # exit(0)

        #u_new = quadprog(H, f, A, b, [], [], [], [], u)

    print("safety index: ", safetyindex(s))
    if safetyindex(s) < 0.03:
        break

    s = step(s, u_new, dt)

    # visualize
    # todo

    #time.sleep(0.1)


# Video
# todo























