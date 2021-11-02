# Gradient Mappi
# ng Sampler
## Gaussian Sampler
# 2D configuration space infeasible set

import numpy as np
import time
from utils import *





def AdamBA(s, u, dt):
    infSet = []
    np.random.seed(0)

    # 2d case, 2 dimensional control signal
    # uniform sampling
    # offset = [0.5 0.5];
    # scale = [0.5 0.5];
    # action = scale. * rand(1, 2) + offset;

    # 这里就不转置了
    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)


    dt = 1e-8
    # limits = [-100, 100, -0.1 * pi / dt, 0.1 * pi / dt] # each row define th limits for one dimensional action
    limits = [[-100, 100], [-100, 100]]  # each row define the limits for one dimensional action
    NP = []

    # no need crop since se only need on sample
    NP = action

    start_time = time.time()

    # generate direction
    NP_vec_dir = []
    NP_vec = []
    sigma_vec = [[1, 0], [0, 1]]
    vec_num = 100

    # num of actions input, default as 1
    action_num = 1
    for t in range(0, NP.shape[0]):
        vec_set = []
        vec_dir_set = []
        for m in range(0, vec_num):
            vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
            vec_dir = vec_dir / np.linalg.norm(vec_dir)
            vec_dir_set.append(vec_dir)
            vec = NP[t]
            vec_set.append(vec)
            # print(vec)
            # print(vec_set)
            # print(vec_dir)
            # print(vec_dir_set)
            # exit(0)

        NP_vec_dir.append(vec_dir_set)
        NP_vec.append(vec_set)

    # print(NP_vec)
    # print(NP_vec_dir)
    # exit(0)
    bound = 0.0001

    # record how many boundary points have been found
    collected_num = 0

    for n in range(0, NP.shape[0]):
        NP_vec_tmp = NP_vec[n]
        NP_vec_dir_tmp = NP_vec_dir[n]
        for v in range(0, vec_num):
            if collected_num >= 2:
                break
            collected_num = collected_num + 1  # one more instance
            # update NP_vec
            NP_vec_tmp_i = NP_vec_tmp[v]
            NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]
            # print(NP_vec_tmp_i)
            # print(NP_vec_dir_tmp_i)
            # exit(0)
            eta = bound
            decrease_flag = 0
            # print(eta)

            while eta >= bound:
                flag = chk_unsafe(s, NP_vec_tmp_i, dt)
                # print(flag)
                # print(eta)
                # exit(0)
                # check if the action is out of limit, if yes, the abandon

                # print(NP_vec_tmp_i)
                if outofbound(limits, NP_vec_tmp_i):
                    # print("\nout\n")
                    collected_num = collected_num - 1  # not found, discard the recorded number
                    break

                # AdamBA procudure
                if flag == 1 and decrease_flag == 0:
                    # outreach
                    NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                    eta = eta * 2
                    continue
                # monitor for 1st reaching out boundary
                if flag == 0 and decrease_flag == 0:
                    decrease_flag = 1
                    eta = eta * 0.25  # make sure decrease step start at 0.5 of last increasing step
                    continue
                # decrease eta
                if flag == 1 and decrease_flag == 1:
                    NP_vec_tmp_i = NP_vec_tmp_i + eta * NP_vec_dir_tmp_i
                    eta = eta * 0.5
                    continue
                if flag == 0 and decrease_flag == 1:
                    NP_vec_tmp_i = NP_vec_tmp_i - eta * NP_vec_dir_tmp_i
                    eta = eta * 0.5
                    continue

            NP_vec_tmp[v] = NP_vec_tmp_i

        # exit(0)
        #discard those points that are out of boundary and not expanded

        NP_vec_tmp_new = []
        print("NP_vec_tmp: ",NP_vec_tmp)

        cnt = 0
        out = 0
        yes = 0
        for vnum in range(0, len(NP_vec_tmp)):
            # print(vnum)
            # print(NP_vec_tmp[vnum])
            # print(NP_vec_tmp)
            # print(len(NP_vec_tmp))
            # exit(0)
            cnt += 1
            if outofbound(limits, NP_vec_tmp[vnum]):
                # print("out")
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0] and NP_vec_tmp[vnum][1] == u[1]:
                # print("yes")
                yes += 1
                continue

            NP_vec_tmp_new.append(NP_vec_tmp[vnum])
        print("out = ", out)
        print("yes = ", yes)
        print("valid = ", cnt - out - yes)
        #update NP_vec

        NP_vec[n] = NP_vec_tmp_new
        # print(NP_vec_tmp_new)
        # exit(0)

    print("collected_num: ",collected_num)
    end_time = time.time()


    # start to get the A and B for the plane
    # print(NP_vec)

    NP_vec_tmp = NP_vec[0]
    # print(NP_vec_tmp)
    # exit(0)
    # print("NP_vec: ", NP_vec)
    # print("NP_vec_tmp: ", NP_vec_tmp)

    assert len(NP_vec_tmp) == 2 # we only need two points
    B = -5
    x1= NP_vec_tmp[0][0]
    y1 = NP_vec_tmp[0][1]
    x2 = NP_vec_tmp[1][0]
    y2 = NP_vec_tmp[1][1]
    a = B*(y1-y2)/(x2*y1 - x1*y2)
    b = B*(x1-x2)/(y2*x1 - y1*x2)
    A = [a, b]
    # print(A)
    # print(B)
    return [A, B]