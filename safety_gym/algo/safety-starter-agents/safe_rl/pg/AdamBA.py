import numpy as np
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import osqp
import copy


from gym.envs.robotics.utils import reset_mocap2body_xpos

import safe_rl.pg.run_agent



def quadprog(H, f, A=None, b=None,
             initvals=None, verbose=False):
    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)

    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status


def chk_unsafe(s, point, dt_ratio, dt_adamba, env, threshold):
    safe_rl.pg.run_agent.CHECK_CNT += 1

    action = [point[0], point[1]]

    # save state of env
    stored_state = copy.deepcopy(env.sim.get_state())


    stored_robot_position = env.robot_pos
    mujoco_id = env.sim.model.body_name2id('robot')
    stored_robot_body_jacp = copy.deepcopy(env.sim.data.body_jacp[mujoco_id])

    cost_now = env.cost()['cost']
    projection_cost_now = env.projection_cost()

    # simulate the action

    s_new = env.step(action, dt_ratio, simulate_in_adamba=True)
    vel_after_tmp_action = env.sim.data.get_body_xvelp('robot')

    cost_future = env.cost()['cost']
    projection_cost_future = env.projection_cost()
    #dphi = cost_future - cost_now
    # projection_dphi
    dphi = projection_cost_future - projection_cost_now

    if dphi <= threshold * dt_adamba:
        flag = 0  # safe
    else:
        flag = 1  # unsafe

    # reset env

    # set qpos and qvel
    env.sim.set_state(stored_state)
    
    # Note that the position-dependent stages of the computation must have been executed for the current state in order for these functions to return correct results. So to be safe, do mj_forward and then mj_jac. If you do mj_step and then call mj_jac, the Jacobians will correspond to the state before the integration of positions and velocities took place.
    env.sim.forward()
        
    
    return flag, env


def outofbound(limit, p):
    # limit, dim*2
    # p, dim
    # flag=1 is out of bound
    flag = 0
    assert len(limit[0]) == 2
    for i in range(len(limit)):
        assert limit[i][1] > limit[i][0]
        if p[i] < limit[i][0] or p[i] > limit[i][1]:
            flag = 1
            break
    return flag




def AdamBA(s, u, env, threshold, dt_ratio=1.0, ctrlrange=10.0):
    infSet = []
    u = np.clip(u, -ctrlrange, ctrlrange)
    np.random.seed(0)

    # 2d case, 2 dimensional control signal
    # uniform sampling
    # offset = [0.5 0.5];
    # scale = [0.5 0.5];
    # action = scale. * rand(1, 2) + offset;

    action_space_num = 2
    action = np.array(u).reshape(-1, action_space_num)

    dt_adamba = 0.002 * env.frameskip_binom_n * dt_ratio
    limits = [[-ctrlrange, ctrlrange], [-ctrlrange, ctrlrange]]  # each row define the limits for one dimensional action
    NP = []

    # no need crop since se only need on sample

    # NP = np.clip(action, np.array([[-1,-1]]), np.array([[1, 1]]))
    NP = action
    # print(NP)
    # exit(0)

    start_time = time.time()

    # generate direction
    NP_vec_dir = []
    NP_vec = []
    sigma_vec = [[1, 0], [0, 1]]
    vec_num = 100

    # num of actions input, default as 1
    for t in range(0, NP.shape[0]):
        vec_set = []
        vec_dir_set = []
        for m in range(0, vec_num):
            vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
            vec_dir = vec_dir / np.linalg.norm(vec_dir)
            vec_dir_set.append(vec_dir)
            vec = NP[t]
            vec_set.append(vec)
        NP_vec_dir.append(vec_dir_set)
        NP_vec.append(vec_set)

    bound = 0.0001

    # record how many boundary points have been found
    collected_num = 0
    valid = 0
    cnt = 0
    out = 0
    yes = 0
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
            eta = bound
            decrease_flag = 0
            # print(eta)
            
            while eta >= bound:

                flag, env = chk_unsafe(s, NP_vec_tmp_i, dt_ratio=dt_ratio, dt_adamba=dt_adamba, env=env,
                                       threshold=threshold)

                # safety gym env itself has clip operation inside
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

        NP_vec_tmp_new = []

        for vnum in range(0, len(NP_vec_tmp)):

            cnt += 1
            if outofbound(limits, NP_vec_tmp[vnum]):
                # print("out")
                out += 1
                continue
            if NP_vec_tmp[vnum][0] == u[0][0] and NP_vec_tmp[vnum][1] == u[0][1]:
                # print("yes")
                yes += 1
                continue

            valid += 1
            NP_vec_tmp_new.append(NP_vec_tmp[vnum])
            
        NP_vec[n] = NP_vec_tmp_new

    end_time = time.time()

    # start to get the A and B for the plane
    NP_vec_tmp = NP_vec[0]

    if valid == 2:
        valid_adamba = "adamba success"
    elif valid == 0 and yes==100:
        valid_adamba = "itself satisfy"
    elif valid == 0 and out==100:
        valid_adamba = "all out"
    elif valid == 1:
        valid_adamba = "one valid"
    else:
        valid_adamba = "exception"
        print("out = ", out)
        print("yes = ", yes)
        print("valid = ", valid)
        
    if len(NP_vec_tmp) == 2:  # we only need two points
        B = threshold
        x1 = NP_vec_tmp[0][0]
        y1 = NP_vec_tmp[0][1]
        x2 = NP_vec_tmp[1][0]
        y2 = NP_vec_tmp[1][1]
        a = B * (y1 - y2) / (x2 * y1 - x1 * y2)
        b = B * (x1 - x2) / (y2 * x1 - y1 * x2)
        A = [a, b]

        return [A, B], valid_adamba
    else:
        return [None, None], valid_adamba
