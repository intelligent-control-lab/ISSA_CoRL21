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


def chk_unsafe(s, point, dt_ratio, dt_adamba, env, threshold, margin, adaptive_k, adaptive_n, adaptive_sigma, trigger_by_pre_execute, pre_execute_coef):
    action = point.tolist()
    # save state of env
    stored_state = copy.deepcopy(env.sim.get_state())
    safe_index_now = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

    # simulate the action
    s_new = env.step(action, simulate_in_adamba=True)

    safe_index_future = env.adaptive_safety_index(k=adaptive_k, sigma=adaptive_sigma, n=adaptive_n)

    dphi = safe_index_future - safe_index_now

    if trigger_by_pre_execute:
        if safe_index_future < pre_execute_coef:
            flag = 0  # safe
        else:
            flag = 1  # unsafe
    else:
        if dphi <= threshold * dt_adamba: #here dt_adamba = dt_env
            flag = 0  # safe
        else:
            flag = 1  # unsafe

    # set qpos and qvel
    env.sim.set_state(stored_state)
    
    # Note that the position-dependent stages of the computation must have been executed for the current state in order for these functions to return correct results. So to be safe, do mj_forward and then mj_jac. If you do mj_step and then call mj_jac, the Jacobians will correspond to the state before the integration of positions and velocities took place.
    env.sim.forward()
    return flag, env


def outofbound(action_limit, action):
    flag = False
    for i in range(len(action_limit)):
        assert action_limit[i][1] > action_limit[i][0]
        if action[i] < action_limit[i][0] or action[i] > action_limit[i][1]:
            flag = True
            break
    return flag



def AdamBA_SC(s, u, env, threshold=0, dt_ratio=1.0, ctrlrange=10.0, margin=0.4, adaptive_k=3, adaptive_n=1, adaptive_sigma=0.04, trigger_by_pre_execute=False, pre_execute_coef=0.0, vec_num=None, max_trial_num =1):
    
    # start_time = time.time()
    infSet = []

    u = np.clip(u, -ctrlrange, ctrlrange)

    #action_space_num = 2
    action_space_num = env.action_space.shape[0]
    action = np.array(u).reshape(-1, action_space_num)

    dt_adamba = env.model.opt.timestep * env.frameskip_binom_n * dt_ratio

    assert dt_ratio == 1

    limits= [[-ctrlrange, ctrlrange]] * action_space_num
    NP = action

    # generate direction
    NP_vec_dir = []
    NP_vec = []
   
    #sigma_vec = [[1, 0], [0, 1]]
    # vec_num = 10 if action_space_num==2 else 
    if action_space_num ==2:
        vec_num = 10 if vec_num == None else vec_num
    elif action_space_num == 12:
        vec_num = 20 if vec_num == None else vec_num
    else:
        raise NotImplementedError

    loc = 0 
    scale = 0.1
    
    # num of actions input, default as 1
    for t in range(0, NP.shape[0]):
        if action_space_num == 2:
            vec_set = []
            vec_dir_set = []
            for m in range(0, vec_num):
                # vec_dir = np.random.multivariate_normal(mean=[0, 0], cov=sigma_vec)
                theta_m = m * (2 * np.pi / vec_num)
                vec_dir = np.array([np.sin(theta_m), np.cos(theta_m)]) / 2
                #vec_dir = vec_dir / np.linalg.norm(vec_dir)
                vec_dir_set.append(vec_dir)
                vec = NP[t]
                vec_set.append(vec)
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)
        else:
            vec_dir_set = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
            vec_set = [NP[t]] * vec_num
            #import ipdb; ipdb.set_trace()
            NP_vec_dir.append(vec_dir_set)
            NP_vec.append(vec_set)
        

    

    bound = 0.0001

    # record how many boundary points have been found
    # collected_num = 0
    valid = 0
    cnt = 0
    out = 0
    yes = 0
    
    max_trials = max_trial_num
    for n in range(0, NP.shape[0]):
        trial_num = 0
        at_least_1 = False
        while trial_num < max_trials and not at_least_1:
            at_least_1 = False
            trial_num += 1
            NP_vec_tmp = copy.deepcopy(NP_vec[n])
            #print(NP_vec)

            if trial_num ==1:
                NP_vec_dir_tmp = NP_vec_dir[n]
            else:
                NP_vec_dir_tmp = np.random.normal(loc=loc, scale=scale, size=[vec_num, action_space_num])
                #print(NP_vec_dir_tmp)
            # print("NP_vec:\n",NP_vec)
            # print("NP_vec_dir_tmp:\n",NP_vec_dir_tmp)
            


            for v in range(0, vec_num):
                NP_vec_tmp_i = NP_vec_tmp[v]

                NP_vec_dir_tmp_i = NP_vec_dir_tmp[v]

                eta = bound
                decrease_flag = 0
                # print(eta)
                
                while True: 
                    #print(eta)
                    # chk_start_time = time.time()
                    flag, env = chk_unsafe(s, NP_vec_tmp_i, dt_ratio=dt_ratio, dt_adamba=dt_adamba, env=env,
                                        threshold=threshold, margin=margin, adaptive_k=adaptive_k, adaptive_n=adaptive_n, adaptive_sigma=adaptive_sigma,
                                        trigger_by_pre_execute=trigger_by_pre_execute, pre_execute_coef=pre_execute_coef)

                    # safety gym env itself has clip operation inside
                    if outofbound(limits, NP_vec_tmp_i):
                        # print("\nout\n")
                        # collected_num = collected_num - 1  # not found, discard the recorded number
                        break

                    if eta <= bound and flag==0:
                        at_least_1 = True
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
        # print("NP_vec_tmp: ",NP_vec_tmp)
        # exit(0)

        # print(u)
        # print(NP_vec_tmp)
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

    NP_vec_tmp = NP_vec[0]

    if valid > 0:
        valid_adamba_sc = "adamba_sc success"
    elif valid == 0 and yes==vec_num:
        valid_adamba_sc = "itself satisfy"
    elif valid == 0 and out==vec_num:
        valid_adamba_sc = "all out"
    else:
        valid_adamba_sc = "exception"
        print("out = ", out)
        print("yes = ", yes)
        print("valid = ", valid)

    if len(NP_vec_tmp) > 0:  # at least we have one sampled action satisfying the safety index 
        norm_list = np.linalg.norm(NP_vec_tmp, axis=1)
        optimal_action_index = np.where(norm_list == np.amin(norm_list))[0][0]
        return NP_vec_tmp[optimal_action_index], valid_adamba_sc, env, NP_vec_tmp
    else:
        return None, valid_adamba_sc, env, None
