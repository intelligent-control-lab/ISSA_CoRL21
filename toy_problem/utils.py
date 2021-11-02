import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import osqp

# auxiliary functions
# dynamics function

def safetyindex(s):
    x = s[0]
    y = s[1]
    theta = s[2]
    x0 = 0.5
    y0 = 1.5
    margin = 0.2
    phi = (1 + 0.5 + margin) ** 2 - ((x0 - x) * np.sin(theta) - (y0 - y) * np.cos(theta)) ** 2
    return phi


# plot cricle
def obj(u, ustar):
    y = np.linalg.norm(u - ustar)
    return y


def circle(x, y, r):
    th = np.linspace(0, 2 * np.pi, 100)
    xunit = r * np.cos(th) + x
    yunit = r * np.sin(th) + y
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(x, y, color="darkred", linewidth=2)

def dphidx(s):
    x0 = 0.5
    y0 = 1.5
    x = s[0]
    y = s[1]
    theta = s[2]
    first_chain = -2 * ((x0-x)*np.sin(theta) - (y0-y)*np.cos(theta))
    dpds = [first_chain*(-np.sin(theta)),
            first_chain*np.cos(theta),
            first_chain*((x0-x)*np.cos(theta) + (y0-y)*np.sin(theta))]
    return dpds

def gx(s):
    # dot s = g(s) * u
    theta = s[2]
    g = [np.cos(theta), 0,
         np.sin(theta), 0,
         0, 1]
    return g


def quadprog(H, f, A=None, b=None,
             initvals=None, verbose=True):

    qp_P = sparse.csc_matrix(H)
    qp_f = np.array(f)
    qp_l = -np.inf * np.ones(len(b))
    qp_A = sparse.csc_matrix(A)
    qp_u = np.array(b)

    # if A is not None:
    #     qp_A = spa.vstack([G, A]).tocsc()
    #     qp_l = np.hstack([l, b])
    #     qp_u = np.hstack([h, b])
    # else:  # no equality constraint
    #     qp_A = G
    #     qp_l = l
    #     qp_u = h
    model = osqp.OSQP()
    model.setup(P=qp_P, q=qp_f,
                A=qp_A, l=qp_l, u=qp_u, verbose=verbose)
    if initvals is not None:
        model.warm_start(x=initvals)
    results = model.solve()
    return results.x, results.info.status











def chk_unsafe(s, point, dt):
    # flag=1 is unsafe, flag=0 is safe

    action = [point[0], point[1]]

    # simulate the action
    s_new = step(s, action, dt)
    dphi = safetyindex(s_new) - safetyindex(s)
    # print("dphi: ", dphi)

    if dphi <= -5*dt:
        flag=0 #safe
    else:
        flag =1 #unsafe

    return flag

def outofbound(limit,p):
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

def step(s, u, dt):
    fx = np.array(s)
    theta = s[2]
    gx = np.array([[np.cos(theta)*dt, 0],
          [np.sin(theta)*dt, 0],
          [0, dt]])
    u = np.array(u).T

    # print(s)
    # print(u)
    # print(dt)
    # print(gx)
    # exit(0)

    s_next = fx + np.dot(gx, u)
    return s_next
