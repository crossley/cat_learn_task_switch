import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import differential_evolution, LinearConstraint


def fit():
    # decision_thresh = 500
    # w_evidence = 0.8
    # w_cognitive_control_switch = 0.6
    # w_cognitive_control_stay = 0.8
    # w_motor_control = 0.1

    bounds = ((1, 1000), (0, 1), (0, 1), (0, 1), (0, 1))

    args = []

    results = differential_evolution(
        func=obj_func,
        bounds=bounds,
        args=args,
        disp=True,
        maxiter=100,
        tol=1e2,
        updating="deferred",
        workers=-1,
    )

    return results


def obj_func(params, *args):
    # [diff, same]
    x_obs = np.array([20, 10])
    x_pred = simulate(params, args)
    x_pred = x_pred.groupby(["effector"])["cost"].mean().to_numpy()
    sse = np.sum((x_obs - x_pred) ** 2)
    return sse


def simulate(params, *args):
    n_subs = 10
    n_trials = 10
    n_steps_max = 1000

    decision_thresh = params[0]
    w_evidence = params[1]
    w_cognitive_control_switch = params[2]
    w_cognitive_control_stay = params[3]
    w_motor_control = params[4]

    dd = {
        "effector": [],
        "sub": [],
        "trial": [],
        "switch_trial": [],
        "cue": [],
        "cat": [],
        "resp": [],
        "rt": [],
    }

    for e in ["same", "diff"]:
        for s in range(n_subs):
            x = np.zeros((4, n_trials, n_steps_max))
            evidence = np.zeros((4, n_trials, n_steps_max))
            cognitive_control = np.zeros((4, n_trials, n_steps_max))
            motor_control = np.zeros((4, n_trials, n_steps_max))

            for i in range(n_trials):
                rt = -1
                for j in range(1, n_steps_max):
                    switch_trial = True if np.random.uniform(0, 1) > 0.5 else False
                    cue = 1 if np.random.uniform(0, 1) > 0.5 else 2
                    cat = 1 if np.random.uniform(0, 1) > 0.5 else 2

                    # assume the categories are well learned
                    if cat == 1:
                        evidence[:, i, j] = [
                            w_evidence,
                            1 - w_evidence,
                            w_evidence,
                            1 - w_evidence,
                        ]
                    else:
                        evidence[:, i, j] = [
                            1 - w_evidence,
                            w_evidence,
                            1 - w_evidence,
                            w_evidence,
                        ]

                    # assume cognitive_control is weaker after switches
                    if switch_trial == True:
                        if cue == 1:
                            cognitive_control[:, i, j] = [
                                w_cognitive_control_switch,
                                w_cognitive_control_switch,
                                1 - w_cognitive_control_switch,
                                1 - w_cognitive_control_switch,
                            ]
                        else:
                            cognitive_control[:, i, j] = [
                                1 - w_cognitive_control_switch,
                                1 - w_cognitive_control_switch,
                                w_cognitive_control_switch,
                                w_cognitive_control_switch,
                            ]
                    else:
                        if cue == 1:
                            cognitive_control[:, i, j] = [
                                w_cognitive_control_stay,
                                w_cognitive_control_stay,
                                1 - w_cognitive_control_stay,
                                1 - w_cognitive_control_stay,
                            ]
                        else:
                            cognitive_control[:, i, j] = [
                                1 - w_cognitive_control_stay,
                                1 - w_cognitive_control_stay,
                                w_cognitive_control_stay,
                                w_cognitive_control_stay,
                            ]

                    # motor control depends on the number of effectors
                    if e == "same":
                        motor_control[0, i, j] = x[1, i, j]
                        motor_control[1, i, j] = x[0, i, j]
                        motor_control[2, i, j] = x[3, i, j]
                        motor_control[3, i, j] = x[2, i, j]
                    elif e == "diff":
                        motor_control[0, i, j] = x[[1, 2, 3], i, j].sum()
                        motor_control[1, i, j] = x[[0, 2, 3], i, j].sum()
                        motor_control[2, i, j] = x[[0, 1, 3], i, j].sum()
                        motor_control[3, i, j] = x[[0, 1, 2], i, j].sum()

                    motor_control[:, i, j] *= w_motor_control

                    # overall state is sum of components
                    xe = evidence[:, i, j]
                    xcc = cognitive_control[:, i, j]
                    xmc = motor_control[:, i, j]
                    x[:, i, j] = x[:, i, j - 1] + xe + xcc + xmc

                    if np.any(x[:, i, j] > decision_thresh):
                        rt = j
                        resp = np.where(x[:, i, j] > decision_thresh)[0][0]
                        break

                if rt == -1:
                    rt = n_steps_max
                    resp = np.argmax(x[:, i, j])

                dd["effector"].append(e)
                dd["sub"].append(s)
                dd["trial"].append(i)
                dd["switch_trial"].append(switch_trial)
                dd["cue"].append(cue)
                dd["cat"].append(cat)
                dd["resp"].append(resp)
                dd["rt"].append(rt)

    dd = pd.DataFrame(dd)
    dd = dd.groupby(["sub", "effector"]).apply(compute_cost)
    dd = dd[["sub", "effector", "cost"]].drop_duplicates()

    return dd


def compute_cost(dd):
    stay = dd.loc[dd["switch_trial"] == False, "rt"].mean()
    switch = dd.loc[dd["switch_trial"] == True, "rt"].mean()
    cost = stay - switch
    dd["cost"] = cost
    return dd


results = fit()
d = pd.DataFrame({"x": results["x"], "fun": results["fun"]})
d.to_csv("../fits/fit.csv", index=False)

p = pd.read_csv("../fits/fit.csv")
params = p["x"].to_numpy()

dd = simulate(params, [])

sns.catplot(data=dd, x="effector", y="cost", capsize=0, kind="point")
plt.show()
