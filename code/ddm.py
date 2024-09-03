import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

n_subs = 15
n_trials = 500
n_steps_max = 1000

decision_thresh = 50
w_evidence = 0.1
w_ic_control = [0, 0.1]
w_ic_motor = [0, 0.001]
effector = ["same", "different"]

a = np.zeros((n_trials, n_steps_max))
b = np.zeros((n_trials, n_steps_max))
c = np.zeros((n_trials, n_steps_max))
d = np.zeros((n_trials, n_steps_max))

dd = {
    "sub": [],
    "trial": [],
    "switch_trial": [],
    "cue": [],
    "cat": [],
    "resp": [],
    "rt": [],
    "effector": [],
    "wic": [],
    "wmc": [],
}

for wic in w_ic_control:
    for wmc in w_ic_motor:
        for eff in effector:
            for s in range(n_subs):
                print(s)

                for i in range(n_trials):
                    rt = -1

                    switch_trial = True if np.random.uniform(0, 1) > 0.5 else False
                    cue = 1 if np.random.uniform(0, 1) > 0.5 else 2
                    cat = 1 if np.random.uniform(0, 1) > 0.5 else 2

                    # assume the categories are well learned
                    if cat == 1:
                        evidence_a = 0.8
                        evidence_b = 0.2
                        evidence_c = 0.8
                        evidence_d = 0.2
                    else:
                        evidence_a = 0.2
                        evidence_b = 0.8
                        evidence_c = 0.2
                        evidence_d = 0.8

                    # assume ic_control is weaker after switches
                    if switch_trial == True:
                        if cue == 1:
                            ic_a = 0.6
                            ic_b = 0.6
                            ic_c = 0.05
                            ic_d = 0.05
                        else:
                            ic_a = 0.05
                            ic_b = 0.05
                            ic_c = 0.6
                            ic_d = 0.6
                    else:
                        if cue == 1:
                            ic_a = 0.8
                            ic_b = 0.8
                            ic_c = 0.01
                            ic_d = 0.01
                        else:
                            ic_a = 0.01
                            ic_b = 0.01
                            ic_c = 0.8
                            ic_d = 0.8

                    for j in range(1, n_steps_max):
                        a[i, j] = a[i, j - 1]
                        b[i, j] = b[i, j - 1]
                        c[i, j] = c[i, j - 1]
                        d[i, j] = d[i, j - 1]

                        a[i, j] += w_evidence * np.random.normal(evidence_a, 1)
                        b[i, j] += w_evidence * np.random.normal(evidence_b, 1)
                        c[i, j] += w_evidence * np.random.normal(evidence_c, 1)
                        d[i, j] += w_evidence * np.random.normal(evidence_d, 1)

                        a[i, j] -= wic * ic_a
                        b[i, j] -= wic * ic_b
                        c[i, j] -= wic * ic_c
                        d[i, j] -= wic * ic_d

                        if eff == "same":
                            # TODO: check to make sure never
                            # more than two responses make
                            # it to the response when this
                            # is true.
                            a[i, j] -= wmc * (b[i, j])
                            b[i, j] -= wmc * (a[i, j])
                            c[i, j] -= wmc * (d[i, j])
                            d[i, j] -= wmc * (c[i, j])
                        else:
                            # TODO: I think the key to why
                            # the motor inhibition model
                            # works is here. Two out of the
                            # three terms in each of these
                            # equations will carry markers
                            # from the cue level inhibition.
                            # The signature of cue level
                            # inhibition is not carried
                            # through in the "same"
                            # conditions.
                            a[i, j] -= wmc * (b[i, j] + c[i, j] + d[i, j])
                            b[i, j] -= wmc * (a[i, j] + c[i, j] + d[i, j])
                            c[i, j] -= wmc * (d[i, j] + a[i, j] + b[i, j])
                            d[i, j] -= wmc * (c[i, j] + a[i, j] + b[i, j])

                        process = np.array([a[i, j], b[i, j], c[i, j], d[i, j]])

                        if np.any(process > decision_thresh):
                            rt = j
                            resp = np.where(process > decision_thresh)[0][0]
                            break

                    if rt == -1:
                        rt = n_steps_max

                    dd["sub"].append(s)
                    dd["trial"].append(i)
                    dd["switch_trial"].append(switch_trial)
                    dd["cue"].append(cue)
                    dd["cat"].append(cat)
                    dd["resp"].append(resp)
                    dd["rt"].append(rt)
                    dd["effector"].append(eff)
                    dd["wic"].append(wic)
                    dd["wmc"].append(wmc)

# t = np.arange(0, n_steps_max, 1)
# plt.plot(t, a[0, :], label='a')
# plt.plot(t, b[0, :], label='b')
# plt.plot(t, c[0, :], label='c')
# plt.plot(t, d[0, :], label='d')
# plt.plot([rt, rt], [-decision_thresh, decision_thresh], '--k', alpha=0.5)
# plt.legend()
# plt.show()

dd = pd.DataFrame(dd)


def compute_cost(dd):
    stay = dd.loc[dd["switch_trial"] == False, "rt"].mean()
    switch = dd.loc[dd["switch_trial"] == True, "rt"].mean()
    cost = stay - switch
    dd["cost"] = cost
    return dd


ddd = dd.groupby(["sub", "effector", "wic", "wmc"]).apply(compute_cost)
ddd = ddd[["sub", "effector", "wic", "wmc", "cost"]].drop_duplicates()

ddd["cnd"] = "none"
ddd.loc[
    (ddd["wic"] == 0.0) & (ddd["wmc"] == 0.0), "cnd"
] = "No Cognitive / Motor Control"
ddd.loc[(ddd["wic"] == 0.1) & (ddd["wmc"] == 0.0), "cnd"] = "Only Cognitive Control"
ddd.loc[(ddd["wic"] == 0.0) & (ddd["wmc"] == 0.001), "cnd"] = "Only Motor Control"
ddd.loc[
    (ddd["wic"] == 0.1) & (ddd["wmc"] == 0.001), "cnd"
] = "Both Cognitive / Motor Control"

ddd = ddd.sort_values(["cnd", "effector"])

dddd = ddd.loc[ddd["cnd"] != "none"]

fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5.1, 5.1))
sns.pointplot(
    data=dddd, x="effector", y="cost", hue="cnd", capsize=0, kind="point", ax=ax[0, 0]
)
plt.legend(loc="best")
plt.tight_layout()
# plt.show()
plt.savefig("../figures/model_all_2.pdf")
plt.close()
