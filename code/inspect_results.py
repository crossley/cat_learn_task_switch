from imports import *
from util_funcs import *

block_size = 28
d = load_data(block_size=block_size)

print(d.groupby(["attention", "effector", "memory"])["cnd"].unique())

# NOTE: compute switch costs across all trials
d = d.groupby(['effector', 'attention', 'memory', 'cnd',
               'sub']).apply(compute_switch_cost).reset_index(drop=True)

d = d.groupby(['effector', 'attention', 'memory', 'cnd', 'sub'
               ]).apply(compute_switch_cost_by_type).reset_index(drop=True)

# NOTE: compute mean by block and fit tanh model to learning curve
dd = d.groupby(
    ["effector", "attention", "memory", "cnd", "sub", "block",
     "cue"])["acc"].mean().reset_index(drop=False)

dd = dd.groupby(["effector", "attention", "memory", "cnd", "sub",
                 "cue"]).apply(fit_func, "acc",
                               tanh_func).reset_index(drop=True)

# NOTE assess tanh fit quality to learning curves
y_obs = dd.groupby(['effector', 'attention', 'memory', 'block'])['acc'].mean()
y_pred = dd.groupby(['effector', 'attention', 'memory',
                     'block'])['acc_pred'].mean()
_, _, r, _, _ = linregress(y_obs, y_pred)
r2 = r**2
print(r2.round(2))

#
# NOTE: figures
#

# generate figure 1
inspect_cats(d)

# generate figure 3
inspect_func_fits(dd)

# generate figure 4
inspect_interaction_threeway(dd)

# generate figure 5
inspect_interaction_switch_costs_iirb(d)

# generate figure 6
inspect_interaction_switch_costs(d)

# generate figure 7
inspect_cats_3(d)

#
# NOTE: stats
#

inspect_interaction(dd, 'fit_a')
inspect_interaction(dd, 'fit_b')
inspect_interaction(dd, 'fit_c')
inspect_interaction(dd, 'fit_ac')
inspect_interaction(d, 'switch_cost_acc')
inspect_interaction(d, 'switch_cost_rt')
inspect_interaction(d, 'switch_cost_acc_cue_diff')
inspect_interaction(d, 'switch_cost_rt_cue_diff')


#
# NOTE: decision bound model analysis
#

report_dbm_results()
