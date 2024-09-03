from imports import *
from util_funcs import load_data

d = load_data()
dd = d[['cnd', 'sub', 'x', 'y', 'cat', 'resp', 'cue',
        'trial']].drop_duplicates().reset_index(drop=True)

n_trials = d.trial.max()

n_steps_max = 1000

n_channels = dd.cat.unique().size

rt_max = 3000

# NOTE: rb system parameters
n_rules = 4

# cues are used to direct cognitive control to select the right partition of
# knowledge
cue = -1
n_cues = 2
rule_ind = 0
current_rule = 0

# activation arrays
act_rb = np.zeros((n_cues, n_rules, n_channels))
act_ii = np.random.uniform(0.1, 0.2, (n_cues, n_channels))

# xc = np.random.normal(50, 15)
# yc = np.random.normal(50, 15)
xc = 30
yc = 30

# NOTE: ii system parameters
vis_dim = 100
vis_width = 35

w_ltp_ii = 1e-1
w_ltd_ii = 1e-1
w_ltp_rb = 5e-2
w_ltd_rb = 5e-2

# NOTE: total motor output parameters
motor_noise_sd = 0.1
resp = np.zeros(n_channels)

# NOTE: parameters to control strength of cognitive and motor control (lateral
# inhibition)
w_evidence_ii = 1e1 * 1
w_evidence_rb = 1e-2 * 0
w_ii_lateral = 1e-4 * 0
w_rb_lateral = 1e-4 * 0
w_rule_selection = 1e-4 * 0
w_cognitive_control_ii = 1e-5 * 0
w_cognitive_control_rb = 1e-5 * 0
w_cognitive_control = 1e-4
w_motor_control = 1e-1
decision_thresh = 100

# NOTE: learning parameters
r = 0
delta = 0
pr_rb = 0
pr_ii = 0
pr_alpha = 0.01

cat = 0
x = 0
y = 0
resp = 0

# NOTE: records
cue_rec = np.zeros(n_trials)
acc_rec = np.zeros(n_trials)
rt_rec = np.zeros(n_trials)
rule_rec = np.zeros(n_trials)
w_cue_rule_rec = np.zeros((n_trials, n_cues, n_rules))
conf_ii_rec = np.zeros(n_trials)
conf_rb_rec = np.zeros(n_trials)
act_resp = np.zeros((n_trials, n_steps_max, n_channels))
w_vis_msn = np.random.normal(0.1, 0.01, (vis_dim**2, n_channels, n_cues))
vis_act = np.zeros(vis_dim**2)
w_cue_rule = np.random.uniform(0.1, 0.3, (n_cues, n_rules))
act_cue = np.zeros((n_cues, 1))
