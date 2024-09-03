from imports import *
from util_funcs import tanh_func
from model_init import *


def reset_buffers():
    global cue_rec, acc_rec, rt_rec, rule_rec, w_cue_rule_rec, conf_ii_rec
    global conf_rb_rec, act_resp, w_vis_msn, vis_act, w_cue_rule, act_cue

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


def simulate_model(cnd):

    dcnd = dd.loc[(dd['cnd'] == cnd)].reset_index()

    for sub in dcnd['sub'].unique():
        dsub = dcnd.loc[dcnd['sub'] == sub]

    cat_arr = dsub['cat'].to_numpy()
    x_arr = dsub['x'].to_numpy()
    y_arr = dsub['y'].to_numpy()
    cue_arr = dsub['cue'].to_numpy()

    switch_trial_arr = np.diff(cue_arr, prepend=cue_arr[0])
    switch_trial_arr = switch_trial_arr != 0

    reset_buffers()

    for i in range(n_trials):

        print('trial ' + str(i))

        # init / reset trial
        act_cue = np.zeros((n_cues, 1))
        vis_act = np.zeros(vis_dim**2)
        r = 0
        delta = 0
        resp = -1
        rt = -1

        # current stimulus
        cat = cat_arr[i]
        x = x_arr[i]
        y = y_arr[i]
        cue = cue_arr[i]
        switch_trial = switch_trial_arr[i]

        # get rb activation
        act_rb = get_act_rb(cat, x, y)

        # select current rule
        act_cue[cue, 0] = 1
        current_rule = act_cue * w_cue_rule
        current_rule = current_rule / np.linalg.norm(current_rule)
        current_rule = np.argmax(current_rule[cue, :])

        # inhibit non-selected rules
        act_rb[:, ~np.isin(np.arange(0, n_rules, 1), current_rule
                           ), :] *= w_rule_selection

        # inhibit non-selected contexts (apply cognitive control to rb)
        act_rb[~np.isin(np.arange(0, n_cues, 1), cue
                        ), :, :] *= w_cognitive_control_rb

        # inhibit non-selected responses (apply lateral inhibition to rb)
        rb_ind = act_rb[cue, current_rule, :] == act_rb[cue,
                                                        current_rule, :].max()
        act_rb[cue, current_rule, ~rb_ind] *= w_rb_lateral

        # get ii activation
        act_ii = get_act_ii(cat, x, y)

        # inhibit non-selected contexts (apply cognitive control to ii)
        act_ii[~np.isin(np.arange(0, n_cues, 1), cue
                        ), :] *= w_cognitive_control_ii

        # add noise to II
        act_ii += np.random.normal(0, 0.002, act_ii.shape)

        # inhibit non-selected responses (apply lateral inhibition to ii)
        ii_ind = act_ii[cue, :] == act_ii[cue, :].max()
        act_ii[cue, ~ii_ind] *= w_ii_lateral

        # NOTE: Not elegant. Assume the winning response is activated
        # simultaneously in both contexts.
        # act_ii[~np.isin(np.arange(0, n_cues, 1), cue), :] = act_ii[cue, :]

        # confidence will be the cumulative evidence of each process (see
        # below)
        conf_ii = 0
        conf_rb = 0

        # # compute response via diffusion process
        # act_resp = np.zeros((n_trials, n_steps_max, n_channels))
        # for j in range(1, n_steps_max):
        #     for k in range(n_channels):

        #         # carry over old evidence
        #         act_resp[i, j, k] += act_resp[i, j - 1, k]

        #         for l in range(n_cues):

        #             # separate diffusion processes for rb and ii systems that
        #             # jointly feed into a joint response pool
        #             evidence_sd = 0.1

        #             evidence_ii = np.random.normal(
        #                 w_evidence_ii * act_ii[l, k], evidence_sd)

        #             evidence_rb = np.random.normal(
        #                 w_evidence_rb * act_rb[l, current_rule, k],
        #                 evidence_sd)

        #             # joint evidence is simply the average of ii and rb
        #             evidence = (evidence_ii + evidence_rb) / 2

        #             # track confidence
        #             conf_ii += evidence_ii
        #             conf_rb += evidence_rb

        #             # add new evidence to diffusion process
        #             act_resp[i, j, k] += evidence

        #             # NOTE: Core assumptions about cognitive control embedded
        #             # here. We assume that cognitive control is inhibitory in
        #             # nature, and that it inhibits non-active contexts (cues /
        #             # response sets) more than active contexts. We further
        #             # assume that it does this job more effectively on stay
        #             # trials than on switch trials.
        #             if switch_trial == True:
        #                 if l == cue:
        #                     cc = 0.05
        #                 else:
        #                     cc = 0.6
        #             else:
        #                 if l == cue:
        #                     cc = 0.01
        #                 else:
        #                     cc = 0.8

        #             act_resp[i, j, k] -= w_cognitive_control * cc

        #             # NOTE: Motor control is really just lateral inhibition
        #             # between all response options. The 4-response conditions
        #             # have greater cost because there are more non-zero sources
        #             # in this term.
        #             mc_ind = ~np.isin(np.arange(0, n_channels, 1), k)
        #             mc = np.sum(act_resp[i, j, mc_ind])
        #             act_resp[i, j, k] -= w_motor_control * mc

        #     if act_resp[i, j, :].max() > decision_thresh:
        #         rt = j
        #         resp = np.argmax(act_resp[i, j, :])
        #         break

        #     if j == (n_steps_max - 1):
        #         rt = j
        #         resp = np.argmax(act_resp[i, j, :])
        #         break

        # # prepare for rb / ii update
        # act_rb[cue, current_rule, :] = act_resp[i, rt, :] + act_resp[
        #     i, rt, :].min()

        # act_ii = np.zeros((n_cues, n_channels))
        # act_ii[cue, resp] = act_resp[i, rt, resp]

        resp = np.argmax(act_ii[cue, :])
        rt = 1

        act_ii *= 50

        # update rb / ii
        update_rb(cat, resp, act_cue, act_rb)
        update_ii(cat, resp, act_ii)

        # record trial
        cue_rec[i] = cue
        acc_rec[i] = (cat == resp).astype(np.int)
        rt_rec[i] = rt
        rule_rec[i] = np.argmax(current_rule)
        w_cue_rule_rec[i, :, :] = copy.deepcopy(w_cue_rule)
        conf_ii_rec[i] = conf_ii
        conf_rb_rec[i] = conf_rb

    # compute switch costs
    acc_switch = np.mean(acc_rec[switch_trial_arr])
    acc_stay = np.mean(acc_rec[~switch_trial_arr])
    switch_cost_acc = acc_switch - acc_stay

    rt_switch = np.mean(rt_rec[switch_trial_arr])
    rt_stay = np.mean(rt_rec[~switch_trial_arr])
    switch_cost_rt = rt_switch - rt_stay

    # # plot diffusion process
    # fig, ax = plt.subplots(1, 1, squeeze=False)
    # for k in range(n_channels):
    #     ax[0, 0].plot(act_resp[i, :, k], label=str(k))
    # ax[0, 0].legend()
    # plt.show()

    # plot trial
    # plot_trial(dsub)

    # NOTE: plot for DP proposal
    block_size = 28
    n_blocks = n_trials / block_size
    block = np.repeat(np.arange(0, n_blocks, 1), block_size)
    dp = pd.DataFrame({'cue': cue_rec, 'acc': acc_rec})
    dp['block'] = block

    x0 = dp.loc[dp['cue'] == 0].block.to_numpy()
    y0 = dp.loc[dp['cue'] == 0].acc.to_numpy()
    ppopt, pcov = curve_fit(tanh_func, x0, y0, maxfev=1e5, bounds=(0, 1))
    acc_pred_0 = tanh_func(x0, *ppopt)

    x1 = dp.loc[dp['cue'] == 1].block.to_numpy()
    y1 = dp.loc[dp['cue'] == 1].acc.to_numpy()
    ppopt, pcov = curve_fit(tanh_func, x1, y1, maxfev=1e5, bounds=(0, 1))
    acc_pred_1 = tanh_func(x1, *ppopt)

    tmp = dp.groupby(['block', 'cue'])['acc'].mean().reset_index()
    x0 = tmp.loc[tmp['cue'] == 0, 'block']
    acc_pred_0 = tmp.loc[tmp['cue'] == 0, 'acc']
    x1 = tmp.loc[tmp['cue'] == 1, 'block']
    acc_pred_1 = tmp.loc[tmp['cue'] == 1, 'acc']

    sos = signal.butter(2, 0.4, output='sos')
    acc_pred_0 = signal.sosfiltfilt(sos, acc_pred_0)
    acc_pred_1 = signal.sosfiltfilt(sos, acc_pred_1)

    x0 = x0 + 1
    x1 = x1 + 1

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(3, 3))
    plt.plot(x0, acc_pred_0, '-r')
    plt.plot(x1, acc_pred_1, '-b')
    ax[0, 0].set_xlabel('')
    ax[0, 0].set_ylabel('')
    ax[0, 0].set_ylim([0, 1])
    ax[0, 0].set_xlim([0, 21])
    # ax[0, 0].set_xticks([5, 10, 15, 20])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    fig.savefig('fig_6_.pdf', transparent=True)
    plt.close()

    # return results
    res = pd.DataFrame({
        'cnd': cnd,
        'cue_rec': [cue_rec],
        'acc_rec': [acc_rec],
        'switch_cost_acc': switch_cost_acc,
        'switch_cost_rt': switch_cost_rt
    })

    return res


def update_rb(cat, resp, act_cue, act_rb):
    global w_cue_rule, pr_rb, r, delta

    # NOTE: For now, assume the ideal criterion is known to the system (hence
    # no criterion learning code here), but in general, the criterion for each
    # rule will have to be learned... interaction between criterion and
    # selection / switching is a really cool system to explore. In essence,
    # this is the credit assignment again. E.g., how do you know whether an
    # error is attributable to the selection of an incorrect rule vs an
    # incorrect criterion (i.e., wrong criterion). Very similar feeling to the
    # credit assignment problem between action selection and action
    # execution... just the cognitive version.

    resp_rb = resp

    r = 1 if cat == resp_rb else -1
    delta = r - pr_rb
    pr_rb += pr_alpha * delta

    for ii in range(n_cues):
        for jj in range(n_rules):
            for kk in range(n_cues):
                if delta >= 0:
                    w_cue_rule[ii, jj] += w_ltp_rb * act_cue[ii, 0] * act_rb[
                        kk, jj, :].sum() * delta
                else:
                    w_cue_rule[ii, jj] += w_ltd_rb * act_cue[ii, 0] * act_rb[
                        kk, jj, :].sum() * delta

    np.clip(w_cue_rule, 0.01, 100, out=w_cue_rule)

    # NOTE: add noise to have fair guessing when weights get slammed to 0.01
    w_cue_rule += np.random.uniform(0.01, 0.03, (n_cues, n_rules))


def update_ii(cat, resp, act_ii):
    global w_vis_msn, vis_act, pr_ii, r, delta

    resp_ii = resp

    r = 1 if cat == resp_ii else -1
    delta = r - pr_ii
    pr_ii += pr_alpha * delta

    for ii in range(vis_dim**2):
        for jj in range(n_channels):
            for kk in range(n_cues):
                if delta >= 0:
                    w_vis_msn[ii, jj, kk] += w_ltp_ii * vis_act[ii] * act_ii[
                        kk, jj] * delta

                else:
                    w_vis_msn[ii, jj, kk] += w_ltd_ii * vis_act[ii] * act_ii[
                        kk, jj] * delta

    np.clip(w_vis_msn, 0.01, 1, out=w_vis_msn)


def get_act_rb(cat, x, y):
    global act_rb, current_rule

    # reset act_rb before each trial
    act_rb = np.random.uniform(1, 10, (n_cues, n_rules, n_channels))

    eps = np.random.normal(0, 10)

    # TODO: All possible channel x cue mappings aren't represented here
    # TODO: manually inspect the data and find out which cue values correspond
    # to which response channels

    # uxa
    discrim = x - xc + eps
    if discrim < 0:
        act_rb[0, 0, 0] = np.abs(discrim)
        act_rb[1, 0, 2] = np.abs(discrim)
    else:
        act_rb[0, 0, 1] = np.abs(discrim)
        act_rb[1, 0, 3] = np.abs(discrim)

    # uxb
    discrim = x - xc + eps
    if discrim >= 0:
        act_rb[0, 1, 0] = np.abs(discrim)
        act_rb[1, 1, 2] = np.abs(discrim)
    else:
        act_rb[0, 1, 1] = np.abs(discrim)
        act_rb[1, 1, 3] = np.abs(discrim)

    # uya
    discrim = y - yc + eps
    if discrim < 0:
        act_rb[0, 2, 0] = np.abs(discrim)
        act_rb[1, 2, 2] = np.abs(discrim)
    else:
        act_rb[0, 2, 1] = np.abs(discrim)
        act_rb[1, 2, 3] = np.abs(discrim)

    # uyb
    discrim = y - yc + eps
    if discrim >= 0:
        act_rb[0, 3, 0] = np.abs(discrim)
        act_rb[1, 3, 2] = np.abs(discrim)
    else:
        act_rb[0, 3, 1] = np.abs(discrim)
        act_rb[1, 3, 3] = np.abs(discrim)

    return act_rb


def get_act_ii(cat, x, y):
    global act_ii, vis_act

    xx, yy = np.mgrid[0:vis_dim:1, 0:vis_dim:1]
    pos = np.dstack((xx, yy))
    rf = multivariate_normal([x, y], [[vis_width, 0], [0, vis_width]])
    vis_act = rf.pdf(pos).flatten()

    # make sure previous trial is cleared
    act_ii = np.random.uniform(0.1, 0.2, (n_cues, n_channels))

    for i in range(n_channels):
        for j in range(n_cues):
            act_ii[j, i] = np.inner(w_vis_msn[:, i, j], vis_act)

    return act_ii


def plot_trial(dcnd):

    fig = plt.figure(constrained_layout=True)
    grid = gridspec.GridSpec(ncols=4, nrows=4, figure=fig)

    # ax = fig.add_subplot(grid[0, 0], aspect=1)
    # sns.scatterplot(data=dcnd.loc[dcnd['cue'] == 0],
    #                 x='x',
    #                 y='y',
    #                 hue='cat',
    #                 ax=ax)
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)

    # ax = fig.add_subplot(grid[0, 1], aspect=1)
    # sns.scatterplot(data=dcnd.loc[dcnd['cue'] == 1],
    #                 x='x',
    #                 y='y',
    #                 hue='cat',
    #                 ax=ax)
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)

    # ax = fig.add_subplot(grid[0, 2])
    # ax.imshow(np.reshape(vis_act, (vis_dim, vis_dim)))
    # ax.set_xlim(0, 100)
    # ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[0, 0:2])
    ax.plot(conf_ii_rec, label='II')
    ax.plot(conf_rb_rec, label='RB')
    ax.legend()

    ax = fig.add_subplot(grid[0, 2:])
    block_size = 20
    n_blocks = n_trials / block_size
    block = np.repeat(np.arange(0, n_blocks, 1), block_size)

    ddd = dd.groupby('trial').mean()
    ddd['block'] = block
    ddd['acc_rec'] = acc_rec
    ddd = ddd.groupby('block').mean().reset_index()
    x = ddd.block.to_numpy()
    y = ddd.acc_rec.to_numpy()
    ppopt, pcov = curve_fit(tanh_func, x, y, maxfev=1e5, bounds=(0, 1))
    acc_pred = tanh_func(x, *ppopt)
    ax.plot(acc_pred)
    ax.plot(ddd.acc_rec)

    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(np.reshape(w_vis_msn[:, 0, 0], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(np.reshape(w_vis_msn[:, 1, 0], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[1, 2])
    ax.imshow(np.reshape(w_vis_msn[:, 2, 0], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[1, 3])
    ax.imshow(np.reshape(w_vis_msn[:, 3, 0], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[2, 0])
    ax.imshow(np.reshape(w_vis_msn[:, 0, 1], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[2, 1])
    ax.imshow(np.reshape(w_vis_msn[:, 1, 1], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[2, 2])
    ax.imshow(np.reshape(w_vis_msn[:, 2, 1], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax = fig.add_subplot(grid[2, 3])
    ax.imshow(np.reshape(w_vis_msn[:, 3, 1], (vis_dim, vis_dim)))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    col = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    for i in range(w_cue_rule_rec.shape[1]):
        ax = fig.add_subplot(grid[3, 0:2]) if i == 0 else fig.add_subplot(
            grid[3, 2:])
        fmt = '-' if i == 0 else '--'
        for j in range(w_cue_rule_rec.shape[2]):
            ax.plot(w_cue_rule_rec[:, i, j], fmt, color=col[j], label=str(j))
        ax.legend()

    plt.show()
