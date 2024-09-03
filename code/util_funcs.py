from imports import *


def load_data(block_size):
    d = []
    colnames = ["cat", "x", "y", "cue", "resp", "rt"]
    froot = "../data/all_data/"
    s = 1
    for fn in os.listdir(froot):
        cnd = fn[:5]

        if not np.isin(
                cnd,
                np.array([
                    "cj4ci", "ii2ci", "ii4ci", "ud2su", "cj2su", "cj2ci",
                    "ii2su"
                ]),
        ):
            tmp = pd.read_csv(froot + fn,
                              sep=" ",
                              names=colnames,
                              skipinitialspace=True)

            # rescale stim
            x = tmp["x"].to_numpy()
            y = tmp["y"].to_numpy()

            b = -100 / (np.max(x) / np.min(x) - 1)
            m = (100 - b) / np.max(x)
            tmp["x"] = m * x + b

            b = -100 / (np.max(y) / np.min(y) - 1)
            m = (100 - b) / np.max(y)
            tmp["y"] = m * y + b

            # b = -100 / (range(data$y)[2] / range(data$y)[1] - 1)
            # m = (100 - b) / range(data$y)[2]
            # data[, y := m*y + b]

            # add condition indicator
            tmp["cnd"] = cnd

            # add subject indicator
            tmp["sub"] = s
            s += 1

            # add accuracy
            tmp["acc"] = tmp["cat"] == tmp["resp"]
            tmp["acc"] = tmp["acc"].astype(int)

            # add trial
            n_trials = tmp.shape[0]
            tmp["trial"] = np.arange(1, n_trials + 1, 1)

            # add block
            n_blocks = n_trials // block_size
            tmp["block"] = np.repeat(np.arange(1, n_blocks + 1, 1), block_size)

            tmp["subtask"] = "none"
            st1 = cnd[:2]
            st2 = "".join([x for x in cnd if not x.isdigit()][2:])
            st2 = st2 if st2 != "cr" else st1
            tmp.loc[tmp["cue"] == 0, "subtask"] = st1
            tmp.loc[tmp["cue"] == 1, "subtask"] = st2

            # add memory memory indicator (within, between)
            cnd_between = np.array(["cjii2", "cjii4", "udii2", "udii4"])
            ms = "between" if np.isin(cnd, cnd_between) else "within"
            tmp["memory"] = ms

            # add attention indicator (1d, 2d)
            cnd_1d = np.array(["udcj2", "udcj4", "udii2", "udii4"])
            atn = "1d" if np.isin(cnd, cnd_1d) else "2d"
            tmp["attention"] = atn

            # add effector indicator (same, diff)
            eff_same = np.array(
                ["cjii2", "udcj2", "udii2", "cj2ci", "cj2cr", "ii2cr"])
            eff = "same" if np.isin(cnd, eff_same) else "diff"
            tmp["effector"] = eff

            # add switch indicators
            tmp["switch"] = tmp.groupby([
                "cnd", "sub"
            ])["cue"].transform(lambda x: np.concatenate(([0], np.diff(x))))
            tmp["stay"] = tmp["switch"] == 0

            d.append(tmp)

    d = pd.concat(d)
    d = d[[
        "cnd",
        "sub",
        "attention",
        "memory",
        "effector",
        "block",
        "trial",
        "x",
        "y",
        "cat",
        "resp",
        "acc",
        "rt",
        "cue",
        "subtask",
        "switch",
        "stay",
    ]]

    return d


def compute_switch_cost_following_correct(d):
    # accuracy switch cost following correct feedback
    ind_acc = d.loc[d["acc"] == 1].index
    ind_acc = ind_acc if ind_acc[-1] < d.shape[0] - 1 else ind_acc[:-1]
    dd = d.iloc[ind_acc + 1]
    stay = dd.loc[(dd["stay"] == True), "acc"].mean()
    switch = dd.loc[(dd["stay"] == False), "acc"].mean()
    cost_acc_after_correct = stay - switch
    d["switch_cost_acc_after_correct"] = cost_acc_after_correct

    ind_acc = d.loc[d["acc"] == 0].index
    ind_acc = ind_acc if ind_acc[-1] < d.shape[0] - 1 else ind_acc[:-1]
    dd = d.iloc[ind_acc + 1]
    stay = dd.loc[(dd["stay"] == True), "acc"].mean()
    switch = dd.loc[(dd["stay"] == False), "acc"].mean()
    cost_acc_after_incorrect = stay - switch
    d["switch_cost_acc_after_incorrect"] = cost_acc_after_incorrect

    d["switch_cost_acc_diff"] = cost_acc_after_correct - cost_acc_after_incorrect

    # rt switch cost following correct feedback
    ind_acc = d.loc[d["acc"] == 1].index
    ind_acc = ind_acc if ind_acc[-1] < d.shape[0] - 1 else ind_acc[:-1]
    dd = d.iloc[ind_acc + 1]
    stay = dd.loc[(dd["stay"] == True), "rt"].mean()
    switch = dd.loc[(dd["stay"] == False), "rt"].mean()
    cost_rt_after_correct = stay - switch
    d["switch_cost_rt_after_correct"] = cost_rt_after_correct

    ind_acc = d.loc[d["acc"] == 0].index
    ind_acc = ind_acc if ind_acc[-1] < d.shape[0] - 1 else ind_acc[:-1]
    dd = d.iloc[ind_acc + 1]
    stay = dd.loc[(dd["stay"] == True), "rt"].mean()
    switch = dd.loc[(dd["stay"] == False), "rt"].mean()
    cost_rt_after_incorrect = stay - switch
    d["switch_cost_rt_after_incorrect"] = cost_rt_after_incorrect

    d["switch_cost_rt_diff"] = cost_rt_after_correct - cost_rt_after_incorrect

    return d


def compute_switch_cost_by_type(d):
    # accuracy switch cost by switch type
    dd = d.loc[d["switch"] != 1]
    stay = dd.loc[dd["stay"] == True, "acc"].mean()
    switch = dd.loc[dd["stay"] == False, "acc"].mean()
    switch_cost_acc_10 = stay - switch
    d["switch_cost_acc_10"] = switch_cost_acc_10

    dd = d.loc[d["switch"] != -1]
    stay = dd.loc[dd["stay"] == True, "acc"].mean()
    switch = dd.loc[dd["stay"] == False, "acc"].mean()
    switch_cost_acc_01 = stay - switch
    d["switch_cost_acc_01"] = switch_cost_acc_01

    d["switch_cost_acc_cue_diff"] = switch_cost_acc_10 - switch_cost_acc_01

    # rt switch cost by switch type
    dd = d.loc[d["switch"] != 1]
    stay = dd.loc[dd["stay"] == True, "rt"].mean()
    switch = dd.loc[dd["stay"] == False, "rt"].mean()
    switch_cost_rt_10 = stay - switch
    d["switch_cost_rt_10"] = switch_cost_rt_10

    dd = d.loc[d["switch"] != -1]
    stay = dd.loc[dd["stay"] == True, "rt"].mean()
    switch = dd.loc[dd["stay"] == False, "rt"].mean()
    switch_cost_rt_01 = stay - switch
    d["switch_cost_rt_01"] = switch_cost_rt_01

    d["switch_cost_rt_cue_diff"] = switch_cost_rt_10 - switch_cost_rt_01

    return d


def compute_switch_cost(d):
    # generic accuracy switch cost
    stay = d.loc[d["stay"] == True, "acc"].mean()
    switch = d.loc[d["stay"] == False, "acc"].mean()
    d["switch_cost_acc"] = stay - switch

    # generic rt switch cost
    stay = d.loc[d["stay"] == True, "rt"].mean()
    switch = d.loc[d["stay"] == False, "rt"].mean()
    d["switch_cost_rt"] = stay - switch

    return d


def power_func(x, a, b, c):
    res = a * x**b + c
    return res


def tanh_func(x, a, b, c):
    res = a * np.tanh(b * (x - 1)) + c
    return res


def fit_func(d, dv, func):
    x = d["block"].to_numpy()
    y = d[dv].to_numpy()
    ppopt, pcov = curve_fit(func, x, y, maxfev=1e5, bounds=(0, 1))
    acc_pred = func(x, *ppopt)
    _, _, r, _, _ = linregress(y, acc_pred)
    r2 = r**2
    # plt.plot(x, y)
    # plt.plot(x, func(x, *ppopt))
    # plt.show()
    d["fit_a"] = ppopt[0]
    d["fit_b"] = ppopt[1]
    d["fit_c"] = ppopt[2]
    d["fit_ac"] = ppopt[0] + ppopt[2]
    d[dv + "_pred"] = acc_pred
    d["r2"] = r2
    return d


def report_pairwise(x):
    x = x.round(2)
    print()
    for i in range(x.shape[0]):
        contrast = x.iloc[i]["Contrast"]
        t = x.iloc[i]["T"].astype("U")
        df = x.iloc[i]["dof"].astype("U")
        p_cor = x.iloc[i]["p-corr"]
        p_unc = x.iloc[i]["p-unc"]
        p = str(p_cor) if ~np.isnan(p_cor) else str(p_unc)
        d = x.iloc[i]["cohen"].astype("U")
        bf = x.iloc[i]["BF10"]

        rep = contrast
        rep += ": t(" + df + ") = " + t
        rep += ", p = " + p
        rep += ", d = " + d

        print(rep)


def report_aov(x):
    for i in range(x.shape[0] - 1):
        source = x.iloc[i]["Source"]
        df1 = x.iloc[i]["DF"].astype("U")
        df2 = x.iloc[-1]["DF"].astype("U")
        f = x.iloc[i]["F"].astype("U")
        p = x.iloc[i]["p-unc"].astype("U")
        np2 = x.iloc[i]["np2"].astype("U")

        rep = source
        rep += ": F(" + df1 + ", " + df2 + ") = " + f
        rep += ", p = " + p
        rep += ", part_eta_sq = " + np2

        print(rep)


def inspect_interaction(d, dv):
    d = d[["effector", "attention", "memory", "cnd", "sub",
           dv]].drop_duplicates()

    res = pg.anova(
        data=d,
        dv=dv,
        between=["effector", "memory", "attention"],
        ss_type=3,
        effsize="np2",
    ).round(2)

    print("DV = " + dv)
    print(res)
    print()
    report_aov(res)

    sns.catplot(
        data=d,
        x="effector",
        y=dv,
        hue="memory",
        col="attention",
        capsize=0,
        kind="point",
    )
    plt.show()


def inspect_interaction_threeway(dd):
    dd = dd[[
        "effector", "attention", "memory", "cnd", "sub", "fit_c", "fit_ac",
        "fit_b"
    ]].drop_duplicates()

    dvs = ["fit_c", "fit_ac", "fit_b"]
    labs = ["Initial Accuracy", "Learning Asymtote", "Learning Rate"]
    fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(8, 8))
    for i, dv in enumerate(dvs):
        sns.pointplot(
            data=dd[dd["attention"] == "1d"],
            x="effector",
            y=dv,
            hue="memory",
            ax=ax[i, 0],
            legend=False,
        )
        sns.pointplot(
            data=dd[dd["attention"] == "2d"],
            x="effector",
            y=dv,
            hue="memory",
            ax=ax[i, 1],
            legend=False,
        )
        ax[i, 0].set_ylabel(labs[i])
        ax[i, 1].set_ylabel(labs[i])
        ax[i, 0].legend(loc="upper center")
        ax[i, 1].legend(loc="upper center")
    labs = ["A", "B", "C", "d", "E", "F"]
    for i, curax in enumerate(ax.flatten()):
        curax.text(
            -0.15,
            1.05,
            labs[i],
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center",
            transform=curax.transAxes,
        )
    ax[0, 0].set_title("1D")
    ax[0, 1].set_title("2D")
    plt.tight_layout()
    plt.savefig("../figures/param_fits.pdf")


def inspect_switch_cost_per_block(d):
    # NOTE: tanh probably not the right func to fit for switch cost. Decaying
    # exponential / inverse power or something seems better.
    # dd = d.groupby(['effector', 'attention', 'memory', 'cnd', 'sub',
    #                 'block']).apply(compute_switch_cost)
    # dd = dd.groupby(['effector', 'attention', 'memory', 'cnd',
    #                  'sub']).apply(fit_func, 'switch_cost_acc', tanh_func)

    # y_obs = dd.groupby(['effector', 'attention', 'memory',
    #                     'block'])['switch_cost_acc'].mean()
    # y_pred = dd.groupby(['effector', 'attention', 'memory',
    #                      'block'])['switch_cost_acc_pred'].mean()

    # plt.plot(np.arange(0, y_obs.shape[0]), y_obs)
    # plt.plot(np.arange(0, y_pred.shape[0]), y_pred)
    # plt.show()

    # _, _, r, _, _ = linregress(y_obs, y_pred)
    # r2 = r**2
    # print(r2.round(2))

    dv = "switch_cost_acc"
    res = pg.rm_anova(data=dd,
                      dv=dv,
                      within="block",
                      subject="sub",
                      effsize="np2").round(2)

    print("DV = " + dv)
    print(res)

    dv = "switch_cost_rt"
    res = pg.rm_anova(data=dd,
                      dv=dv,
                      within="block",
                      subject="sub",
                      effsize="np2").round(2)

    print("DV = " + dv)
    print(res)

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 8))

    sns.lineplot(
        data=dd[dd["attention"] == "1d"],
        x="block",
        y="switch_cost_acc",
        hue="memory",
        style="effector",
        ax=ax[0, 0],
    )

    sns.lineplot(
        data=dd[dd["attention"] == "2d"],
        x="block",
        y="switch_cost_acc",
        hue="memory",
        style="effector",
        ax=ax[0, 1],
    )

    sns.lineplot(
        data=dd[dd["attention"] == "1d"],
        x="block",
        y="switch_cost_rt",
        hue="memory",
        style="effector",
        ax=ax[1, 0],
    )

    sns.lineplot(
        data=dd[dd["attention"] == "2d"],
        x="block",
        y="switch_cost_rt",
        hue="memory",
        style="effector",
        ax=ax[1, 1],
    )

    plt.savefig("../figures/switch_costs_blk.pdf")


def inspect_func_fits_big(d, dd):
    # dbm = pd.read_csv('../dbm_fits/dbm_results.csv')
    # dbm = dbm.loc[dbm['model'] == dbm['best_model']]

    fig, ax = plt.subplots(5, 4, squeeze=True, figsize=(8, 10))

    conditions1 = ["udii2", "udcj2", "cjii2", "cj2cr", "ii2cr"]
    conditions2 = ["udii4", "udcj4", "cjii4", "cj4cr", "ii4cr"]

    for i in range(len(conditions1)):
        c1 = conditions1[i]
        c2 = conditions2[i]

        for j, c in enumerate([c1, c2]):
            for k in range(2):
                d0 = dd.loc[
                    (dd["cnd"] == c) & (dd["cue"] == k),
                    [
                        "effector", "attention", "memory", "block", "acc",
                        "acc_pred"
                    ],
                ]

                sns.lineplot(data=d0,
                             x="block",
                             y="acc",
                             legend=False,
                             ax=ax[i, 2 * j + k])

                ax[i, 2 * j + k].set_title("subtask " + str(k), fontsize=10)

                axins = inset_axes(ax[i, 2 * j + k],
                                   "35%",
                                   "35%",
                                   loc="lower right",
                                   borderpad=0.5)

                axins.set_xlabel("")
                axins.set_ylabel("")
                axins.set_xlabel("")
                axins.set_ylabel("")
                axins.set_xticks([])
                axins.set_yticks([])
                axins.set_xticks([])
                axins.set_yticks([])

                d0 = d.loc[(d["cnd"] == c) & (d["cue"] == k),
                           ["x", "y", "cat", "sub"]]

                s = d0["sub"].unique()[0]
                d0 = d0.loc[d0["sub"] == s].drop_duplicates()

                sns.scatterplot(
                    data=d0,
                    x="x",
                    y="y",
                    hue="cat",
                    style="cat",
                    size=0.2,
                    alpha=1,
                    legend=False,
                    ax=axins,
                )

                # dbm0 = dbm.loc[(dbm['cnd'] == c) & (dbm['cue'] == k) &
                #                (dbm['block'] == 5), ['sub', 'best_model', 'p']]
                # plot_dbm(dbm0, axins)

            ax[i, 2 * j + 0].set_xlabel("")
            ax[i, 2 * j + 0].set_ylabel(c)
            ax[i, 2 * j + 1].set_xlabel("")
            ax[i, 2 * j + 1].set_ylabel("")
            # ax[i, 2 * j + 0].set_xticks([])
            # ax[i, 2 * j + 0].set_yticks([])
            # ax[i, 2 * j + 1].set_xticks([])
            # ax[i, 2 * j + 1].set_yticks([])

    [x.set_ylim(0, 1) for x in ax.flatten()]

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    # plt.show()
    plt.tight_layout()
    plt.savefig("../figures/func_fits_big.pdf")
    plt.close()


def inspect_func_fits(dd):
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 8))

    sns.lineplot(
        data=dd[dd["attention"] == "1d"],
        x="block",
        y="acc",
        hue="memory",
        style="effector",
        legend=False,
        ax=ax[0, 0],
    )

    sns.lineplot(
        data=dd[dd["attention"] == "1d"],
        x="block",
        y="acc_pred",
        hue="memory",
        style="effector",
        legend=False,
        ax=ax[0, 1],
    )

    sns.lineplot(
        data=dd[dd["attention"] == "2d"],
        x="block",
        y="acc",
        hue="memory",
        style="effector",
        legend=False,
        ax=ax[1, 0],
    )

    sns.lineplot(
        data=dd[dd["attention"] == "2d"],
        x="block",
        y="acc_pred",
        hue="memory",
        style="effector",
        legend=False,
        ax=ax[1, 1],
    )

    [x.set_ylim(0, 1) for x in ax.flatten()]
    [x.set_xlabel("Block") for x in ax.flatten()]
    [
        x.set_ylabel("1D Mean Accuracy \n (proportion correct)")
        for x in ax[0, :].flatten()
    ]
    [
        x.set_ylabel("2D Mean Accuracy \n (proportion correct)")
        for x in ax[1, :].flatten()
    ]
    ax[0, 0].set_title("Observed")
    ax[0, 1].set_title("Best-fitting tanh")
    labs = ["A", "B", "C", "D"]
    for i, curax in enumerate(ax[:2, :].flatten()):
        curax.text(
            -0.15,
            1.05,
            labs[i],
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center",
            transform=curax.transAxes,
        )

    handles, labels = ax[1, 1].get_legend_handles_labels()
    line_1 = Line2D(
        [0],
        [0],
        label="between memory systems & different motor plans",
        color="C0",
        linestyle="-",
    )
    line_2 = Line2D(
        [0],
        [0],
        label="between memory systems & same motor plans",
        color="C0",
        linestyle="--",
    )
    line_3 = Line2D(
        [0],
        [0],
        label="within memory system & different motor plans",
        color="C1",
        linestyle="-",
    )
    line_4 = Line2D(
        [0],
        [0],
        label="within memory system & same motor plans",
        color="C1",
        linestyle="--",
    )
    handles = [line_1, line_2, line_3, line_4]
    lgd = plt.legend(handles=handles, loc=(0, 0), bbox_to_anchor=(-1.0, -0.5))

    fig.subplots_adjust(wspace=0.3)

    plt.savefig("../figures/func_fits.pdf",
                bbox_extra_artists=(lgd, ),
                bbox_inches="tight")


def inspect_interaction_switch_costs(d):
    d = d[[
        "effector",
        "attention",
        "memory",
        "cnd",
        "sub",
        "switch_cost_acc",
        "switch_cost_rt",
    ]].drop_duplicates()

    dvs = ["switch_cost_acc", "switch_cost_rt"]
    labs = [
        "Accuracy Switch Cost \n (proportion correct)",
        "RT Switch Cost \n (seconds)",
    ]
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5.33))
    for i, dv in enumerate(dvs):
        sns.pointplot(
            data=d[d["attention"] == "1d"],
            x="effector",
            y=dv,
            hue="memory",
            ax=ax[i, 0],
            legend=False,
        )
        sns.pointplot(
            data=d[d["attention"] == "2d"],
            x="effector",
            y=dv,
            hue="memory",
            ax=ax[i, 1],
            legend=False,
        )
        ax[i, 0].set_ylabel(labs[i])
        ax[i, 1].set_ylabel(labs[i])
        ax[i, 0].legend(loc="upper center")
        ax[i, 1].legend(loc="upper center")
    labs = ["A", "B", "C", "D"]
    for i, curax in enumerate(ax.flatten()):
        curax.text(
            -0.15,
            1.05,
            labs[i],
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center",
            transform=curax.transAxes,
        )
    ax[0, 0].set_title("1D")
    ax[0, 1].set_title("2D")
    plt.tight_layout()
    plt.savefig("../figures/switch_costs.pdf")


def inspect_interaction_switch_costs_iirb(d):
    dd = d.loc[np.isin(d["cnd"], ["cjii4", "cj4cr", "ii4cr"])]

    # NOTE: switch types
    # -1: 0|1 -> cj|ii
    #  1: 1|0 -> ii|cj

    # NOTE: cue == 0 is cj and cue == 1 is ii
    # def inspect_cue(tmp):
    #     fig, ax = plt.subplots(1, 2, squeeze=False)
    #     sns.scatterplot(data=tmp[tmp['cue'] == 0],
    #                     x='x',
    #                     y='y',
    #                     hue='cat',
    #                     ax=ax[0, 0])
    #     sns.scatterplot(data=tmp[tmp['cue'] == 1],
    #                     x='x',
    #                     y='y',
    #                     hue='cat',
    #                     ax=ax[0, 1])
    #     plt.show()
    # dd.groupby(['cnd']).apply(inspect_cue)

    dd = (dd.groupby(["cnd", "sub", "memory",
                      "switch"])[["acc", "rt"]].mean().reset_index())

    # dvs = ['acc', 'rt']
    # labs = ['Accuracy', 'RT']
    dvs = ["acc"]
    labs = ["Proportion correct"]
    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(8, 4))
    for i, dv in enumerate(dvs):
        for j, mem in enumerate(dd["memory"].unique()):
            sns.pointplot(
                data=dd[dd["memory"] == mem],
                x="switch",
                y=dv,
                join=False,
                ax=ax[i, j],
            )
            if mem == "within":
                ax[i, j].set_xticklabels(["2|1", "stay", "1|2"])
                ax[i, j].set_title("Within system")
            if mem == "between":
                ax[i, j].set_xticklabels(["ii|rb", "stay", "rb|ii"])
                ax[i, j].set_title("Between systems")
            ax[i, j].set_ylabel(labs[i])
    [x.set_xlabel("Type of switch") for x in ax.flatten()]
    plt.tight_layout()
    plt.savefig("../figures/switch_costs_iirb.pdf")
    plt.close("all")


#     for dv in ["acc", "rt"]:
#         aov = pg.mixed_anova(
#             data=dd,
#             dv=dv,
#             within="switch",
#             subject="sub",
#             between="memory",
#             correction=True,
#             effsize="np2",
#         ).round(2)
#
#         print()
#         for i in range(aov.shape[0]):
#             df1 = aov.iloc[i]["DF1"].astype("U")
#             df2 = aov.iloc[-1]["DF2"].astype("U")
#             f = aov.iloc[i]["F"].astype("U")
#             p = aov.iloc[i]["p-unc"].astype("U")
#             np2 = aov.iloc[i]["np2"].astype("U")
#
#             rep = dv
#             rep += ": F(" + df1 + ", " + df2 + ") = " + f
#             rep += ", p = " + p
#             rep += ", part_eta_sq = " + np2
#
#             print(rep)
#
#     for dv in ["acc", "rt"]:
#         for mem in dd["memory"].unique():
#             tt = pg.pairwise_ttests(
#                 data=dd[dd["memory"] == mem],
#                 dv=dv,
#                 within="switch",
#                 subject="sub",
#                 effsize="cohen",
#                 padjust="bonf",
#             ).round(3)
#
#             print()
#             for i in range(tt.shape[0]):
#                 A = tt.iloc[i]["A"].astype("U")
#                 B = tt.iloc[i]["B"].astype("U")
#                 t = tt.iloc[i]["T"].astype("U")
#                 df = tt.iloc[i]["dof"].astype("U")
#                 p_cor = tt.iloc[i]["p-corr"].astype("U")
#                 p_unc = tt.iloc[i]["p-unc"].astype("U")
#                 d = tt.iloc[i]["cohen"].astype("U")
#                 bf = tt.iloc[i]["BF10"]
#
#                 rep = mem + " " + dv
#                 rep += ": " + A + " - " + B
#                 rep += ": t(" + df + ") = " + t
#                 rep += ", p = " + p_cor
#                 rep += ", d = " + d
#
#                 print(rep)


def inspect_cats(d):
    fig, ax = plt.subplots(1, 3, squeeze=True, figsize=(8, 3))

    dd = d.loc[(d["cnd"] == "udcj2") & (d["cue"] == 0),
               ["x", "y", "cat"]].drop_duplicates()

    sns.scatterplot(data=dd,
                    x="x",
                    y="y",
                    hue="cat",
                    style="cat",
                    legend=False,
                    ax=ax[0])

    dd = d.loc[(d["cnd"] == "udcj2") & (d["cue"] == 1),
               ["x", "y", "cat"]].drop_duplicates()
    sns.scatterplot(data=dd,
                    x="x",
                    y="y",
                    hue="cat",
                    style="cat",
                    legend=False,
                    ax=ax[1])

    dd = d.loc[(d["cnd"] == "udii2") & (d["cue"] == 1),
               ["x", "y", "cat"]].drop_duplicates()
    sns.scatterplot(data=dd,
                    x="x",
                    y="y",
                    hue="cat",
                    style="cat",
                    legend=False,
                    ax=ax[2])

    ax[0].plot([50, 50], [-5, 105], "--k", alpha=0.5)
    ax[1].plot([29, 29], [29, 105], "--k", alpha=0.5)
    ax[1].plot([29, 105], [29, 29], "--k", alpha=0.5)
    ax[2].plot([-5, 105], [105, -5], "--k", alpha=0.5)

    ax[0].set_title("1D RB")
    ax[1].set_title("2D RB")
    ax[2].set_title("II")

    [x.set_xlim(-5, 105) for x in ax]
    [x.set_ylim(-5, 105) for x in ax]
    [x.set_aspect("equal") for x in ax]
    [x.set_xticks([]) for x in ax]
    [x.set_yticks([]) for x in ax]
    [x.set_xlabel("Major Axis Length") for x in ax]
    [x.set_ylabel("Major Axis Orientation") for x in ax]
    labs = ["A", "B", "C"]
    for i, curax in enumerate(ax.flatten()):
        curax.text(
            -0.15,
            1.05,
            labs[i],
            fontsize=16,
            horizontalalignment="center",
            verticalalignment="center",
            transform=curax.transAxes,
        )

    plt.tight_layout()
    plt.savefig("../figures/categories.pdf")


def inspect_cats_2(d):
    fig, ax = plt.subplots(5, 4, squeeze=True, figsize=(8, 3))

    conditions1 = ["udii2", "udcj2", "cjii2", "cj2cr", "ii2cr"]
    conditions2 = ["udii4", "udcj4", "cjii4", "cj4cr", "ii4cr"]

    for i in range(len(conditions1)):
        c1 = conditions1[i]
        c2 = conditions2[i]

        for j, c in enumerate([c1, c2]):
            d0 = d.loc[(d["cnd"] == c) & (d["cue"] == 0),
                       ["x", "y", "cat", "sub"]]

            s = d0["sub"].unique()[0]
            d0 = d0.loc[d0["sub"] == s].drop_duplicates()

            sns.scatterplot(
                data=d0,
                x="x",
                y="y",
                hue="cat",
                style="cat",
                legend=False,
                ax=ax[i, 2 * j + 0],
            )

            d1 = d.loc[(d["cnd"] == c) & (d["cue"] == 1),
                       ["x", "y", "cat", "sub"]]

            s = d1["sub"].unique()[0]
            d1 = d1.loc[d1["sub"] == s].drop_duplicates()

            sns.scatterplot(
                data=d1,
                x="x",
                y="y",
                hue="cat",
                style="cat",
                legend=False,
                ax=ax[i, 2 * j + 1],
            )
            ax[i, 2 * j + 0].set_xlabel("")
            ax[i, 2 * j + 0].set_ylabel(c)
            ax[i, 2 * j + 1].set_xlabel("")
            ax[i, 2 * j + 1].set_ylabel("")
            ax[i, 2 * j + 0].set_xticks([])
            ax[i, 2 * j + 0].set_yticks([])
            ax[i, 2 * j + 1].set_xticks([])
            ax[i, 2 * j + 1].set_yticks([])

    plt.savefig('../figures/categories_2.pdf')
    plt.close()


def plot_dbm(dbm, ax):
    for s in dbm["sub"].unique():
        x = dbm.loc[dbm["sub"] == s]

        best_model = x["best_model"].to_numpy()[0]

        if best_model in ("nll_unix_0", "nll_unix_1"):
            xc = x["p"].to_numpy()[0]
            ax.plot([xc, xc], [0, 100], "--k", linewidth=0.75)

        elif best_model in ("nll_uniy_0", "nll_uniy_1"):
            yc = x["p"].to_numpy()[0]
            ax.plot([0, 100], [yc, yc], "--k", linewidth=0.75)

        elif best_model in ("nll_glc_0", "nll_glc_1"):
            a1 = x["p"].to_numpy()[0]
            a2 = np.sqrt(1 - a1**2)
            b = x["p"].to_numpy()[1]
            ax.plot([0, 100], [-b / a2, -(100 * a1 + b) / a2],
                    "-k",
                    linewidth=0.75)

        elif best_model in ("nll_gcc_eq_0", "nll_gcc_3"):
            xc = x["p"].to_numpy()[0]
            yc = x["p"].to_numpy()[1]
            ax.plot([0, xc], [yc, yc], "-k")
            ax.plot([xc, xc], [0, yc], "-k")

        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)


def inspect_cats_3(d):
    dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
    dbm = dbm.loc[dbm["model"] == dbm["best_model"]]

    fig, ax = plt.subplots(5, 4, squeeze=True, figsize=(8, 10))

    conditions1 = ["udii2", "udcj2", "cjii2", "cj2cr", "ii2cr"]
    conditions2 = ["udii4", "udcj4", "cjii4", "cj4cr", "ii4cr"]

    for i in range(len(conditions1)):
        c1 = conditions1[i]
        c2 = conditions2[i]

        for j, c in enumerate([c1, c2]):
            for k in range(2):
                d0 = d.loc[(d["cnd"] == c) & (d["cue"] == k),
                           ["x", "y", "cat", "sub"]]

                s = d0["sub"].unique()[0]
                d0 = d0.loc[d0["sub"] == s].drop_duplicates()

                sns.scatterplot(
                    data=d0,
                    x="x",
                    y="y",
                    hue="cat",
                    style="cat",
                    alpha=0.5,
                    legend=False,
                    ax=ax[i, 2 * j + k],
                )

                dbm0 = dbm.loc[
                    (dbm["cnd"] == c) & (dbm["cue"] == k) &
                    (dbm["block"] == 5),
                    ["sub", "best_model", "p"],
                ]
                plot_dbm(dbm0, ax[i, 2 * j + k])

                ax[i, 2 * j + k].set_title("subtask " + str(k), fontsize=10)

            ax[i, 2 * j + 0].set_xlabel("")
            ax[i, 2 * j + 0].set_ylabel(c)
            ax[i, 2 * j + 1].set_xlabel("")
            ax[i, 2 * j + 1].set_ylabel("")
            ax[i, 2 * j + 0].set_xticks([])
            ax[i, 2 * j + 0].set_yticks([])
            ax[i, 2 * j + 1].set_xticks([])
            ax[i, 2 * j + 1].set_yticks([])

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    # plt.show()
    plt.savefig("../figures/categories_3.pdf")


def relabel_best_model(x):
    best_model = x["best_model"].to_numpy()[0]
    if best_model == "nll_guess":
        best_model = "guess"
    if best_model == "nll_biased_guess":
        best_model = "guess"
    if best_model == "nll_unix_0":
        best_model = "1D"
    if best_model == "nll_unix_1":
        best_model = "1D"
    if best_model == "nll_uniy_0":
        best_model = "1D"
    if best_model == "nll_uniy_1":
        best_model = "1D"
    if best_model == "nll_glc_0":
        best_model = "2D"
    if best_model == "nll_glc_1":
        best_model = "2D"
    if best_model == "nll_gcc_eq_0":
        best_model = "2D"
    if best_model == "nll_gcc_eq_1":
        best_model = "2D"
    if best_model == "nll_gcc_eq_2":
        best_model = "2D"
    if best_model == "nll_gcc_eq_3":
        best_model = "2D"
    x["best_model_lumped"] = best_model
    return x


def add_optimal_model(x):
    cnd = x["cnd"].to_numpy()[0]
    cue = x["cue"].to_numpy()[0]

    if cnd == "udii2" and cue == 0:
        opt_mod = "1D"
    if cnd == "udcj2" and cue == 0:
        opt_mod = "1D"
    if cnd == "cjii2" and cue == 0:
        opt_mod = "2D"
    if cnd == "cj2cr" and cue == 0:
        opt_mod = "2D"
    if cnd == "ii2cr" and cue == 0:
        opt_mod = "2D"
    if cnd == "udii4" and cue == 0:
        opt_mod = "1D"
    if cnd == "udcj4" and cue == 0:
        opt_mod = "1D"
    if cnd == "cjii4" and cue == 0:
        opt_mod = "2D"
    if cnd == "cj4cr" and cue == 0:
        opt_mod = "2D"
    if cnd == "ii4cr" and cue == 0:
        opt_mod = "2D"

    if cnd == "udii2" and cue == 1:
        opt_mod = "2D"
    if cnd == "udcj2" and cue == 1:
        opt_mod = "2D"
    if cnd == "cjii2" and cue == 1:
        opt_mod = "2D"
    if cnd == "cj2cr" and cue == 1:
        opt_mod = "2D"
    if cnd == "ii2cr" and cue == 1:
        opt_mod = "2D"
    if cnd == "udii4" and cue == 1:
        opt_mod = "2D"
    if cnd == "udcj4" and cue == 1:
        opt_mod = "2D"
    if cnd == "cjii4" and cue == 1:
        opt_mod = "2D"
    if cnd == "cj4cr" and cue == 1:
        opt_mod = "2D"
    if cnd == "ii4cr" and cue == 1:
        opt_mod = "2D"

    x["opt_mod"] = opt_mod

    return x


def inspect_dbm_counts():
    sns.barplot(data=d0, x="best_model_lumped", y="sub", ax=ax[i, 2 * j + k])
    ax[i, 2 * j + k].set_title("subtask " + str(k), fontsize=10)

    ax[i, 2 * j + 0].set_xlabel("")
    ax[i, 2 * j + 0].set_ylabel(c)
    ax[i, 2 * j + 1].set_xlabel("")
    ax[i, 2 * j + 1].set_ylabel("")
    ax[i, 2 * j + 0].set_xticks([])
    ax[i, 2 * j + 0].set_yticks([])
    ax[i, 2 * j + 1].set_xticks([])
    ax[i, 2 * j + 1].set_yticks([])

    plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05)
    # plt.show()
    plt.savefig("../figures/dbm_counts.pdf")
    plt.close()


def inspect_tanh():
    # NOTE: tanh_func parameters
    # asymtote = a + c
    # rate = b
    # initial accuracy = c

    a = [0.1, 0.4]
    b = [1]
    c = [0.1, 0.35]
    x = np.arange(1, 24, 1)
    for aa in a:
        for bb in b:
            for cc in c:
                lab = "a= " + str(aa) + ", b= " + str(bb) + ", c= " + str(cc)
                # plt.plot(x, aa * x**bb + cc, label=lab)
                plt.plot(x, aa * np.tanh(bb * (x - 1)) + cc, label=lab)
                plt.legend()
    plt.show()


def report_stats(ddd):
    x = ddd[ddd["effector"] == "same buttons"]["fit_c"].to_numpy()
    y = ddd[ddd["effector"] == "different buttons"]["fit_c"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    x = ddd[(ddd["effector"] == "same buttons")
            & (ddd["attention"] == "2d")]["fit_ac"].to_numpy()
    y = ddd[(ddd["effector"] == "different buttons")
            & (ddd["attention"] == "2d")]["fit_ac"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    x = ddd[(ddd["effector"] == "same buttons")
            & (ddd["attention"] == "1d")]["fit_ac"].to_numpy()
    y = ddd[(ddd["effector"] == "different buttons")
            & (ddd["attention"] == "1d")]["fit_ac"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    x = ddd[(ddd["effector"] == "same buttons")
            & (ddd["attention"] == "2d")
            & (ddd["memory"] == "between")]["fit_ac"].to_numpy()
    y = ddd[(ddd["effector"] == "different buttons")
            & (ddd["attention"] == "2d")
            & (ddd["memory"] == "within")]["fit_ac"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    x = ddd[(ddd["effector"] == "same buttons")
            & (ddd["attention"] == "2d")
            & (ddd["memory"] == "between")]["fit_b"].to_numpy()
    y = ddd[(ddd["effector"] == "different buttons")
            & (ddd["attention"] == "2d")
            & (ddd["memory"] == "within")]["fit_b"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    x = ddd[(ddd["effector"] == "same buttons")
            & (ddd["memory"] == "between")]["fit_b"].to_numpy()
    y = ddd[(ddd["effector"] == "different buttons")
            & (ddd["memory"] == "between")]["fit_b"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    x = ddd[(ddd["effector"] == "same buttons")
            & (ddd["memory"] == "within")]["fit_b"].to_numpy()
    y = ddd[(ddd["effector"] == "different buttons")
            & (ddd["memory"] == "within")]["fit_b"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    # NOTE: Accuracy Switch costs
    x = ddd[(ddd["memory"] == "between")]["switch_cost_acc"].to_numpy()
    y = ddd[(ddd["memory"] == "within")]["switch_cost_acc"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)

    # NOTE: RT Switch costs
    x = ddd[(ddd["effector"] == "same buttons")]["switch_cost_rt"].to_numpy()
    y = ddd[(
        ddd["effector"] == "different buttons")]["switch_cost_rt"].to_numpy()
    res = pg.ttest(x,
                   y,
                   paired=False,
                   alternative="two-sided",
                   correction="auto")
    rep = "t(" + res["dof"].round(2).astype("U") + ") = "
    rep += res["T"].round(2).astype("U")
    rep += ", p = " + res["p-val"].round(2).astype("U")
    rep += ", d = " + res["cohen-d"].round(2).astype("U")
    print(rep)


def fit_dbm(d, model_func, side, k, n, model_name):
    fit_args = {
        "obj_func": None,
        "bounds": None,
        "disp": False,
        "maxiter": 3000,
        "popsize": 20,
        "mutation": 0.7,
        "recombination": 0.5,
        "tol": 1e-3,
        "polish": False,
        "updating": "deferred",
        "workers": -1,
    }

    obj_func = fit_args["obj_func"]
    bounds = fit_args["bounds"]
    maxiter = fit_args["maxiter"]
    disp = fit_args["disp"]
    tol = fit_args["tol"]
    polish = fit_args["polish"]
    updating = fit_args["updating"]
    workers = fit_args["workers"]
    popsize = fit_args["popsize"]
    mutation = fit_args["mutation"]
    recombination = fit_args["recombination"]

    cnd = d["cnd"]
    sub = d["sub"]
    cue = d["cue"]

    drec = []
    for m, mod in enumerate(model_func):
        dd = d[(d["sub"] == sub) & (d["cnd"] == cnd) &
               (d["cue"] == cue)][["cat", "x", "y", "resp"]]

        cat = dd.cat.to_numpy()
        x = dd.x.to_numpy()
        y = dd.y.to_numpy()
        resp = dd.resp.to_numpy()

        # nll funcs expect resp to be [0, 1]
        n_zero = np.sum(resp == 0)
        n_one = np.sum(resp == 1)
        n_two = np.sum(resp == 2)
        n_three = np.sum(resp == 3)

        if np.argmax([n_zero, n_one, n_two, n_three]) > 1:
            resp = resp - 2

        # rescale x and y to be [0, 100]
        range_x = np.max(x) - np.min(x)
        x = ((x - np.min(x)) / range_x) * 100
        range_y = np.max(y) - np.min(y)
        y = ((y - np.min(y)) / range_y) * 100

        # compute glc bnds
        yub = np.max(y) + 0.1 * range_y
        ylb = np.min(y) - 0.1 * range_y
        bub = 2 * np.max([yub, -ylb])
        blb = -bub
        nlb = 0.001
        nub = np.max([range_x, range_y]) / 2

        if "unix" in model_name[m]:
            bnd = ((0, 100), (nlb, nub))
        elif "uniy" in model_name[m]:
            bnd = ((0, 100), (nlb, nub))
        elif "glc" in model_name[m]:
            bnd = ((-1, 1), (blb, bub), (nlb, nub))
        elif "gcc" in model_name[m]:
            bnd = ((0, 100), (0, 100), (nlb, nub))

        z_limit = 3

        args = (z_limit, cat, x, y, resp, side[m])

        results = differential_evolution(
            func=mod,
            bounds=bnd,
            args=args,
            disp=disp,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            tol=tol,
            polish=polish,
            updating=updating,
            workers=workers,
        )

        tmp = np.concatenate((results["x"], [results["fun"]]))
        tmp = np.reshape(tmp, (tmp.shape[0], 1))

        # a1*x + a2*y + b = 0
        # y = -(a1*x + b) / a2
        # a1 = results['x'][0]
        # a2 = np.sqrt(1 - a1**2)
        # b = results['x'][1]

        # fig, ax = plt.subplots(1, 1, squeeze=False)
        # ax[0, 0].scatter(x, y, c=resp)
        # ax[0, 0].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], '--k')
        # ax[0,0].set_xlim(-5, 105)
        # ax[0,0].set_ylim(-5, 105)
        # plt.show()

        tmp = pd.DataFrame(results["x"])
        tmp.columns = ["p"]
        tmp["nll"] = results["fun"]
        tmp["bic"] = k[m] * np.log(n) + 2 * results["fun"]
        # tmp['aic'] = k[m] * 2 + 2 * results['fun']
        tmp["model"] = model_name[m]
        drec.append(tmp)

    drec = pd.concat(drec)
    return drec


def report_dbm_results():

    dbm = pd.read_csv("../dbm_fits/dbm_results.csv")
    dbm = dbm.loc[dbm["model"] == dbm["best_model"]]
    dbm = dbm.groupby(["cnd", "sub", "block",
                       "cue"]).apply(relabel_best_model).reset_index(drop=True)
    dbm = dbm.groupby(["cnd", "sub", "block",
                       "cue"]).apply(add_optimal_model).reset_index(drop=True)
    dbm = dbm.loc[dbm["block"] == 5,
                  ["cnd", "sub", "cue", "best_model_lumped", "opt_mod"
                   ]].drop_duplicates()

    def print_dbm_stats(d):
        n = d.shape[0]
        k = np.sum(d["best_model_lumped"] == d["opt_mod"])
        p = 0.5
        pval = 1 - binom.cdf(k - 1, n, p)
        pval = np.round(pval, 2)
        d["n"] = n
        d["k"] = k
        d["pval"] = pval
        dd = d[["cnd", "cue", "n", "k", "pval"]].drop_duplicates()
        return dd

    dd = dbm.groupby(["cnd",
                      "cue"]).apply(print_dbm_stats).reset_index(drop=True)

    custom_order = [
        "udii4",
        "udcj4",
        "udii2",
        "udcj2",
        "cjii4",
        "cj4cr",
        "ii4cr",
        "cjii2",
        "cj2cr",
        "ii2cr",
    ]
    dd["cnd"] = pd.Categorical(dd["cnd"],
                               categories=custom_order,
                               ordered=True)
    dd = dd.sort_values(by=["cnd", "cue"])

    def rename_cue(d):
        if d["cue"].to_numpy()[0] == 0:
            x = d["cnd"].to_numpy()[0][0:2]

        elif d["cue"].to_numpy()[0] == 1:
            x = d["cnd"].to_numpy()[0][2:4]

        if x == "ud":
            x = "1D RB"

        elif x == "ii":
            x = "II"

        elif x == "cj":
            x = "2D RB"

        d["cue"] = x
        return d

    dd = dd.groupby(["cnd", "cue"]).apply(rename_cue).reset_index(drop=True)

    column_name_mapping = {"cnd": "Condition", "cue": "Sub-Task"}

    dd = dd.rename(columns=column_name_mapping)
    dd["b"] = (r"$\mathcal{B}(" + dd["n"].astype(str) + ", 0.5) = " +
               dd["k"].astype(str) + ", p = " + dd["pval"].astype(str) + r"$")
    dd = dd[["Condition", "Sub-Task", "b"]]
    print(dd)

    ddl = dd.to_latex(index=False, escape=False)
    print(ddl)
    with open("../write/dbm_table.tex", "w") as f:
        f.write(ddl)


def nll_unix(params, *args):
    """
    - returns the negative loglikelihood of the unidimensional X bound fit
    - params format:  [bias noise] (so x=bias is boundary)
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    xc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    zscoresX = (x - xc) / noise
    zscoresX = np.clip(zscoresX, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresX, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresX, 0.0, 1.0)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + sum(log_B_probs))

    return nll


def nll_uniy(params, *args):
    """
    - returns the negative loglikelihood of the unidimensional Y bound fit
    - params format:  [bias noise] (so y=bias is boundary)
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    yc = params[0]
    noise = params[1]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    zscoresY = (y - yc) / noise
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscoresY, 0.0, 1.0)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscoresY, 0.0, 1.0)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + sum(log_B_probs))

    return nll


def nll_glc(params, *args):
    """
    - returns the negative loglikelihood of the GLC
    - params format: [a1 b noise]
    -- a1*x+a2*y+b=0 is the linear bound
    -- assumes without loss of generality that:
    --- a2=sqrt(1-a1^2)
    --- a2 >= 0
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    a1 = params[0]
    a2 = np.sqrt(1 - params[0]**2)
    b = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    z_coefs = np.array([[a1, a2, b]]).T / params[2]
    data_info = np.array([x, y, np.ones(np.shape(x))]).T
    zscores = np.dot(data_info, z_coefs)
    zscores = np.clip(zscores, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscores)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscores)
        prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + np.sum(log_B_probs))

    return nll


def nll_gcc_eq(params, *args):
    """
    returns the negative loglikelihood of the 2d data for the General
    Conjunctive Classifier with equal variance in the two dimensions.

    Parameters:
    params format: [biasX biasY noise] (so x = biasX and
    y = biasY make boundary)
    data row format:  [subject_response x y correct_response]
    z_limit is the z-score value beyond which one should truncate
    """

    xc = params[0]
    yc = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    if side == 0:
        zscoresX = (x - xc) / noise
        zscoresY = (y - yc) / noise
    if side == 1:
        zscoresX = (xc - x) / noise
        zscoresY = (y - yc) / noise
    if side == 2:
        zscoresX = (x - xc) / noise
        zscoresY = (yc - y) / noise
    else:
        zscoresX = (xc - x) / noise
        zscoresY = (yc - y) / noise

    zscoresX = np.clip(zscoresX, -z_limit, z_limit)
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    pXB = norm.cdf(zscoresX)
    pYB = norm.cdf(zscoresY)

    prB = pXB * pYB
    prA = 1 - prB

    log_A_probs = np.log(prA[A_indices])
    log_B_probs = np.log(prB[B_indices])

    nll = -(np.sum(log_A_probs) + np.sum(log_B_probs))

    return nll


def val_gcc_eq(params, *args):
    """
    Generates model responses for 2d data for the General Conjunctive
    Classifier with equal variance in the two dimensions.

    Parameters:
    params format: [biasX biasY noise] (so x = biasX and
    y = biasY make boundary)
    data row format:  [subject_response x y correct_response]
    z_limit is the z-score value beyond which one should truncate
    """

    xc = params[0]
    yc = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    if side == 0:
        zscoresX = (x - xc) / noise
        zscoresY = (y - yc) / noise
    if side == 1:
        zscoresX = (xc - x) / noise
        zscoresY = (y - yc) / noise
    if side == 2:
        zscoresX = (x - xc) / noise
        zscoresY = (yc - y) / noise
    else:
        zscoresX = (xc - x) / noise
        zscoresY = (yc - y) / noise

    zscoresX = np.clip(zscoresX, -z_limit, z_limit)
    zscoresY = np.clip(zscoresY, -z_limit, z_limit)

    pXB = norm.cdf(zscoresX)
    pYB = norm.cdf(zscoresY)

    prB = pXB * pYB
    prA = 1 - prB

    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp


def val_glc(params, *args):
    """
    Generates model responses for 2d data in the GLC.
    - params format: [a1 b noise]
    -- a1*x+a2*y+b=0 is the linear bound
    -- assumes without loss of generality that:
    --- a2=sqrt(1-a1^2)
    --- a2 >= 0
    - z_limit is the z-score value beyond which one should truncate
    - data columns:  [cat x y resp]
    """

    a1 = params[0]
    a2 = np.sqrt(1 - params[0]**2)
    b = params[1]
    noise = params[2]

    z_limit = args[0]
    cat = args[1]
    x = args[2]
    y = args[3]
    resp = args[4]
    side = args[5]

    n = x.shape[0]
    A_indices = np.where(resp == 0)
    B_indices = np.where(resp == 1)

    z_coefs = np.array([[a1, a2, b]]).T / params[2]
    data_info = np.array([x, y, np.ones(np.shape(x))]).T
    zscores = np.dot(data_info, z_coefs)
    zscores = np.clip(zscores, -z_limit, z_limit)

    if side == 0:
        prA = norm.cdf(zscores)
        prB = 1 - prA
    else:
        prB = norm.cdf(zscores)
        prA = 1 - prB

    resp = np.random.uniform(size=prB.shape) < prB
    resp = resp.astype(int)

    return cat, x, y, resp
