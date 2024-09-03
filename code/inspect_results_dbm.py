from imports import *
from util_funcs import *

block_size = 112
d = load_data(block_size=block_size)

print(d.groupby(["attention", "effector", "memory"])["cnd"].unique())

c = "ii4cr"
d = d.loc[d["cnd"] == c]

models = [
    nll_unix,
    nll_unix,
    nll_uniy,
    nll_uniy,
    nll_glc,
    nll_glc,
    nll_gcc_eq,
    nll_gcc_eq,
    nll_gcc_eq,
    nll_gcc_eq,
]
side = [0, 1, 0, 1, 0, 1, 0, 1, 2, 3]
k = [2, 2, 2, 2, 3, 3, 3, 3, 3, 3]
n = block_size
model_names = [
    "nll_unix_0",
    "nll_unix_1",
    "nll_uniy_0",
    "nll_uniy_1",
    "nll_glc_0",
    "nll_glc_1",
    "nll_gcc_eq_0",
    "nll_gcc_eq_1",
    "nll_gcc_eq_2",
    "nll_gcc_eq_3",
]


def assign_best_model(x):
    model = x["model"].to_numpy()
    bic = x["bic"].to_numpy()
    best_model = np.unique(model[bic == bic.min()])[0]
    x["best_model"] = best_model
    return x


# NOTE: Fit the DBM models
if not os.path.exists("../dbm_fits/dbm_results.csv"):
    dbm = (d.groupby(["cnd", "sub", "cue",
                      "block"]).apply(fit_dbm, models, side, k, n,
                                      model_names).reset_index())

    dbm.to_csv("../dbm_fits/dbm_results.csv")

    dbm = dbm.groupby(["cnd", "sub", "cue", "block"]).apply(assign_best_model)

else:
    dbm = pd.read_csv("../dbm_fits/dbm_results.csv")

dd = dbm.loc[dbm["model"] == dbm["best_model"]]
ddd = dd[["cnd", "sub", "cue", "block", "best_model"]].drop_duplicates()
dcat = d[["cue", "x", "y", "cat"]].drop_duplicates()

# NOTE: inspect results (not used for the figures in the paper)
fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(12, 8))
sns.scatterplot(data=dcat[dcat["cue"] == 0],
                x="x",
                y="y",
                hue="cat",
                ax=ax[0, 0])
sns.scatterplot(data=dcat[dcat["cue"] == 1],
                x="x",
                y="y",
                hue="cat",
                ax=ax[0, 1])
ax[0, 0].get_legend().remove()
ax[0, 1].get_legend().remove()

for s in dd["sub"].unique():
    x = dd.loc[(dd["sub"] == s) & (dd["cue"] == 0) & (dd["block"] == 5)]

    best_model = x["best_model"].to_numpy()[0]

    if best_model in ("nll_unix_0", "nll_unix_1"):
        xc = x["p"].to_numpy()[0]
        ax[0, 0].plot([xc, xc], [0, 100], "--k")

    elif best_model in ("nll_uniy_0", "nll_uniy_1"):
        yc = x["p"].to_numpy()[0]
        ax[0, 0].plot([0, 100], [yc, yc], "--k")

    elif best_model in ("nll_glc_0", "nll_glc_1"):
        # a1 * x + a2 * y + b = 0
        # y = -(a1 * x + b) / a2
        a1 = x["p"].to_numpy()[0]
        a2 = np.sqrt(1 - a1**2)
        b = x["p"].to_numpy()[1]
        ax[0, 0].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], "-k")

    elif best_model in ("nll_gcc_eq_0", "nll_gcc_eq_1", "nll_gcc_eq_2",
                        "nll_gcc_eq_3"):
        xc = x["p"].to_numpy()[0]
        yc = x["p"].to_numpy()[1]
        ax[0, 0].plot([xc, xc], [0, 100], "-k")
        ax[0, 0].plot([0, 100], [yc, yc], "-k")

    ax[0, 0].set_xlim(-5, 105)
    ax[0, 0].set_ylim(-5, 105)

    x = dd.loc[(dd["sub"] == s) & (dd["cue"] == 1) & (dd["block"] == 5)]

    best_model = x["best_model"].to_numpy()[0]

    if best_model in ("nll_unix_0", "nll_unix_1"):
        xc = x["p"].to_numpy()[0]
        ax[0, 1].plot([xc, xc], [0, 100], "--k")

    elif best_model in ("nll_uniy_0", "nll_uniy_1"):
        yc = x["p"].to_numpy()[0]
        ax[0, 1].plot([0, 100], [yc, yc], "--k")

    elif best_model in ("nll_glc_0", "nll_glc_1"):
        # a1 * x + a2 * y + b = 0
        # y = -(a1 * x + b) / a2
        a1 = x["p"].to_numpy()[0]
        a2 = np.sqrt(1 - a1**2)
        b = x["p"].to_numpy()[1]
        ax[0, 1].plot([0, 100], [-b / a2, -(100 * a1 + b) / a2], "-k")

    elif best_model in ("nll_gcc_eq_0", "nll_gcc_eq_1", "nll_gcc_eq_2",
                        "nll_gcc_eq_3"):
        xc = x["p"].to_numpy()[0]
        yc = x["p"].to_numpy()[1]
        ax[0, 1].plot([xc, xc], [0, 100], "-k")
        ax[0, 1].plot([0, 100], [yc, yc], "-k")

    ax[0, 1].set_xlim(-5, 105)
    ax[0, 1].set_ylim(-5, 105)

# sns.countplot(data=ddd[ddd['cue'] == 0],
#               x='block',
#               hue='best_model',
#               ax=ax[1, 0])
# sns.countplot(data=ddd[ddd['cue'] == 1],
#               x='block',
#               hue='best_model',
#               ax=ax[1, 1])
# ax[1, 0].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
# ax[1, 1].legend(bbox_to_anchor=(1.01, 1), borderaxespad=0)
# plt.show()
plt.tight_layout()
plt.savefig("../figures/fig_dbm_" + str(c) + ".pdf")
