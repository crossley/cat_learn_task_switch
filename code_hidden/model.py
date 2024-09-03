from imports import *
from model_init import *
from model_util import *

# TODO: Get RB system working for 4 category conditions

# TODO: Already getting the switch cost results we want without lateral
# inhibition etc?

n_sims = 1

# cnd_list = ['cj2cr', 'ii2cr', 'cjii4', 'cj4cr', 'udcj4', 'udii2', 'udii4',
#             'cjii2', 'ii4cr', 'udcj2']
# cnd_list = ['cj2cr', 'cj4cr']
# cnd_list = ['ii2cr', 'ii4cr']

cnd_list = ['ii4cr']

res = []

for i in range(n_sims):
    for cnd in cnd_list:
        res.append(simulate_model(cnd))
res = pd.concat(res)
res.to_csv('../fits/fits_covis.csv')

# res = pd.read_csv('../fits/fits_covis.csv')

res.reset_index(inplace=True, drop=True)

res_mean = res.groupby(['cnd'])['acc_rec'].apply(np.mean).reset_index()

for cnd in cnd_list:
    print(cnd)
    res_cnd = res_mean.loc[res['cnd'] == cnd]
    acc = res_cnd.acc_rec.to_numpy()[0]
    block_size = 28
    n_blocks = n_trials / block_size
    block = np.repeat(np.arange(0, n_blocks, 1), block_size)
    d = pd.DataFrame({'block': block, 'acc': acc, 'cue': cue})
    dd = d.groupby(['block']).mean().reset_index()
    x = dd.block.to_numpy()
    y = dd.acc.to_numpy()
    ppopt, pcov = curve_fit(tanh_func, x, y, maxfev=1e5, bounds=(0, 1))
    acc_pred = tanh_func(x, *ppopt)

    fig, ax = plt.subplots(1, 1, squeeze=False)
    ax[0, 0].plot(acc_pred)
    ax[0, 0].plot(y)
    plt.show()

# fig, ax = plt.subplots(1, 2, squeeze=False)
# sns.pointplot(data=res, x='cnd', y='switch_cost_rt', legend=True, ax=ax[0, 0])
# sns.pointplot(data=res, x='cnd', y='switch_cost_acc', legend=True, ax=ax[0, 1])
# plt.show()

# cost is: switch - stay
