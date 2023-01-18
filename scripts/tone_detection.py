import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from sklearn.metrics import auc, roc_curve
from taskchain.task import Config
from tqdm import trange

from signalai import config
from signalai.tools.visualization import plot_binary_map


ALL_PIANO_TONES = [
    '$C_0$', '$C^{\\#}_0$', '$D_0$', '$D^{\\#}_0$', '$E_0$', '$F_0$', '$F^{\\#}_0$', '$G_0$',
    '$G^{\\#}_0$', '$A_0$', '$A^{\\#}_0$', '$B_0$', '$C_1$', '$C^{\\#}_1$', '$D_1$', '$D^{\\#}_1$',
    '$E_1$', '$F_1$', '$F^{\\#}_1$', '$G_1$', '$G^{\\#}_1$', '$A_1$', '$A^{\\#}_1$', '$B_1$', '$C_2$',
    '$C^{\\#}_2$', '$D_2$', '$D^{\\#}_2$', '$E_2$', '$F_2$', '$F^{\\#}_2$', '$G_2$', '$G^{\\#}_2$',
    '$A_2$', '$A^{\\#}_2$', '$B_2$', '$C_3$', '$C^{\\#}_3$', '$D_3$', '$D^{\\#}_3$', '$E_3$', '$F_3$',
    '$F^{\\#}_3$', '$G_3$', '$G^{\\#}_3$', '$A_3$', '$A^{\\#}_3$', '$B_3$', '$C_4$', '$C^{\\#}_4$',
    '$D_4$', '$D^{\\#}_4$', '$E_4$', '$F_4$', '$F^{\\#}_4$', '$G_4$', '$G^{\\#}_4$', '$A_4$',
    '$A^{\\#}_4$', '$B_4$', '$C_5$', '$C^{\\#}_5$', '$D_5$', '$D^{\\#}_5$', '$E_5$', '$F_5$',
    '$F^{\\#}_5$', '$G_5$', '$G^{\\#}_5$', '$A_5$', '$A^{\\#}_5$', '$B_5$', '$C_6$', '$C^{\\#}_6$',
    '$D_6$', '$D^{\\#}_6$', '$E_6$', '$F_6$', '$F^{\\#}_6$', '$G_6$', '$G^{\\#}_6$', '$A_6$',
    '$A^{\\#}_6$', '$B_6$', '$C_7$']


config_path = config.CONFIGS_DIR / 'models' / f'{sys.argv[1]}' / f'{sys.argv[2]}' / f'{sys.argv[3]}.yaml'

conf = Config(
    config.TASKS_DIR,  # where Taskchain data should be stored
    config_path,
    global_vars=config,  # set global variables
)
chain = conf.chain()

length, fs = chain.echo_info.value
if fs is None:
    fs = 48000

x_max = 1000 * length / fs

# chain.train_model.force()

dc = chain.evaluate_model.force().value
dt = dc.pop('items')

nex_dir = config.BASE_DIR / 'eventlist' / f'{sys.argv[2]}'

json_file = nex_dir / 'json' / f'{sys.argv[1]}.json'
with json_file.open('w') as f:
    json.dump(dc, f)


img_dir = nex_dir / 'img' / f'{sys.argv[1]}'
img_dir.mkdir(exist_ok=True, parents=True)

for j in trange(30, desc=f'Plotting in {img_dir} '):
    th = .5
    plot_binary_map(dt[j][0], all_object_labels=ALL_PIANO_TONES, savefigs=[img_dir / f'{j}-binary_map_full.pdf',
                                        img_dir / f'{j}-binary_map_full.pgf'],
                    figsize=(7, 3), yshift=3.8, x_max=x_max, show=False, object_name='Tone')

    plot_binary_map(convolve2d((dt[j][1] > th).astype(int), np.ones((1, 5)), 'same') >= 3,
                    all_object_labels=ALL_PIANO_TONES,
                    savefigs=[img_dir / f'{j}-binary_map_pr_c.pdf', img_dir / f'{j}-binary_map_pr_c.pgf'],
                    figsize=(7, 3), yshift=3.8, x_max=x_max, show=False, object_name='Tone')

    plot_binary_map(dt[j][1] > th, all_object_labels=ALL_PIANO_TONES,
                    savefigs=[img_dir / f'{j}-binary_map_pred.pdf', img_dir / f'{j}-binary_map_pred.pgf'],
                    figsize=(7, 3), yshift=3.8, x_max=x_max, show=False, object_name='Tone')


mp = np.concatenate([i[0] for i in dt], axis=1)
pred = np.concatenate([i[1] for i in dt], axis=1)

fpr, tpr, _ = roc_curve(mp.reshape(-1), pred.reshape(-1))
roc_auc = auc(fpr, tpr)


plt.figure(figsize=(10, 3))
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label=f"ROC curve (area = {roc_auc:.04f})",
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
plt.savefig(img_dir / f'roc_auc.pdf', bbox_inches='tight', pad_inches=0.)
plt.savefig(img_dir / f'roc_auc.pgf', bbox_inches='tight', pad_inches=0.)
plt.close()
