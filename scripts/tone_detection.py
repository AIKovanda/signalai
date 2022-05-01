import json
import sys

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from sklearn.metrics import auc, roc_curve
from taskchain.task import Config
from tqdm import trange

from signalai import config
from signalai.tools import plot_binary_map
from signalai.tools.visualization import ALL_PIANO_TONES

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
