import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def plot_binary_map(y: np.ndarray, all_object_labels: list, x_max=1000 * 131584 / 48000, savefigs=(), figsize=(7, 4),
                    xshift=0, yshift=2.9, show=True, object_name='Object'):
    tone_info = []
    assert len(y.shape) == 2
    for i, (tone, np_tone) in enumerate(zip(all_object_labels, y)):
        where_ = np.where(np_tone == 1.)[0]
        if len(where_) > 1:
            starting = where_[0]
            last = where_[0]
            for j in where_[1:]:
                if j > last + 1:
                    tone_info.append((i, tone, starting * x_max / len(np_tone), last * x_max / len(np_tone)))
                    starting = j

                last = j

            tone_info.append((i, tone, starting * x_max / len(np_tone), last * x_max / len(np_tone)))

    fig, ax = plt.subplots(figsize=figsize)
    for i, tone, start, stop in tone_info:
        ax.add_patch(Rectangle((start, i), stop - start, 1, color='k'))
        plt.text((stop + start) / 2 + xshift, i - yshift, s=tone, horizontalalignment='center', verticalalignment='center')

    plt.xlim((0, x_max))
    plt.ylim((len(all_object_labels), 0))
    plt.xlabel(f'Time [ms]')
    plt.ylabel(f'{object_name} id')
    for i in savefigs:
        plt.savefig(i, bbox_inches='tight', pad_inches=0.)

    if show:
        plt.show()
    else:
        plt.close()
