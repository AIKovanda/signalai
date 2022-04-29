import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


def plot_piano_tones(y, x_max=1000 * 131584 / 48000, savefigs=(), figsize=(7, 4), yshift=2.9, show=True):
    all_tones = ['$C_0$', '$C^{\\#}_0$', '$D_0$', '$D^{\\#}_0$', '$E_0$', '$F_0$', '$F^{\\#}_0$', '$G_0$',
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

    tone_info = []
    assert len(y.shape) == 2
    for i, (tone, np_tone) in enumerate(zip(all_tones, y)):
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
        plt.text((stop + start) / 2, i - yshift, s=tone, horizontalalignment='center', verticalalignment='center')

    plt.xlim((0, x_max))
    plt.ylim((85, 0))
    plt.xlabel(f'Time [ms]')
    plt.ylabel('Tone id')
    for i in savefigs:
        plt.savefig(i, bbox_inches='tight', pad_inches=0.)

    if show:
        plt.show()
    else:
        plt.close()
