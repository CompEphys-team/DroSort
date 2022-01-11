from functions import *
import sssio
from pathlib import Path

import numpy as np
def get_neighbors_time(asig, st, n_samples, n_neighbors):
    """Calculates the window time for n neighbors based on 
        the duration of a spike and the stimated ISI.
        spike_time * neighbors + ISI mean
    """
    dt = asig.times[1] - asig.times[0]
    dt = dt.item()
    spike_time = n_samples * dt
    isis = [b - a for a, b in zip(st.times[:-1], st.times[1:])]
    # print(np.mean(isis),spike_time)

    return spike_time * n_neighbors + np.mean(isis) * n_neighbors

def get_spikes_ids(id_ref, time, SpikeInfo, spike_train):
    """ Get ids from spikes in a certain time window 
    """
    times_all = SpikeInfo['time']

    idx_t = times_all.values[id_ref]

    ini = idx_t - time
    end = idx_t + time
    # times = times_all.index[np.where((times_all.values > ini) & (times_all.values < end) & (times_all.values != idx_t))]

    # neighbors = times[np.where(SpikeInfo.loc[times, unit_column].values==unit)]
    ids = np.where((spike_train.times > ini) & (spike_train.times < end) & (spike_train.times != idx_t))

    return ids


def align_spikes(spikes, mode):
    """Align all spikes given using a certain mode"""
    return np.array([align_to(spike, mode) for spike in spikes.T]).T


def get_averages_from_units(aligned_spikes, units, SpikeInfo, unit_column, verbose=False):
    """Returns the average of the spikes for each unit 
    """
    average_spikes = []
    for unit in units:
        try:
           ix = SpikeInfo.groupby(unit_column).get_group(unit)['id']

           templates = aligned_spikes[:, ix].T
           if verbose:
               print_msg("Averaging %d spikes for cluster %s" % (len(templates),unit))
           average_spike = np.average(templates, axis=0)
        except Exception as e:
            if verbose:
                print_msg("Exception in averaging" + str(e.args) + " generating\
                zeros average for %.3f s" % SpikeInfo['time'].iloc[len(SpikeInfo) // 2])
            average_spike = np.zeros(aligned_spikes.shape[0])

        average_spikes.append(average_spike)

    return average_spikes



def get_combined_templates(average_spikes, dt_c, max_len, mode):
    """Gets all possible combinations of spikes and return the waveforms 
    of the templates and the labels. 
    a, b; b, a; b, b; a, a; a; b; a+b; """

    combined_templates = []
    A, B = average_spikes
    #Add AB templates
    combine_templates(combined_templates, A, B, dt_c, max_len, mode)
    n = len(combined_templates)
    templates_labels = [['a','b']]*n

    #Add BA templates
    combine_templates(combined_templates, B, A, dt_c, max_len, mode)
    n = len(combined_templates) - n
    templates_labels+=[['b', 'a']] * n

    #TODO: fix BB and AA, last combination removed to avoid A+A aligned
    #might mix with c spike. 
    #Add BB templates
    prev_n = len(combined_templates)
    combine_templates(combined_templates, B, B, dt_c, max_len, mode)
    combined_templates = combined_templates[:-24]  # TODO: hardcoded
    n = len(combined_templates) - prev_n
    templates_labels += [['b', 'b']] * n

    # #Add AA templates
    # prev_n = len(combined_templates)
    # combine_templates(combined_templates, A, A, dt_c, max_len, mode)
    # combined_templates = combined_templates[:-25] #TODO: hardcoded
    # n = len(combined_templates) -prev_n
    # templates_labels+=[['a','a']]*n

    #Add sum of two
    comb_t = np.array(np.concatenate((A, [A[-1]] * max_len)) + np.concatenate((B, [B[-1]] * max_len)))
    combined_templates.append(np.array(align_to(comb_t, mode)))
    templates_labels.append(['c'])

    #Add simple A
    comb_t = np.array(np.concatenate((A, [A[-1]] * max_len)))
    combined_templates.append(np.array(align_to(comb_t, mode)))
    templates_labels.append(['a'])

    #Add simple B
    comb_t = np.array(np.concatenate((B, [B[-1]] * max_len)))
    combined_templates.append(np.array(align_to(comb_t, mode)))
    templates_labels.append(['b'])

    return np.array(combined_templates), templates_labels


def combine_templates(combined_templates, A, B, dt, max_len, align_mode):
    # n_samples = np.sum(w_samples)
    # max_len = w_samples[1]-1
    for dt in np.arange(0, max_len, dt):
        long_a = np.concatenate((A, [A[-1]] * (max_len)))
        long_b = np.concatenate(([B[0]] * abs(max_len - dt), B, [B[-1]] * dt))

        comb_t = np.array(long_a + long_b)
        combined_templates.append(np.array(align_to(comb_t, align_mode)))

def plot_combined_templates(combined_templates, templates_labels, ncols=5, org_spike=[],distances=None, title='',save=None):
    nrows = int(np.ceil(combined_templates.shape[0] / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols * 2, nrows), num=1, clear=True)

    for c, ct in enumerate(combined_templates):
        i, j = c // ncols, c % ncols

        axes[i, j].plot(org_spike, color='k')
        axes[i, j].plot(ct)

        # peak_inds = signal.argrelmax(ct)[0]
        # peak_inds = peak_inds[np.argsort(ct[peak_inds])[-len(templates_labels[c]):]]

        # axes[i, j].plot(peak_inds, ct[peak_inds],'.')
        x = 0.6
        y = 0.7

        axes[i, j].text(x,y, str(templates_labels[c]), transform=axes[i,j].transAxes)

        if distances is not None:
            if distances[c] == min(distances) or distances[c] == np.sort(distances)[1]:
                color = 'r'
                change_ax_color(axes[i, j], color)
            else:
                color = 'k'

            y = 0.2
            axes[i, j].text(x, y, "%.3f" % distances[c], color=color, transform=axes[i,j].transAxes)

        # plt.ylim(-2, 0)

    plt.suptitle(title)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save)
        # plt.close(fig)
        # plt.close()

        # WARNING: do not add plt.close; figure clears by definition
        #         (arg: num=1, clear=True) adding plt.close leaks memory


def plot_combined_templates_bests(combined_templates, templates_labels, org_spike, distances, n_bests=2, title='', save=None):

    fig, axes = plt.subplots(nrows=1, ncols=n_bests, sharex=True, sharey=True, num=1, clear=True)
    inds = np.argsort(distances)[:2]

    for i, ind in enumerate(inds):
        axes[i].plot(org_spike, color='k', label="spike")
        axes[i].plot(combined_templates[ind], label="template")

        axes[i].text(x=0.8, y=0.8, s=str(templates_labels[ind]), fontsize=15, transform=axes[i].transAxes)
        axes[i].text(x=0.6, y=0.2, s="distance = %.3f" % distances[ind], fontsize=15, color='k', transform=axes[i].transAxes)

    plt.legend()
    plt.suptitle(title)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save)
        # WARNING: do not add plt.close; figure clears by definition
        #         (arg: num=1, clear=True) adding plt.close leaks memory

        # plt.close(fig)
        # plt.close()

    return fig, axes


def change_ax_color(ax, color):
    ax.spines['left'].set_color(color)
    ax.spines['top'].set_color(color)
    ax.spines['right'].set_color(color)
    ax.spines['bottom'].set_color(color)


# Defined here and not in sssio due to dependencies with functions.py

def save_all(results_folder, Config, SpikeInfo, Blk, units, Frates=False):
    # store SpikeInfo
    outpath = results_folder / 'SpikeInfo.csv'
    print_msg("saving SpikeInfo to %s" % outpath)
    SpikeInfo.to_csv(outpath)

    # store Block
    outpath = results_folder / 'result.dill'
    print_msg("saving Blk as .dill to %s" % outpath)
    sssio.blk2dill(Blk, outpath)

    print_msg("data is stored")


    # output csv data
    if Config.getboolean('output','csv'):
        print_msg("writing csv")

        # SpikeTimes
        for i, Seg in enumerate(Blk.segments):
            seg_name = Path(Seg.annotations['filename']).stem
            for j, unit in enumerate(units):
                St, = select_by_dict(Seg.spiketrains, unit=unit)
                outpath = results_folder / ("Segment_%s_unit_%s_spike_times.txt" % (seg_name, unit))
                np.savetxt(outpath, St.times.magnitude)

        if Frates:
            # firing rates - full res
            for i, Seg in enumerate(Blk.segments):
                FratesDf = pd.DataFrame()
                seg_name = Path(Seg.annotations['filename']).stem
                for j, unit in enumerate(units):
                    asig, = select_by_dict(Seg.analogsignals, kind='frate_fast', unit=unit)
                    FratesDf['t'] = asig.times.magnitude
                    FratesDf[unit] = asig.magnitude.flatten()

                outpath = results_folder / ("Segment_%s_frates.csv" % seg_name)
                FratesDf.to_csv(outpath)


def add_spikes_to_SpikeTrain(Blk, new_times, new_waveforms):
    st_times = Blk.segments[0].spiketrains[0].times
    times = np.append(st_times,new_times)

    st_waveforms = Blk.segments[0].spiketrains[0].waveforms
    waveforms = np.append(st_waveforms, new_waveforms)
    print(waveforms.shape)

    AnalogSignal = Blk.segments[0].analogsignals[0]

    SpikeTrain = neo.core.SpikeTrain(times,
                                     t_start=AnalogSignal.t_start,
                                     t_stop=AnalogSignal.t_stop,
                                     sampling_rate=AnalogSignal.sampling_rate,
                                     waveforms=waveforms,
                                     sort=True)

    Blk.segments[0].spiketrains[0] = SpikeTrain

    return Blk