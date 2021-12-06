from functions import *

import numpy as np
def get_neighbors_time(asig,st,n_samples,n_neighbors):
    dt = asig.times[1]-asig.times[0]
    dt = dt.item()
    spike_time = n_samples*dt
    isis = [ b-a for a,b in zip(st.times[:-1],st.times[1:])]
    # print(np.mean(isis),spike_time)

    return spike_time*n_neighbors + np.mean(isis)

def get_spikes_ids(id_ref,time,SpikeInfo,spike_train):
    times_all = SpikeInfo['time']

    idx_t = times_all.values[id_ref]

    ini = idx_t - time
    end = idx_t + time
    # times = times_all.index[np.where((times_all.values > ini) & (times_all.values < end) & (times_all.values != idx_t))]

    # neighbors = times[np.where(SpikeInfo.loc[times,unit_column].values==unit)]
    ids = np.where((spike_train.times > ini) & (spike_train.times < end) & (spike_train.times != idx_t))


    return ids

def align_spikes(spikes,mode):

    return np.array([align_to(spike, mode) for spike in spikes.T]).T



def get_averages_from_units(aligned_spikes,units,SpikeInfo,unit_column,verbose=False):
    """
    """
    average_spikes = []
    for unit in units:
        try:
           ix = SpikeInfo.groupby(unit_column).get_group(unit)['id']

           templates = aligned_spikes[:,ix].T
           if verbose:
               print_msg("Averaging %d spikes for cluster %s"%(len(templates),unit))
           average_spike = np.average(templates, axis=0)
        except Exception as e:
            print_msg("Exception in averaging"+str(e.args)+" generating zeros average for %.3f s"%SpikeInfo['time'].iloc[len(SpikeInfo)//2])
            average_spike = np.zeros(aligned_spikes.shape[0])

        average_spikes.append(average_spike)

    return average_spikes



def get_combined_templates(average_spikes,dt_c,max_len,mode):
    combined_templates = []
    A,B = average_spikes
    #Add AB templates
    combine_templates(combined_templates,A,B,dt_c,max_len,mode)
    n = len(combined_templates)
    templates_labels = [['a','b']]*n

    #Add BA templates
    combine_templates(combined_templates,B,A,dt_c,max_len,mode)
    n = len(combined_templates) -n
    templates_labels+=[['b','a']]*n

    #Add sum of two
    comb_t =np.array(np.concatenate((A,[A[-1]]*max_len))+np.concatenate((B,[B[-1]]*max_len)))
    # comb_t =np.array(np.concatenate(([A[0]]*(n_samples//2),A,[A[-1]]*(n_samples//2)))+np.concatenate(([B[0]]*(n_samples//2),B,[B[-1]]*(n_samples//2))))
    combined_templates.append(np.array(align_to(comb_t,mode)))
    templates_labels.append(['c'])

    #Add simple A
    comb_t =np.array(np.concatenate((A,[A[-1]]*max_len)))
    # comb_t =np.array(np.concatenate(([A[0]]*(n_samples//2),A,[A[-1]]*(n_samples//2))))
    combined_templates.append(np.array(align_to(comb_t,mode)))
    templates_labels.append(['a'])

    #Add simple B
    comb_t =np.array(np.concatenate((B,[B[-1]]*max_len)))
    # comb_t =np.array(np.concatenate(([B[0]]*(n_samples//2),B,[B[-1]]*(n_samples//2))))
    combined_templates.append(np.array(align_to(comb_t,mode)))
    templates_labels.append(['b'])

    return combined_templates,templates_labels


#TODO superpos spikes needed????
#from superpos_functions import align_to
def combine_templates(combined_templates,A,B,dt,max_len,align_mode):
    # n_samples = np.sum(w_samples)
    # max_len = w_samples[1]-1
    for dt in np.arange(0,max_len,dt):
        long_a = np.concatenate((A,[A[-1]]*(max_len)))
        long_b = np.concatenate(([B[0]]*abs(max_len-dt),B,[B[-1]]*dt))

        comb_t = np.array(long_a+long_b)
        combined_templates.append(np.array(align_to(comb_t,align_mode)))

def plot_combined_templates(combined_templates,templates_labels,ncols=5):
    # ncols = 5 
    nrows = int(np.ceil(len(combined_templates)/ncols))

    fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True,figsize=(ncols*2,nrows))
    # fig, axes= plt.subplots(nrows=nrows,ncols=ncols, sharex=True, sharey=True)

    for c,ct in enumerate(combined_templates):
        i,j = c//ncols,c%ncols

        axes[i,j].plot(ct)

        peak_inds = signal.argrelmax(ct)[0]
        peak_inds = peak_inds[np.argsort(ct[peak_inds])[-len(templates_labels[c]):]]

        axes[i,j].plot(peak_inds,ct[peak_inds],'.')
        axes[i,j].text(0.25, 0.75,str(templates_labels[c]))
        # plt.ylim(-2,0)

    plt.tight_layout()
