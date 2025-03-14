# sys
from pathlib import Path

# sci
import scipy as sp
from scipy import signal, stats
import numpy as np

# ephys
import neo
import quantities as pq
import elephant as ele

# vis
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# own
from tools.functions import *

def get_colors(units, palette='tab10', desat=None, keep=True):
    """ return dict mapping unit labels to colors """
    if len(units) == 0:
        units= ['A', 'B']
    if 'A' in units or 'B' in units:
        n_colors = 2
        units= [ str(x) for x in units ]
    else:
        if keep:
            n_colors = np.array(units).astype('int32').max()+1
        else:
            n_colors = len(units)
    colors = sns.color_palette(palette, n_colors=n_colors, desat=desat)
    # unit_ids = sp.arange(n_colors).astype('U')
    sunits= units.copy()
    sunits.sort()
    return dict(zip(sunits,colors))

def plot_Model(Model, max_rate=None, N=5, ax=None):
    """ plots a single model on ax """
    if ax is None:
        fig, ax = plt.subplots()

    colors = sns.color_palette('inferno',n_colors=N)
    if max_rate is None:
        max_rate = np.clip(np.max(Model.frates),0,200)

    frates = sp.linspace(0, max_rate, N)
    for j,f in enumerate(frates):
        ax.plot(Model.predict(f),color=colors[j])
    ax.text(0.05, 0.05, "%.2f"%max_rate, horizontalalignment='left', verticalalignment='baseline', transform=ax.transAxes,fontsize='small')
    return ax

def plot_Models(Models, max_rates=None, N=5, unit_order=None, save=None, colors=None):
    """ plots all models """
    units = list(Models.keys())

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(ncols=len(units), sharey=True, figsize=[len(units),2])
    for i,unit in enumerate(units):
        if max_rates is not None:
            max_rate = max_rates[i]
        else:
            max_rate = None
        axes[i] = plot_Model(Models[unit], ax=axes[i], max_rate=max_rate)
        axes[i].set_title(unit,color=colors[unit])
        
    axes[0].set_ylabel('amplitude')
    sns.despine(fig)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
    return fig, axes

def plot_templates(Templates, SpikeInfo, unit_column=None, unit_order=None, N=100, save=None, colors=None):
    """ plots all templates """

    if unit_column is None:
        unit_column = 'unit'

    units = get_units(SpikeInfo, unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(ncols=len(units), sharey=True,  figsize=[len(units),2])

    for i, unit in enumerate(units):
        #AG prints the spikes marked as good in color and rest in black
        try:
            ix = SpikeInfo.groupby([unit_column,'good']).get_group((unit,True))['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Templates[:,ix]
            axes[i].plot(T, color=colors[unit],alpha=0.5,lw=1)
        except:
            pass

        try:
            ix = SpikeInfo.groupby([unit_column,'good']).get_group((unit,False))['id']
            if N is not None and ix.shape[0] > N:
                ix = ix.sample(N)
            T = Templates[:,ix]

            axes[i].plot(T, color='k',alpha=0.5,lw=1,zorder=-1)
        except:
            pass
        
        axes[i].set_title(unit)

    sns.despine()
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes

def plot_segment(Seg, units, sigma=0.05, zscore=False, save=None, colors=None):
    """ inspect plots a segment """
    if colors is None:
        colors = get_colors(units)

    fig, axes = plt.subplots(nrows=3,sharex=True)
    asig = Seg.analogsignals[0]
    for i, unit in enumerate(units):
        St, = select_by_dict(Seg.spiketrains, unit=unit)
        for t in St.times:
            axes[0].plot([t,t],[i-0.4, i+0.4], lw= 0.5, color=colors[unit])
        tvec = sp.linspace(asig.times.magnitude[0], asig.times.magnitude[-1], 1000)
        fr = est_rate(St.times.magnitude, tvec, sigma)
        if zscore:
            fr = ele.signal_processing.zscore(fr)
        axes[1].plot(tvec,fr, lw= 0.5, color=colors[unit])
    axes[2].plot(asig.times, asig.data, color='k',lw=0.3)

    # deco
    axes[0].set_yticks(range(len(units)))
    axes[0].set_yticklabels(units)
    axes[1].set_ylabel('firing rate (Hz)')
    axes[1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes

#TODO reduce complexity and combine to plot_compare spike events
def plot_spike_events(Segment,thres=2,max_window=1,max_row=5,save=None,save_format='.png',show=False,st=None,rejs=None):
    plt.rcParams.update({'font.size': 5})
    for asig in Segment.analogsignals:  
        max_window = int(max_window*asig.sampling_rate) #FIX conversion from secs to points

        asig=asig.reshape(asig.shape[0])
        asig_max= np.amax(asig)
        asig_min= np.amin(asig)
        n_rows = asig.shape[0]//max_window #compute number of rows needed to plot complete signal
        n_plots = n_rows//max_row+ int(not(n_rows/max_row).is_integer())

        #Plot max_row rows per window, plots n_plots to plot the complete signal
        for i_fig in range(0,n_plots):
            fig, axes = plt.subplots(nrows=max_row,sharey=True)

            for idx in range(0,max_row): #plot the max_row axes.
                #plot analog signal
                ini = i_fig*max_window*max_row+idx * max_window
                end = i_fig*max_window*max_row+idx * max_window+max_window
                
                end = min(end,asig.data.shape[0])-1
                if ini >= asig.data.shape[0]:
                    break

                axes[idx].plot(asig.times[ini:end],asig.data[ini:end],linewidth=1,color='k')
                
                if st is None:
                    st = Segment.spiketrains[0] #get spike trains (assuming there's only one spike train)

                t_ini = asig.times[ini]; t_end = asig.times[end]
                #get events in this chunk
                t_events = st.times[np.where((st.times > t_ini) & (st.times < t_end))]
                #get events amplitude value (spike)
                a_events = st.waveforms[np.where((st.times > t_ini) & (st.times < t_end))] 
                a_events = [max(a) for a in a_events]
                if rejs is not None:
                    chunk_r = rejs[np.where((rejs > t_ini) & (rejs < t_end))]
                    axes[idx].plot(chunk_r,asig_max*np.ones(chunk_r.shape),'|',markersize=5,color='r',label='rejected_spikes')

                axes[idx].plot(t_events,a_events,'|',markersize=5,label='detected_spikes',c=[ 0.5, 1.0, 0.5])
                axes[idx].plot(asig.times[ini:end],np.ones(asig.times[ini:end].shape)*thres,linewidth=0.5)
                axes[idx].set_ylim(asig_min*1.1,asig_max*1.1)
            if idx ==0:
                break

            #set plot info:
            fig.suptitle("Spike Detection (%d/%d)"%(i_fig+1,n_plots))
            axes[idx].set_xlabel("Time (%s)"%str(asig.times.units).split()[-1])
            v_unit = str(asig.units).split()[-1]
            if v_unit == 'dimensionless':
                v_unit = 'V'
            axes[idx//2].set_ylabel("Voltage (%s)"%v_unit)

            plt.tight_layout()

            if save is not None:
                fig.savefig(str(save)+"_%d"%i_fig+save_format)

            if show:
                plt.show()
            else:
                plt.close(fig)

    # return fig, axes

def plot_compared_spike_events(Segment1,Segment2,thres=2,max_window=1,max_row=5,save=None,save_format='.png',show=False):

    for asig1,asig2 in zip(Segment1.analogsignals,Segment2.analogsignals):  
        max_window = int(max_window*asig1.sampling_rate) #FIX conversion from secs to points

        asig1=asig1.reshape(asig1.shape[0])
        n_rows = asig1.shape[0]//max_window
        n_plots = n_rows//max_row+ int(not(n_rows/max_row).is_integer())

        #Plot max_row rows per window, plots n_plots to plot the complete signal
        for i_fig in range(0,n_plots):
            fig, axes = plt.subplots(nrows=max_row)

            for idx in range(0,max_row):
                #plot analog signal
                ini = i_fig*max_window*max_row+idx * max_window
                end = i_fig*max_window*max_row+idx * max_window+max_window
                
                end = min(end,asig1.data.shape[0])-1
                if ini >= asig1.data.shape[0]:
                    break

                axes[idx].plot(asig1.times[ini:end],asig1.data[ini:end],linewidth=1)

                #plot spike events
                st = Segment1.spiketrains[0] #get spike trains

                t_ini = asig1.times[ini]; t_end = asig1.times[end]
                #get events in this chunk
                t_events = st.times[np.where((st.times > t_ini) & (st.times < t_end))]
                #get events amplitude value (spike)
                a_events = st.waveforms[np.where((st.times > t_ini) & (st.times < t_end))] 
                a_events = [max(a) for a in a_events]

                axes[idx].plot(t_events,a_events,'.',markersize=1)


                #plot spike events from second Segment
                st = Segment2.spiketrains[0] #get spike trains
               
                #get events in this chunk
                t_events = st.times[np.where((st.times > t_ini) & (st.times < t_end))]
                #get events amplitude value (spike)
                a_events = st.waveforms[np.where((st.times > t_ini) & (st.times < t_end))] 
                a_events = [max(a) for a in a_events]

                axes[idx].plot(t_events,a_events,'.',markersize=1)


                axes[idx].plot(asig1.times[ini:end],np.ones(asig1.times[ini:end].shape)*thres,linewidth=0.5)

            if idx ==0:
                break
            #set plot info:
            fig.suptitle("Spike Detection (%d/%d)"%(i_fig+1,n_plots))
            axes[idx].set_xlabel("Time (%s)"%str(asig1.times.units).split()[-1])
            v_unit = str(asig1.units).split()[-1]
            if v_unit == 'dimensionless':
                v_unit = 'V'
            axes[idx//2].set_ylabel("Voltage (%s)"%v_unit)

            plt.tight_layout()

            if save is not None:
                fig.savefig(str(save)+"_%d"%i_fig+save_format)

            if show:
                plt.show()
            else:
                plt.close(fig)

    # return fig, axes

def plot_fitted_spikes(Segment, Models, SpikeInfo, unit_column, unit_order=None, zoom=None, box= None, save=None, colors=None,wsize=40,rejs=None, spike_label_interval= 0):
    """ plot to inspect fitted spikes """
    fig, axes = plt.subplots(nrows=2, sharex=True, sharey=True, num=1, clear=True, figsize= [ 4, 3 ])

    asig = Segment.analogsignals[0]
    fs = asig.sampling_rate
    as_min= np.amin(asig)
    as_max= np.amax(asig)
    ylim= [ 1.1*as_min, 1.1*as_max ]
    
    #if zoom is not None:
        #left= max(int(zoom[0]*fs),0)
        #right= min(int(zoom[1]*fs),len(asig.data))
    #else:
        #left= 0
        #right= len(asig.data)

    #axes[0].plot(asig.times[left:right], asig.data[left:right], color='k', lw=0.5)
    #axes[1].plot(asig.times[left:right], asig.data[left:right], color='k', lw=0.5)
    axes[0].plot(asig.times, asig.data, color='k', lw=0.5)
    axes[1].plot(asig.times, asig.data, color='k', lw=0.5)

    if zoom is not None:
        for ax in axes:
            ax.set_xlim(zoom)

    for ax in axes:
        ax.set_ylim(ylim)

    st = Segment.spiketrains[0]  # get all spike trains (assuming there's only one spike train)
    if rejs is None:
        rejs= SpikeInfo["time"][SpikeInfo[unit_column] == '-2']
    units = get_units(SpikeInfo,unit_column)
    fac= axes[1].get_ylim()[1]*(0.95-len(units)*0.05)
    axes[1].plot(rejs, np.ones(rejs.shape)*fac, '|', markersize=2, color='k', label="rejected spike")

    plot_by_unit(axes[1], st, asig, Models, SpikeInfo, unit_column, unit_order,
                 colors, wsize)
            
    if box is not None:
        plot_box(axes[0], box)
        plot_box(axes[1], box)
       
    plot_spike_labels(axes[0], SpikeInfo, spike_label_interval)
    plot_spike_labels(axes[1], SpikeInfo, spike_label_interval)
    fig.tight_layout()
    #fig.subplots_adjust(top=0.9)
    #sns.despine(fig)
    
    fig.legend()

    if save is not None:
        fig.savefig(save)
        # plt.close(fig)

    return fig, axes

def plot_fitted_spikes_complete(seg, Models, SpikeInfo, unit_column,max_window, plots_folder, fig_format, unit_order=None, save=None, colors=None,wsize=40,extension='',plot_function=plot_fitted_spikes,rejs=None,spike_label_interval=0):
    
    asig = seg.analogsignals[0]

    max_window = int(max_window*asig.sampling_rate) #FIX conversion from secs to points
    n_plots = asig.shape[0]//max_window

    for n_plot in range(0,n_plots):
        outpath = plots_folder / ('fitted_spikes%s_%s_%d'%(extension,max_window,n_plot) + fig_format)
        ini = n_plot*max_window 
        end = ini + max_window
        zoom = [ini,end]/asig.sampling_rate

        if rejs is None:
            rejs= SpikeInfo["time"][SpikeInfo[unit_column] == '-2']

        plot_function(seg, Models, SpikeInfo, unit_column, zoom=zoom, save=outpath,wsize=wsize, rejs=rejs,spike_label_interval=spike_label_interval)

def plot_by_unit(ax,st, asig,Models, SpikeInfo, unit_column, unit_order=None, colors=None,wsize=40):

    try:
        left= wsize[0]
        right= wsize[1]
    except:
        left= wsize//2
        right= wsize//2

    units = get_units(SpikeInfo,unit_column)
    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    fs = asig.sampling_rate

    for u, unit in enumerate(units):
        # St, = select_by_dict(Segment.spiketrains, unit=unit)
        fac= ax.get_ylim()[1]*(0.95-0.05*u)
        times= SpikeInfo["time"][SpikeInfo[unit_column] == unit]
        ax.plot(times, np.ones(times.shape)*fac, '|', markersize=2, linewidth= 0.5, label=unit+' spikes',color=colors[unit])
        asig_recons = sp.zeros(asig.shape[0])
        asig_recons[:] = sp.nan 

        # inds = (St.times * fs).simplified.magnitude.astype('int32')

        # get times from SpikeInfo so units are extracted 
        # from Spike and not from annotation in spiketrains
        times = SpikeInfo.groupby([unit_column]).get_group((unit))['time'].values

        # TN: I don't see the point in this & it breaks where new spikes are inserted
        #fr = (asig.times[1]-asig.times[0]).simplified.magnitude.astype('int32')
        #Inds = [np.where(np.isclose(t,np.array(st.times),atol=fr))[0][0] for t in np.array(times)]

        #inds = (st.times[Inds]*fs).simplified.magnitude.astype('int32')
        inds = (times*fs).simplified.magnitude.astype('int32')
        
        offset = (st.t_start * fs).simplified.magnitude.astype('int32')
        inds = inds - offset

        try:
            if type(Models).__name__=='dict':
                frates = SpikeInfo.groupby([unit_column]).get_group((unit))['frate_fast'].values
                pred_spikes = [Models[unit].predict(f) for f in frates]
            else:
                Templates = Models
                ix = SpikeInfo.groupby([unit_column]).get_group((unit))['id']
                pred_spikes = Templates[:,ix].T

            for i, spike in enumerate(pred_spikes):
                try:
                    asig_recons[inds[i]-left:inds[i]+right] = spike
                
                except ValueError as e:
                    print("In plot by unit exception:",e.args)
                    # thrown when first or last spike smaller than reconstruction window
                    continue
            ax.plot(asig.times, asig_recons, lw=1.0, color=colors[unit], alpha=0.8)
                  
        except KeyError:
            # thrown when no spikes are present in this segment
            pass
        
#TODO add "rej" spikes in function
def plot_compared_fitted_spikes(Segment, j, Models, SpikeInfo, unit_columns, unit_order=None, zoom=None, save=None, colors=None,wsize=40,rejs=None,title=None):
    """ plot to inspect fitted spikes """
    fig, axes =plt.subplots(nrows=2, sharex=True, sharey=True, num=1, clear=True)

    asig = Segment.analogsignals[0]

    axes[0].plot(asig.times, asig.data, color='k', lw=1)
    axes[1].plot(asig.times, asig.data, color='k', lw=1)

    st = Segment.spiketrains[0]  # get all spike trains (assuming there's only one spike train)
    # get events amplitude value (spike)
    # a_events = st.waveforms
    # a_events = [max(a) for a in a_events]
    # axes[1].plot(st.times,np.ones(st.times.shape),'|',markersize=2)

    if '-1' in SpikeInfo[unit_columns[1]]:
        removed = SpikeInfo.groupby(unit_columns[1]).get_group('-1')['time']
        axes[1].plot(removed, np.ones(removed.shape), '|', markersize=3, color='r')

    plt.title(unit_columns[0])
    plot_by_unit(axes[0], st, asig, Models, SpikeInfo, unit_columns[0], unit_order, colors, wsize)
    plt.title(unit_columns[1])
    plot_by_unit(axes[1], st, asig, Models, SpikeInfo, unit_columns[1], unit_order, colors, wsize)

    if zoom is not None:
        for ax in axes:
            ax.set_xlim(zoom)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        # plt.close(fig)

        # WARNING: do not add plt.close; figure clears by definition
        #         (arg: num=1, clear=True) adding plt.close leaks memory

    return fig, axes


def plot_templates_on_trace(Segment, j, Templates, zoom=None, save=None,wsize=40):
    """ plot to inspect fitted spikes """
    fig, axes =plt.subplots(nrows=2, sharex=True, sharey=True)
    try:
        left= wsize[0]
        right= wsize[1]
    except:        
        left= wsize//2
        right= wsize//2

    asig = Segment.analogsignals[0]
    axes[0].plot(asig.times, asig.data, color='k', lw=1)
    axes[1].plot(asig.times, asig.data, color='k', lw=1)

    st = Segment.spiketrains[0] #get spike trains (assuming there's only one spike train)

    fs = asig.sampling_rate

    asig_recons = sp.zeros(asig.shape[0])
    asig_recons[:] = sp.nan

    inds = (st.times * fs).simplified.magnitude.astype('int32')
    offset = (st.t_start * fs).simplified.magnitude.astype('int32')
    inds = inds - offset

    pred_spikes = Templates[:, :].T

    for i, spike in enumerate(pred_spikes):
        asig_recons[inds[i] - left:inds[i] + right] = spike

    axes[1].plot(asig.times, asig_recons, lw=1.5, color='b')

    # get events amplitude value (spike)
    a_events = st.waveforms
    a_events = [max(a) for a in a_events]

    axes[1].plot(st.times, a_events, '.', markersize=2, color='r')

    if zoom is not None:
        for ax in axes:
            ax.set_xlim(zoom)
            
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    sns.despine(fig)

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes


def plot_convergence(ScoresSum, save=None):
    """ convergence check """
    fig, axes = plt.subplots(figsize=[4,3])

    its = range(len(ScoresSum))
    axes.plot(its, ScoresSum, ':', color='k')
    axes.plot(its, ScoresSum, 'o')
    axes.set_title('convergence plot')
    axes.set_xlabel('iteration')
    axes.set_ylabel('total sum of best Rss')
    
    sns.despine(fig)
    fig.tight_layout()
    if save is not None:
        fig.savefig(save)
        plt.close(fig)

def plot_clustering(Templates, SpikeInfo, unit_column, n_components=5, N=300, save=None, colors=None, unit_order=None):
    """ clustering inspect """
    units = get_units(SpikeInfo,unit_column)

    if unit_order is not None:
        units = [units[i] for i in unit_order]

    if colors is None:
        colors = get_colors(units)

    spike_labels = SpikeInfo[unit_column]

    # pca
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(Templates.T)

    fig, axes = plt.subplots(figsize=[8,8], nrows=n_components, ncols=n_components,sharex=True,sharey=True)

    for unit in units:
        ix = sp.where(spike_labels == unit)[0]
        x = X[ix,:]
        if N > x.shape[0]:
            N = x.shape[0]
        for i in range(n_components):
            for j in range(n_components):
                ix = sp.random.randint(0,x.shape[0],size=N)
                axes[i,j].plot(x[ix,i],x[ix,j],'.',color=colors[unit],markersize=1, alpha=0.5)

    for ax in axes.flatten():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    sns.despine(fig)
    fig.tight_layout()

    if save is not None:
        fig.savefig(save)
        plt.close(fig)

    return fig, axes

def plot_averages(average_spikes,SpikeInfo,unit_column,units,colors=None,title=None):
    fig, axes =plt.subplots(ncols=len(average_spikes), sharex=True, sharey=True)

    # units = get_units(SpikeInfo,unit_column)
    
    if colors is None:
        colors = get_colors(units)

    for i,average_spike in enumerate(average_spikes):
        axes[i].plot(average_spike,color=colors[units[i]])
        if title is None:
            axes[i].set_title(units[i])
        else:
            axes[i].set_title(title[i])

    fig.suptitle("Average of clusters")
    fig.tight_layout()

    return fig,axes



def plot_averages_with_spike(spike,average_spikes,SpikeInfo,unit_column,unit,colors=None):
    fig, axes =plt.subplots(ncols=len(average_spikes), sharex=True, sharey=True)

    units = get_units(SpikeInfo,unit_column)
    
    if colors is None:
        colors = get_colors(units)

    for i,average_spike in enumerate(average_spikes):
        axes[i].plot(spike,color=colors[unit])
        axes[i].plot(average_spike,color=colors[units[i]])
        axes[i].set_title(units[i])

    fig.suptitle("Average of clusters")
    fig.tight_layout()

    return fig,axes


def plot_means(means,units,template_a,template_b,asigs,outpath=None,show=False,colors=None):
    fig, axes = plt.subplots(ncols=len(units), figsize=[len(units)*3,4])

    if colors is None:
        colors = {'A':'b','B':'g','?':'r'}

    for i,(mean,unit) in enumerate(zip(means,units)):
        axes[i].plot(mean,label=unit,color='k',linewidth=0.7)
        axes[i].plot(mean,color=colors[asigs[unit]],alpha=0.3,linewidth=5)
        axes[i].plot(template_a,label="A",color=colors['A'],linewidth=0.7)
        axes[i].plot(template_b,label="B",color=colors['B'],linewidth=0.7)
        axes[i].legend()

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath)
    if show:
        plt.show()

def plot_box(ax, box):
    bot= ax.get_ylim()[0]
    top= ax.get_ylim()[1]
    left= box[0]-box[1]/2
    right= box[0]+box[1]/2
    x= [ left, left, right, right, left ]
    y= [ bot, top, top, bot, bot ]
    ax.plot(x,y, 'r', lw= 0.5)
    
def plot_spike_labels(ax, SpikeInfo, spike_label_interval):
    if spike_label_interval > 0:
        xrg= ax.get_xlim() 
        lbl= SpikeInfo['id'][::spike_label_interval]
        xpo= SpikeInfo['time'][::spike_label_interval]
        lbl= lbl[np.logical_and(xpo > xrg[0], xpo < xrg[1])]
        xpo= xpo[np.logical_and(xpo > xrg[0], xpo < xrg[1])]
        fac= ax.get_ylim()[0]*0.9
        ypo= np.ones(len(xpo))*fac
        for x, y, s in zip(xpo,ypo,lbl):
            ax.text(x, y, str(s),ha='center',fontsize=4)
