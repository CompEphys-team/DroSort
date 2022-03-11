import matplotlib.pyplot as plt
import pandas as pd
from sssio import * 
from plotters import *
from functions import *
from sys import path
from postprocessing_functions import *

import configparser

################################################################
##  
##              Load clustering result
##
################################################################

# get config
config_path = Path(os.path.abspath(sys.argv[1]))
sssort_path = os.path.dirname(os.path.abspath(sys.argv[0]))
Config = configparser.ConfigParser()
Config.read(config_path)
print_msg('config file read from %s' % config_path)

# handling paths and creating outunits.sort()put directory
data_path = Path(Config.get('path', 'data_path'))
if not data_path.is_absolute():
    data_path = config_path.parent / data_path

exp_name = Config.get('path', 'experiment_name')
results_folder = config_path.parent / exp_name / 'results'
plots_folder = results_folder / 'plots' / 'post_process_all' 

os.makedirs(plots_folder, exist_ok=True)

Blk=get_data(results_folder/"result.dill")
SpikeInfo = pd.read_csv(results_folder/"SpikeInfo.csv")

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo, unit_column)

#plot config
plotting_changes = Config.getboolean('postprocessing','plot_changes')
mpl.rcParams['figure.dpi'] = Config.get('output','fig_dpi')
fig_format = Config.get('output','fig_format')

st = Blk.segments[0].spiketrains[0]
Seg = Blk.segments[0]

# recalculate the latest firing rates according to spike assignments in unit_column
kernel_slow = Config.getfloat('kernels','sigma_slow')
kernel_fast = Config.getfloat('kernels','sigma_fast')
calc_update_frates(Blk.segments, SpikeInfo, unit_column, kernel_fast, kernel_slow)

templates_path = config_path.parent / results_folder / "Templates_ini.npy"
Templates= np.load(templates_path)
print_msg('templates read from %s' % templates_path)
n_model_comp = Config.getint('spike model','n_model_comp')
Models = train_Models(SpikeInfo, unit_column, Templates, n_comp=n_model_comp, verbose=True)

unit_ids= SpikeInfo[unit_column]
units = get_units(SpikeInfo, unit_column)
frate= {}
for unit in units:
    frate[unit]= SpikeInfo['frate_from_'+unit]

sz_wd= Config.getfloat('postprocessing','spike_window_width')
align_mode= Config.get('postprocessing','vertical_align_mode')
tol= Config.getfloat('postprocessing','spike_position_tolerance')
tol= int(tol*10) # in time steps

asig= Seg.analogsignals[0]
asig= asig.reshape(asig.shape[0])
asig= align_to(asig,align_mode)
n_wd= int(sz_wd*10)
n_wdh= n_wd//2
for i in range(len(unit_ids)):
    start= int((float(st.times[i])*1000-sz_wd/2)*10)
    stop= start+n_wd
    if (start > 0) and (stop < len(asig)):   # only do something if the spike is not too close to the start or end of the recording, otherwise ignore
        v= asig[start:stop]
        d= []
        sh= []
        un= []
        templates= {}
        for unit in units[:2]:
            templates[unit]= make_single_template(Models[unit], frate[unit][i])
            templates[unit]= align_to(templates[unit],align_mode)
            for pos in range(n_wdh-tol,n_wdh+tol):
                d.append(dist(v,templates[unit],pos))
                sh.append(pos)
                un.append(unit)
        #print(d)
        #print(sh)
        #print(un)
        d2= []
        sh2= []
        for pos1 in range(n_wd):
            for pos2 in range(n_wd):
                # one of the spikes must be close to the spike time under consideration
                if (abs(pos1-n_wdh) <= tol) or (abs(pos2-n_wdh) <= tol):
                    d2.append(compound_dist(v,templates['a'],templates['b'],pos1,pos2))
                    sh2.append((pos1,pos2))
        fig, ax= plt.subplots(1,2, sharey= True)
        best= np.argmin(d)
        dist(v,templates[un[best]],sh[best],ax[0])
        best2= np.argmin(d2)
        compound_dist(v,templates['a'],templates['b'],sh2[best2][0],sh2[best2][1],ax[1])
        plt.show()
        
