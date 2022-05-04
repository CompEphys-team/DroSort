import sys
sys.path.append('../')

import pandas as pd
from tools.sssio import * 
from tools.plotters import *
from tools.functions import *

# path = sys.argv[1]

results_folder = Path(os.path.abspath(sys.argv[1]))
fig_format = '.png'
Blk = get_data(results_folder / "result.dill")
mpl.rcParams['figure.dpi'] = 300


SpikeInfo = pd.read_csv(results_folder / "SpikeInfo.csv")

Templates = np.load(results_folder / "Templates_final.npy")

plots_folder = results_folder / 'plots' / 'final_result'
os.makedirs(plots_folder, exist_ok=True)

n_samples = Templates.shape[0]

unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
print(SpikeInfo[unit_column].value_counts())

SpikeInfo = SpikeInfo.astype({unit_column: str})

units = get_units(SpikeInfo, unit_column)
# print(units)

colors = get_colors(units)

# plot all sorted spikes
for j, Seg in enumerate(Blk.segments):
    seg_name = Path(Seg.annotations['filename']).stem
    outpath = plots_folder / (seg_name + '_final_overview' + fig_format)
    plot_segment(Seg, units, save=outpath)

# plot all sorted spikes
max_window = 0.3  # AG: TODO add to config file
plot_fitted_spikes_complete(Blk.segments[0], Templates, SpikeInfo, unit_column, max_window, plots_folder, fig_format, wsize=n_samples, extension='_final')


if len(sys.argv) > 2:
    zoom = sp.array(sys.argv[2].split(','), dtype='float32')
    Seg = Blk.segments[0]
    outpath = plots_folder / (seg_name + '_fitted_spikes_final_zoom' + fig_format)
    plot_fitted_spikes(Seg, 0, Templates, SpikeInfo, unit_column, zoom=zoom, save=outpath,wsize=n_samples)
