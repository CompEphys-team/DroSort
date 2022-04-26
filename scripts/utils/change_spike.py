import pandas as pd
from sssio import * 
from plotters import *
from functions import *
import argparse

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--path", required=True, help="Path to the experiment")
ap.add_argument("-id", "--id", required=True, help="Spike id")
ap.add_argument("-u", "--unit", required=True, help="New unit: 'a' or 'b'")
# ap.add_argument("-r","--range",required=False,default=None, help="Range of data format 1.2,1.3")

args = vars(ap.parse_args())


path = args['path']
spike_id = int(args['id'])
new_unit = args['unit']


results_folder = Path(os.path.abspath(path))
# fig_format = '.png'
Blk = get_data(results_folder / "result.dill")


SpikeInfo = pd.read_csv(results_folder / "SpikeInfo.csv")

Templates = np.load(results_folder / "Templates_final.npy")

# plots_folder = results_folder / 'plots' / 'final_result'
# os.makedirs(plots_folder, exist_ok=True)

n_samples = Templates.shape[0]
unit_column = [col for col in SpikeInfo.columns if col.startswith('unit')][-1]
SpikeInfo = SpikeInfo.astype({unit_column: str})
units = get_units(SpikeInfo, unit_column)

# New column for result
new_labels = copy.deepcopy(SpikeInfo[unit_column].values)
new_column = 'unit_changed_%d' % spike_id
SpikeInfo[new_column] = new_labels

print_msg("Changing from %s to %s" % (SpikeInfo[unit_column].iloc[spike_id], new_unit))

SpikeInfo[new_column].iloc[spike_id] = new_unit


print_msg("Saving SpikeInfo, Blk and Spikes into disk")

units = get_units(SpikeInfo, new_column)
Blk = populate_block(Blk, SpikeInfo, new_column, units)
output_csv = True
save_all(results_folder, output_csv, SpikeInfo, Blk, units)


peak = SpikeInfo['time'][spike_id]

zoom = [peak - 0.2, peak + 0.2]

Seg = Blk.segments[0]
fig, axes = plot_compared_fitted_spikes(Seg, 0, Templates[:40, :], SpikeInfo,
		[unit_column, new_column], zoom=zoom, save=None, title='')

plt.show()
