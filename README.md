# Drosophila Spike Sorting
***Based on SSSort by Georg Raiser***
## Authors
**Alicia Garrido Peña**

**Thomas Nowotny**
## Description
## 0. Variables type
### NEO
https://neo.readthedocs.io/en/stable/

Library for neuroscience data. Recordings are organized into blocks and each block might contain several segments. En each segment you have the signal info including the signal trace and the spikes detected you have added into SpikeTrain. The spikes might be saved as time instant reference and as waveform. 

- Block --> Load from original data. See sssio.py
- Segment --> Access as seg = Blk.segment[0]; Note: In this code there was suppose to be only one segment. 
	+ Analog signal --> seg.analogsignal[0]
	+ Spike train --> st = seg.spiketrains[0]
		* Time --> st.times
		* waveforms --> st.waveforms

### SpikeInfo
Dataframe saving info for each spike detected. This contains the info of the cluster assignation in each iteration, labeled as unit_n. **The number of spikes in SpikeInfo must match the numbers of Templates.** 

Columns:

* 'id' --> spike reference
* 'time' --> spike time in s
* 'segment' --> segment where the spike is
* 'unit' --> first unit cluster assignation
* 'good', 
* 'frate_fast'
* 'frate_from_n'
* 'unit_n'
* 'unit_labeled' --> changed spikes from "cluster_identification.py"
* 'unit_amplitude'--> changed spikes from "post_processing_amplitude.py"
* 'unit_templates'--> changed spikes from "post_processing_templates.py"

**The labels**. Clusters will have an assigned **positive** integer number from the beggining and during merges some of them may disappear. The numbers in the final result will not necessarily be consecutive nor in order. When a spike is unassigned, it will be marked as -1. 

* 0-n Corresponding cluster unit.
* -1 unassigned spike
* -2 unknown cluster identified in "cluster_identification.py". Only when sssort ends at 3 clusters.

### Templates

## 1. Spike detection
This is run by "templates_extraction.py"
Detects spikes, reject non-spikes and saves templates in the ws form. 

Note: The spikes not detected here will not be used in the following steps of the algorithm. The spikes waveforms are saved into Templates_ini.npy and will be used in the following process.

- Positive: gets spikes based on a threshold.
- Negative: gets spikes based on a threshold from inverse signal.
- Double: gets spikes combining positive and negative detection. 

Spike rejection:
From the original detection removes spikes that are:

* First point is much smaller than last.
* The trace crosses mid point only once.
* Amplitude is too small for a spike.
* Duration is too short for a spike

########TODO insert examples images.

![](../images/bad_spike_example.png)

## 2. Spike sorting by clustering

The clustering is performed following the original algorithm. A model is generated based on the spike shape and its firing rate. 
1. Firing rates are estimated for each unit (cluster)
2. A model is predicted based on the linear relationship between PCA of the waveform and the predicted firing rate. 
3. It is checked how well do the spikes in the unit fit the prediction "Score". Each spike is assigned the unit with the min score. 

This is performed in each iteration. After this is done, the clusters are compared for a possible merge. A merge is performed when two clusters are so similar, based on a pairwise distance and a probability parameter clust_alpha.

### Changes and restrictions
The initial number of clusters is set in the config file at the beggining. This could be for example 9. From the initial clustering, the process avobe generating models is performed. Instead of n iterations, the loop will run until it converges to 2 or 3 clusters. 

For the definition of the algorithm, there is an ideal number of clusters when spikes will fit the model and the firing rate, for example when there are two stimulations, for each neuron, the ideal number of clusters could be 4: neuron a, neuron a stimulated, neuron b and neuron b stimulated. Up to this point, merge should be done so the clusters which spikes are similar will be together, but reassign unit based on the model prediction based on the firing rate might missassign spikes.

For that purpose, the parameter 'cluster_limit_train' would set the number of clusters after which there should be no more training and it should only merge. 

The algorithm might get stacked in a certain number of clusters, to avoid that, the merge probability parameter is modified after n unsuccessful merges (n is defined by it_no_merge').

#######TODO image

## 3. Post-processing
The aim of the postprocessing is to fix some spikes that might have been bad assigned. The postprocessing relay on an overall good spike clustering. If there are many spikes bad clustered it might fail.
1. Cluster identification: when there are 3 clusters, they will be label to a, b or unknown. The SpikeInfo is saved with a -2 for unknown.
2. Reassign by amplitude 
3. Reassign by composed spikes

### 3.1 Cluster identification
Identifies clusters from a loaded template. Necessary to distinguish between a cluster, b cluster and unknown cluster. 
Assigns a '-2' value to unknown cluster in a new unit column in SpikeInfo.

### 3.2 Assign unknown by composed templates.
If there is a third cluster, the average is loaded or calculated from the two identified clusters, and the combined spikes are calculated. The spikes in unit '-2' are assigned to the cluster which less distance.

### 3.3 Reassign by amplitude
Analyzes each spike and its neighbors amplitude by cluster. If the amplitude of an spike is more similar to their neighbors from the other cluster, the spike is reassigned.

### 3.4 Reassign by composed spikes
A matrix with all the possible combination of spikes is done. a+b; b+a; a, b and a+b at the same point. 
1. Computes the average of the 2 types of spikes or loads the default templates. 
2. Generates the general combined grid.
3. Generates specific template matrix based on the neighbors average.
4. Calculates the distance of the spike to each template
5. Gets best fit and compares to original situation. 
6. Saves all plots. 



## Instalation
This software runs in Python 3. There are some packages that need to be installed. Instalation options:
### 1. Conda
You can run Spike Sorting with Conda ([https://docs.anaconda.com/anaconda/install/index.html]) and use the environment on environment.yml in this repository. After creating the environment you might have all the packages necesaries. **You will have to activate the environment everytime you want to use it**.

Create conda environment

	conda env create -n SpikeSort -f environment.yml

Activate conda SpikeSort

    conda activate SpikeSort

For finishing using this environment:

	conda deactivate

### 2. Manual installation

If you want to manually install the packages instead of using conda, you can either run the scripts and install required packages or try running:

	./install_dependencies.sh

This bash script for Linux has the main dependencies listed to install with pip. (You can also use this as a reference)

## Use
If you are using conda run first:

	conda activate SpikeSort

Move to scripts path
		
	cd scripts

Convert file to .dill using:
	
	python smr2dill.py ../data/examples/sorttest.smr

Create config file with parameters from template: model.ini. Then run run_all.py script as:
	python run_all.py ../data/examples/model.ini

That will run the following scripts in order:

	python templates_extraction.py ../data/examples/model.ini

	python sssort.py ../data/examples/model.ini

	python cluster_identification.py ../data/examples/model.ini

	python label_unknown_spikes.py ../data/examples/model.ini

	python post_processing_amplitude.py ../data/examples/model.ini

    python post_processing_templates.py ../data/examples/model.ini

Every scripts uses a config file, you can see a commented example in model.ini

**Note:** Create one config file for each data file, ideally call the ini file same as the data .dill file.

## Utils

**Move to utils path before execute**

To plot the general result of a concrete experiment run plot_result with the path of the trial as argument:
    
    python plot_result.py ../../data/examples/sorttest/results/

To change one concrete spike you can run change_spike.py and specify the id and the label you want for the spike:
    
    python change_spike.py -p ../../data/examples/sorttest/results/ -id 392 -u b

Note: the arguments value must be specified with the argument flag as -p, -id or -u for path, spike id and unit, respectively.


## TODO
1. Spike detection:

	- Missing spikes !!!!!
	- "double" detection super slow: 
		
		a. save min and max at detection
		
		b. use positive with low threshold
	- Review spike rejection... Parameters hardcoded
	
2. Spike sort:

	- Spike rejection "bad spikes" overlaps with the initial rejection.
	- Remove small cluster... Not so necessary

2. Cluster identification

	- label spikes in SpikeInfo

4. postprocessing templates

	- Review outcome in different situations
	- Spike a+b add b spike to SpikeInfo and Templates

5. Script to change single spike given a reference
6. Script to plot final result
	




	

