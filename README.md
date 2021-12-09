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
* 'unit_amplitude'--> changed spikes from "pos_processing_amplitude.py"
* 'unit_templates'--> changed spikes from "pos_processing_templates.py"

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

## 2. Spike sorting by clustering

## 3. Pos-processing
### 3.1 Cluster identification
Identifies clusters from a loaded template. Necessary to distinguis between a cluster, b cluster and unknown cluster. 
Assigns a '-2' value to unknown cluster in a new unit column in SpikeInfo.

### 3.2 Reassign by amplitude
### 3.3 Reassign by composed spikes
### 3.4 Reassign by neighbors


## Use
Create config file with parameters from template: model.ini. Then run run_all.py script as:
	
	python3 run_all.py a_path/model.ini

That will run the following scripts in order:
python3 run 


## TODO
1. Spike detection:
	- Missing spikes !!!!!^
	- "double" detection super slow: 
		
		a. save min and max at detection
		
		b. use positive with low threshold
	- Review spike rejection... Parameters hardcoded
	
2. Spike sort:

	- Unused spike rejection --> -1 spikes complicated.
	- Remove small cluster tricky

2. Cluster identification

	- See possible -1 unit.
3. Posprocessing amplitude
4. Posprocessing templates
	- General template
		1. Add -2 unit alternative
		2. Review outcome
		3. Analyze single spikes too???
	- Neighbors
		1. Spike templates in b spike have a spike... ?!?!?!


	

