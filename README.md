# Drosophila Spike Sorting
***Based on SSSort by Georg Raiser***
## Authors
**Alicia Garrido Peña**

**Thomas Nowotny**
## Description
## 0. Variables type
### Elephant
- Block
- Segment
	+ Analog signal
	+ Spike train
		* Time
		* waveforms

### SpikeInfo
Dataframe saving info for each spike detected. Columns:
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
### Templates


## 1. Spike detection
Detects spikes, reject non-spikes and saves templates in the ws form. 

- Positive: gets spikes based on a threshold 
- Negative: gets spikes based on a threshold from inverse signal
- Double: gets spikes combining positive and negative detection. 

Spike rejection:


## 2. Spike sorting by clustering
## 3. Pos-processing
### 3.1 Cluster identification
Identifies clusters from a loaded template. Necessary to distinguis between a cluster, b cluster and unknown cluster. 
Assigns a '-2' value to unknown cluster in a new unit column in SpikeInfo.

### 3.2 Reassign by amplitude
### 3.3 Reassign by composed spikes
### 3.4 Reassign by neighbors

![](../meetings/8-Oct-2021/bad_spike_example.png)
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


	

