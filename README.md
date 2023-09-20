# LipenOptimization
 This is the code for the LipenOptimisation" Project. It is an extention of the LipenProject  <a href="https://github.com/theATM/LipenProject"> (avaiable here)</a>.

 Main goal is to optimize the model using pruning and quantisation.


## Trening 
To train run the train.py script

### Standard Trening Arguments
* labels -  pass csv file with matching labels (is specific format)
* num-classes - pas the numer of classes (3 or 6)
* run-dir - dir for tensorblard logs
* save-dir - dir to save model
* class-names - names of the classes used for the confusion matrix  eq. "WritingTool, Rubber, MeasurementTool"

### Training Configurations

#### Original
train.py args:
* labels ./dataset/UniformDatasetLabel.csv
* num-classes 6
* run-dir checkpoints
* save-dir checkpoints
* class-names "Pen, Pencil, Rubber, Ruler, Triangle, None"
* seed xxxxxx


#### Baseline
train.py args:
* labels ./dataset/ReducedDatasetLabel.csv
* run-dir checkpoints
* save-dir checkpoints
* weights weights.pt
* class-names "WritingTool, Rubber, MeasurementTool"
* num-classes 3
* freeze
* seed xxxxxx
* weighted-loss


#### Pruning
train.py args:
* labels ./dataset/ReducedDatasetLabel.csv
* run-dir checkpoints
* save-dir checkpoints
* weights weights.pt
* class-names "WritingTool, Rubber, MeasurementTool"
* num-classes 3
* freeze
* prune
* seed xxxxxx
* weighted-loss
* non-zero-params 0.2

#### Quantization
train.py args:
* labels ./dataset/ReducedDatasetLabel.csv
* run-dir checkpoints
* save-dir checkpoints
* weights weights.pt
* class-names "WritingTool, Rubber, MeasurementTool" 
* num-classes 3
* freeze
* seed xxxxxx
* weighted-loss
* quantization


## Test
Test is done using the test_model_py script, using the "weights" arg to laod the model.


## Pruning and Quantization Results

Small CNN model have been optimised without large loss in the Accuracy as seen in the table below.


| Optimized Model | Result | Comparison |
|-|-|-|
| Interference Time on CPU|          164,.7 ms|  75.7% reduction| 
| One Epoch Training Time on GPU  |  0.1 s |     16,7% reduction|	  
| One Epoch Training Time on CPU |   1.68 s |    3% reduction|
| Non-Zero Parameters            |   68869  |    19.9% of all parameters|
| Model Size                     |   0.36 MB	|   74% reduction in size|
| Validation Accurac             |   56,22%	 |  1,6% decrease|
| Test Accuracy	                 |   51,89%	|   3,5% decrease|



## Additional Info

Project Documentation (in Polish):
https://docs.google.com/document/d/1dtHW9IPJ4MOiHkUW2RpSrk-0lBKWL2s_HENMbopEVR8/edit?usp=sharing
