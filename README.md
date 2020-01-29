# Results for MINT: MI-based Neuron Trimming

### Experiment 1: [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

--------------------------------------------------------------
| Method                       | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| Baseline  (ours)             |       N/A     |    98.59    |
| Structured Sparsity Learning |       83.5    |    98.47    |
| Network Slimming             |       84.4    |    98.51    |
| MINT (a) (ours)              |       --.-    |    --.--    |
| MINT (b) (ours)              |       --.-    |    --.--    |
--------------------------------------------------------------

### Results Compilation
#### Group variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     5        |       250           |    90.19        |     98.53        | (Requested Prune Percent: 0.307) 
|   (b)   |     N/A      |     10       |       250           |    83.33        |     98.55        | (Requested Prune Percent: 0.340) 
|   (b)   |     N/A      |     20       |       250           |    84.07        |     98.53        | (Requested Prune Percent: 0.324)
|   (b)   |     N/A      |     30       |       250           |    84.49        |     98.50        | (Requested Prune Percent: 0.398)
|   (b)   |     N/A      |     40       |       250           |    89.45        |     98.65        | (Requested Prune Percent: 0.848)
|   (b)   |     N/A      |     50       |       250           |    92.81        |     98.50        | (Requested Prune Percent: 0.460)
----------------------------------------------------------------------------------------------------

#### Sample variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     50       |       100           |    --.--        |     --.--        | (Requested Prune Percent: 0.307) 
|   (b)   |     N/A      |     50       |       150           |    --.--        |     --.--        | (Requested Prune Percent: 0.340) 
|   (b)   |     N/A      |     50       |       200           |    --.--        |     --.--        | (Requested Prune Percent: 0.324)
|   (b)   |     N/A      |     50       |       250           |    84.49        |     98.50        | (Requested Prune Percent: 0.398)
|   (b)   |     N/A      |     50       |       300           |    --.--        |     --.--        | (Requested Prune Percent: 0.848)
|   (b)   |     N/A      |     50       |       350           |    --.--        |     --.--        | (Requested Prune Percent: 0.460)
----------------------------------------------------------------------------------------------------
### Notes
- Anomaly is Groups=5 result since that doesn't follow the trend.


### Experiment 2: [Pruning filters for efficient Convnets](https://openreview.net/pdf?id=rJqFGTslg)


--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| VGG16(ours)                  |       N/A     |    93.98    |
| VGG16-pruned-A               |      64.00    |    93.40    |
|**MINT (a) (ours)**           |      --.--    |    --.--    |
|**MINT (b) (ours)**           |      --.--    |    --.--    |
--------------------------------------------------------------


### VGG16 Layer Breakdown (Baseline)

----------------------------------------------------------------------------
| Layer pairs      |  Total Number of Params   |  Reduced Number of Params |
|:----------------:|:-------------------------:|:-------------------------:|
| Input - Conv1    |         = 1728            |         1728              |
|*Conv1 - Conv2*   |         = 36864           |                           |
| Conv2 - Conv3    |         = 73728           |         73728             |
| Conv3 - Conv4    |         = 147456          |         147456            | 
| Conv4 - Conv5    |         = 294912          |         294912            |
| Conv5 - Conv6    |         = 589824          |         589824            | 
| Conv6 - Conv7    |         = 589824          |         589824            | 
| Conv7 - Conv8    |         = 1179648         |         1179648           | 
|*Conv8 - Conv9*   |         = 2359296         |                           | 
|*Conv9 - Conv10*  |         = 2359296         |                           | 
|*Conv10 - Conv11* |         = 2359296         |                           | 
|*Conv11 - Conv12* |         = 2359296         |                           | 
|*Conv12 - Conv13* |         = 2359296         |                           | 
|*Conv13 - Linear1*|         = 262144          |                           |
| Linear1- Linear2 |         = 5120            |         5120              |
| Total            |       **14977728**        |                           | 
----------------------------------------------------------------------------
Untouched params = 2882240/14977728 = 19.24\%

### Results Compilation
#### Group variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     85       |     8        |       250           |    47.05        |     93.51        | (Requested prune percent: 0.508) 
|   (b)   |     85       |     16       |       250           |    48.58        |     93.43        | (Requested prune percent: 0.592) 
|   (b)   |     85       |     32       |       250           |    26.85        |     93.46        | (Requested prune percent: 0.376)
|   (b)   |     85       |     64       |       250           |    50.72        |     93.49        | (Requested prune percent: 0.388) 
----------------------------------------------------------------------------------------------------

#### Sample variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     85       |     64       |       150           |    --.--        |     --.--        | (Requested prune percent: -.---) 
|   (b)   |     85       |     64       |       250           |    --.--        |     --.--        | (Requested prune percent: -.---) 
|   (b)   |     85       |     64       |       350           |    --.--        |     --.--        | (Requested prune percent: -.---)
|   (b)   |     85       |     64       |       450           |    --.--        |     --.--        | 
----------------------------------------------------------------------------------------------------

### Notes
