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
|   (b)   |     N/A      |     5        |       250           |    86.27        |     98.58        | (Requested Prune Percent: 0.292) 
|   (b)   |     N/A      |     10       |       250           |    88.23        |     98.56        | (Requested Prune Percent: 0.415) 
|   (b)   |     N/A      |     20       |       250           |    88.48        |     98.53        | (Requested Prune Percent: 0.340)
|   (b)   |     N/A      |     50       |       250           |    91.87        |     98.52        | (Requested Prune Percent: 0.376)
----------------------------------------------------------------------------------------------------

#### Sample variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     20       |       150           |    85.35        |     98.58        | (Requested Prune Percent: 0.352) 
|   (b)   |     N/A      |     20       |       250           |    88.48        |     98.53        | (Requested Prune Percent: 0.340)
|   (b)   |     N/A      |     20       |       350           |    88.48        |     98.55        | (Requested Prune Percent: 0.364)
|   (b)   |     N/A      |     20       |       450           |    88.72        |     98.51        | (Requested Prune Percent: 0.405)
|   (b)   |     N/A      |     20       |       550           |    79.41        |     98.54        | (Requested Prune Percent: 0.335)
|   (b)   |     N/A      |     20       |       650           |    --.--        |     98.5-        | (Requested Prune Percent: 0.---)
----------------------------------------------------------------------------------------------------
### Notes
- Anomaly is that increasing groups doesn't linearly help. There is a slight dip in params pruned. 
- There is almost a point afterwhich adding samples DOES NOT help. Instead of being stable it decreases, which is very interesting. Also, it is low by quite a margin.
- Average performance for 100s experiments = 91.73\% but how can we compare the accuracy portion?
- Possibly average the MI estimates over the runs and then compute accuracy. Maybe that will be stable...er?
- Group variation patterns seem non existent, possible due to conflict with number of samples.
- Sample variations seem to show opposite of expected trend, i.e., including more samples would be beneficial but in our case including makes us remove less nodes. Why?
- 1 possible explanation is that for 30 and 40 groups, they don't divide the number of nodes evenly and hence their results need to be discounted.
- Removed 30 and 40 groups and ran results on 1 system, hyaloidcanal. Results seem more consistent. 
- Current weird trend is that in increasing samples beyond a point, number of params to prune seems to dip. Confirming this by running 650samples/class experiment.

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
|   (b)   |     85       |     8        |       250           |    47.05        |     93.53        | (Requested prune percent: 0.508) 
|   (b)   |     85       |     16       |       250           |    --.--        |     93.--        | (Requested prune percent: 0.---) 
|   (b)   |     85       |     32       |       250           |    --.--        |     93.--        | (Requested prune percent: 0.---)
|   (b)   |     85       |     64       |       250           |    --.--        |     93.--        | (Requested prune percent: 0.---) 
----------------------------------------------------------------------------------------------------

#### Sample variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     85       |     64       |       150           |    --.--        |     93.--        | (Requested prune percent: 0.---) 
|   (b)   |     85       |     64       |       250           |    --.--        |     93.--        | (Requested prune percent: 0.---) 
|   (b)   |     85       |     64       |       350           |    --.--        |     93.--        | (Requested prune percent: 0.---)
|   (b)   |     85       |     64       |       450           |    --.--        |     --.--        | 
----------------------------------------------------------------------------------------------------

### Notes
- Big red flag in both group variations and sample variations. There is a sudden dip in performance (params pruned) which buckle the expected trend.
