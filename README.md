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
| VGG16                        |       N/A     |    93.25    |
| VGG16-pruned-A               |      64.00    |    93.40    |
|**MINT (a) (ours)**           |      --.--    |    --.--    | (My own baseline: 94.01)
|**MINT (b) (ours)**           |      --.--    |    --.--    | (My own baseline: 94.01)
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

----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (a)   |     75       |     8        |       250           |    --.--        |     --.--        | 
|   (a)   |     75       |     15       |       250           |    --.--        |     --.--        | 
|   (a)   |     75       |     20       |       250           |    --.--        |     --.--        | 
|   (a)   |     75       |     25       |       250           |    --.--        |     --.--        | 
|   ---   |     --       |     --       |       ---           |    -----        |     -----        | 
|   (b)   |     75       |     8        |       250           |    --.--        |     --.--        | 
|   (b)   |     75       |     15       |       250           |    --.--        |     --.--        | 
|   (b)   |     75       |     20       |       250           |    --.--        |     --.--        | 
|   (b)   |     75       |     25       |       250           |    --.--        |     --.--        | 
----------------------------------------------------------------------------------------------------

### Notes
- Between 8 and 15 groups, consistently 15 groups provided higher performance, even when overall params pruned was comparable.
- Why is 0.999 equating to only 39.53 or 44.61 of the params? The number of unique I\_parent values is smaller than the number of repeat values, Re-running with unique values used to compute percentage.
- > 70 % params pruned with performance > 93.40 achieved on vgg16 on CIFAR-10.
- Updating upper limit to 80% to observe results. Doesn't matter its a question of optimal parameters.
- Updating results with algorithm version b
- Found  bug in setup masks which used only 7 layers. Redid it to include all layers and rerunning vgg16 results.
- All layers seems to be worse, moving back to 7 layer, modified combine.py and rerunning for final results
