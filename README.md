# Results for MINT: MI-based Neuron Trimming

### Experiment 1: [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

--------------------------------------------------------------
| Method                       | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| Baseline                     |       N/A     |    98.57    |
| Structured Sparsity Learning |       83.5    |    98.47    |
| Network Slimming             |       84.4    |    98.51    |
| *MINT (ours)*                |      *93.87*  |   *98.54*   |
--------------------------------------------------------------

### Results Compilation

------------------------------------------------------------------------------------------
| Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|     N/A      |     10       |       250           |    88.23        |     98.56        | (Requested Prune Percent: 0.443) 
|     N/A      |     20       |       250           |   *93.87*       |     98.54        | (Requested Prune Percent: 0.631)
------------------------------------------------------------------------------------------


### Experiment 2: [Pruning filters for efficient Convnets](https://openreview.net/pdf?id=rJqFGTslg)


--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| VGG16                        |       N/A     |    93.25    |
| VGG16-pruned-A               |      64.00    |    93.40    |
| MINT (ours)                  |      --.--    |    --.--    | (My own baseline: 94.01)
--------------------------------------------------------------


### VGG16 Layer Breakdown (Baseline)

----------------------------------------------------------------------------
| Layer pairs      |  Total Number of Params   |  Reduced Number of Params |
|:----------------:|:-------------------------:|:-------------------------:|
| Input - Conv1    |         = 1728            |                           |
|*Conv1 - Conv2*   |         = 36864           |                           |
| Conv2 - Conv3    |         = 73728           |                           |
| Conv3 - Conv4    |         = 147456          |                           | 
| Conv4 - Conv5    |         = 294912          |                           |
| Conv5 - Conv6    |         = 589824          |                           | 
| Conv6 - Conv7    |         = 589824          |                           | 
| Conv7 - Conv8    |         = 1179648         |                           | 
|*Conv8 - Conv9*   |         = 2359296         |                           | 
|*Conv9 - Conv10*  |         = 2359296         |                           | 
|*Conv10 - Conv11* |         = 2359296         |                           | 
|*Conv11 - Conv12* |         = 2359296         |                           | 
|*Conv12 - Conv13* |         = 2359296         |                           | 
|*Conv13 - Linear1*|         = 262144          |                           |
| Linear1- Linear2 |         = 5120            |                           |
| Total            |        *14977728*         |                           | 
----------------------------------------------------------------------------

### Results Compilation

------------------------------------------------------------------------------------------
| Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|     75       |     8        |       250           |    34.58        |     93.45        | 
|     75       |     15       |       250           |    71.40        |     93.50        | 
|     75       |     20       |       250           |    --.--        |     --.--        | 
|     75       |     25       |       250           |    --.--        |     --.--        | 
------------------------------------------------------------------------------------------

### Notes
- Between 8 and 15 groups, consistently 15 groups provided higher performance, even when overall params pruned was comparable.
- Why is 0.999 equating to only 39.53 or 44.61 of the params? The number of unique I\_parent values is smaller than the number of repeat values, Re-running with unique values used to compute percentage.
- > 70 % params pruned with performance > 93.40 achieved on vgg16 on CIFAR-10.
