# Results for MINT: MI-based Neuron Trimming

### Experiment 1: [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

--------------------------------------------------------------
| Method                       | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| Baseline                     |       N/A     |    98.57    |
| Structured Sparsity Learning |       83.5    |    98.47    |
| Network Slimming             |       84.4    |    98.51    |
| *MINT (ours)*                |      *87.2*   |   *98.54*   |
--------------------------------------------------------------

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
| Input - Conv1    |     64*3*3*3    = 1728    |          1728             |
|*Conv1 - Conv2*   |     64*64*3*3   = 36864   |          12384            |
| Conv2 - Conv3    |     64*128*3*3  = 73728   |          24768            |
| Conv3 - Conv4    |     128*128*3*3 = 147456  |          47808            | 
| Conv4 - Conv5    |     128*256*3*3 = 294912  |          89280            |
| Conv5 - Conv6    |     256*256*3*3 = 589824  |          150255           | 
| Conv6 - Conv7    |     256*256*3*3 = 589824  |          147654           | 
| Conv7 - Conv8    |     256*512*3*3 = 1179648 |          305712           | 
|*Conv8 - Conv9*   |     512*512*3*3 = 2359296 |          580212           | 
|*Conv9 - Conv10*  |     512*512*3*3 = 2359296 |          590616           | 
|*Conv10 - Conv11* |     512*512*3*3 = 2359296 |          580212           | 
|*Conv11 - Conv12* |     512*512*3*3 = 2359296 |          590616           | 
|*Conv12 - Conv13* |     512*512*3*3 = 2359296 |          590616           | 
|*Conv13 - Linear1*|     512*512     = 262144  |          63312            |
| Linear1- Linear2 |     512*10      = 5120    |          1414             |
| Total            |     *14977728*            |        *3774859*          | 
----------------------------------------------------------------------------

### Results Compilation

------------------------------------------------------------------------------------------
| Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|     60       |     8        |       250           |    48.05        |     93.44        | (Invalid) 
|     60       |     15       |       250           |    49.31        |     93.83        | (Invalid)
|    ---       |    ---       |       ---           |    -----        |     -----        |
|     75       |     8        |       250           |    68.13        |     93.04        | 
|     75       |     15       |       250           |    74.79        |     93.49        | (In Progress)
|     75       |     20       |       250           |    71.10        |     93.50        | (In Progress)
|     75       |     25       |       250           |    71.79        |     93.47        | (In Progress)
|    ---       |    ---       |       ---           |    -----        |     -----        | 
|     75       |     8        |       500           |    --.--        |     10.00        | (Re-running)
------------------------------------------------------------------------------------------

### Notes
- Between 8 and 15 groups, consistently 15 groups provided higher performance, even when overall params pruned was comparable.
- Why is 0.999 equating to only 39.53 or 44.61 of the params? The number of unique I\_parent values is smaller than the number of repeat values, Re-running with unique values used to compute percentage.
- > 70 % params pruned with performance > 93.40 achieved on vgg16 on CIFAR-10.
