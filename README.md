# Results for MINT: MI-based Neuron Trimming

# Experiment 1: [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)

--------------------------------------------------------------
| Method                       | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| Baseline                     |       N/A     |    98.57    |
| Structured Sparsity Learning |       83.5    |    98.47    |
| Network Slimming             |       84.4    |    98.51    |
| *MINT (ours)*                |      *87.2*   |   *98.54*   |
--------------------------------------------------------------

# Experiment 2: [Pruning filters for efficient Convnets](https://openreview.net/pdf?id=rJqFGTslg)


--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| VGG16                        |       N/A     |    93.25    |
| VGG16-pruned-A               |      64.00    |    93.40    |
| MINT (ours)                  |      --.--    |    --.--    | (My own baseline: 94.01)
--------------------------------------------------------------


# VGG16 Layer Breakdown (Baseline)
------------------------------------------------
| Layer pairs      |  Total Number of Params   |
|:----------------:|:-------------------------:|
| Input - Conv1    |     64*3*3*3    = 1728    |
|*Conv1 - Conv2*   |     64*64*3*3   = 36864   |
| Conv2 - Conv3    |     64*128*3*3  = 73728   |
| Conv3 - Conv4    |     128*128*3*3 = 147456  | 
| Conv4 - Conv5    |     128*256*3*3 = 294912  |
| Conv5 - Conv6    |     256*256*3*3 = 589824  | 
| Conv7 - Conv8    |     256*512*3*3 = 1179648 | 
|*Conv8 - Conv9*   |     512*512*3*3 = 2359296 | 
|*Conv9 - Conv10*  |     512*512*3*3 = 2359296 | 
|*Conv10 - Conv11* |     512*512*3*3 = 2359296 | 
|*Conv11 - Conv12* |     512*512*3*3 = 2359296 | 
|*Conv12 - Conv13* |     512*512*3*3 = 2359296 | 
|*Conv13 - Linear1*|     512*512     = 262144  |
| Linear1- Linear2 |     512*10      = 5120    |
| Total            |     *14387904*            | 
------------------------------------------------

# Results Compilation

------------------------------------------------------------------------------------------
| Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|     60       |     8        |       250           |    48.05        |     93.44        |
|     60       |     15       |       250           |    49.31        |     93.83        |
|    ---       |    ---       |       ---           |    -----        |     -----        |
|     75       |     8        |       250           |    --.--        |     --.--        |
|     75       |     15       |       250           |    --.--        |     --.--        |
|     75       |     20       |       250           |    --.--        |     --.--        |
|    ---       |    ---       |       ---           |    -----        |     -----        |
|     75       |     8        |       500           |    --.--        |     --.--        |
------------------------------------------------------------------------------------------

## Notes
- Between 8 and 15 groups, consistently 15 groups provided higher performance, even when overall params pruned was comparable.
