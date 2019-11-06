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
| VGG16-pruned-A               |      64.0     |    93.40    |
| MINT (ours)                  |      --.-     |    --.--    |
--------------------------------------------------------------
