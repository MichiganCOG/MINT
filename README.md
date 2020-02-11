# Results for MINT: MI-based Neuron Trimming

### Experiment 1: [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)
```diff
- Incomplete 
```
----------------------------------------------------------------------
| Method                       |    Params Pruned      | Performance |
|:----------------------------:|:---------------------:|:-----------:|
| Baseline  (ours)             |       N/A             |    98.59    |
| Structured Sparsity Learning |       90.61 (83.5)    |    98.47    | Original paper edits all layers, to be fair we evaluate pruning beyond layer 1 only.
| Network Slimming             |       95.68 (84.4)    |    98.51    | Original paper edits all layers, to be fair we evaluate pruning beyond layer 1 only.
| MINT (b) (ours)              |       --.-            |    --.--    |
----------------------------------------------------------------------

### Results Compilation
#### Group variations 
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     5        |       250           |    86.47        |     98.50        | (Requested Prune Percent: 0.424) 
|   (b)   |     N/A      |     10       |       250           |    88.23        |     98.56        | (Requested Prune Percent: 0.415) 
|   (b)   |     N/A      |     20       |       250           |    88.48        |     98.53        | (Requested Prune Percent: 0.340)
|   (b)   |     N/A      |     50       |       250           |    91.87        |     98.52        | (Requested Prune Percent: 0.376)
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     0.8      |     5        |       250           |    86.47        |     98.50        | (Requested Prune Percent: 0.424) 
|   (b)   |     0.8      |     10       |       250           |    86.00        |     98.53        | (Requested Prune Percent: 0.676)
|   (b)   |     0.8      |     20       |       250           |    76.96        |     98.62        | (Requested Prune Percent: 0.287)
|   (b)   |     0.8      |     50       |       250           |    77.32        |     98.60        | (Requested Prune Percent: 0.316)
----------------------------------------------------------------------------------------------------

#### Sample variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     20       |       150           |    85.35        |     98.58        | (Requested Prune Percent: 0.352) 
|   (b)   |     N/A      |     20       |       250           |    88.48        |     98.53        | (Requested Prune Percent: 0.340)
|   (b)   |     N/A      |     20       |       350           |    88.48        |     98.55        | (Requested Prune Percent: 0.364)
|   (b)   |     N/A      |     20       |       450           |    88.72        |     98.51        | (Requested Prune Percent: 0.405)
|   (b)   |     N/A      |     20       |       550           |    79.41        |     98.54        | (Requested Prune Percent: 0.333)
|   (b)   |     N/A      |     20       |       650           |    89.70        |     98.53        | (Requested Prune Percent: 0.429)
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
- Current weird trend is that in increasing samples beyond a point, number of params to prune seems to dip. Confirming this by running 650samples/class experiment. 550samples/class seems to an anomaly rather than the trend.
- Imposing an 80% limit still gives 13\% pruning somehow! Look into this ASAP. Although this ceiling seems to hold, in terms of fixed upper limit (whatever is computed) 

### Experiment 2: [Pruning filters for efficient Convnets](https://openreview.net/pdf?id=rJqFGTslg)
```diff
- Incomplete 
```
--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| VGG16(ours)                  |       N/A     |    93.98    |
| VGG16-pruned-A               |      64.00    |    93.40    |
| MINT (b) (ours)              |      68.54    |    93.41    |
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
| Conv7 - Conv8    |         = 1179648         |                           | 
|*Conv8 - Conv9*   |         = 2359296         |                           | 
|*Conv9 - Conv10*  |         = 2359296         |                           | 
|*Conv10 - Conv11* |         = 2359296         |                           | 
|*Conv11 - Conv12* |         = 2359296         |                           | 
|*Conv12 - Conv13* |         = 2359296         |                           | 
|*Conv13 - Linear1*|         = 262144          |                           |
| Linear1- Linear2 |         = 5120            |         5120              |
| Total            |       **14977728**        |                           | 
----------------------------------------------------------------------------
Untouched params = 14977728 - 13275136  = 1702592

### Notes
- Still compiling results for Samples and Groups variations for VGG16. Slightly delayed but we are moving ahead with ResNet56 setup for CIFAR-10. 
- The results from Group and sample variations seems to not follow any particular trend. Could this be because of the extra upper limit / layer being imposed? 
- Moving to MLP with upper limit / layer to observe behaviour more quickly.
- Limit now is 85\%, lets change it to 75\%? Maximum limit for pruning percentage is 66.47. Lets do 80\% first, this translates to 70.90\%.


### Experiment 3: [Pruning blocks for CNN Compression and Acceleration via Online Ensemble Distillation](https://ieeexplore.ieee.org/iel7/6287639/8600701/08918410.pdf)
```diff
+ Experiment Run Complete 
```
--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| ResNet56(ours)               |       N/A     |    93.98    |
| GAL                          |      11.80    |    93.38    |
| ResNet56-Pruned-A            |      14.10    |    93.06    |
| NISP                         |      42.40    |    93.01    |
| OED 0.04                     |      43.50    |    93.29    |
| MINT (b) (ours)              |      52.41    |    93.47    | (Prune: 43.363957, Acc: 93.47, Params: 785664)
| MINT (b) (ours)              |      55.39    |    93.02    | (Prune: 40.145024, Acc: 93.02, Params: 785664)
--------------------------------------------------------------

### ResNet56 Layer Breakdown (Baseline)

----------------------------------------------------------------------------
| Layer pairs      |  Total Number of Params   |  Reduced Number of Params |
|:----------------:|:-------------------------:|:-------------------------:|
| Input - Conv1    |         = 432             |         432               |
|*Conv1 - Conv2*   |         = 2304            |                           |
| Conv2 - Conv3    |         = 2304            |                           |
| Conv3 - Conv4    |         = 2304            |                           | 
| Conv4 - Conv5    |         = 2304            |                           |
| Conv5 - Conv6    |         = 2304            |                           | 
| Conv6 - Conv7    |         = 2304            |                           | 
| Conv7 - Conv8    |         = 2304            |                           | 
|*Conv8 - Conv9*   |         = 2304            |                           | 
|*Conv9 - Conv10*  |         = 2304            |                           | 
|*Conv10 - Conv11* |         = 2304            |                           | 
|*Conv11 - Conv12* |         = 2304            |                           | 
|*Conv12 - Conv13* |         = 2304            |                           | 
|*Conv13 - Conv14* |         = 2304            |                           | 
|*Conv14 - Conv15* |         = 2304            |                           | 
|*Conv15 - Conv16* |         = 2304            |         2304              | 
|*Conv16 - Conv17* |         = 2304            |                           | 
|*Conv17 - Conv18* |         = 2304            |                           | 
|*Conv18 - conv19* |         = 2304            |                           |
|*Conv19 - conv20* |         = 4608            |         4608              |
|*Conv20 - Conv21* |         = 9216            |                           | 
|*Conv21 - Conv22* |         = 9216            |                           | 
|*Conv22 - Conv23* |         = 9216            |                           | 
|*Conv23 - Conv24* |         = 9216            |                           | 
|*Conv24 - Conv25* |         = 9216            |                           | 
|*Conv25 - Conv26* |         = 9216            |                           | 
|*Conv26 - Conv27* |         = 9216            |                           | 
|*Conv27 - Conv28* |         = 9216            |                           | 
|*Conv28 - conv29* |         = 9216            |                           |
|*Conv29 - conv30* |         = 9216            |                           |
|*Conv30 - Conv31* |         = 9216            |                           | 
|*Conv31 - Conv32* |         = 9216            |                           | 
|*Conv32 - Conv33* |         = 9216            |                           | 
|*Conv33 - Conv34* |         = 9216            |                           | 
|*Conv34 - Conv35* |         = 9216            |                           | 
|*Conv35 - Conv36* |         = 9216            |                           | 
|*Conv36 - Conv37* |         = 9216            |                           | 
|*Conv37 - Conv38* |         = 18432           |         18432             | 
|*Conv38 - conv39* |         = 36864           |                           |
|*Conv39 - conv40* |         = 36864           |                           |
|*Conv40 - Conv41* |         = 36864           |                           | 
|*Conv41 - Conv42* |         = 36864           |                           | 
|*Conv42 - Conv43* |         = 36864           |                           | 
|*Conv43 - Conv44* |         = 36864           |                           | 
|*Conv44 - Conv45* |         = 36864           |                           | 
|*Conv45 - Conv46* |         = 36864           |                           | 
|*Conv46 - Conv47* |         = 36864           |                           | 
|*Conv47 - Conv48* |         = 36864           |                           | 
|*Conv48 - conv49* |         = 36864           |                           |
|*Conv49 - conv50* |         = 36864           |                           |
|*Conv50 - Conv51* |         = 36864           |                           | 
|*Conv51 - Conv52* |         = 36864           |                           | 
|*Conv52 - Conv53* |         = 36864           |         36864             | 
|*Conv53 - Conv54* |         = 36864           |                           | 
|*Conv54 - Conv55* |         = 36864           |         640               | 
| Conv55 - Linear1 |         = 640             |                           |
| Total            |       **848944**          |                           | 
----------------------------------------------------------------------------

Untouched params = 848944 - 785664 = 63280
### Notes
- Using maximum grouping levels for conv layers (64 being max for 3rd portion) with 500 samples/class.

### Experiment 4: [Pruning blocks for CNN Compression and Acceleration via Online Ensemble Distillation](https://ieeexplore.ieee.org/iel7/6287639/8600701/08918410.pdf)
```diff
- Incomplete 
```
--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| ResNet50(ours)               |       N/A     |    --.--    |
| GAL                          |      16.86    |    71.95    |
| OED 0.04                     |      25.68    |    73.55    |
| SSS                          |      27.05    |    74.18    |
| ThiNet-50                    |      51.45    |    71.01    |
| MINT (b) (ours)              |      --.--    |    --.--    | (Prune: --.--, Acc: --.--, Params: ------)
| MINT (b) (ours)              |      --.--    |    --.--    | (Prune: --.--, Acc: --.--, Params: ------)
--------------------------------------------------------------

### ResNet56 Layer Breakdown (Baseline)

----------------------------------------------------------------------------
| Layer pairs      |  Total Number of Params   |  Reduced Number of Params |
|:----------------:|:-------------------------:|:-------------------------:|
| Input - Conv1    |         = 432             |         432               |
|*Conv1 - Conv2*   |         = 2304            |                           |
| Conv2 - Conv3    |         = 2304            |                           |
| Conv3 - Conv4    |         = 2304            |                           | 
| Conv4 - Conv5    |         = 2304            |                           |
| Conv5 - Conv6    |         = 2304            |                           | 
| Conv6 - Conv7    |         = 2304            |                           | 
| Conv7 - Conv8    |         = 2304            |                           | 
|*Conv8 - Conv9*   |         = 2304            |                           | 
|*Conv9 - Conv10*  |         = 2304            |                           | 
|*Conv10 - Conv11* |         = 2304            |                           | 
|*Conv11 - Conv12* |         = 2304            |                           | 
|*Conv12 - Conv13* |         = 2304            |                           | 
|*Conv13 - Conv14* |         = 2304            |                           | 
|*Conv14 - Conv15* |         = 2304            |                           | 
|*Conv15 - Conv16* |         = 2304            |         2304              | 
|*Conv16 - Conv17* |         = 2304            |                           | 
|*Conv17 - Conv18* |         = 2304            |                           | 
|*Conv18 - conv19* |         = 2304            |                           |
|*Conv19 - conv20* |         = 4608            |         4608              |
|*Conv20 - Conv21* |         = 9216            |                           | 
|*Conv21 - Conv22* |         = 9216            |                           | 
|*Conv22 - Conv23* |         = 9216            |                           | 
|*Conv23 - Conv24* |         = 9216            |                           | 
|*Conv24 - Conv25* |         = 9216            |                           | 
|*Conv25 - Conv26* |         = 9216            |                           | 
|*Conv26 - Conv27* |         = 9216            |                           | 
|*Conv27 - Conv28* |         = 9216            |                           | 
|*Conv28 - conv29* |         = 9216            |                           |
|*Conv29 - conv30* |         = 9216            |                           |
|*Conv30 - Conv31* |         = 9216            |                           | 
|*Conv31 - Conv32* |         = 9216            |                           | 
|*Conv32 - Conv33* |         = 9216            |                           | 
|*Conv33 - Conv34* |         = 9216            |                           | 
|*Conv34 - Conv35* |         = 9216            |                           | 
|*Conv35 - Conv36* |         = 9216            |                           | 
|*Conv36 - Conv37* |         = 9216            |                           | 
|*Conv37 - Conv38* |         = 18432           |         18432             | 
|*Conv38 - conv39* |         = 36864           |                           |
|*Conv39 - conv40* |         = 36864           |                           |
|*Conv40 - Conv41* |         = 36864           |                           | 
|*Conv41 - Conv42* |         = 36864           |                           | 
|*Conv42 - Conv43* |         = 36864           |                           | 
|*Conv43 - Conv44* |         = 36864           |                           | 
|*Conv44 - Conv45* |         = 36864           |                           | 
|*Conv45 - Conv46* |         = 36864           |                           | 
|*Conv46 - Conv47* |         = 36864           |                           | 
|*Conv47 - Conv48* |         = 36864           |                           | 
|*Conv48 - conv49* |         = 36864           |                           |
|*Conv49 - conv50* |         = 36864           |                           |
|*Conv50 - Conv51* |         = 36864           |                           | 
|*Conv51 - Conv52* |         = 36864           |                           | 
|*Conv52 - Conv53* |         = 36864           |         36864             | 
|*Conv53 - Conv54* |         = 36864           |                           | 
|*Conv54 - Conv55* |         = 36864           |         640               | 
| Conv55 - Linear1 |         = 640             |                           |
| Total            |       **848944**          |                           | 
----------------------------------------------------------------------------

Untouched params = 848944 - 785664 = 63280
