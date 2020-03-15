# Results for MINT: MI-based Neuron Trimming

### Experiment 1: [Learning Efficient Convolutional Networks through Network Slimming](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.pdf)
```diff
- Incomplete 
```
------------------------------------------------------------------------------
| Method                       |        Params Pruned          | Performance |
|:----------------------------:|:-----------------------------:|:-----------:|
| Baseline  (ours)             |       N/A                     |    98.59    |
| Structured Sparsity Learning |       90.61 (83.5) (90.95)    |    98.47    | Original paper edits all layers, to be fair we evaluate pruning beyond layer 1 only.
| Network Slimming             |       95.68 (84.4) (96.00)    |    98.51    | Original paper edits all layers, to be fair we evaluate pruning beyond layer 1 only.
| MINT (b) (ours)              |       96.01                   |    98.47    | (Requested Prune percent: 0.645800, True Prune Percent: 3.792, UL: 0.99, Parameters: 150000)
------------------------------------------------------------------------------

### Results Compilation
#### Group variations 
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     5        |       250           |    86.27        |     98.--        |  
|   (b)   |     N/A      |     10       |       250           |    88.23        |     98.56        |  
|   (b)   |     N/A      |     20       |       250           |    88.48        |     98.53        | 
|   (b)   |     N/A      |     50       |       250           |    91.87        |     98.52        | 
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     0.8      |     5        |       250           |    86.54        |     98.--        |  
|   (b)   |     0.8      |     10       |       250           |    86.00        |     98.53        | 
|   (b)   |     0.8      |     20       |       250           |    76.96        |     98.62        | 
|   (b)   |     0.8      |     50       |       250           |    77.32        |     98.60        | 
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     0.1      |     5        |       250           |    11.96        |     98.--        |  
|   (b)   |     0.2      |     10       |       250           |    86.00        |     98.53        | 
|   (b)   |     0.4      |     20       |       250           |    76.96        |     98.62        | 
|   (b)   |     1.0      |     50       |       250           |    77.32        |     98.60        | 
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     0.0      |     5        |       250           |    3.96         |     98.--        |  
|   (b)   |     0.04     |     10       |       250           |    86.00        |     98.53        | 
|   (b)   |     0.16     |     20       |       250           |    76.96        |     98.62        | 
|   (b)   |     1.0      |     50       |       250           |    77.32        |     98.60        | 
----------------------------------------------------------------------------------------------------

#### Sample variations
----------------------------------------------------------------------------------------------------
| Version | Upper limit  |   Groups     |  Samples per class  | Params Pruned   |    Performance   |
|:-------:|:------------:|:------------:|:-------------------:|:---------------:|:----------------:|
|   (b)   |     N/A      |     20       |       150           |    85.35        |     98.58        |  
|   (b)   |     N/A      |     20       |       250           |    88.48        |     98.53        | 
|   (b)   |     N/A      |     20       |       350           |    88.48        |     98.55        | 
|   (b)   |     N/A      |     20       |       450           |    88.72        |     98.51        | 
|   (b)   |     N/A      |     20       |       550           |    79.41        |     98.54        | 
|   (b)   |     N/A      |     20       |       650           |    89.70        |     98.53        | 
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


- Command to get current best results
python retrain\_stage3.py --Epoch 30 --Batch\_size 256 --Lr 0.1 --Dataset MNIST --Dims 10 --Expt\_rerun 1 --Milestones 10 20 --Opt sgd --Weight\_decay 0.0001 --Model mlp --Gamma 0.1 --Nesterov --Device\_ids 0 --Retrain BASELINE\_MNIST\_MLP\_FULL/0/logits\_best.pkl --Retrain\_mask BASELINE\_MNIST\_MLP\_FULL/0/I\_parent\_peak\_1b.npy --Labels\_file BASELINE\_MNIST\_MLP\_FULL/0/Labels\_peak\_1b.npy --Labels\_children\_file BASELINE\_MNIST\_MLP\_FULL/0/Labels\_children\_peak\_1b.npy --parent\_key fc1.weight --children\_key fc2.weight --parent\_clusters 250 --children\_clusters 100 --upper\_prune\_limit 0.99 --upper\_prune\_per 0.65 --lower\_prune\_per 0.639 --prune\_per\_step 0.0001 --Save\_dir BASELINE\_MNIST\_MLP\_FULL\_RETRAIN\_1 --key\_id 111



### Experiment 2: [Pruning filters for efficient Convnets](https://openreview.net/pdf?id=rJqFGTslg)
```diff
- Incomplete  
```
--------------------------------------------------------------
| Model                        | Params Pruned | Performance |
|:----------------------------:|:-------------:|:-----------:|
| VGG16(ours)                  |       N/A     |    93.98    |
| VGG16-pruned-A               |      64.00    |    93.40    |
| MINT (b) (ours)              |      65.24    |    93.47    | (Prune: , Acc: 93.47, Params: )
| SSS                          |      66.70    |    93.63    |
| SSS                          |      73.80    |    93.02    |
| GAL-0.05                     |      77.60    |    93.77    |
| GAL-0.1                      |      82.20    |    93.42    |
| MINT (b) (ours)              |      83.43    |    93.43    | (Prune: 15.035583, Params: 14712832)
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
| ResNet56(ours)               |       N/A     |    92.55    |
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
| MINT (b) (ours)              |      --.--    |    71.54    | (Requested Prune: 0.11201, Retained percent: 46.693903, Epochs: 130)
| MINT (b) (ours)              |      --.--    |    71.41    | (Requested Prune: 0.114,   Retained percent: 46.020009, Epochs: 130)
| MINT (b) (ours)              |      --.--    |    71.--    | (Requested Prune: 0.---,   Retained percent: 43.142143, Epochs: 100)
| MINT (b) (ours)              |      --.--    |    71.--    | (Requested Prune: 0.114,   Retained percent: )
| MINT (b) (ours)              |      --.--    |    71.--    | (Requested Prune: 0.114,   Retained percent: )
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
- ``` python python retrain_stage3.py --Epoch 100 --Batch_size 64 --Lr 0.1 --Dataset IMAGENET --Dims 1000 --Model resnet --Milestones 30 60 90 --Weight_decay 0.0001 --Gamma 0.1 --Device_ids 0 --Retrain BASELINE_IMAGENET2012_RESNET50/0/logits_best.pkl --Retrain_mask BASELINE_IMAGENET2012_RESNET50/0/I_parent_64g_5s_1b.npy --Labels_file BASELINE_IMAGENET2012_RESNET50/0/Labels_64g_5s_1b.npy --Labels_children_file BASELINE_IMAGENET2012_RESNET50/0/Labels_children_64g_5s_1b.npy --parent_key conv1.weight conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv16.weight conv17.weight conv18.weight conv19.weight conv20.weight conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv37.weight conv38.weight conv39.weight conv40.weight conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight conv51.weight conv52.weight --children_key  conv2.weight conv3.weight conv4.weight conv5.weight conv6.weight conv7.weight conv8.weight conv9.weight conv10.weight conv11.weight conv12.weight conv13.weight conv14.weight conv15.weight conv16.weight conv17.weight conv18.weight conv19.weight conv20.weight conv21.weight conv22.weight conv23.weight conv24.weight conv25.weight conv26.weight conv27.weight conv28.weight conv29.weight conv30.weight conv31.weight conv32.weight conv33.weight conv34.weight conv35.weight conv36.weight conv37.weight conv38.weight conv39.weight conv40.weight conv41.weight conv42.weight conv43.weight conv44.weight conv45.weight conv46.weight conv47.weight conv48.weight conv49.weight conv50.weight conv51.weight conv52.weight conv53.weight --parent_clusters 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 --children_clusters 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 64 --upper_prune_limit 0.55 --upper_prune_per 0.95 --lower_prune_per 0.11201 --prune_per_step 0.01 --Save_dir BASELINE_IMAGENET2012_RESNET50_RETRAIN_1 --key_id 1```
