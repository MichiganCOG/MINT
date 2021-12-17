# Mutual Information-based Neuron Trimming (MINT)
This repository contains code relevant to the article titled: ``[Mutual Information-based Neuron Trimming](https://doi.org/10.1109/ICPR48806.2021.9412590)''  


## Main Results

### Experiment 1: Comparison against SOTA one-shot pruning methods
----------------------------------------------------------------------------------------
| Dataset-DNN |  Methods  |   Params Pruned(\%) |    Performance(\%)   |  Memory (Mb)  |
|:-----------:|:---------:|:-------------------:|:--------------------:|:-------------:|
|  MNIST-MLP  |  Baseline |         N/A         |        98.59         |     0.537     |
|  MNIST-MLP  |  SSL      |         90.95       |        98.47         |     N/A       |
|  MNIST-MLP  |  Nw. Slim |         96.00       |        98.51         |     N/A       |
|  MNIST-MLP  |  **MINT** |         96.20       |        98.47         |     0.022     |
----------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------
| Dataset-DNN |  Methods  |   Params Pruned(\%) |    Performance(\%)   |  Memory (Mb)  |
|:-----------:|:---------:|:-------------------:|:--------------------:|:-------------:|
|CIFAR10-VGG16|  Baseline |         N/A         |        93.98         |     53.868    |
|CIFAR10-VGG16|  Pruning F|         64.00       |        93.40         |     N/A       |
|CIFAR10-VGG16|  SSS      |         73.80       |        93.02         |     N/A       |
|CIFAR10-VGG16|  GAL      |         82.20       |        93.42         |     N/A       |
|CIFAR10-VGG16|  **MINT** |         83.46       |        93.43         |     9.020     |
----------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------
| Dataset-DNN |  Methods  |   Params Pruned(\%) |    Performance(\%)   |  Memory (Mb)  |
|:-----------:|:---------:|:-------------------:|:--------------------:|:-------------:|
|CIFAR10-Res56|  Baseline |         N/A         |        92.55         |     3.109     |
|CIFAR10-Res56|  GAL      |         11.80       |        93.38         |     N/A       |
|CIFAR10-Res56|  Pruning F|         13.70       |        93.06         |     N/A       |
|CIFAR10-Res56|  OED      |         43.50       |        93.29         |     N/A       |
|CIFAR10-Res56|  NISP     |         43.68       |        93.01         |     N/A       |
|CIFAR10-Res56|  **MINT** |         52.41       |        93.47         |     1.552     |
|CIFAR10-Res56|  **MINT** |         57.01       |        93.02         |     1.461     |
----------------------------------------------------------------------------------------

-----------------------------------------------------------------------------------------
| Dataset-DNN  |  Methods  |   Params Pruned(\%) |    Performance(\%)   |  Memory (Mb)  |
|:------------:|:---------:|:-------------------:|:--------------------:|:-------------:|
|ILSVRC12-Res50|  Baseline |         N/A         |        76.13         |     91.157    |
|ILSVRC12-Res50|  GAL      |         16.86       |        71.95         |     N/A       |
|ILSVRC12-Res50|  OED      |         25.68       |        73.55         |     N/A       |
|ILSVRC12-Res50|  SSS      |         27.05       |        74.18         |     N/A       |
|ILSVRC12-Res50|  NISP     |         43.82       |        71.99         |     N/A       |
|ILSVRC12-Res50|  ThiNet   |         51.45       |        71.01         |     N/A       |
|ILSVRC12-Res50|  **MINT** |         43.01       |        71.50         |     52.365    |
|ILSVRC12-Res50|  **MINT** |         49.00       |        71.12         |     47.513    |
|ILSVRC12-Res50|  **MINT** |         49.62       |        71.05         |     46.925    |
-----------------------------------------------------------------------------------------

#### Notes:
- For the MNIST-MLP experiment, only layer 2 is compared.

### Experiment 2: Empirical analysis of filter group size and sample size
---------------------------------------------------------------------------
| Dataset-DNN |  Group Size  |   Params Pruned(\%) |    Performance(\%)   |
|:-----------:|:------------:|:-------------------:|:--------------------:|
|  MNIST-MLP  |      5       |         86.27       |        98.55         |
|  MNIST-MLP  |      10      |         87.25       |        98.52         |
|  MNIST-MLP  |      20      |         88.48       |        98.55         |
|  MNIST-MLP  |      50      |         91.87       |        98.55         |
---------------------------------------------------------------------------

---------------------------------------------------------------------------
| Dataset-DNN |  Sample Size |   Params Pruned(\%) |    Performance(\%)   |
|:-----------:|:------------:|:-------------------:|:--------------------:|
|  MNIST-MLP  |      150     |         85.35       |        98.58         |
|  MNIST-MLP  |      250     |         88.48       |        98.53         |
|  MNIST-MLP  |      450     |         88.72       |        98.51         |
|  MNIST-MLP  |      650     |         89.70       |        98.53         |
---------------------------------------------------------------------------

#### Notes:
- For filter group size experiments, number of samples per class is 250.
- For sample size experiments, number of groups is 20.


## Usage Instructions
The code provided within each DNN folder has 3 core scripts for training, pruning and re-training.
To run the training scripts, for mlp, execute a command similar to:

```
python train_stage1.py --Epoch $num_epochs --Batch_size $batch_size --Lr $learning_rate --Save_dir $save_directory --Dataset MNIST --Dims $num_labels --Expt_rerun $experiment_reruns --Milestones $lr_stepping_schedule --Opt sgd --Weight_decay 0.0001 --Model mlp --Gamma $lr_decay_factor --Nesterov --Device_ids $gpu_id
```

To run the pruning scripts, for mlp, execute a command similar to:

```
python pruning_1b.py --model mlp --dataset MNIST --weights_dir $save_directory/rerun_number/ --cores $num_cores --dims $num_labels --key_id $layer_id --samples_per_class $num_samples_per_class --parent_clusters $parent_filter_group_size --children_clusters $children_filter_group_size --name_postfix $postfix
```

Finally, to run the re-training scripts, for mlp, execute a command similar to:

```
python retrain_stage3.py --Epoch $num_epochs --Batch_size $batch_size --Lr $learning_rate --Dataset MNIST --Dims $num_labels --Expt_rerun $experiment_reruns --Milestones $lr_stepping_schedule --Opt sgd --Weight_decay 0.0001 --Model mlp --Gamma $lr_decay_factor --Nesterov --Device_ids $gpu_id --Retrain $save_directory/rerun_number/logits_best.pkl --Retrain_mask $save_directory/rerun_number/I_parent_postfix.npy --Labels_file $save_directory/rerun_number/Labels_postfix.npy --Labels_children_file $save_directory/rerun_number/Labels_children_postfix.npy --parent_key  $[list of parent layers] --children_key $[list of children layers] --parent_clusters $[list of parent groups] --children_clusters $[list of children groups] --upper_prune_limit $upper_pruning_limit_all_layers --upper_prune_per $top_pruning_limit_for_gridsearch --lower_prune_per $lowest_pruning_limit_for_gridsearch --prune_per_step $step_size_for_gridsearch --Save_dir $new_save_directory --key_id $pruning_selection_idx
```

#### Notes:
- All commands have to be executed under the main DNN directory.
- The code assumes save directory has an additional folder for number of experiment reruns. Please ensure atleast 1 directory '0' exists under the results folder.
- parent and children filter group sizes are flexible. However, ensure their values correspond across the entire DNN.
- ```key_id``` in the pruning script corresponds to the layer index. Code assumes that 0 measures the dependency between layer 0 - layer 1.
- children group size must match the total number of labels in the dataset for the last layer.
- Examples of parent and children layer lists are: [fc1.weight fc2.weight] and [fc2.weight final.weight]
- ``` new_save_directory``` is an alternative save directory used for the new pruned DNN. It has to be different from the original ```save_directory```.
- ```key_id``` in the retraining script is used to select the pruning value requested from the list of pruning values generated between ```upper_prune_per``` and ```lower_prune_per``` at a step size of ```prune_per_step```

## BibTeX citation
```
@inproceedings{DBLP:conf/icpr/GaneshCS20,
  author    = {Madan Ravi Ganesh and
               Jason J. Corso and
               Salimeh Yasaei Sekeh},
  title     = {{MINT:} Deep Network Compression via Mutual Information-based Neuron
               Trimming},
  booktitle = {25th International Conference on Pattern Recognition, {ICPR} 2020,
               Virtual Event / Milan, Italy, January 10-15, 2021},
  pages     = {8251--8258},
  publisher = {{IEEE}},
  year      = {2020},
  url       = {https://doi.org/10.1109/ICPR48806.2021.9412590},
  doi       = {10.1109/ICPR48806.2021.9412590},
  timestamp = {Fri, 07 May 2021 12:53:57 +0200},
  biburl    = {https://dblp.org/rec/conf/icpr/GaneshCS20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
