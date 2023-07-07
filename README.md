# DeepGD: A Deep Learning Framework for Graph Drawing Using GNN
This repo contains a simple demonstration for the IEEE CG&A21 paper entitled "[DeepGD: A Deep Learning Framework for Graph Drawing Using GNN](https://arxiv.org/abs/2106.15347)". It includes:

* a dataloader for Rome Graphs dataset,
* a basic implementation of DeepGD model,
* and a demo notebook that shows how to train DeepGD models from scratch with minimal amount of codes.

## Environment
This code has been tested on python3.10 + cuda11.8 + pytorch2.0 + pyg2.3. Anaconda is suggested for managing dependencies, as installing pyg with pip can be tricky. 

## Configuration
The default hyper-parameters of the model have been configured to reproduce the best performance reported in the [DeepGD paper](https://arxiv.org/abs/2106.15347). 

However, the layout initializer for the dataset is by default `nx.drawing.random_layout`, which **is not** PivotMDS that yields the best results shown in the paper (random initialization already produces good enough results though). Feel free to modify this behavior by passing a different initializer with `dataset = RomeDataset(layout_initializer=something_else)`. A custom initializer can be any function that follows the behavior of `nx.drawing.random_layout`.

## Training
With Nvidia V100, each training epoch takes 30s on average. It takes around 600 epochs to completly converge.

## Evaluation
For evaluation on custom data, the easiest way is to subclass `RomeDataset` and override `raw_file_names` and `process_raw` methods.
> **Caveat**: Even though the behavior of `process` do not need to be overriden, it is required to have a dummy `def process(self): super().process()` defined in the subclasses to make it work properly. For details, please refer to `pyg.data.InMemoryDataset` [documentation](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.InMemoryDataset).

## Citation
If you used our code or find our work useful in your research, please consider citing:
```
@article{deepgd,
author={Wang, Xiaoqi and Yen, Kevin and Hu, Yifan and Shen, Han-Wei},
journal={IEEE Computer Graphics and Applications},
title={DeepGD: A Deep Learning Framework for Graph Drawing Using GNN},
year={2021},
volume={41},
number={5},
pages={32-44},
doi={10.1109/MCG.2021.3093908}
}
```
