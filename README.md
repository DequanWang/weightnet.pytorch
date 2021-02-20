# weightnet.pytorch

The unoffical PyTorch implementation of [WeightNet](https://arxiv.org/abs/2007.11823),
based on [OpenMMLab](https://openmmlab.com/) [mmclassification](https://github.com/open-mmlab/mmclassification).

## Installation

See [`install.md`](docs/install.md) and [`getting_started.md`](docs/getting_started.md).
Learn more at [mmcls's documentation](https://mmclassification.readthedocs.io/en/latest/).

## Experiments

| Model   | #Params. | FLOPs | Top-1 err. (offical)| Top-1 err. (ours)|
|---------|----------|-------|---------------------|------------------|
|[ShuffleNetV2 (1×)](ckpts/shufflenet_1x_b256x4_imagenet/)| 2.2M | 138M | 30.9 | **30.5** |
|[+ WeightNet (1×)](ckpts/weightnet_1x_b256x4_imagenet/)  | 2.4M | 139M | 28.8 | **29.1** |

```bash
bash tools/dist_train.sh configs/shufflenet_v2/shufflenet_1x_b256x4_imagenet.py 4
# ckpts/shufflenet_1x_b256x4_imagenet
# top1 accuracy: 69.5 top5 accuracy: 88.8
# reference top1 accuracy: 69.1
bash tools/dist_test.sh configs/shufflenet_v2/shufflenet_1x_b256x4_imagenet.py ckpts/shufflenet_1x_b256x4_imagenet/shufflenet_1x_b256x4_imagenet.pth 4 --metrics accuracy
```

```bash
bash tools/dist_train.sh configs/shufflenet_v2/weightnet_1x_b256x4_imagenet.py 4
# ckpts/weightnet_1x_b256x4_imagenet
# top1 accuracy: 70.9 top5 accuracy: 89.9
# reference top1 accuracy: 71.2
bash tools/dist_test.sh configs/shufflenet_v2/weightnet_1x_b256x4_imagenet.py ckpts/weightnet_1x_b256x4_imagenet/weightnet_1x_b256x4_imagenet.pth 4 --metrics accuracy
```

## Acknowledgement

- [WeightNet](https://github.com/megvii-model/WeightNet)
- [ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series)
- [mmclassification](https://github.com/open-mmlab/mmclassification)

## License

This project is released under the [Apache 2.0 license](LICENSE).
