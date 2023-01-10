from __future__ import print_function
import numpy as np
import torch
import os
import glob
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
import argparse
import itertools
import logging
from torchvision import datasets, transforms
import copy
import client_base
import torchmetrics
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models
import pytorchvideo.models.resnet
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from slurm import copy_and_run_with_config
from torch.utils.data import DistributedSampler, RandomSampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
class Aggregator:
    def __init__(self, args):  
        self.args = args
        

    def _make_transforms(self):
        args = self.args
        if self.args.data_type == "video":
            transform = [
                self._video_transform(),
                RemoveKey("audio"),
            ]
        else:
            raise Exception(f"{self.args.data_type} not supported")
        return Compose(transform)
    def _video_transform(self):
        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(args.video_num_subsampled),
                    Normalize(args.video_means, args.video_stds),
                ]
                + (
                    [
                        RandomShortSideScale(
                            min_size=args.video_min_short_side_scale,
                            max_size=args.video_max_short_side_scale,
                        ),
                        RandomCrop(args.video_crop_size),
                        RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                        ShortSideScale(args.video_min_short_side_scale),
                        CenterCrop(args.video_crop_size),
                    ]
                )
            ),
        )
    def merge(self, no_cuda, clients, total_dataset):
        args = self.args
        weights = [client.model for client in clients]
        total_data_size = 0 
        for i in range(len(total_dataset)):
            total_data_size = total_data_size + total_dataset[i]
        print('total data size:',total_data_size)
        factors = []
        for i in total_dataset:
            factors.append( i/total_data_size )
        merged = {}
        for key in weights[0].keys():
            merged[key] = sum([w[key] * f for w, f in zip(weights, factors)])
        use_cuda = not no_cuda and torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        sampler = RandomSampler
        test_transform = self._make_transforms()
        test_data = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=test_transform,
            video_sampler=sampler,
        )  
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        trainer = pytorch_lightning.Trainer.from_argparse_args(args,default_root_dir="./model_save/aggregate")
        classification_module = client_base.VideoClassificationLightningModule(args,"aggregator")
        classification_module.model.load_state_dict(copy.deepcopy(merged))
        trainer.validate( classification_module, dataloaders=test_loader)
        return merged
     