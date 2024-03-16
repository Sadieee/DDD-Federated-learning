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
import pytorchvideo.models.x3d
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
#===============this cause error=================
#from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
class Aggregator:
    def __init__(self, args, name):  
        self.others = []
        self.args = args
        self.name = name
        

    def _make_transforms(self):
        """
        ##################
        # PTV Transforms #
        ##################

        # Each PyTorchVideo dataset has a "transform" arg. This arg takes a
        # Callable[[Dict], Any], and is used on the output Dict of the dataset to
        # define any application specific processing or augmentation. Transforms can
        # either be implemented by the user application or reused from any library
        # that's domain specific to the modality. E.g. for video we recommend using
        # TorchVision, for audio we recommend TorchAudio.
        #
        # To improve interoperation between domain transform libraries, PyTorchVideo
        # provides a dictionary transform API that provides:
        #   - ApplyTransformToKey(key, transform) - applies a transform to specific modality
        #   - RemoveKey(key) - remove a specific modality from the clip
        #
        # In the case that the recommended libraries don't provide transforms that
        # are common enough for PyTorchVideo use cases, PyTorchVideo will provide them in
        # the same structure as the recommended library. E.g. TorchVision didn't
        # have a RandomShortSideScale video transform so it's been added to PyTorchVideo.
        """
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
        """
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
        args = self.args
        if self.args.arch == "video_slowfast": 
            side_size = 256
            mean = [0.45, 0.45, 0.45]
            std = [0.225, 0.225, 0.225]
            crop_size = 256
            num_frames = 32
            sampling_rate = 2
            frames_per_second = 30
            alpha = 4
            return  ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames),
                            Lambda(lambda x: x/255.0),
                            NormalizeVideo(mean, std),
                            ShortSideScale(
                                size=side_size
                            ),
                            CenterCropVideo(crop_size),
                            client_base.PackPathway()
                        ]
                    ),
                )
        else:
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


    def merge(self, no_cuda, clients, total_dataset, neighbor_list, communication):
        args = self.args
        #total_dataset為一個數字seqence，放每一個client的data sample數量\
        """
        # 把鄰居weight
        neighbor_index = 0
        weights = []
        for client in rang(len(clients)):
            if neighbor_index >= len(neighbor_list):
                break
            elif client == neighbor_list[neighbor_index]:
                weights.append(clients[client].model
                neighbor_index += 1
        """
        weights = [client for client in clients]

        # total_data_size = sum(m['size'] for m in models)
        total_data_size = 0 
        for i in range(len(total_dataset)):
            isNeighbor = False
            for j in neighbor_list :
                if j == i:
                    isNeighbor = True
            if isNeighbor:
                total_data_size = total_data_size + total_dataset[i]
        print('total data size:',total_data_size)

        #除了鄰居以外 其他人權重0
        factors = []
        for i in range(len(total_dataset)):
            isNeighbor = False
            for j in neighbor_list :
                if j == i:
                    isNeighbor = True
            if isNeighbor:
                factors.append( total_dataset[i]/total_data_size )
            else:
                factors.append( 0 )

        merged = {}
        for key in weights[0].keys():
            merged[key] = sum([w[key] * f for w, f in zip(weights, factors)])

        # 加載test dataset 做測試
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
        #dataloader
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        #===============紀錄結果==================
        #RESULT檔在不在
        output_file = './result_{}.txt'.format("aggregator{}".format(self.name))
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             
        output_file.write('Inter-node communication round{}: '.format( communication ))
        output_file.close()
        #===============紀錄結束==================

        #Load model
        trainer = pytorch_lightning.Trainer.from_argparse_args(args,default_root_dir="./model_save/aggregate{}".format(self.name))
        classification_module = client_base.VideoClassificationLightningModule(args,"aggregator{}".format(self.name))
        # copy一份merged並載入模型
        classification_module.model.load_state_dict(copy.deepcopy(merged))
        # 進行測試
        #trainer = pytorch_lightning.Trainer.from_argparse_args(args)
        trainer.validate( classification_module, dataloaders=test_loader)
        #trainer.save_checkpoint("aggregate.ckpt")
        #metrics = self.__test( model, args.devices, test_loader):

        return merged
     
