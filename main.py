import threading
import argparse
import client_base
import aggregator_base
import copy
import pytorch_lightning
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision import models
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.utils.data import DistributedSampler, RandomSampler
import torchmetrics
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models
import pytorchvideo.models.x3d
import pytorchvideo.models.resnet
import torch.nn.functional as F
from client_base import MetricsCallback
from torchvision import datasets, transforms
import os
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
#from torchaudio.transforms import MelSpectrogram, Resample
from slurm import copy_and_run_with_config
from pytorch_lightning import Callback
class Handler:
    def __init__(self, args):  
        self.args = args

    def _make_transforms(self):
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

def getLoss( i, model, test_client, args ):
    handler = Handler(args)
    # 加載test dataset 做測試
    sampler = RandomSampler
    train_transform = handler._make_transforms()
    train_dataset = pytorchvideo.data.Kinetics(
        data_path=os.path.join(test_client.resume, "train.csv"),
        clip_sampler=pytorchvideo.data.make_clip_sampler(
            "random", args.clip_duration
        ),
        video_path_prefix=args.video_path_prefix,
        transform=train_transform,
        video_sampler=sampler,
    )
    #dataloader
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers,
    )

    #Load model
    trainer = pytorch_lightning.Trainer.from_argparse_args(args,default_root_dir="./model_save/neighbor_selection{}".format(i))
    classification_module = client_base.VideoClassificationLightningModule(args,"neighbor_selection{}".format(i))
    # copy一份merged並載入模型
    classification_module.model.load_state_dict(copy.deepcopy(model))
    # 進行測試
    #trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    trainer.validate( classification_module, dataloaders=train_loader)
    #m = MetricsCallback()
    metrics = trainer.callback_metrics
    return metrics["val_loss"]

def give_neighbor( i, num_user, clients, args ):
    #環狀鄰居-與自己相鄰之兩個節點為鄰居 0起算
    #建立loss table
    loss_table = []
    for client in range(len(clients)):
        if client == i:
            loss_table.append(100)
        else:
            loss_table.append(getLoss( i, clients[i].model, clients[client], args ))
    #挑兩個最小loss的index
    neighbor = []
    smallest_idx = 0
    smallest = loss_table[0]
    smallest_idx2 = 1
    smallest2 = loss_table[1]
    print(loss_table)
    for index in range(2, len(loss_table)):
        if smallest > loss_table[index]:
            if smallest < smallest2 :
                smallest_idx2 = smallest_idx
                smallest2 = smallest
            smallest = loss_table[index]
            smallest_idx = index
        elif smallest2 > loss_table[index]:
            if smallest2 < smallest :
                smallest = smallest2
                smallest_idx = smallest_idx2
            smallest2 = loss_table[index]
            smallest_idx2 = index
    neighbor.append(smallest_idx)
    neighbor.append(smallest_idx2)
    #自己也丟進去
    neighbor.append(i)
    print(neighbor)
    return neighbor


def federated_learning():
    ###################################################################            
    # Initial
    parser = argparse.ArgumentParser()
    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)
    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="video_resnet",
        choices=["video_resnet", "audio_resnet","video_slowfast","video_vgg16"],
        type=str,
    )
    # Data parameters.
    parser.add_argument("--data_path", default='data', type=str)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_num_subsampled", default=8, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)
    parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    parser.add_argument("--audio_mel_window_size", default=32, type=int)
    parser.add_argument("--audio_mel_step_size", default=16, type=int)
    parser.add_argument("--audio_num_mels", default=80, type=int)
    parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
    parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    parser.add_argument("--audio_logmel_std", default=4.66, type=float)
    # 自己加的fl參數
    parser.add_argument("--num_class", default=36, type=int)
    parser.add_argument("--num_user", default=10, type=int)
    parser.add_argument("--round", default=100, type=int)
    parser.add_argument("--communication", default=10, type=int)
    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=3,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    clients = []
    total_dataset = [] 
    global_model = []
    client_model_list = []
    neighbor_list = [] #list中有list

    #用迴圈把7個client丟進list 並初始化client，各自的aggregator也在這邊初始化
    for i in range(args.num_user):
        #放入預訓練模型
        global_model.append(None)
        client_model_list.append(None)
        neighbor_list.append(None)
        total_dataset.append(None)
        clients.append(client_base.Client(str(i + 1),args))
        
    
    ###################################################################
    print('user : ', args.num_user)
    print('round : ', args.round)
    print('epochs : ', args.max_epochs)
    print('lr : ', args.lr)
    print('batch_size : ', args.batch_size)
    no_cuda = False
    if args.devices == 'cpu' :
        no_cuda = True
    print('no_cuda : ', no_cuda)

    for round_idx in range(args.round):
        print('----------------------------Now Round ', str(round_idx + 1), '----------------------------')
        
        mode = 'FL_iterator_Training'
        if mode == 'FL_Threading_Training':   # Threading
            threads = []
            for i in clients:
                threads.append(threading.Thread(target=i.train, args=(epochs, global_model[i], lr, batch_size, no_cuda,)))
                threads[-1].start()
            for i in range(user):
                threads[i].join()
   
        elif mode == 'FL_iterator_Training':  # iterator
            #跑過每一client做訓練
            for i in range( args.num_user):
                # if global_model is None:
                #     print(global_model)
                clients[i].main_train( global_model[i])
                #站存model
                client_model_list[i] = clients[i].model
                total_dataset[i] = clients[i].sample
 
        #找出i的鄰居index list
        for i in range(args.num_user):
            neighbor_list[i] = give_neighbor( i, args.num_user-1, clients, args )
        if args.num_user > 1 :
            #跑過每一個client做他們各自的aggregtor
            #     (在每個client訓練完後將各自的gradient傳給其他client並暫存)
            print('---Global model---')
            communication_round = args.communication
            for j in range( communication_round):
                for i in range(args.num_user):               
                    #聚合        
                    print('aggregator:', i+1)
                    print('Client{} neighbor is:'.format(i), neighbor_list[i])
                    global_model[i] = clients[i].aggregator.merge(no_cuda, client_model_list, total_dataset, neighbor_list[i], j)
                #一次inter_communication完成 更新client_model_list
                for i in range(args.num_user):
                    client_model_list[i] = global_model[i]
        else :
            global_model[0] = clients[0].model

    print('Train Finish')

if __name__ == "__main__":
    federated_learning()
