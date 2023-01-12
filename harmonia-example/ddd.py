from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import logging
#----------------------------------------------
from net import ConvLstm
import os
import glob
import torch.utils.data
import torch.utils.data.distributed
from tqdm import tqdm
import copy
import argparser
import itertools
import logging
from net import ConvLstm
import torchmetrics
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models
import pytorchvideo.models.x3d
import pytorchvideo.models.resnet
import torch
import torch.nn.functional as F
from pytorch_lightning import Callback
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

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        return metrics["val_loss"]

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args, name):
        """
        This LightningModule implementation constructs a PyTorchVideo ResNet,
        defines the train and val loss to be trained with (cross_entropy), and
        configures the optimizer.
        """
        self.args = args
        self.name = name
        super().__init__()
        #===============this cause error=================
        #self.train_accuracy = pytorch_lightning.metrics.Accuracy()
        #self.val_accuracy = pytorch_lightning.metrics.Accuracy()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.matrics = []

        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                model_num_class=self.args.num_class,
            )
            self.batch_key = "video"
        elif self.args.arch == "video_vgg16":
            self.model = ConvLstm(768, 256, 2, True, self.args.num_class)
            self.batch_key = "video"
        elif self.args.arch == "audio_resnet":
            self.model = pytorchvideo.models.resnet.create_acoustic_resnet(
                input_channel=1,
                model_num_class=self.args.num_class,
            )
            self.batch_key = "audio"
        else:
            raise Exception("{self.args.arch} not supported")

    def get_model(self):
        return self.model

    def on_train_epoch_start(self):
        """
        For distributed training we need to set the datasets video sampler epoch so
        that shuffling is done correctly
        """
        epoch = self.trainer.current_epoch
        #if self.trainer.use_ddp:
        #    self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        #if self.argsw.arch == "video_slowfast":
            #print("==============",x.shape)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        #print('batch is',batch_idx)
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_idx
        )

        return {'loss':loss, 'acc':acc}
 
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc']for x in outputs]).mean()
        logging.info('loss is: {},on epoch: {}'.format(avg_loss, self.current_epoch))
        #self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the evaluation cycle. For this
        simple example it's mostly the same as the training loop but with a different
        metric name.
        """
        x = batch[self.batch_key]
        #if self.args.arch == "video_slowfast":
            #print("=========", x.shape)
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=False, batch_size=batch_idx, prog_bar=True, sync_dist=True
        )

        return {'val_loss':loss, 'val_acc':acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc']for x in outputs]).mean()
        logging.info('loss is: {},on epoch: {}'.format( avg_loss, self.current_epoch))
        #self.logger.experiment.add_scalar('loss', avg_loss, self.current_epoch)
        
    def configure_optimizers(self):
        """
        We use the SGD optimizer with per step cosine annealing scheduler.
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]


class KineticsDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, args, resume):
        self.args = args
        self.resume = resume #train set的path
        super().__init__()

    def _make_transforms(self, mode: str):

        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        else:
            raise Exception(f"{self.args.data_type} not supported")

        return Compose(transform)

    def _video_transform(self, mode: str):
        """
        This function contains example transforms using both PyTorchVideo and TorchVision
        in the same Callable. For 'train' mode, we use augmentations (prepended with
        'Random'), for 'val' mode we use the respective determinstic function.
        """
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
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(args.video_min_short_side_scale),
                        CenterCrop(args.video_crop_size),
                    ]
                )
            ),
        )

    def train_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = RandomSampler
        train_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path=os.path.join(self.resume, "train.csv"),
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        """
        Defines the train DataLoader that the PyTorch Lightning Trainer trains/tests with.
        """
        sampler = RandomSampler
        val_transform = self._make_transforms(mode="val")
        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=val_transform,
            video_sampler=sampler,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class LimitDataset(torch.utils.data.Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        alpha = 4
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

class Client:
    def __init__(self, epoch):
    #初始化類、對數據加載
    #加載就是針對不同的數據，把其data和label讀入到內存中
        self.name = 'edge'
        self.sample = None
        self.metrics = None
        self.args = argparser.args(epoch)
        self.resume = '/app/data/' 

    def train(self, output,global_model):
        args = self.args
        trainer = pytorch_lightning.Trainer.from_argparse_args(args,default_root_dir="./saved_model_{}".format(self.name))
        logging.info("Train init success...")
        if global_model is not None:
            logging.info("Global model loading...")
            classification_module = VideoClassificationLightningModule.load_from_checkpoint( global_model,name = self.name, map_location='cpu',args=args )
        else:
            logging.info("No global model...")
            classification_module = VideoClassificationLightningModule(args,self.name)
        data_module = KineticsDataModule(args, self.resume)
        logging.info("Dataloader success...")
        trainer.fit(classification_module, data_module)
        #存model
        logging.info("[DDD] Save Weights... ")
        trainer.save_checkpoint(output)
        self.sample =  len(data_module.train_dataset)
        metrics = trainer.callback_metrics
        logging.info("val_loss: {}, val_acc: {}".format(metrics["val_loss"], metrics["val_acc"]))
        return metrics
    """
    def setup_logger(self):
        ch = logging.StreamHandler()
    """
        #formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    """
        ch.setFormatter(formatter)
        logger = logging.getLogger("pytorchvideo")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)
    """
    #resume->global model/output->訓練完model name
    def main_train( self, output: str, global_model=''):
        logging.info("Main_train start...")
        #self.setup_logger()
        pytorch_lightning.trainer.seed_everything()
        #load global_model
        metrics = self.train(output, global_model)
        logging.info("End of {}".format(self.name))
        return metrics




