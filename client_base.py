import os
import glob
import torch.utils.data
import torch.utils.data.distributed
import aggregator_base
from tqdm import tqdm
import copy
import argparse
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


"""
This video classification example demonstrates how PyTorchVideo models, datasets and
transforms can be used with PyTorch Lightning module. Specifically it shows how a
simple pipeline to train a Resnet on the Kinetics video dataset can be built.

Don't worry if you don't have PyTorch Lightning experience. We'll provide an explanation
of how the PyTorch Lightning module works to accompany the example.

The code can be separated into three main components:
1. VideoClassificationLightningModule (pytorch_lightning.LightningModule), this defines:
    - how the model is constructed,
    - the inner train or validation loop (i.e. computing loss/metrics from a minibatch)
    - optimizer configuration

2. KineticsDataModule (pytorch_lightning.LightningDataModule), this defines:
    - how to fetch/prepare the dataset
    - the train and val dataloaders for the associated dataset

3. pytorch_lightning.Trainer, this is a concrete PyTorch Lightning class that provides
  the training pipeline configuration and a fit(<lightning_module>, <data_module>)
  function to start the training/validation loop.

All three components are combined in the train() function. We'll explain the rest of the
details inline.
"""
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
        #self.save_hyperparameters(args)
        self.matrics = []


        #############
        # PTV Model #
        #############

        # Here we construct the PyTorchVideo model. For this example we're using a
        # ResNet that works with Kinetics (e.g. 400 num_classes). For your application,
        # this could be changed to any other PyTorchVideo model (e.g. for SlowFast use
        # create_slowfast).
        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                model_num_class=self.args.num_class,
            )
            self.batch_key = "video"
        elif self.args.arch == "video_slowfast":
            self.model = pytorchvideo.models.slowfast.create_slowfast(
                #input_channels=3,
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
        """
        Forward defines the prediction/inference actions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        This function is called in the inner loop of the training epoch. It must
        return a loss that is used for loss.backwards() internally. The self.log(...)
        function can be used to log any training metrics.

        PyTorchVideo batches are dictionaries containing each modality or metadata of
        the batch collated video clips. Kinetics contains the following notable keys:
           {
               'video': <video_tensor>,
               'audio': <audio_tensor>,
               'label': <action_label>,
           }

        - "video" is a Tensor of shape (batch, channels, time, height, Width)
        - "audio" is a Tensor of shape (batch, channels, time, 1, frequency)
        - "label" is a Tensor of shape (batch, 1)

        The PyTorchVideo models and transforms expect the same input shapes and
        dictionary structure making this function just a matter of unwrapping the dict and
        feeding it through the model/loss.
        """
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
        self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)
        #===============紀錄結果==================
        #RESULT檔在不在
        output_file = './result_{}.txt'.format(self.name)
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             
        output_file.write('epoch={}, Train loss {:.8f}, Train acc {:.3f}\n'.format( self.current_epoch, avg_loss, avg_acc))
        output_file.close()
        #===============紀錄結束==================

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
        #print(outputs[0].shape)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc']for x in outputs]).mean()
        self.logger.experiment.add_scalar('loss', avg_loss, self.current_epoch)
        #===============紀錄結果==================
        #RESULT檔在不在
        output_file = './result_{}.txt'.format(self.name)
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             
        output_file.write('epoch={}, Val loss {:.8f}, Val acc {:.3f}\n'.format( self.current_epoch, avg_loss, avg_acc))
        output_file.close()
        #===============紀錄結束==================
        
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
    """
    This LightningDataModule implementation constructs a PyTorchVideo Kinetics dataset for both
    the train and val partitions. It defines each partition's augmentation and
    preprocessing transforms and configures the PyTorch DataLoaders.
    """

    def __init__(self, args, resume):
        self.args = args
        self.resume = resume #train set的path
        super().__init__()

    def _make_transforms(self, mode: str):
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
        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        elif self.args.data_type == "audio":
            transform = [
                self._audio_transform(),
                RemoveKey("video"),
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
                            PackPathway()
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
                        ]
                        if mode == "train"
                        else [
                            ShortSideScale(args.video_min_short_side_scale),
                            CenterCrop(args.video_crop_size),
                        ]
                    )
                ),
            )

    def _audio_transform(self):
        """
        This function contains example transforms using both PyTorchVideo and TorchAudio
        in the same Callable.
        """
        args = self.args
        n_fft = int(
            float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size
        )
        hop_length = int(
            float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size
        )
        eps = 1e-10
        return ApplyTransformToKey(
            key="audio",
            transform=Compose(
                [
                    Resample(
                        orig_freq=args.audio_raw_sample_rate,
                        new_freq=args.audio_resampled_rate,
                    ),
                    MelSpectrogram(
                        sample_rate=args.audio_resampled_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=args.audio_num_mels,
                        center=False,
                    ),
                    Lambda(lambda x: x.clamp(min=eps)),
                    Lambda(torch.log),
                    UniformTemporalSubsample(args.audio_mel_num_subsample),
                    Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                    Lambda(
                        lambda x: x.view(1, x.size(0), 1, x.size(1))
                    ),  # (T, F) -> (1, T, 1, F)
                    Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
                ]
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
    def __init__(self, name,args):
    #初始化類、對數據加載
    #加載就是針對不同的數據，把其data和label讀入到內存中
        self.name = 'edge' + name
        self.sample = None
        self.model = None
        self.metrics = None
        self.args = args
        self.resume = './data/edge' + str(name)
        #聚合器
        self.aggregator = aggregator_base.Aggregator(args, name)

    def train(self,global_model):
        args = self.args
        trainer = pytorch_lightning.Trainer.from_argparse_args(args,default_root_dir="./saved_model_{}".format(self.name))
        classification_module = VideoClassificationLightningModule(args,self.name)
        if global_model is not None:
            print("Global model loading...")
            classification_module.model.load_state_dict(global_model)
        data_module = KineticsDataModule(args, self.resume)
        trainer.fit(classification_module, data_module)
        #存model
        trainer.save_checkpoint("best_model_{}.ckpt".format(self.name))
        model = classification_module.get_model()
        self.model = model.state_dict()
        self.sample =  len(data_module.train_dataset)

    def setup_logger(self):
        ch = logging.StreamHandler()
        formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        ch.setFormatter(formatter)
        logger = logging.getLogger("pytorchvideo")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(ch)

    def main_train(self, global_model):
        args = self.args
        self.setup_logger()
        pytorch_lightning.trainer.seed_everything()
        #load global_model
        self.train(global_model)

        print("End of {}".format(self.name))




