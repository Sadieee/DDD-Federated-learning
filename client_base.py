import os
import glob
import torch.utils.data
import torch.utils.data.distributed
import aggregator_base
from tqdm import tqdm
import argparse
import itertools
import logging
import torchmetrics
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models
import pytorchvideo.models.x3d
import pytorchvideo.models.resnet
import torch
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
class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args, name):
        self.args = args
        self.name = name
        super().__init__()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                model_num_class=2,
            )
            self.batch_key = "video"
        elif self.args.arch == "video_slowfast":
            self.model = pytorchvideo.models.slowfast.create_slowfast(
                model_num_class=2,
            )
            self.batch_key = "video"
        elif self.args.arch == "audio_resnet":
            self.model = pytorchvideo.models.resnet.create_acoustic_resnet(
                input_channel=1,
                model_num_class=2,
            )
            self.batch_key = "audio"
        else:
            raise Exception("{self.args.arch} not supported")

    def get_model(self):
        return self.model

    def on_train_epoch_start(self):
        epoch = self.trainer.current_epoch
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_idx
        )
        return {'loss':loss, 'acc':acc}
 
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc']for x in outputs]).mean()
        self.logger.experiment.add_scalar('loss',avg_loss, self.current_epoch)
        output_file = './result_{}.txt'.format(self.name)
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             ]
        output_file.close()

    def validation_step(self, batch, batch_idx):
        x = batch[self.batch_key]
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
        self.logger.experiment.add_scalar('loss', avg_loss, self.current_epoch)
        output_file = './result_{}.txt'.format(self.name)
        if os.path.exists(output_file):
            output_file = open(output_file, 'a')
        else :
            output_file = open(output_file, 'w')
            print("Create result.txt.")             
        output_file.write('epoch={}, Val loss {:.8f}, Val acc {:.3f}\n'.format( self.current_epoch, avg_loss, avg_acc))
        output_file.close()
        
    def configure_optimizers(self):
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
        self.resume = resume
        super().__init__()

    def _make_transforms(self, mode: str):
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
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        alpha = 4
        fast_pathway = frames
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
        self.name = 'edge' + name
        self.sample = None
        self.model = None
        self.metrics = None
        self.args = args
        self.resume = './data/edge' + str(name)

    def train(self,global_model):
        args = self.args
        trainer = pytorch_lightning.Trainer.from_argparse_args(args,default_root_dir="./saved_model_{}".format(self.name))
        classification_module = VideoClassificationLightningModule(args,self.name)
        if global_model is not None:
            print("Global model loading...")
            classification_module.model.load_state_dict(global_model)
        data_module = KineticsDataModule(args, self.resume)
        trainer.fit(classification_module, data_module)
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
        self.train(global_model)
        print("End of {}".format(self.name))




