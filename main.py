import threading
import argparse
import client_base
import aggregator_base
import pytorch_lightning
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision import models
from pytorch_lightning.callbacks import LearningRateMonitor

def federated_learning():
    parser = argparse.ArgumentParser()
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="video_resnet",
        choices=["video_resnet", "audio_resnet","video_slowfast","video_vgg16"],
        type=str,
    )
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
    parser.add_argument("--num_class", default=36, type=int)
    parser.add_argument("--num_user", default=1, type=int)
    parser.add_argument("--round", default=1, type=int)
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=3,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
    )
    args = parser.parse_args()
    clients = []
    total_dataset = [] 
    global_model = None
    aggregator = aggregator_base.Aggregator(args)
    for i in range(args.num_user):
        clients.append(client_base.Client(str(i + 1),args))
        total_dataset.append(None)
 
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
            for i in range( args.num_user):
                clients[i].main_train( global_model)
                total_dataset[i] = clients[i].sample
        print('---Global model---')
        global_model = aggregator.merge(no_cuda,clients, total_dataset)

    print('Train Finish')

if __name__ == "__main__":
    federated_learning()
