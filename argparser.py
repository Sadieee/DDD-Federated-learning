import logging
class args:
    def __init__(self, epoch=3):
        logging.info("argparse building...")
        # Model parameters.
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay =1e-4
        self.arch = "video_vgg16"

        # Data parameters.
        self.data_path = '/app/data'
        self.workers = 0
        self.batch_size =4
        self.clip_duration =5 
        # 自己加的fl參數
        self.num_class = 2 
        self.round =100 
        self.communication =2 
        self.num_user = 3
        self.devices = None
        self.accelerator = None
        # Trainer parameters.
        self.max_epochs=epoch

        self.video_path_prefix = ""
        self.data_type = "video"
        self.video_num_subsampled =8 
        self.video_means = (0.45, 0.45, 0.45) 
        self.video_stds = (0.225, 0.225, 0.225) 
        self.video_crop_size = 224 
        self.video_min_short_side_scale =256 
        self.video_max_short_side_scale =320 
        self.video_horizontal_flip_p =0.5 
        self.audio_raw_sample_rate =44100 
        self.audio_resampled_rate =16000 
        self.audio_mel_window_size =32 
        self.audio_mel_step_size =16 
        self.audio_num_mels =80
        self.audio_mel_num_subsample =128 
        self.audio_logmel_mean =-7.03 
        self.audio_logmel_std =4.66 
