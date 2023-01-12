from concurrent import futures
import logging
import threading
import os
import random
import grpc
import service_pb2
import service_pb2_grpc
#---------------------------------------------------------
import argparser
import ddd
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
import pytorchvideo.models.resnet
import torch.nn.functional as F
from ddd import MetricsCallback
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
OPERATOR_URI = os.getenv('OPERATOR_URI', "localhost:8787")
APPLICATION_URI = "0.0.0.0:7878"
STOP_EVENT = threading.Event()

def train(baseModel, output_model_path, epochs=1):
    logging.info("Train start...")
    #output = os.path.join("/repos", output_model_path, 'best_model.ckpt')
    output = os.path.join("/app/best_model.ckpt")
    logging.info(f'input path: [{baseModel.path}]')
    logging.info(f'output path: [{output}]')
    logging.info(f'epochs: {epochs}')

    #base_weight_path = os.path.join("/repos", baseModel.path, "best_model.ckpt")
    base_weight_path = os.path.join("/app/best_model.ckpt")
    #建立client
    client = ddd.Client(epochs)
    logging.info("client build success...")
    metrics = client.main_train( output, global_model=base_weight_path)


    # Send finish message
    logging.info(f"GRPC_CLIENT_URI: {OPERATOR_URI}")
    try:
        channel = grpc.insecure_channel(OPERATOR_URI)
        stub = service_pb2_grpc.EdgeOperatorStub(channel)
        result = service_pb2.LocalTrainResult(
            error=0,
            datasetSize=2500,
            metrics=metrics
        )

        response = stub.LocalTrainFinish(result)
    except grpc.RpcError as rpc_error:
        logging.error("grpc error: {}".format(rpc_error))
    except Exception as err:
        logging.error('got error: {}'.format(err))

    logging.debug("sending grpc message succeeds, response: {}".format(response))


class EdgeAppServicer(service_pb2_grpc.EdgeAppServicer):
    def TrainInit(self, request, context):
        logging.info("TrainInit")
        resp = service_pb2.Empty()
        logging.info(f"Sending response: {resp}")
        return resp

    def LocalTrain(self, request, context):
        logging.info("LocalTrain")
        logging.info("{},{},{}".format(request.baseModel, request.localModel.path, request.EpR))
        threading.Thread(
            target=train,
            args=(request.baseModel, request.localModel.path, request.EpR),
            daemon=True
        ).start()
        logging.info("Thread start...")
        resp = service_pb2.Empty()
        logging.info("Sending response: {}".format(resp))
        return resp

    def TrainInterrupt(self, _request, _context):
        # Not Implemented
        return service_pb2.Empty()

    def TrainFinish(self, _request, _context):
        logging.info("TrainFinish")
        STOP_EVENT.set()
        return service_pb2.Empty()


def serve():
    logging.basicConfig(level=logging.DEBUG)

    logging.info("Start server... {}".format(APPLICATION_URI))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    logging.info("Start ThreadPoolExecutor... ")
    service_pb2_grpc.add_EdgeAppServicer_to_server(
        EdgeAppServicer(), server)
    logging.info("Add edgeAppServicer... ")
    server.add_insecure_port(APPLICATION_URI)
    logging.info("Add insecure port... ")
    server.start()
    logging.info("Server start... ")

    STOP_EVENT.wait()
    logging.info("Server Stop")
    server.stop(None)

if __name__ == "__main__":
    serve()