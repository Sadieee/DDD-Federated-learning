# Video Dataset Federated Learning Simulation

Distributed deep learning training using PyTorchVideo and PyTorchLightning.
The training model is 3D-VGG16 and Dataset is video driver drowsiness detection dataset.
## Requirements
anaconda >= 22.9.0  
python >= 3.8  
torch >= 1.12.1  
pytorchvideo >= 0.1.5  
pytorch_lightning >= 1.7.7  

## Create Environment
> conda create --name envName  
> activate envName

## Dataset Preprocessing
```
data  
│
├───val.csv # test dataset annotation file
├───edge1
│       ├───train.csv # annotation file of edge1
│       ├───0001  # arbitrary video folder naming
│       │     ├───00001.avi
│       │     .
│       │     └───00002.avi
│       └───0002
│             ├───00001.avi
│             .
│             └───00002.avi
│
└───edge2
        ├───train.csv # annotation file of edge2
        ├───0001  # arbitrary video folder naming
        │     ├───00001.avi
        │     .
        │     └───00002.avi
        └───0002
              ├───00001.avi
              .
              └───00002.avi
```
## Simulation
> python main.py --data_path data
