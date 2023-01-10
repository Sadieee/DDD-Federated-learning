# Video Daraset Loading and Federated Learning Simulation

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
> python main.py --parameter
