# UNetPyTorchTutorial
A comprehensive tutorial on how to implement and train variational UNet based ousing PyTorch


## Demo notebooks
1. Go to the [Cell tracking chanllenge website](https://celltrackingchallenge.net/2d-datasets/) to download the *HeLa cells on a flat glass* training and test dataset. The dataset can be downloaded and unzipped manually or use the [PythonDownloadAndUnzip](./PythonDownloadAndUnzip.ipynb) notebook to download programmably.

2. Run the [Preprocess](./Preprocess.ipynb) notebook to perform erosion and spatial weight calculation preprocessing on the dataset. 

3. Run the [TrainSimpleUNetWithWeight](./TrainSimpleUNetWithWeight.ipynb) or [TrainSimpleUNetWithoutWeight](./TrainSimpleUNetWithoutWeight.ipynb) to train UNet with or without spatial weighted loss.

4. Run the [DirectInference](./DirectInference.ipynb) notebook or [OverlapTileInference](./OverlapTileInference.ipynb) notebook to segment new (larger) image using the trained UNet model through direct inference or overlap tile strategy.

5. Run the [Evaluation](./Evaluation.ipynb) notebook to calculate intersection over union (IoU) of the trained UNet model.


## Tutorial


## Example results
Preprocessing result:
![Preprocess result](./Assets/Images/Preprocess.png, "Preprocess result")

UNet model trained without weighted loss function segmentation result (valdiation set IoU = 85.36%): 
![UNet trained without spatial weight](./Assets/Images/NoWeightSegResult.png, "UNet trained without spatial weight")

UNet model trained with weighted loss function segmentation result (valdiation set IoU = 85.61%): 
![UNet trained with spatial weight](./Assets/Images/WeightedSegResult.png, "UNet trained with spatial weight")

Overlap tile strategy implementation result:
![Overlap tile strategy result](./Assets/Images/OverlapTile.png, "Overlap tile strategy result")


## Dependency
This repo has been implemented and tested on the following dependencies:
- Python 3.10.13
- matplotlib 3.8.2
- numpy 1.26.2
- torch 2.1.1+cu118
- torchvision 0.16.1+cu118
- notebook 7.0.6
- opencv-python 4.10.0.84


## Computer requirement
This repo has been tested on a laptop computer with the following specs:
- CPU: Intel(R) Core(TM) i7-9750H CPU
- Memory: 32GB 
- GPU: NVIDIA GeForce RTX 2060

## License

[GPL-3.0 license](./LICENSE)

## Reference

[1] Ronneberger, O., Fischer, P. & Brox, T. U-NET: Convolutional Networks for Biomedical Image Segmentation. in Lecture notes in computer science 234–241 (2015). doi:10.1007/978-3-319-24574-4_28.

[2] Maska, M., (...), de Solorzano, C.O.: A benchmark for comparison of cell tracking
algorithms. Bioinformatics 30, 1609-1617 (2014)

## Resources
[1] [Cell tracking challenge website](https://celltrackingchallenge.net/) URL: https://celltrackingchallenge.net/

## Citation
