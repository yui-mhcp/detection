# :yum: Object detection

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications! :yum:

**Important Note**: The EAST training procedure is not implemented yet for the current post-processing pipeline inspired from [this repo](https://github.com/SakuraRiven/EAST). It can still be used by using the available pretrained weights! :smile:

## Project structure

```bash
├── architectures            : utilities for model architectures
│   ├── layers               : custom layer implementations
│   ├── transformers         : transformer architecture implementations
│   ├── common_blocks.py     : defines common blocks (e.g., Conv + BN + ReLU)
│   ├── east_arch.py         : EAST architecture
│   ├── generation_utils.py  : utilities for text and sequence generation
│   ├── hparams.py           : hyperparameter management
│   ├── simple_models.py     : defines classical models such as CNN / RNN / MLP and siamese
│   └── yolo_arch.py         : YOLOv2 architecture
├── custom_train_objects     : custom objects used in training / testing
│   ├── callbacks            : callbacks loading and implementations
│   ├── generators           : custom data generators
│   ├── losses               : loss functions
│   │   └── yolo_loss.py     : YOLO specific loss
│   ├── metrics              : metrics loading and implementations
│   ├── optimizers           : optimizer loading
│   ├── checkpoint_manager.py: handle model checkpoint management (inspired from `tf.train.CheckpointManager`)
│   └── history.py           : main History class to compute training statistics / track config
├── loggers                  : logging utilities for tracking experiment progress
├── models                   : main directory for model classes
│   ├── detection            : detector implementations
│   │   ├── base_detector.py : abstract base class for all detectors
│   │   ├── east.py          : EAST implementation for text detection
│   │   └── yolo.py          : YOLOv2 implementation for general object detection
│   ├── interfaces           : directories for interface classes
│   └── weights_converter.py : utilities to convert weights between different models
├── tests                    : unit and integration tests for model validation
├── utils                    : utility functions for data processing and visualization
├── detection.ipynb          : notebook demonstrating detection features
├── example_yolo.ipynb       : specific example notebook for YOLO model
├── LICENCE                  : project license file
├── README.md                : this file
└── requirements.txt         : required packages
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes.

## Available features

- **Detection** (module `models.detection`):

| Feature   | Function / class | Description |
| :-------- | :--------------- | :---------- |
| detection | `detect`         | Detect objects on images/videos with multiple saving options (save cropped boxes, detected images, video frames, etc.) |
| stream    | `stream`         | Perform real-time detection using your camera (also allows saving frames) |

The `detection` notebook provides a concrete demonstration of these functions :smile:

## Available models

### Model architectures

Available architectures:
- `detection`:
    - [YOLOv2](https://pjreddie.com/darknet/yolov2/) : You Only Look Once (version 2)
    - [EAST](https://arxiv.org/abs/1704.03155) : Efficient and Accurate Scene Text detector

### Model weights

| Classes   | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |
| [80 classes](https://github.com/pjreddie/darknet/blob/master/data/coco.names) | [COCO](https://cocodataset.org/#home) | `YOLOv2`  | [YOLOv2's author](https://pjreddie.com/darknet/yolov2/)   | [link](https://pjreddie.com/media/files/yolov2.weights) |

`Pretrained backend` for YOLO can be downloaded at [this link](https://drive.google.com/drive/folders/1lv0s8IAg1AWiiGq7o3H13TJnjp0K9Nh8?usp=sharing).

The pretrained version of EAST can be downloaded [from this project](https://github.com/SakuraRiven/EAST). It should be stored in `pretrained_models/pretrained_weights/east_vgg16.pth` (`torch` is required to transfer the weights: `pip install torch`).

## Installation and usage

See [the installation guide](https://github.com/yui-mhcp/blob/master/INSTALLATION.md) for a step-by-step installation :smile:

Here is a summary of the installation procedure, if you have a working python environment :
1. Clone this repository: `git clone https://github.com/yui-mhcp/detection.git`
2. Go to the root of this repository: `cd detection`
3. Install requirements: `pip install -r requirements.txt`
4. Open the `detection` notebook and follow the instructions!

## TO-DO list:

- [x] Make the TO-DO list
- [x] Support pretrained COCO model
- [x] Add weights for face detection
- [x] Add label-based model loading (without manual association)
- [x] Add `producer-consumer` based streaming
- [x] Automatically download the official YOLOv2 pretrained weights (if not loaded)
- [x] Add the Locality-Aware Non Maximum Suppression (NMS) method as described in the `EAST` paper
- [x] Keras 3 support
- [x] Convert the pretrained models to be compatible with Keras 3
- [x] Make comprehensive comparison example between NMS and LANMS

## Difference between `detection` and `segmentation`

The two main methodologies in *object detection* are `detection` with `bounding boxes` and `pixel-wise segmentation`. These approaches both aim to detect the position of objects in an image but with different levels of precision. This difference impacts the model architecture as the required output shape is not the same.

Here is a simple, non-exhaustive comparison of both approaches based on several criteria:

| Criterion | Detection | Segmentation |
| :-------: | :-------: | :----------: |
| Precision | Surrounding bounding boxes | Pixel by pixel |
| Type of output | `[x, y, w, h]` (position of bounding boxes) | Mask ([0, 1] probability score for each pixel) |
| Output shape | `[grid_h, grid_w, nb_box, 4 + 1 + nb_class]`\* | `[image_h, image_w, 1]` |
| Applications | General detection + classification | Medical image detection / object extraction |
| Model architecture | Full CNN 2D downsampling to `(grid_h, grid_w)` | Full CNN with downsampling and upsampling |
| Post processing | Decode output to get position of boxes | Thresholding pixel confidence |
| Model mechanism | Split image into grid and detect boxes in each grid cell | Downsample the image and upsample it to give probability of object for each pixel |
| Support multi-label classification | Yes, by design | Yes, but not its main application |

\* This is the classical output shape of `YOLO` models. The last dimension is `[x, y, w, h, confidence, * class_score]`

More advanced strategies also exist, differing from the standard methodologies described above. This aims to be a simple introduction to object detection and segmentation.

## Notes and references

### GitHub projects

The code for the YOLO part of this project is highly inspired from this repo:
- [experiencor's repository](https://github.com/experiencor/keras-yolo2): TensorFlow 1.x implementation of `YOLOv2` (main inspiration for this repository)

The code for the EAST part of this project is highly inspired from this repo:
- [SakuraRiven pytorch implementation](https://github.com/SakuraRiven/EAST): PyTorch implementation of the EAST paper.


### Object Detection Tutorials

- [TensorFlow Object Detection API Tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) : Step-by-step guide
- [Image segmentation tutorials](https://www.tensorflow.org/tutorials/images/segmentation) : U-Net implementation in TensorFlow + image segmentation tutorial
- [PyTorch Vision Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) : Object detection with PyTorch
- [YOLO Explained](https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088) : Detailed explanation of YOLO architecture
- [Gentle guide on how YOLO object detection works](https://hackernoon.com/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-1-aec99277f56f) : Good tutorial explaining the image detection mechanism

### Papers

- [YOLO9000: Better, Stronger, Faster](https://arxiv.org/abs/1612.08242v1) : The original YOLOv2 paper
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) : U-Net original paper
- [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) : Text detection (with possibly rotated bounding boxes) with a segmentation model (U-Net).


### Video Series

- [Kaggle's Computer Vision Tutorials](https://www.kaggle.com/learn/computer-vision) : Practical computer vision examples
- [Two Minute Papers](https://www.youtube.com/c/K%C3%A1rolyZsolnai) : Quick explanations of recent deep learning papers


### Datasets

- [COCO](https://cocodataset.org/#home) dataset : 80 labels dataset for object detection in real context
- [COCO Text](https://vision.cornell.edu/se3/coco-text-2/) dataset : An extension of COCO for text detection
- [Wider Face](http://shuoyang1213.me/WIDERFACE/) dataset : Face detection dataset
- [kangaroo](https://github.com/experiencor/kangaroo) dataset: Funny tiny dataset to train fast a powerful model (fun to have fast results)

## Contacts and licence

Contacts:
- **Mail**: `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)**: yui0732

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for details.

This license allows you to use, modify, and distribute the code, as long as you include the original copyright and license notice in any copy of the software/source. Additionally, if you modify the code and distribute it, or run it on a server as a service, you must make your modified version available under the same license.

For more information about the AGPL-3.0 license, please visit [the official website](https://www.gnu.org/licenses/agpl-3.0.html)

## Citation

If you find this project useful in your work, please add this citation to give it more visibility! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```