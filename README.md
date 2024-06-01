# :yum: Object detection

Check the [CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) file to have a global overview of the latest modifications ! :yum:

**Important Note** : the EAST training procedure is not implemented yet for the current post-processing pipeline inspired from [this repo](https://github.com/SakuraRiven/EAST). It can still be used by using the available pretrained weights ! :smile:

## Project structure

```bash
├── custom_architectures
│   ├── east_arch.py        : defines the EAST architecture with VGG16 backbone
│   └── yolo_arch.py        : defines the YOLOv2 architecture
├── custom_layers
├── custom_train_objects
│   ├── losses
│   │   └── yolo_loss.py        : main YOLOLoss class
├── loggers
├── models
│   ├── detection
│   │   ├── base_detector.py    : abstract class for object detection models
│   │   ├── east.py         : main EAST class (rotated bounding box detection based on U-Net model)
│   │   └── yolo.py         : main YOLO class (object detection)
├── unitests
├── utils
├── detection.ipynb     : illustrates the use of pretrained models for object / text detection
└── example_yolo.ipynb  : illustrates a complete YOLOv2 training procedure
```

Check [the main project](https://github.com/yui-mhcp/base_dl_project) for more information about the unextended modules / structure / main classes. 

## Available features

- **Detection** (module `models.detection`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| detection | `detect`  | detect objects on images / videos and allow multiple saving types (save cropped boxes, detected images, video frames, ...)    |
| stream    | `stream`  | perform detection on your camera (also allow to save frames) |

The `detection` notebook provides a concrete demonstration of these functions :smile:

## Available models

### Model architectures

Available architectures : 
- `detection` :
    - [YOLOv2](https://pjreddie.com/darknet/yolov2/)
    - [EAST](https://arxiv.org/abs/1704.03155)

### Model weights

| Classes   | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |
| [80 classes](https://github.com/pjreddie/darknet/blob/master/data/coco.names) | [COCO](https://cocodataset.org/#home) | `YOLOv2`  | [YOLOv2's author](https://pjreddie.com/darknet/yolov2/)   | [link](https://pjreddie.com/media/files/yolov2.weights) |


`pretrained backend` for YOLO can be downloaded at [this link](https://drive.google.com/drive/folders/1lv0s8IAg1AWiiGq7o3H13TJnjp0K9Nh8?usp=sharing). 

The pretrained version of EAST can be downloaded [from this project](https://github.com/SakuraRiven/EAST). It should be stored in `pretrained_models/pretrained_weights/east_vgg16.pth` (`torch` is required to transfer the weights : `pip install torch`).

**The previously available `yolo_faces` models wil be published in the next update. Some further `tensorflow -> keras` convertion checks have to be performed.**

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/detection.git`
2. Go to the root of this repository : `cd detection`
3. Install requirements : `pip install -r requirements.txt`
4. Open `detection` notebook and follow the instructions !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Support pretrained COCO model
- [x] Add weights for face detection
- [x] Add label-based model loading (without manual association)
- [x] Add `producer-consumer` based prediction / streaming
- [x] Automatically downloads the official YOLOv2 pretrained weights (if not loaded)
- [x] Add the Locality-Aware Non Maximum Suppression (NMS) method as described in the `EAST` paper
- [x] Keras 3 support
- [ ] Convert the pretrained models to be compatible with Keras 3
- [x] Make comprehensive comparison example between NMS and LANMS

## Difference between `detection` and `segmentation`

The 2 main methodologies in *object detection* are `detection` with `bounding boxes` and `pixel-wise segmentation`. These 2 approaches tends to detect position of objects in an image but with different level of precision. This difference has an impact on the model architecture as the required output shape is not thesame. 

Here is a simple, non-exhaustive comparison of both approaches based on some criteria :

| Criterion     | Detection                     | Segmentation      |
| :-----------: | :---------------------------: | :---------------: |
| Precision     | Surrounding bounding boxes    | pixel by pixel    |
| Type of output    | `[x, y, w, h]` (position of bounding boxes)   | mask ([0, 1] probability score for each pixel)  |
| Output shape      | `[grid_h, grid_w, nb_box, 4 + 1 + nb_class]`\*  | `[image_h, image_w, 1]`   |
| Applications      | General detection + classification    | Medical image detection / object extraction   |
| Model architecture    | Full CNN 2D downsampling to `(grid_h, grid_w)`    | Full CNN with downsampling and upsampling |
| Post processing       | Decode output to get position of boxes    | Thresholding pixel confidence |
| Model mechanism       | Split image in grid and detect boxes in each grid cell    | downsample the image and upsample it to give probability of object for each pixel of the image    |
| Support multi-label classification    | yes, by design        | I guess yes but not its main application  |

\* This is the classical output shape of `YOLO` models. The last dimension is `[x, y, w, h, confidence, * class_score]`

More advanced strategies also exist, differing from the standard methodology described above ;) This aims to be a simple introduction to object detection / segmentation. 

## Contacts and licence

Contacts :
- **Mail** : `yui-mhcp@tutanota.com`
- **[Discord](https://discord.com)** : yui0732

### Terms of use

The goal of these projects is to support and advance education and research in Deep Learning technology. To facilitate this, all associated code is made available under the [GNU Affero General Public License (AGPL) v3](AGPLv3.licence), supplemented by a clause that prohibits commercial use (cf the [LICENCE](LICENCE) file).

These projects are released as "free software", allowing you to freely use, modify, deploy, and share the software, provided you adhere to the terms of the license. While the software is freely available, it is not public domain and retains copyright protection. The license conditions are designed to ensure that every user can utilize and modify any version of the code for their own educational and research projects.

If you wish to use this project in a proprietary commercial endeavor, you must obtain a separate license. For further details on this process, please contact me directly.

For my protection, it is important to note that all projects are available on an "As Is" basis, without any warranties or conditions of any kind, either explicit or implied. However, do not hesitate to report issues on the repository's project, or make a Pull Request to solve it :smile: 

### Citation

If you find this project useful in your work, please add this citation to give it more visibility ! :yum:

```
@misc{yui-mhcp
    author  = {yui},
    title   = {A Deep Learning projects centralization},
    year    = {2021},
    publisher   = {GitHub},
    howpublished    = {\url{https://github.com/yui-mhcp}}
}
```

## Notes and references 

The code for the YOLO part of this project is highly inspired from this repo :
- [1] [experiencor's repository](https://github.com/experiencor/keras-yolo2) : tensorflow 1.x implementation of `YOLOv2` (main inspiration for this repository)
The code for the EAST part of this project is highly inspired from this repo :
- [2] [SakuraRiven pytorch implementation](https://github.com/SakuraRiven/EAST) : pytorch implementation of the EAST paper.


Papers and tutorials :
- [3] [YOLO9000 : better, stronger, faster](https://arxiv.org/abs/1612.08242v1) : the original YOLOv2 paper
- [4] [darknet's website](https://pjreddie.com/darknet/yolov2/) : the official releases of YOLOv2
- [5] [Gentle guide on how YOLO object detection works](https://hackernoon.com/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-1-aec99277f56f) : good tutorial explaning the image detection mechanism
- [6] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) : U-net original paper
- [7] [image segmentation tutorials](https://www.tensorflow.org/tutorials/images/segmentation) : U-net implementation in tensorflow + image segmentation tutorial
- [8] [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155) : text detection (with possibly rotated bounding-boxes) with a segmentation model (U-Net). 

Datasets :
- [COCO](https://cocodataset.org/#home) dataset : 80 labels dataset for object detection in real context
- [COCO Text](https://vision.cornell.edu/se3/coco-text-2/) dataset : an extension of COCO for text detection
- [Wider Face](http://shuoyang1213.me/WIDERFACE/) dataset : face detection dataset
- [kangaroo](https://github.com/experiencor/kangaroo) dataset : funny tiny dataset to train fast a powerful model (fun to have fast results)
