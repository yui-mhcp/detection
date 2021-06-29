# :yum: Object detection

## Project structure

```bash
├── custom_architectures/   : custom architectures
├── custom_layers/          : custom layers
├── custom_train_objects/   : custom objects for training
│   ├── callbacks/          : custom callbacks
│   ├── generators/         : custom data generators
│   ├── losses/             : custom losses
│   ├── optimizers/         : custom optimizers / lr schedulers
├── datasets/               : utilities for dataset loading / processing
│   ├── custom_datasets/    : where to save custom datasets processing
├── hparams/                : utility class to define modulable hyper-parameters
├── models/                 : main `BaseModel` subclasses directory
│   ├── detection/          : directory for detection models
├── pretrained_models/      : saving directory for pretrained models
└── utils/                  : utilities for data processing
```

See [my data_processing repo](https://github.com/yui-mhcp/data_processing) for more information on the `utils` module and `data processing` features.

See [my base project](https://github.com/yui-mhcp/base_dl_project) for more information on the `BaseModel` class, supported datasets, project extension, ...

## Available features

- **Detection** (module `models.detection`) :

| Feature   | Fuction / class   | Description |
| :-------- | :---------------- | :---------- |
| detection | `detect`  | detect objects on images / videos and allow multiple saving types (save cropped boxes, detected images, video frames, ...)    |
| stream    | `stream`  | perform detection on your camera (also allow to save frames) |

You can check the `detection` notebook for a concrete demonstration

## Available models

### Model architectures

Available architectures : 
- `detection` :
    - [YOLOv2](https://pjreddie.com/darknet/yolov2/)

### Model weights

| Classes   | Dataset   | Architecture  | Trainer   | Weights   |
| :-------: | :-------: | :-----------: | :-------: | :-------: |
| [80 classes](https://github.com/pjreddie/darknet/blob/master/data/coco.names) | [COCO](https://cocodataset.org/#home) | `YOLOv2`  | [YOLOv2's author](https://pjreddie.com/darknet/yolov2/)   | [link](https://pjreddie.com/media/files/yolov2.weights) |
| `face`    | [Wider Face](http://shuoyang1213.me/WIDERFACE/)    | `YOLOv2`  | [me](https://github.com/yui-mhcp)   | [Google Drive](https://drive.google.com/drive/folders/1nQ4MpCxgikO-mU6gkQRsh0famnUS6Dzs?usp=sharing)    |
| `kangaroo`    | [experiencor kangaroo](https://github.com/experiencor/kangaroo)  | `YOLOv2`  | [me](https://github.com/yui-mhcp)   | [Google Drive](https://drive.google.com/drive/folders/1CykpBAHdJhd_blpyHG_U8458nIpXz-RR?usp=sharing)  |

Models must be unzipped in the `pretrained_models/` directory !

**Important Note** : the official pretrained model is available as a `yolov2.weights` file. You can find how to create a `YOLO` pretrained model based on these weights in the `detection` notebook.

You can download `pretrained backend` at [this link](https://drive.google.com/drive/folders/1lv0s8IAg1AWiiGq7o3H13TJnjp0K9Nh8?usp=sharing). 

## Installation and usage

1. Clone this repository : `git clone https://github.com/yui-mhcp/detection.git`
2. Go to the root of this repository : `cd detection`
3. Install requirements : `pip install -r requirements.txt`
4. Open `detection` notebook and follow the instructions !

## TO-DO list :

- [x] Make the TO-DO list
- [x] Support pretrained COCO model
- [x] Add weights for face detection
- [ ] Add label-based model loading (without manual association)
- [ ] Improve the face detection model
- [ ] Add more detection models
- [ ] Add support vor `YOLOv3` and `YOLOv4` models
- [ ] Add segmentation models

## Difference between `detection` and `segmentation`

Main methods in object *detection* in image are `detection` with `bounding boxes` and `pixel-wise segmentation`. These 2 approaches tends to detect position of objects in an image but with different level of precision. This difference has an impact on the model architecture as the required output shape is not thesame. 

Here is a comparison of both approaches based on some criteria :

| Criteria      | Detection                     | Segmentation      |
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

## Contacts and licence

You can contact [me](https://github.com/yui-mhcp) at yui-mhcp@tutanota.com or on [discord](https://discord.com) at `yui#0732`

The objective of these projects is to facilitate the development and deployment of useful application using Deep Learning for solving real-world problems and helping people. 
For this purpose, all the code is under the [GNU GPL v3 licence](LICENCE)

Furthermore, you **cannot** use any of these projects for commercial purpose without my permission. You can use, modify, distribute and use any of my projects for production as long as you respect the terms of the [licence](LICENCE) and use it for non-commercial purposes (i.e. free applications / research). 

If you use this project in your work, please cite this project to give it more visibility ! :smile:

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

The code for this project is highly inspired from this repo :
- [1] [experiencor's repository](https://github.com/experiencor/keras-yolo2) : tensorflow 1.x implementation of `YOLOv2` (main inspiration for this repository)

Papers and tutorials :
- [2] [YOLO9000 : better, stronger, faster](https://arxiv.org/abs/1612.08242v1) : the original YOLOv2 paper
- [3] [darknet's website](https://pjreddie.com/darknet/yolov2/) : the official releases of YOLOv2
- [4] [Gentle guide on how YOLO object detection works](https://hackernoon.com/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-1-aec99277f56f) : good tutorial explaning the image detection mechanism

Datasets :
- [COCO](https://cocodataset.org/#home) dataset : 80 labels dataset for object detection in real context
- [Wider Face](http://shuoyang1213.me/WIDERFACE/) dataset : face detection dataset
- [kangaroo](https://github.com/experiencor/kangaroo) dataset : funny tiny dataset to train fast a powerful model (fun to have fast results)
