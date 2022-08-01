
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import DEV, TIME_LEVEL, timer
from custom_architectures import get_architecture
from models.interfaces.base_image_model import BaseImageModel
from utils import load_json, dump_json, normalize_filename, plot
from utils.image import _video_formats, _image_formats, load_image, save_image, stream_camera, BASE_COLORS, get_video_infos
from utils.image.box_utils import *
from utils.thread_utils import Producer, ThreadedDict

time_logger = logging.getLogger('timer')

DEFAULT_ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

COCO_CONFIG = {
    'labels' : ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    'anchors' : DEFAULT_ANCHORS
}
VOC_CONFIG  = {
    'labels' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'anchors'   : DEFAULT_ANCHORS #[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
}

def update_box_labels(images, boxes, classifier, ** kwargs):
    img_boxes = []
    for img, box in zip(images, boxes):
        img_boxes.extend(crop_box(img, box, ** kwargs))
    
    labels = classifier.predict(frame_boxes)
    
    label_idx = 0
    for box in boxes:
        for b in box:
            b.label = labels[label_idx]
            label_idx += 1

class YOLO(BaseImageModel):
    def __init__(self,
                 labels, 
                 max_box_per_image  = 100,
                 
                 input_size = 416,
                 nb_class   = None,
                 backend    = "FullYolo",
                 
                 anchors    = DEFAULT_ANCHORS,

                 obj_threshold  = 0.35,
                 nms_threshold  = 0.2,

                 **kwargs
                ):
        assert len(anchors) % 2 == 0

        self._init_image(input_size = input_size, ** kwargs)
        
        self.backend    = backend
        self.anchors    = anchors
        self.max_box_per_image = max_box_per_image

        self.labels   = list(labels) if not isinstance(labels, str) else [labels]
        self.nb_class = max(2, nb_class if nb_class is not None else len(self.labels))
        if self.nb_class > len(self.labels):
            self.labels += [''] * (self.nb_class - len(self.labels))
        
        self.np_anchors = np.reshape(np.array(anchors), [self.nb_box, 2])
        
        self.obj_threshold  = obj_threshold
        self.nms_threshold  = nms_threshold
        
        super().__init__(** kwargs)

    def _init_folders(self):
        super()._init_folders()
        os.makedirs(self.stream_dir, exist_ok = True)
        os.makedirs(self.pred_img_dir, exist_ok = True)
        os.makedirs(self.pred_boxes_dir, exist_ok = True)
        os.makedirs(self.stream_img_dir, exist_ok = True)
        os.makedirs(self.stream_boxes_dir, exist_ok = True)
    
    def init_train_config(self, * args, ** kwargs):
        super().init_train_config(* args, ** kwargs)
        
        if hasattr(self, 'model_loss'):
            self.model_loss.seen = self.current_epoch
            
    def _build_model(self, flatten = True, randomize = True, ** kwargs):
        feature_extractor = get_architecture(
            architecture_name = self.backend, input_shape = self.input_size, include_top = False,
            ** kwargs
        )
        
        super()._build_model(model = {
            'architecture_name' : 'yolo',
            'feature_extractor' : feature_extractor,
            'input_shape'       : self.input_size,
            'nb_class'      : self.nb_class,
            'nb_box'        : self.nb_box,
            'flatten'       : flatten,
            'randomize'     : randomize
        })
    
    @property
    def stream_dir(self):
        return os.path.join(self.folder, "stream")
        
    @property
    def pred_img_dir(self):
        return os.path.join(self.pred_dir, "images")
    
    @property
    def pred_boxes_dir(self):
        return os.path.join(self.pred_dir, "boxes")
    
    @property
    def stream_img_dir(self):
        return os.path.join(self.stream_dir, "images")
    
    @property
    def stream_boxes_dir(self):
        return os.path.join(self.stream_dir, "boxes")
    
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(
                shape =(None, self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class), dtype = tf.float32
            ),
            tf.TensorSpec(
                shape = (None, 1, 1, 1, self.max_box_per_image, 4), dtype = tf.float32
            )
        )
    
    @property
    def nb_box(self):
        return len(self.anchors) // 2
    
    @property
    def grid_h(self):
        return self.output_shape[1]
    
    @property
    def grid_w(self):
        return self.output_shape[2]
        
    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_image)
    
    def __str__(self):
        des = super().__str__()
        des += self._str_image()
        des += "Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        des += "Feature extractor : {}\n".format(self.backend)
        return des
    
    @timer(name = 'inference')
    def detect(self, image, get_boxes = False, training = False, ** kwargs):
        """
            Perform prediction on `image` and returns either the model's output either the boxes
            
            Arguments :
                - image : tf.Tensor of rank 3 or 4 (single / batched image(s))
                - get_boxes : bool, whether to decode the model's output or not
                - training  : whether to make prediction in training mode
                - kwargs    : given to `decode_output` if `get_boxes`
            Return :
                if `get_boxes == False` :
                    model's output of shape (B, grid_h, grid_w, nb_box, 5 + nb_class)
                else:
                    list of boxes (where boxes is the list of BoundingBox for detected objects)
                
        """
        if not isinstance(image, tf.Tensor): image = tf.cast(image, tf.float32)
        if len(tf.shape(image)) == 3: image = tf.expand_dims(image, axis = 0)
        
        outputs = self(image, training = training)
        
        if not get_boxes: return outputs
        
        return [self.decode_output(out, ** kwargs) for out in outputs]
    
    def compile(self, loss_config = {}, **kwargs):
        kwargs['loss'] = 'YoloLoss'
        
        loss_config.update({'anchors' : self.anchors})
        
        super().compile(loss_config = loss_config, ** kwargs)
    
    def get_input(self, filename):
        return self.get_image(filename)
    
    def get_output_fn(self, boxes, labels, nb_box, image_h, image_w, ** kwargs):
        if hasattr(boxes, 'numpy'): boxes = boxes.numpy()
        if hasattr(image_h, 'numpy'): image_h, image_w = image_h.numpy(), image_w.numpy()
        
        output      = np.zeros((self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class))
        true_boxes  = np.zeros((1, 1, 1, self.max_box_per_image, 4))
        
        logging.debug("Image with shape ({}, {}) and {} boxes :".format(image_h, image_w, len(boxes)))
        
        for i in range(nb_box):
            x, y, w, h = boxes[i]
            label_idx = self.labels.index(labels[i]) if labels[i] in self.labels else 0

            center_y = ((y + 0.5 * h) / image_h) * self.grid_h
            center_x = ((x + 0.5 * w) / image_w) * self.grid_w
            
            w = (w / image_w) * self.grid_w  # unit: grid cell
            h = (h / image_h) * self.grid_h  # unit: grid cell
            
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            logging.debug("Boxes {} ({}) go to grid ({}, {})".format(i, boxes[i], grid_y, grid_x))
            
            if w > 0. and h > 0. and grid_x < self.grid_w and grid_y < self.grid_h:
                box = np.array([center_x, center_y, w, h])
                yolo_box = np.array([center_x, center_y, w, h, 1.])
                
                true_boxes[0, 0, 0, i % self.max_box_per_image, :] = box
                
                # find the anchor that best predicts this box
                
                box_wh = np.repeat(np.array([[w, h]]), self.nb_box, axis = 0)
                
                intersect = np.minimum(box_wh, self.np_anchors)
                intersect = intersect[:,0] * intersect[:,1]
                union = (self.np_anchors[:,0] * self.np_anchors[:,1]) +  (box_wh[:,0] * box_wh[:,1]) - intersect
                
                iou = intersect / union
                
                best_anchor = np.argmax(iou)
                
                logging.debug("Normalized box {} with label {} to anchor idx {} with score {}".format(box, label_idx, best_anchor, iou[best_anchor]))
                
                if iou[best_anchor] > 0.:
                    output[grid_y, grid_x, best_anchor, :5] = yolo_box
                    output[grid_y, grid_x, best_anchor, 5 + label_idx] = 1
        
        return output, true_boxes
    
    def get_output(self, infos):
        output, true_boxes = tf.py_function(
            self.get_output_fn, 
            [infos['box'], infos['label'], infos['nb_box'], infos['height'], infos['width']],
            Tout = [tf.float32, tf.float32]
        )

        output.set_shape([self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class])
        true_boxes.set_shape([1, 1, 1, self.max_box_per_image, 4])
        
        return output, true_boxes
    
    @timer(name = 'output decoding')
    def decode_output(self, output, obj_threshold = None, nms_threshold = None, ** kwargs):
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        
        grid_h, grid_w, nb_box = output.shape[:3]
        nb_class = output.shape[3] - 5

        # decode the output by the network
        conf    = tf.sigmoid(output[..., 4])
        classes = tf.nn.softmax(output[..., 5:], axis = -1)
        
        scores  = tf.expand_dims(conf, axis = -1) * classes
        scores  = scores * tf.cast(scores > obj_threshold, scores.dtype)
        
        class_scores = tf.reduce_sum(scores, axis = -1).numpy()
        
        conf    = conf.numpy()
        output  = output.numpy()
        classes = classes.numpy()

        boxes = []
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    if class_scores[row, col, b] > 0.:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = output[row, col, b, :4]

                        x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                        w = self.anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = self.anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height

                        box = BoundingBox(
                            x1 = max(0., x - w / 2), y1 = max(0., y - h / 2), 
                            x2 = min(1., x + w / 2), y2 = min(1., y + h / 2),
                            conf = conf[row, col, b], classes = classes[row, col, b]
                        )
                        if box.area > 0: boxes.append(box)

        # suppress non-maximal boxes
        for c in range(nb_class):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

            for i, index_i in enumerate(sorted_indices):
                if boxes[index_i].classes[c] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    iou = bbox_iou(boxes[index_i], boxes[index_j])
                    if iou >= nms_threshold:
                        boxes[index_j].classes[c] = 0

        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.score > 0]
        return boxes
    
    def encode_data(self, data):
        image   = self.get_input(data)
        outputs = self.get_output(data)
        
        return image, outputs
    
    def augment_data(self, image, output):
        image = self.augment_image(image)

        return image, output
    
    def preprocess_data(self, image, output):
        image = self.preprocess_image(image)

        return image, output
    
    @timer(name = 'drawing')
    def draw_prediction(self, image, boxes, labels = None, as_mask = False, ** kwargs):
        if len(boxes) == 0: return image
        
        if as_mask:
            return mask_boxes(image, boxes, ** kwargs)
        
        kwargs.setdefault('color', BASE_COLORS)
        kwargs.setdefault('use_label', True)
        kwargs.setdefault('labels', labels if labels is not None else self.labels)

        return draw_boxes(image, boxes, ** kwargs)
             
    @timer(name = 'saving')
    def save_prediction(self,
                        image,
                        boxes,
                        labels      = None,
                        detected    = None,
                        dezoom_factor   = 1.,
                        
                        img_dir     = None,
                        boxes_dir   = None,
                        detected_dir    = None,
                        
                        filename        = 'image_{}.jpg',
                        detected_name   = '{}_detected.jpg',
                        box_name        = '{}_box_{}.jpg',
                        
                        save_detected   = True,
                        save_boxes      = False,
                        extract_boxes   = False,
                        
                        ** kwargs
                       ):
        """
            Save a prediction with many possible options
            
            Arguments :
                - image     : the original image (raw or filename)
                - boxes     : detected boxes
                - labels    : custom labels to use
                - detected  : detected image (image with drawed boxes)
                - dezoom_factor     : dezoom factor for boxes normalization
                
                - img_dir   : where to save original / detected images
                - detected_dir  : where to save detected image (default to 'img_dir')
                - boxes_dir : where to save boxes images
                
                - filename      : original image filename format
                - detected_name : detected image filename format
                - box_name      : box name format
                
                - save_image    : whether to save the original image (if `filename` does not exist)
                - save_detected : whether to save detected image or not
                - save_boxes    : whether to save boxes' information or not
                - extract_boxes : whether to save boxes' images or not
            Return :
                - infos_pred, infos_boxes   : dict, dict
                    - infos_pred  : dict {filename : infos} where `infos` is a dict containing general information about the original image such as {filename, width, height, detected, boxes}
                    
                    - infos_boxes   : {box_filename : box_infos} where `box_infos` is a dict containing information on the extracted box such as {original, box, label, width, height} and `box_filename` is the filename for this specific box
        """
        if not (save_detected or save_boxes or extract_boxes): return {}, {}
        
        # Define default variables / directories
        if labels is None:          labels  = self.labels
        if img_dir is None:         img_dir = self.pred_img_dir
        if detected_dir is None:    detected_dir = img_dir
        if boxes_dir is None:       boxes_dir = self.pred_boxes_dir
        
        if isinstance(image, str): filename = image
        
        # Normalize boxes
        image_h, image_w = get_image_size(image)
        normalized_boxes = [get_box_pos(
            box, image_h = image_h, image_w = image_w, labels = labels,
            with_label = True, dezoom_factor = dezoom_factor,
            normalize_mode = NORMALIZE_WH
        )[:5] for box in boxes]
        
        # Output information
        infos_pred, infos_boxes = {}, {}
        # Save frame (if filename does not exists) if required to have the original image
        if save_boxes:
            if not os.path.exists(filename):
                files = os.listdir(img_dir)
                img_num = max(
                    len([f for f in files if '_detected' in f]),
                    len([f for f in files if '_detected' not in f])
                )

                filename    = os.path.join(img_dir, filename.format(img_num))
                save_image(filename = filename, image = image)
            
            infos_pred.setdefault(filename, {})
            infos_pred[filename].update({
                'filename' : filename, 'height' : image_h, 'width' : image_w,
                'nb_box' : len(normalized_boxes), 'box' : normalized_boxes
            })
        # Save detected image
        if save_detected:
            # Draw boxes (if required to save detected)
            if detected is None:
                detected = self.draw_prediction(image, boxes, labels)
            # It means the frame hasnot been saved (not save_boxes)
            if filename not in infos_pred:
                files = os.listdir(img_dir)
                img_num = max(
                    len([f for f in files if '_detected' in f]),
                    len([f for f in files if '_detected' not in f])
                )

                filename    = os.path.join(img_dir, filename.format(img_num))
            # Format detected_name with the `filename` basename
            detected_name = os.path.join(
                detected_dir, detected_name.format(os.path.basename(filename)[:-4])
            )
            save_image(filename = detected_name, image = detected)
            # If save_boxes, add infos to `infos_pred`
            if isinstance(filename, str) and filename in infos_pred:
                infos_pred[filename]['detected'] = detected_name
        
        # Extract boxes images and save them
        if extract_boxes:
            image = load_image(image)
            
            if os.path.exists(filename):
                basename = os.path.basename(filename)[:-4]
            else:
                no_img_num = len(set([
                    f.split('_')[2] for f in os.listdir(boxes_dir) 
                    if f.startswith('no_img_')
                ]))
                basename = 'no_img_{}'.format(no_img_num)
            
            for i, (x, y, w, h, label) in enumerate(normalized_boxes):
                box_img = image[y : y + h, x : x + w]

                box_filename = os.path.join(
                    boxes_dir, box_name.format(basename, i)
                )
                save_image(filename = box_filename, image = box_img)
                
                infos_boxes[box_filename] = {
                    'original'  : filename if os.path.exists(filename) else None,
                    'box'       : [x, y, w, h],
                    'label'     : label,
                    'height'    : h,
                    'width'     : w
                }
        
        return infos_pred, infos_boxes
    
    def get_pipeline(self,
                     classifier   = None,

                     save         = False,
                     save_empty   = False,
                     save_boxes   = False,
                     save_detected    = False,

                     directory    = None,
                     map_directory    = None,
                     default_filename_format    = 'image_{}.jpg',

                     batch_size   = 1,
                     ** kwargs
                    ):
        """
            Get the inference pipeline with drawing and saving as a list of dict (valid `utils.thread_utils.Pipeline` argument)
            
            Pipeline :
            name        input                       output
            -----------|---------------------------|------------------------------------
            _detect     frame                       (frame, output)
            _decode     (frame, output)             (frame, boxes)
            _draw       (frame, boxes)              (frame, boxes, detected)
                _save   (frame, boxes, detected)    (image_infos, extracted_boxes_infos)
            
            Arguments :
                - classifier    : classifier to re-label boxes (not fully supported yet)
                
                - save          : whether to save frames with boxes' mapping
                - save_empty    : whether to save information about frames with 0 box
                - save_boxes    : whether to save each detected box as an image
                - save_detected : whether to save frames with drawn boxes
                
                - overwrite : whether to erase existing images / detections / boxes
                - directory : the main directory in which to create subdirs to save images and boxes
                - map_directory : where to save mapping files
                - default_filename_format   : default filename for raw images (if saved)
                
                - batch_size    : the number of images to perform detection in parallel
                - kwargs    : given to `self.decode_output`, `self.save_prediction` etc.
            
            The final directory structure is (with all saving options to `True`) :
                directory/
                    images/
                        {default_filename_format}.jpg
                        ...
                    detected/
                        {default_filename_format}_detected.jpg
                        ...
                    boxes/
                        {default_filename_format}_box_0.jpg
                        ...
                    map.json            // map frames to boxes' information
                    map_boxes.json      // map boxes' image to its information
        """
        @timer
        def _detect(frame):
            """ Takes a frame and returns its model's output """
            outputs = self.detect(self.get_input(frame))
            
            if isinstance(frame, list):
                return [(f, out) for f, out in zip(frame, outputs)]
            return (frame, outputs[0])
        
        @timer
        def _decode_with_update(f_out):
            """ Decode a model's output and returns corresponding boxes """
            frame, output = f_out
            boxes = self.decode_output(output, ** kwargs)
            
            if classifier is not None:
                update_box_labels(frame, boxes, classifier = classifier, ** kwargs)
            
            return (frame, boxes)
        
        def _save(f_draw):
            """ Save the output (if required) """
            frame, boxes, detected  = f_draw
            if len(boxes) == 0 and not save_empty: return {}, {}
            
            infos_i, infos_boxes_i = self.save_prediction(
                image   = frame,
                boxes   = boxes,
                detected    = detected,
                
                filename    = default_filename_format,
                img_dir     = img_dir,
                boxes_dir   = boxes_dir,
                detected_dir    = detected_dir,
                
                map_file    = map_file,
                map_box_file    = map_box_file,
                
                save_boxes  = save,
                save_detected   = save_detected,
                extract_boxes   = save_boxes,
                ** kwargs
            )
            infos_pred.update(infos_i)
            infos_boxes.update(infos_boxes_i)
            
            return infos_i, infos_boxes_i
        
        @timer
        def _init_dir(subdir, map_filename, should_save):
            if not should_save: return None, None, {}

            dir_name    = os.path.join(directory, subdir)
            map_file    = os.path.join(map_directory, map_filename) if map_filename else None
            
            os.makedirs(dir_name, exist_ok = True)
            infos   = load_json(map_file, default = {}) if map_file else {}
            
            return map_file, dir_name, infos
        
        if directory is None:   directory = self.pred_dir
        if map_directory is None:   map_directory = directory
        
        map_file,     img_dir,   infos_pred     = _init_dir('images',   'map.json', save)
        _,            detected_dir, _           = _init_dir('detected', None, save_detected)
        map_box_file, boxes_dir, infos_boxes    = _init_dir('boxes',    'map_boxes.json', save_boxes)
        
        infos_pred  = ThreadedDict(** infos_pred)
        infos_boxes = ThreadedDict(** infos_boxes)
        
        return [
            {
                'consumer'  : _detect,
                'batch_size'    : batch_size,
                'max_workers'   : min(1, kwargs.get('max_workers', 0)),
                'name'      : 'detection'
            },
            {
                'consumer'  : _decode_with_update,
                'name'      : 'decoding'
            },
            {
                'consumer'  : lambda f_box: (
                    (f_box[0], f_box[1], self.draw_prediction(f_box[0], f_box[1], ** kwargs))
                ),
                'name'      : 'drawing',
                'consumers' : {
                    'consumer'  : _save,
                    'max_workers'   : min(kwargs.get('max_workers', 0), 0),
                    'name'      : 'saving',
                    'stop_listeners'    : [
                        lambda: dump_json(map_file, infos_pred, indent = 4) if save else None,
                        lambda: dump_json(map_box_file, infos_boxes, indent = 4) if save_boxes else None
                    ]
                }
            }
        ], infos_pred, infos_boxes

    @timer
    def stream(self, batch_size = 8, stream_name = 'stream_{}', ** kwargs):
        """
            Performs streaming either on camera (default) or on filename (by specifying the `cam_id` kwarg
        )"""
        if stream_name:
            directory = kwargs.get('directory', self.stream_dir)
            kwargs.update({
                'directory' : os.path.join(directory, stream_name.format(
                    len(sub for sub in os.listdir(directory) if sub.startswith('stream_'))
                )),
                'map_directory' : directory
            })
        else:
            kwargs.setdefault('directory', self.stream_dir)
        
        kwargs['transform_fn'] = self.get_pipeline(batch_size = batch_size, ** kwargs)[0]
        kwargs['transform_fn'].append({
            'consumer' : lambda f_draw: f_draw[-1], 'max_workers' : -1, 'name' : 'identity'
        })
        
        return stream_camera(** kwargs)
    
    @timer
    def predict(self, images, batch_size = 16, show = True, save = True, overwrite = False,
                verbose = 1, tqdm = lambda x: x, plot_kwargs = {}, ** kwargs):
        """
            Perform prediction on `images` and compute some time statistics
            
            Arguments :
                - images : (list of) images (either path / raw image) to detect objects on
                
                - show  : number of images to show (-1 or True to show all images)
                - save  : whether to save result or not
                - overwrite     : whether to overwrite (or not) already predicted image
                
                - tqdm      : progress bar (put to False if show > 0 or verbose)
                - verbose   : verbosity level (0, 1, 2 or 3)
                    - 0 : silent
                    - 1 : plot detected image
                    - 2 : plot individual boxes
                    - 3 : print boxes informations
                
                - plot_kwargs       : kwargs for `plot()` calls
                - kwargs            : kwargs for `get_pipeline()` call
            Return :
                - list of tuple [(path, infos), ...]
                    - path  : original image (path / raw image*)
                    - infos : general information on the prediction
                        {
                            detected    : detected image (if not save) / path (if save)
                            width / height  : dimension of original image
                            box     : normalized (to w/h) boxes
                        }
                
                * if `save` and it is a raw image, it will save it so if `save` is True, all `path` will be str
            
            Note that even if `save == False`, already predicted images are loaded to not re-predict them.
            i.e. put `save` to False will not overwrite.
        """
        @timer
        def _display(f_draw, img_num = 0):
            image, box, detected = f_draw
            image   = load_image(image)
            
            if verbose and img_num < show:
                # Show individual boxes
                if verbose > 1:
                    if verbose == 3:
                        logging.info("{} boxes found :\n{}".format(
                            len(box), '\n'.join(str(b) for b in box)
                        ))

                    show_boxes(image, box, labels = kwargs.get('labels', self.labels), ** plot_kwargs)
                # Show original image with drawed boxes
                plot(detected, title = '{} object(s) detected'.format(len(box)), ** plot_kwargs)
            return f_draw, (img_num + 1, )
        
        images = normalize_filename(images)
        if not isinstance(images, (list, tuple)): images = [images]
        if not save or show in (True, -1): show = len(images)
        if show > 0: tqdm = lambda x: x
        
        kwargs.setdefault('directory', self.pred_dir)
        
        # Get 1 time eachimage to not detect multiple times (as prediction is deterministic)
        requested_images = images
        images = list(set(images))
        
        pipeline, infos_pred, _ = self.get_pipeline(
            batch_size = batch_size, save = save, ** kwargs
        )
        
        videos = [img for img in images if isinstance(img, str) and img.endswith(_video_formats)]
        images = [
            img for img in images if not isinstance(img, str) or (
                img.endswith(_image_formats) and (overwrite or img not in infos_pred)
            )
        ]
        # Predict on videos (if any)
        if len(videos) > 0:
            videos_infos = self.predict_video(
                videos, batch_size = batch_size, overwrite = overwrite,
                tqdm = tqdm, show = show, ** kwargs
            )
            # Update information on video prediction
            for video, infos in videos_infos:
                infos_pred[video] = infos
        
        if len(images) > 0:
            prod    = Producer(images, run_main_thread = kwargs.get('max_workers', 0) < 0)
            pipe    = prod.add_consumer(pipeline, ** kwargs)

            if verbose:
                pipe.add_consumer(_display, stateful = True, start = True, link_stop = True)

            prod.start()
            prod.join(recursive = True)
        
        return [
            (img, infos_pred.get(img, {})) for img in requested_images if isinstance(img, str)
        ]

    def evaluate(self, 
                 generator, 
                 iou_threshold=0.3,
                 score_threshold=0.3,
                 max_detections=100,
                 save_path=None):
        """ Evaluate a given dataset using a given model.
        code originally from https://github.com/fizyr/keras-retinanet

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            model           : The model to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
        # Returns
            A dict mapping class names to mAP scores.
        """    
        return -1.
        # gather all detections and annotations
        generator.batch_size = 1
        all_detections     = [[None for i in range(len(self.labels))] for j in range(len(generator))]
        all_annotations    = [[None for i in range(len(self.labels))] for j in range(len(generator))]

        for i in range(len(generator)):
            inputs, _ = generator.__getitem__(i)
            raw_image, _ = inputs
            raw_height, raw_width, raw_channels = raw_image.shape

            # make the boxes and the labels
            pred_boxes  = self._predict(inputs, get_boxes=True)
            
            score = np.array([box.get_score() for box in pred_boxes])
            pred_labels = np.array([box.get_label() for box in pred_boxes])        
            
            if len(pred_boxes) > 0:
                pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height, box.xmax*raw_width, box.ymax*raw_height, box.score] for box in pred_boxes])
            else:
                pred_boxes = np.array([[]])  
            
            # sort the boxes and the labels according to scores
            score_sort = np.argsort(-score)
            pred_labels = pred_labels[score_sort]
            pred_boxes  = pred_boxes[score_sort]
            
            # copy detections to all_detections
            for label in range(len(self.labels)):
                all_detections[i][label] = pred_boxes[pred_labels == label, :]
                
            annotations = generator.load_annotation(i)
            
            # copy detections to all_annotations
            for label in range(generator.num_classes()):
                all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                
        # compute mAP by comparing all detections and all annotations
        average_precisions = {}
        
        for label in range(generator.num_classes()):
            false_positives = np.zeros((0,))
            true_positives  = np.zeros((0,))
            scores          = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections           = all_detections[i][label]
                annotations          = all_annotations[i][label]
                num_annotations     += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)
                        continue

                    overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap         = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives  = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives  = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0
                continue

            # sort by score
            indices         = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives  = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives  = np.cumsum(true_positives)

            # compute recall and precision
            recall    = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision  = compute_ap(recall, precision)  
            average_precisions[label] = average_precision

        return average_precisions    
    
    @timer
    def predict_video(self,
                      videos,
                      
                      save_frames   = False,
                      save_boxes    = False,
                      save_video    = True,
                      directory     = None,
                      overwrite     = False,
                
                      tqdm    = lambda x: x,
                
                      ** kwargs
                     ):
        """
            Perform prediction on `videos` (with `self.stream` method)
            
            Arguments :
                - videos    : (list of) videos' filename
                
                - save_frames   : whether to save individual frames (with boxes' infos) or not
                - save_boxes    : whether to save boxes as individual images
                - save_detected : whether to save frames with drawn detection
                - save_video    : whether to save detection as a video
                - directory     : where to save result
                - overwrite     : whether to overwrite (or not) already predicted image
                
                - tqdm          : progress bar
                
                - kwargs        : kwargs for `self.stream` call (such as `batch_size`, etc.)
            Return :
                - list of tuple [(path, infos), ...]
                    - path  : original video path
                    - infos : general information on the prediction
                        {
                            detected    : detected video (if `save_video` else None)
                            width / height / fps / nb_frames : video's information
                            frames  : filename for the frames' information mapping
                        }
        """
        kwargs.setdefault('show', False)
        kwargs.setdefault('max_time', -1)
        
        videos = normalize_filename(videos)
        if not isinstance(videos, (list, tuple)): videos = [videos]
        
        # get saving directory
        if directory is None: directory = self.pred_dir
        
        map_frames_file = os.path.join(directory, 'map.json') if save_frames or save_boxes else None
        
        map_file    = os.path.join(directory, 'map_videos.json')
        infos_videos    = load_json(map_file, default = {})
        
        video_dir = None
        if save_video:
            video_dir   = os.path.join(directory, 'videos')
            os.makedirs(video_dir, exist_ok = True)
        
        # Get 1 time eachimage to not detect multiple times (as prediction is deterministic)
        requested_videos = [video for video in videos if video.endswith(_video_formats)]
        videos = list(set(requested_videos))
        
        # for each batch
        for path in tqdm(videos):
            video_name, ext  = os.path.splitext(os.path.basename(path))
            # Maybe skip because already predicted
            if not overwrite and path in infos_videos:
                if not save_frames or (save_frames and infos_videos[path]['frames'] is not None):
                    if not save_video or (save_video and infos_videos[path]['detected'] is not None):
                        continue
            
            out_file = None if not save_video else os.path.join(
                video_dir, '{}_detected{}'.format(video_name, ext)
            )
            self.stream(
                cam_id  = path,
                save    = save_frames,
                save_boxes  = save_boxes,
                overwrite   = overwrite,
                stream_name = None,
                directory   = os.path.join(directory, 'images', video_name),
                map_directory   = directory,
                output_file = out_file,
                ** kwargs
            )
            
            infos_videos[path] = {
                'detected'  : out_file,
                'frames'    : map_frames_file,
                ** get_video_infos(path).__dict__
            }
        
            dump_json(map_file, infos_videos, indent = 4)
        
        return [(video, infos_videos[video]) for video in requested_videos]
                                
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update({
            ** self.get_config_image(),
            'anchors'   : self.anchors,
            'backend'   : self.backend,
            
            'labels'    : self.labels,
            'nb_class'  : self.nb_class,
            'max_box_per_image' : self.max_box_per_image
        })
        
        return config

    @classmethod
    def from_darknet_pretrained(cls,
                                weight_path = 'yolov2.weights',
                                nom     = 'coco_pretrained',
                                labels  = COCO_CONFIG['labels'],
                                ** kwargs
                               ):
        instance = cls(
            nom = nom, labels = labels, max_to_keep = 1, pretrained_name = weight_path, ** kwargs
        )
        
        decode_darknet_weights(instance.get_model(), weight_path)
        
        instance.save()
        
        return instance

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_darknet_weights(model, wt_path):
    #Chargement des poids
    weight_reader = WeightReader(wt_path)
    weight_reader.reset()
    nb_conv = 23
    for i in range(1, nb_conv+1):
        name = 'conv_{}'.format(i) if i < 23 else 'detection_layer'
        if i < nb_conv and len(model.layers) < 10:
            conv_layer = model.layers[1].get_layer(name = name)
            norm_layer = model.layers[1].get_layer(
                'batch_normalization_{}'.format(i-1) if i > 1 else 'batch_normalization'
            )
        else:
            conv_layer = model.get_layer(name = name)
            norm_layer = model.get_layer(
                'batch_normalization_{}'.format(i-1) if i > 1 else 'batch_normalization'
            ) if i < nb_conv else None
        
        if (i < nb_conv):
            size = np.prod(norm_layer.get_weights()[0].shape)
            
            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])
        
        if (len(conv_layer.get_weights()) > 1):
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2,3,1,0])
            conv_layer.set_weights([kernel])

class WeightReader:
    def __init__(self, path):
        self.offset = 4
        self.all_weights = np.fromfile(path, dtype = 'float32')
    
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size : self.offset]
    
    def reset(self):
        self.offset = 4
