
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
import glob
import time
import shutil
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from loggers import timer
from utils.thread_utils import Pipeline
from custom_architectures import get_architecture
from models.interfaces.base_image_model import BaseImageModel
from utils import load_json, dump_json, normalize_filename, download_file, plot
from utils.image import _video_formats, _image_formats, load_image, save_image, stream_camera, BASE_COLORS, get_video_infos
from utils.image.box_utils import *

logger      = logging.getLogger(__name__)

PRETRAINED_COCO_URL = 'https://pjreddie.com/media/files/yolov2.weights'

DEFAULT_ANCHORS = [
    0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
]

COCO_CONFIG = {
    'labels' : ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    'anchors' : DEFAULT_ANCHORS
}
VOC_CONFIG  = {
    'labels' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'anchors'   : DEFAULT_ANCHORS #[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
}

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
        
        self.obj_threshold  = obj_threshold
        self.nms_threshold  = nms_threshold
        
        self.np_anchors = np.reshape(np.array(anchors), [self.nb_box, 2])

        super().__init__(** kwargs)

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
    def output_signature(self):
        return (
            tf.TensorSpec(
                shape =(None, self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class),
                dtype = tf.float32
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
    
    @timer(name = 'inference', log_if_root = False)
    def detect(self, image, get_boxes = False, training = False, ** kwargs):
        """
            Performs prediction on `image` and returns either the model's output either the boxes (if `get_boxes = True`)
            
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
    
    def compile(self, loss = 'YoloLoss', loss_config = {}, ** kwargs):
        loss_config.update({'anchors' : self.anchors})
        
        super().compile(loss = 'YoloLoss', loss_config = loss_config, ** kwargs)
    
    @timer(name = 'output decoding')
    def decode_output(self, output, obj_threshold = None, nms_threshold = None, ** kwargs):
        if obj_threshold is None: obj_threshold = self.obj_threshold
        if nms_threshold is None: nms_threshold = self.nms_threshold
        
        grid_h, grid_w, nb_box = output.shape[:3]
        nb_class = output.shape[3] - 5

        # decode the output by the network
        conf    = tf.sigmoid(output[..., 4:5])
        classes = tf.nn.softmax(output[..., 5:], axis = -1)
        
        scores  = conf * classes
        scores  = tf.where(scores > obj_threshold, scores, 0.)
        
        class_scores = tf.reduce_sum(scores, axis = -1)

        pos     = output[..., :4].numpy()
        conf    = conf.numpy()[..., 0]
        classes = classes.numpy()

        candidates  = tf.where(class_scores > 0.).numpy()
        
        boxes = []
        for row, col, b in candidates:
            # first 4 elements are x, y, w, and h
            x, y, w, h = pos[row, col, b]
            
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

    def get_input(self, filename, ** kwargs):
        return self.get_image(filename)
    
    def augment_input(self, image, ** kwargs):
        return self.augment_image(image)
    
    def preprocess_input(self, image, ** kwargs):
        return self.preprocess_image(image)
    
    def get_output_fn(self, boxes, labels, nb_box, image_h, image_w, ** kwargs):
        if hasattr(boxes, 'numpy'): boxes = boxes.numpy()
        if hasattr(image_h, 'numpy'): image_h, image_w = image_h.numpy(), image_w.numpy()
        
        output      = np.zeros((self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class))
        true_boxes  = np.zeros((1, 1, 1, self.max_box_per_image, 4))
        
        logger.debug("Image with shape ({}, {}) and {} boxes :".format(image_h, image_w, len(boxes)))
        
        for i in range(nb_box):
            x, y, w, h = boxes[i]
            label_idx = self.labels.index(labels[i]) if labels[i] in self.labels else 0

            center_y = ((y + 0.5 * h) / image_h) * self.grid_h
            center_x = ((x + 0.5 * w) / image_w) * self.grid_w
            
            w = (w / image_w) * self.grid_w  # unit: grid cell
            h = (h / image_h) * self.grid_h  # unit: grid cell
            
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            logger.debug("Boxes {} ({}) go to grid ({}, {})".format(i, boxes[i], grid_y, grid_x))
            
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
                
                logger.debug("Normalized box {} with label {} to anchor idx {} with score {}".format(box, label_idx, best_anchor, iou[best_anchor]))
                
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
    
    def encode_data(self, data):
        return self.get_input(data), self.get_output(data)
    
    def augment_data(self, image, output):
        return self.augment_input(image), output
    
    def preprocess_data(self, image, output):
        return self.preprocess_input(image), output
    
    @timer(name = 'drawing')
    def draw_prediction(self, image, boxes, labels = None, as_mask = False, ** kwargs):
        """ Calls `draw_boxes` or `mask_boxes` depending on `as_mask` and returns the result """
        if len(boxes) == 0: return image
        
        if as_mask:
            return mask_boxes(image, boxes, ** kwargs)
        
        kwargs.setdefault('color', BASE_COLORS)
        kwargs.setdefault('use_label', True)
        kwargs.setdefault('labels', labels if labels is not None else self.labels)

        return draw_boxes(image, boxes, ** kwargs)
    
    def save_image(self, image, directory = None, filename = 'image_{}.jpg', ** kwargs):
        """ Saves `image` to `directory` (default {self.pred_dir}/images/) """
        if isinstance(image, str): return image
        if directory is None: directory = os.path.join(self.pred_dir, 'images')
        
        if not filename.startswith(directory): filename = os.path.join(directory, filename)
        if '{}' in filename:
            filename = filename.format(len(glob.glob(filename.replace('{}', '*'))))
        
        save_image(filename = filename, image = image)
        
        return filename
    
    def save_detected(self,
                      detected  = None,
                      image = None,
                      boxes = None,
                      dezoom_factor = 1.,
                      
                      directory = None,
                      filename  = '{}_detected.jpg',
                      original_filename = None,
                      
                      ** kwargs
                     ):
        if detected is None:
            assert boxes is not None and image is not None
            
            image_h, image_w = get_image_size(image)
            normalized_boxes = [get_box_pos(
                box, image_h = image_h, image_w = image_w, labels = labels,
                with_label = True, dezoom_factor = dezoom_factor,
                normalize_mode = NORMALIZE_WH
            )[:5] for box in boxes]
            
            kwargs.setdefault('labels', self.labels)
            detected = self.draw_prediction(image, boxes, ** kwargs)

        if directory is None: directory = os.path.join(directory, 'detected')
        if not filename.startswith(directory): filename = os.path.join(directory, filename)
        
        if '{}' in filename:
            if isinstance(original_filename, str):
                img_name    = os.path.splitext(os.path.basename(original_filename))[0]
            else:
                img_name    = 'image_{}'.format(len(glob.glob(filename.replace('{}', '*'))))
            
            filename = filename.format(img_name)
        
        save_image(filename = filename, image = detected)
        
        return filename

    def get_pipeline(self,
                     batch_size = 8,
                     
                     post_processing    = None,
                     drawing_consumers  = None,

                     save   = False,
                     save_empty = False,
                     save_boxes = False,
                     save_detected  = False,
                     track_items    = None,

                     directory    = None,
                     map_directory    = None,
                     
                     image_format   = 'image_{}.jpg',
                     detected_format    = '{}_detected.jpg',
                     box_format     = '{}_box_{}.jpg',
                     ** kwargs
                    ):
        """
            Get the inference pipeline with drawing and saving
            
            Arguments :
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
        def preprocess(filename, ** kwargs):
            """ Loads and process the image """
            file_id = None
            if isinstance(filename, dict):
                file_id = filename.get('filename', filename.get('id', filename.get('image', None)))
            
            image   = load_image(filename)
            processed   = self.preprocess_input(self.get_input(image))
            
            image_h, image_w = image.shape[:2]
            return {
                'filename'  : filename if file_id is None else file_id,
                'image'     : image.numpy(),
                'processed' : processed,
                'height'    : image_h,
                'width'     : image_w
            }
        
        def detect(infos, ** kwargs):
            """ Takes a frame and returns its model's output """
            if not isinstance(infos, list):
                infos['output'] = self.detect(infos.pop('processed'))[0]
            else:
                outputs = self.detect(tf.stack([info.pop('processed') for info in infos]))
                for info, out in zip(infos, outputs): info['output'] = out
            return infos
        
        def decode(infos, ** kwargs):
            """ Decode a model's output and returns corresponding boxes """
            kwargs.setdefault('labels', self.labels)
            boxes = self.decode_output(infos.pop('output'), ** kwargs)
            infos.update({'nb_box' : len(boxes), 'box' : boxes})
            return infos
        
        def draw(infos, ** kw):
            """ Draw prediction """
            infos['detected_image']   = self.draw_prediction(
                infos['image'].copy(), infos['box'], ** {** kwargs, ** kw}
            ) if infos['nb_box'] > 0 else infos['image']
            return infos
        
        @timer
        def save_image(infos, ** kw):
            if not isinstance(infos['filename'], str):
                infos['filename'] = self.save_image(
                    image       = infos['filename'],
                    directory   = img_dir,
                    filename    = image_format
                ) if infos['nb_box'] > 0 or save_empty else None
            elif not os.path.exists(infos['filename']):
                infos['filename'] = self.save_image(
                    image       = infos['image'],
                    directory   = img_dir,
                    filename    = infos['filename']
                ) if infos['nb_box'] > 0 or save_empty else None
            
            return infos
        
        @timer
        def save_detected_fn(infos, ** kw):
            infos['detected'] = self.save_detected(
                detected    = infos['detected_image'],
                image   = infos['image'],
                boxes   = infos['box'],
                directory   = detected_dir,
                filename    = detected_format,
                original_filename   = infos['filename'],
                ** {** kwargs, ** kw}
            ) if save_empty or infos['nb_box'] > 0 else None
            return infos
        
        @timer
        def save_boxes_fn(infos, ** kw):
            if not isinstance(infos['filename'], str) and infos['nb_box'] > 0:
                raise RuntimeError('`filename` must be a string when `save_boxes = True` otherwise the result will not be saved. It can occur when you pass a raw image in the pipeline and `save = False`. In this case, pass a dict {id:, image:} as input (the image will not be saved and `id` will be used to save boxes\' information in `map.json`).')
            
            infos['extracted'] = extract_boxes(
                filename    = infos['filename'],
                image   = infos['image'],
                boxes   = infos['box'],
                directory   = boxes_dir,
                file_format = box_format,
                ** {** kwargs, ** kw}
            ) if infos['nb_box'] > 0 else {}
            
            return infos
        
        if directory is None:   directory = self.pred_dir
        if map_directory is None:   map_directory = directory
        map_file = os.path.join(map_directory, 'map.json') if save or save_boxes else None

        expected_keys       = ['nb_box', 'box']
        do_not_save_keys    = ['image', 'processed', 'output', 'detected_image']
        if not save_detected: do_not_save_keys.append('detected')
        
        final_functions = []
        if post_processing is not None:
            final_functions.append(post_processing)
        
        final_functions.append({'consumer' : draw, 'consumers' : drawing_consumers})
        
        if save:
            img_dir     = os.path.join(directory, 'images')
            os.makedirs(img_dir, exist_ok = True)
            
            expected_keys.append('filename')
            final_functions.append({'consumer' : save_image, 'allow_multithread' : False})
        else:
            do_not_save_keys.append('filename')

        if save_detected:
            detected_dir    = os.path.join(directory, 'detected')
            os.makedirs(detected_dir, exist_ok = True)
            
            expected_keys.append('detected')
            final_functions.append({'consumer' : save_detected_fn, 'allow_multithread' : False})
        
        if save_boxes:
            boxes_dir   = os.path.join(directory, 'boxes')
            os.makedirs(boxes_dir, exist_ok = True)

            expected_keys.append('extracted')
            final_functions.append({'consumer' : save_boxes_fn, 'allow_multithread' : False})
        
        pipeline = Pipeline(** {
            ** kwargs,
            'name'  : 'detection_pipeline',
            'filename'  : map_file,
            'track_items'   : track_items,
            'expected_keys' : expected_keys,
            'do_not_save_keys'  : do_not_save_keys,
            
            'tasks' : [
                preprocess,
                {
                    'consumer'  : detect,
                    'batch_size'    : batch_size,
                    'allow_multithread' : False,
                    'name'      : 'detection'
                },
                {'consumer' : decode, 'name' : 'decoding'},
            ] + final_functions
        })
        pipeline.start()
        
        return pipeline

    @timer
    def stream(self, stream_name = 'stream_{}', save = False, save_boxes = False, ** kwargs):
        """
            Performs streaming either on camera (default) or on filename (by specifying the `cam_id` kwarg)
        """
        kwargs.update({'save' : save, 'save_boxes' : save_boxes, 'track_items' : False})
        if stream_name:
            directory = kwargs.get('directory', self.stream_dir)
            if '{}' in stream_name:
                stream_name = stream_name.format(len(
                    glob.glob(os.path.join(directory, stream_name.replace('{}', '*')))
                ))
            kwargs.update({
                'directory' : os.path.join(directory, stream_name),
                'map_directory' : directory
            })
        else:
            kwargs.setdefault('directory', self.stream_dir)
        
        pipeline    = self.get_pipeline(** kwargs)
        drawing_consumer    = pipeline.get_consumer('draw')
        detected_producer   = drawing_consumer.add_consumer(
            lambda infos, ** kwargs: infos['detected_image'], max_workers = -2, name = 'identity'
        )
        kwargs.update({'transform_fn' : pipeline, 'transformer_prod' : detected_producer})
        
        if save or save_boxes:
            kwargs['id_format'] = os.path.join(
                kwargs['directory'], 'images', kwargs.get('image_format', 'frame_{}.jpg')
            )

        return stream_camera(** kwargs)
    
    @timer
    def predict(self, images, display = True, save = True, verbose = 1,
                plot_kwargs = {}, ** kwargs):
        """
            Perform prediction on `images` and compute some time statistics
            
            Arguments :
                - images : (list of) images (either path / raw image) to detect objects on
                
                - display   : number of images to display (-1 or True to display all images)
                - save  : whether to save result or not
                - overwrite     : whether to overwrite (or not) already predicted image
                
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
        def show_result(infos, img_num = 0, ** kwargs):
            image, box, detected = infos['image'], infos['box'], infos['detected_image']
            if verbose and img_num < display:
                # Show individual boxes
                if verbose > 1:
                    if verbose == 3:
                        logger.info("{} boxes found :\n{}".format(
                            len(box), '\n'.join(str(b) for b in box)
                        ))

                    show_boxes(image, box, labels = kwargs.get('labels', self.labels), ** plot_kwargs)
                # Show original image with drawed boxes
                plot(detected, title = '{} object(s) detected'.format(len(box)), ** plot_kwargs)
            return infos, (img_num + 1, )
        
        images = normalize_filename(images)
        if not isinstance(images, (list, tuple)): images = [images]
        if not save or display in (True, -1): display = len(images)
        
        kwargs.setdefault('directory', self.pred_dir)
        
        videos = [img for img in images if isinstance(img, str) and img.endswith(_video_formats)]
        images = [
            img for img in images if not isinstance(img, str) or img.endswith(_image_formats)
        ]
        
        results, video_results = [], []
        # Predicts on videos (if any)
        if len(videos) > 0:
            video_results = self.predict_video(videos, ** kwargs)
        
        if len(images) > 0:
            show_cons = {
                'consumer' : show_result, 'max_workers' : -2, 'batch_size' : 1, 'stateful' : True
            }
            pipeline = self.get_pipeline(
                save = save, drawing_consumers = show_cons if verbose else None, ** kwargs
            )
            
            results = pipeline.extend_and_wait(images, stop = True, ** kwargs)

        return video_results + [(img, res) for img, res in zip(images, results)]

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
                      save_detected = False,
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
        
        map_file    = os.path.join(directory, 'map_videos.json')
        infos_videos    = load_json(map_file, default = {})
        
        video_dir = os.path.join(directory, 'videos')
        
        # Filters files that do not end with a valid video extension
        videos = [video for video in videos if video.endswith(_video_formats)]
        
        for path in tqdm(set(videos)):
            video_name, ext  = os.path.splitext(os.path.basename(path))
            # Maybe skip because already predicted
            if not overwrite and path in infos_videos:
                if not save_frames or (save_frames and infos_videos[path]['frames'] is not None):
                    if not save_video or (save_video and infos_videos[path]['detected'] is not None):
                        continue
            
            save_dir    = os.path.join(video_dir, video_name)
            out_file = None if not save_video else os.path.join(
                save_dir, '{}_detected{}'.format(video_name, ext)
            )
            map_frames = os.path.join(save_dir, 'map.json') if save_frames or save_boxes else None
            
            if os.path.exists(save_dir): shutil.rmtree(save_dir)
            if out_file and os.path.exists(out_file): os.remove(out_file)
            if out_file: os.makedirs(save_dir, exist_ok = True)
            
            self.stream(
                cam_id  = path,
                save    = save_frames,
                save_detected   = save_detected,
                save_boxes  = save_boxes,
                overwrite   = overwrite,
                stream_name = None,
                
                directory   = save_dir,
                output_file = out_file,
                
                ** kwargs
            )
            
            infos_videos[path] = {
                'detected'  : out_file,
                'frames'    : map_frames,
                ** get_video_infos(path).__dict__
            }
        
            dump_json(map_file, infos_videos, indent = 4)
        
        return [(video, infos_videos[video]) for video in videos]
                                
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
        if not os.path.exists(weight_path) and weight_path.endswith('yolov2.weights'):
            weight_path = download_file(PRETRAINED_COCO_URL, filename = weight_path)
            
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
