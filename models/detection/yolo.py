import os
import cv2
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from models.base_model import BaseModel
from custom_architectures import get_architecture
from utils import load_json, dump_json, time_to_string, normalize_filename, plot
from utils.image import load_image, save_image, stream_camera, BASE_COLORS, augment_image, copy_audio
from utils.image.box_utils import *

DEFAULT_ANCHORS = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

COCO_CONFIG = {
    'labels' : ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'brocoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],
    'anchors' : DEFAULT_ANCHORS
}
VOC_CONFIG  = {
    'labels' : ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    'anchors'   : DEFAULT_ANCHORS #[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
}



class YOLO(BaseModel):
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
        if isinstance(input_size, int): input_size = (input_size, input_size, 3)
        
        self.input_size = tuple(input_size)
        self.backend    = backend
        self.anchors    = anchors
        self.max_box_per_image = max_box_per_image

        self.labels   = list(labels) if not isinstance(labels, str) else [labels]
        self.nb_class = nb_class if nb_class is not None else len(self.labels)
        self.nb_class = max(2, self.nb_class)
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
    
    def init_train_config(self,
                          augment_methods = ['hue', 'brightness', 'saturation', 'contrast', 'noise'],
                          ** kwargs
                          ):
        self.augment_methods    = augment_methods

        super().init_train_config(** kwargs)
        
        if hasattr(self, 'model_loss'):
            self.model_loss.seen = self.current_epoch
            
    def _build_model(self, flatten = True, randomize = True, ** kwargs):
        feature_extractor = get_architecture(
            architecture_name = self.backend, input_image = self.input_size, ** kwargs
        )
        
        super()._build_model(model = {
            'architecture_name' : 'yolo',
            'feature_extractor' : feature_extractor,
            'input_size'        : self.input_size,
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
    def input_signature(self):
        return tf.TensorSpec(
            shape = (None,) + self.input_size, dtype = tf.float32
        )
    
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
    def training_hparams(self):
        return super().training_hparams(
            augment_methods = ['hue', 'brightness', 'saturation', 'contrast', 'noise']
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
        
    def __str__(self):
        des = super().__str__()
        des += "Labels (n = {}) : {}\n".format(len(self.labels), self.labels)
        des += "Feature extractor : {}\n".format(self.backend)
        return des
    
    def detect(self, image, get_boxes = False, ** kwargs):
        if tf.rank(image) == 3: image = tf.expand_dims(image, axis = 0)
        
        #start = time.time()
        outputs = self(image)
        #print("Inference time : {:.3f}".format(time.time() - start))
        
        if not get_boxes: return outputs
        
        #start = time.time()
        result = [self.decode_output(out, ** kwargs) for out in outputs]
        #print("Decoding time : {:.3f}".format(time.time() - start))
        return result
    
    def compile(self, loss_config = {}, **kwargs):
        kwargs['loss'] = 'YoloLoss'
        
        loss_config.update({'anchors' : self.anchors})
        
        super().compile(loss_config = loss_config, ** kwargs)
    
    def get_input(self, filename):
        if isinstance(filename, list):
            return tf.stack([self.get_input(f) for f in filename])
        elif isinstance(filename, pd.DataFrame):
            return tf.stack([self.get_input(row) for idx, row in filename.iterrows()])
        
        image = load_image(filename, target_shape = self.input_size, mode = 'rgb')

        return image
    
    def get_output_fn(self, boxes, labels, nb_box, image_h, image_w, debug = False, 
                      ** kwargs):
        if hasattr(boxes, 'numpy'): boxes = boxes.numpy()
        if hasattr(image_h, 'numpy'): image_h, image_w = image_h.numpy(), image_w.numpy()
        
        output      = np.zeros((self.grid_h, self.grid_w, self.nb_box, 5 + self.nb_class))
        true_boxes  = np.zeros((1, 1, 1, self.max_box_per_image, 4))
        
        if debug:
            print("Image with shape ({}, {}) and {} boxes :".format(image_h, image_w, len(boxes)))
        
        for i in range(nb_box):
            x, y, w, h = boxes[i]
            label_idx = self.labels.index(labels[i]) if labels[i] in self.labels else 0

            center_y = ((y + 0.5 * h) / image_h) * self.grid_h
            center_x = ((x + 0.5 * w) / image_w) * self.grid_w
            
            w = (w / image_w) * self.grid_w  # unit: grid cell
            h = (h / image_h) * self.grid_h  # unit: grid cell
            
            grid_x = int(np.floor(center_x))
            grid_y = int(np.floor(center_y))
            
            if debug:
                print("Boxes {} ({}) go to grid ({}, {})".format(i, boxes[i], grid_y, grid_x))
            
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
                
                if debug:
                    print("Normalized box {} with label {} to anchor idx {} with score {}".format(box, label_idx, best_anchor, iou[best_anchor]))
                
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
    
    def decode_output(self, output, obj_threshold = None, nms_threshold = None):
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
                            max(0., x - w / 2), max(0., y - h / 2), 
                            min(1., x + w / 2), min(1., y + h / 2),
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
        image   = load_image(data, target_shape = self.input_size[:2])
        outputs = self.get_output(data)
        
        return image, outputs
    
    def augment_data(self, image, output):
        image = augment_image(
            image, self.augment_methods, self.augment_prct / len(self.augment_methods)
        )
        return image, output
    
    def train_step(self, batch):
        inputs, target = batch
                
        loss_fn     = self.model_loss
        optimizer   = self.model_optimizer
        variables   = self.model.trainable_variables
        
        with tf.GradientTape() as tape:
            pred = self(inputs, training = True)
            losses = loss_fn(target, pred)
            loss = losses[0]
        
        gradients = tape.gradient(loss, variables)
                
        optimizer.apply_gradients(zip(gradients, variables))
        
        return self.update_metrics(target, pred)
    
    def draw_prediction(self, image, boxes, labels = None, as_mask = False,
                        ** kwargs):
        if as_mask:
            return mask_boxes(image, boxes, **kwargs)
        
        kwargs.setdefault('color', BASE_COLORS)
        kwargs.setdefault('use_label', True)
        kwargs.setdefault('labels', labels if labels is not None else self.labels)

        return draw_boxes(image, boxes, ** kwargs)
             
    def save_prediction(self,
                        image,
                        boxes,
                        labels      = None,
                        detected    = None,
                        dezoom_factor   = 1.,
                        
                        img_dir     = None,
                        detected_dir    = None,
                        boxes_dir   = None,
                        
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
                
                - save_detected : whether to save detected image or not
                - save_boxes    : whether to save boxes information or not
                - extract_boxes : whether to save boxes images or not
            Return :
                - infos_pred, infos_boxes   : dict, dict
                    - infos_pred  : dict {filename : infos} where `infos` is a dict containing general information about the original image such as {filename, width, height, detected, boxes}
                    
                    - infos_boxes   : {box_filename : box_infos} where `box_infos` is a dict containing information on the extracted box such as {original, box, label, width, height} and `box_filename` is the filename for this specific box
        """
        if not save_detected and not save_boxes and not extract_boxes: return {}, {}
        
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
    
    def process_frame(self, frame, map_file = None, map_box_file = None, box_kwargs = {},
                      obj_threshold = None, nms_threshold = None, ** kwargs):
        """
            Detect objects on a single frame and save it (if required)
            
            Arguments :
                - frame     : image to process
                - map_file  : where to save information on boxes
                - map_box_file  : where to save information about extracted boxes
                - kwargs    : kwargs passed to `self.draw_prediction` and `self.save_prediction`
            Return :
                - processed : the processed frame with drawed boxes
        """
        processed = self.get_input(frame)

        boxes = self.detect(
            processed, get_boxes = True,
            obj_threshold = obj_threshold, nms_threshold = nms_threshold
        )[0]
        
        detected = self.draw_prediction(frame, boxes, ** box_kwargs)
        
        if map_file is not None or map_box_file is not None:
            infos_pred, infos_boxes = self.save_prediction(
                frame, boxes, detected = detected, ** kwargs
            )
            if map_file is not None:
                infos = load_json(map_file)
                infos.update(infos_pred)
                dump_json(map_file, infos, indent = 4)
            
            if map_box_file is not None:
                infos = load_json(map_box_file)
                infos.update(infos_boxes)
                dump_json(map_box_file, infos, indent = 4)

        return detected
    
    def stream(self, save = False, save_boxes = False, overwrite = False,
               directory = None, ** kwargs):
        map_file, map_box_file, img_dir, boxes_dir = None, None, None, None
        
        if directory is None: directory = self.stream_dir
        
        if save:
            img_dir     = os.path.join(directory, 'images')
            map_file    = os.path.join(directory, 'map.json')
            
            if overwrite and os.path.exists(map_file):
                os.remove(map_file)
                shutil.rmtree(img_dir)

            os.makedirs(img_dir, exist_ok = True)
        
        if save_boxes:
            boxes_dir   = os.path.join(directory, 'boxes')
            map_box_file    = os.path.join(directory, 'map_boxes.json')
            
            if overwrite and os.path.exists(map_box_file):
                os.remove(map_box_file)
                shutil.rmtree(boxes_dir)

            os.makedirs(boxes_dir, exist_ok = True)
        
        kwargs.update({
            'map_file'  : map_file,
            'map_box_file'  : map_box_file,
            
            'filename'      : 'frame_{}.jpg',
            'img_dir'       : img_dir,
            'boxes_dir'     : boxes_dir,
            
            'save_boxes'    : save,
            'save_detected' : False,
            'extract_boxes' : save_boxes,
            
            'transform_fn'  : self.process_frame
        })
        
        stream_camera(** kwargs)
    
    def predict(self,
                images,
                
                labels  = None,
                batch_size  = 16,
                
                obj_threshold   = None,
                nms_threshold   = None,
                
                show    = 0,
                save    = True,
                save_empty  = False,
                save_boxes  = False,
                directory   = None,
                overwrite   = False,
                
                tqdm = lambda x: x,
                verbose = 1,
                debug   = False,
                
                plot_kwargs = {},
                box_plot_kwargs = {},
                videos_kwargs   = {},
                **kwargs
               ):
        """
            Perform prediction on `images` and compute some time statistics
            
            Arguments :
                - images : (list of) images (either path / raw image) to detect objects on
                
                - labels : custom labels to use
                - batch_size    : batch_size for prediction
                
                - show  : number of images to show (-1 or True to show all images)
                - save  : whether to save result or not
                - save_empty    : whether to save image where no objectdetected
                - save_boxes    : whether to save boxes' image or not
                - directory     : where to save result
                - overwrite     : whether to overwriteor not already predicted image
                
                - tqdm      : progress bar (put to False if show > 0 or verbose)
                - verbose   : verbosity level (0, 1, 2 or 3)
                    - 0 : silent
                    - 1 : plot detectedimage
                    - 2 : plot individual boxes
                    - 3 : print boxes informations
                - debug     : whether to show time information or not
                
                - plot_kwargs       : kwargs for `plot()` calls
                - box_plot_kwargs   : kwargs for `show_boxes()` call
                - videos_kwargs     : kwargs for video predictions (call to `self.predict_video`)
                - kwargs            : kwargs for `draw_prediction()` call
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
            
            Note that even if `save == False` we load already predicted images to not re-predict them.
            i.e. put `save` to False will not overwrite.
        """
        images = normalize_filename(images)
        if labels is None: labels = self.labels
        if not isinstance(images, (list, tuple)): images = [images]
        if not save or show in (True, -1): show = len(images)
        if show > 0: tqdm = lambda x: x
        
        t_process, t_infer, t_save, t_show = 0., 0., 0., 0.
        
        # get saving directory
        
        if directory is None: directory = self.pred_dir
        map_file    = os.path.join(directory, 'map.json')
        map_box_file    = os.path.join(directory, 'map_boxes.json')
        img_dir     = os.path.join(directory, 'images')
        boxes_dir   = os.path.join(directory, 'boxes')
        
        if save: os.makedirs(img_dir, exist_ok = True)
        if save_boxes: os.makedirs(boxes_dir, exist_ok = True)
        
        # load previous generated (if any)
        infos_pred = {}
        if os.path.exists(map_file):
            infos_pred = load_json(map_file)
        
        infos_boxes = {}
        if os.path.exists(map_box_file):
            infos_boxes = load_json(map_box_file)
        
        # Get 1 time eachimage to not detect multiple times (as prediction is deterministic)
        requested_images = images
        images = list(set(images))
        
        videos = [img for img in images if img.endswith('.mp4')]
        images = [img for img in images if not img.endswith('.mp4')]
        # Predict on videos (if any)
        if len(videos) > 0:
            videos_infos = self.predict_video(
                videos, directory = directory, overwrite = overwrite, 
                save_boxes = save_boxes,
                batch_size = batch_size, labels = labels, 
                obj_threshold = obj_threshold, nms_threshold = nms_threshold,
                tqdm = tqdm, debug = debug, ** videos_kwargs, ** kwargs
            )
            # Update information on video prediction
            for video, infos in videos_infos:
                for frame_path, frame_infos in infos['frames'].items():
                    infos_pred[frame_path] = frame_infos
        
        # for each batch
        img_num = 0
        for start_idx in tqdm(range(0, len(images), batch_size)):
            start_process = time.time()
            
            # Load images and process them
            batch       = images[start_idx : start_idx + batch_size]
            batch_images    = [load_image(img) for img in batch]
            processed   = self.get_input(batch_images)
            
            start_infer = time.time()
            t_process += start_infer - start_process
            # Detect boxes
            boxes = self.detect(
                processed, get_boxes = True,
                obj_threshold = obj_threshold, nms_threshold = nms_threshold
            )
            
            t_infer += time.time() - start_infer
            
            # Process each prediction
            for path, image, box in zip(batch, batch_images, boxes):
                # Maybe skip if no box detected
                if len(box) == 0 and not save_empty:
                    if verbose: print("No box found on {}, skip it !".format(path))
                    continue
                
                # Draw prediction on image
                detected = self.draw_prediction(image, box, labels = labels, ** kwargs)
                
                # Show predictions (if verbose > 1)
                if verbose and img_num < show:
                    start_show = time.time()

                    img_num += 1
                    # Show individual boxes
                    if verbose > 1:
                        if verbose == 3:
                            print("{} boxes found :\n{}".format(
                                len(box), '\n'.join([str(b) for b in box])
                            ))

                        show_boxes(
                            image, box, labels = labels,
                            ** box_plot_kwargs, ** plot_kwargs
                        )
                    # Show original image with drawed boxes
                    plot(
                        detected, title = '{} object detected'.format(len(box)), 
                        plot_type = 'imshow', ** plot_kwargs
                    )
                    
                    t_show += time.time() - start_show
                
                # Save original image (if raw image) and save detected image
                detected_path = detected
                
                infos_pred_i, infos_box_i = {}, {}
                if save or save_boxes:
                    start_save = time.time()

                    infos_pred_i, infos_box_i = self.save_prediction(
                        image, box, detected = detected, labels = labels,
                        filename = path, img_dir = img_dir, boxes_dir = boxes_dir,
                        save_detected = save, save_boxes = save,
                        extract_boxes = save_boxes
                    )
                    
                    t_save += time.time() - start_save
                
                infos_pred.update(infos_pred_i)
                infos_boxes.update(infos_box_i)
        
        # Save information
        if save or save_boxes:
            start_save = time.time()
            
            if save: dump_json(map_file, infos_pred, indent = 4)
            if save_boxes: dump_json(map_box_file, infos_boxes, indent = 4)
            
            t_save += time.time() - start_save
        
        # Show time information
        if debug:
            print("Total time : {}\n- Processing time : {}\n- Inference time : {}\n- Show time : {}\n- Saving time : {}".format(
                time_to_string(t_process + t_infer + t_show + t_save),
                time_to_string(t_process),
                time_to_string(t_infer),
                time_to_string(t_show),
                time_to_string(t_save)
            ))
        
        return [
            (img, infos_pred.get(img, {})) for img in requested_images
            if isinstance(img, str)
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
    
    def predict_video(self,
                      videos,
                
                      labels  = None,
                      batch_size  = 16,
                      
                      obj_threshold = None,
                      nms_threshold = None,
                      
                      save_frames = False,
                      save_boxes  = False,
                      save_detected = False,
                      save_video  = True,
                      save_empty  = False,
                      directory   = None,
                      overwrite   = False,
                
                      tqdm    = lambda x: x,
                      debug   = False,
                
                      **kwargs
                     ):
        """
            Perform prediction on `images` and compute some time statistics
            
            Arguments :
                - images : (list of) images (either path / raw image) to detect objects on
                
                - labels : custom labels to use
                - batch_size    : batch_size for prediction
                
                - save_frames   : whether to save individual frames or not
                - save_video    : whether to save detection as a video
                - save_empty    : whether to save image where no objectdetected
                - directory     : where to save result
                - overwrite     : whether to overwriteor not already predicted image
                
                - tqdm      : progress bar (put to False if show > 0 or verbose)
                - debug     : whether to show time information or not
                
                - kwargs            : kwargs for `draw_prediction()` call
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
            
            Note that even if `save == False` we load already predicted images to not re-predict them.
            i.e. put `save` to False will not overwrite.
        """
        videos = normalize_filename(videos)
        if labels is None: labels = self.labels
        if not isinstance(videos, (list, tuple)): videos = [videos]
                
        t_process, t_infer, t_save, t_show = 0., 0., 0., 0.
        
        # get saving directory
        if directory is None: directory = self.pred_dir
        map_file        = os.path.join(directory, 'map_videos.json')
        map_frame_file  = os.path.join(directory, 'images.json')
        map_box_file    = os.path.join(directory, 'map_boxes.json')
        
        video_dir, frame_dir, boxes_dir = None, None, None
        if save_frames or save_detected:
            frame_dir   = os.path.join(directory, 'images')
            os.makedirs(frame_dir, exist_ok = True)
        
        if save_boxes:
            boxes_dir   = os.path.join(directory, 'boxes')
            os.makedirs(frame_dir, exist_ok = True)
        
        if save_video:
            video_dir   = os.path.join(directory, 'videos')
            os.makedirs(video_dir, exist_ok = True)
        
        # load previous generated (if any)
        infos_pred = {}
        if os.path.exists(map_file):
            infos_pred = load_json(map_file)
        
        infos_boxes = {}
        if os.path.exists(map_box_file):
            infos_boxes = load_json(map_box_file)
        # Get 1 time eachimage to not detect multiple times (as prediction is deterministic)
        requested_videos = [video for video in videos if video.endswith('.mp4')]
        videos = list(set(requested_videos))
        
        # for each batch
        for path in videos:
            video_name  = os.path.basename(path)[:-4]
            # Maybe skip because already predicted
            if not overwrite and path in infos_pred:
                if not save_frames or (save_frames and infos_pred[path]['frames'] is not None):
                    if not save_video or (save_video and infos_pred[path]['detected'] is not None):
                        continue
            
            video_filename, video_frame_dir, video_boxes_dir = None, None, None
            if save_frames or save_detected:
                video_frame_dir = os.path.join(frame_dir, video_name)
                if os.path.exists(video_frame_dir):
                    shutil.rmtree(video_frame_dir)
                os.makedirs(video_frame_dir, exist_ok = True)
            
            if save_boxes:
                video_boxes_dir = os.path.join(frame_dir, video_name + '_boxes')
                if os.path.exists(video_boxes_dir):
                    shutil.rmtree(video_boxes_dir)
                os.makedirs(video_boxes_dir, exist_ok = True)
            
            if save_video:
                video_filename = os.path.join(
                    video_dir, video_name + '_detected.mp4'
                )
            
            video_reader = cv2.VideoCapture(path)
            
            nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps     = video_reader.get(cv2.CAP_PROP_FPS)
            
            video_writer = None
            if save_video:
                if os.path.exists(video_filename):
                    os.remove(video_filename)
                
                video_writer = cv2.VideoWriter(
                    video_filename, cv2.VideoWriter_fourcc(*'MPEG'), 
                    fps, (frame_w, frame_h)
                )
            
            frame_infos     = {}
            infos_pred[path] = {
                'detected'  : video_filename,
                'frames'    : frame_infos,
                'height'    : frame_h,
                'width'     : frame_w,
                'nb_frames' : nb_frames
            }
            if save_boxes:
                infos_boxes[path] = {}
            
            for start_idx in tqdm(range(0, nb_frames, batch_size)):
                start_process = time.time()
                # Get `batch_size` frames
                frames = []
                for i in range(start_idx, min(start_idx + batch_size, nb_frames)):
                    _, frame = video_reader.read()
                    frames.append(frame[:, :, ::-1])
                processed   = self.get_input(frames)
                
                start_infer = time.time()
                t_process += start_infer - start_process
                # Detect boxes
                boxes = self.detect(
                    processed, get_boxes = True,
                    obj_threshold = obj_threshold, nms_threshold = nms_threshold
                )

                t_infer += time.time() - start_infer
                
                for i, (frame, box) in enumerate(zip(frames, boxes)):
                    # Draw prediction on image
                    detected = self.draw_prediction(
                        frame, box, labels = labels, ** kwargs
                    ) if len(box) > 0 else frame
                    
                    if video_writer is not None:
                        frame_to_save = load_image(detected, dtype = tf.uint8).numpy()
                        video_writer.write(frame_to_save[:, :, ::-1])
                        
                    if len(box) == 0 and not save_empty: continue
                    
                    if save_frames or save_boxes or save_detected:
                        start_save = time.time()
                        
                        infos_pred_i, infos_box_i = self.save_prediction(
                            frame, box, detected = detected, labels = labels,
                            filename = 'frame_{}.jpg', img_dir = video_frame_dir, 
                            boxes_dir = video_boxes_dir,
                            save_detected = save_detected, save_boxes = save_frames,
                            extract_boxes = save_boxes
                        )
                        
                        if save_frames: frame_infos.update(infos_pred_i)
                        if save_boxes: infos_boxes[path].update(infos_box_i)
                        
                        t_save += time.time() - start_save
        
            video_reader.release()
            if video_writer is not None:
                video_writer.release()
                copy_audio(path, video_filename)
        
        # Save information
        if save_frames or save_boxes:
            start_save = time.time()
            
            if save_frames: dump_json(map_file, infos_pred, indent = 4)
            if save_boxes: dump_json(map_box_file, infos_boxes, indent = 4)
            
            t_save += time.time() - start_save
        
        
        
        # Show time information
        if debug:
            print("Total time : {}\n- Processing time : {}\n- Inference time : {}\n- Show time : {}\n- Saving time : {}".format(
                time_to_string(t_process + t_infer + t_save),
                time_to_string(t_process),
                time_to_string(t_infer),
                time_to_string(t_show),
                time_to_string(t_save)
            ))
        
        return [(video, infos_pred[video]) for video in requested_videos]
                                
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['anchors']       = self.anchors
        config['backend']       = self.backend
        config['input_size']    = self.input_size

        config['labels']        = self.labels
        config['nb_class']      = self.nb_class
        config['max_box_per_image'] = self.max_box_per_image
        
        return config

    @classmethod
    def build_from_darknet(cls, weight_path, nom, labels, ** kwargs):
        instance = cls(labels = labels, nom = nom, ** kwargs)
        
        decode_darknet_weights(instance.model, weight_path)
        
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

