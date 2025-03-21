# Copyright (C) 2025-now yui-mhcp project author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import keras
import keras.ops as K

from .loss_with_multiple_outputs import LossWithMultipleOutputs

@keras.saving.register_keras_serializable('yolo')
class YoloLoss(LossWithMultipleOutputs):
    def __init__(self,
                 coord_scale        = 1.0,
                 object_scale       = 5.0,
                 no_object_scale    = 1.0,
                 class_scale        = 1.0,
                 
                 anchors    = None,
                 warmup_epochs  = None,
                 
                 ** kwargs
                ):
        super().__init__(** kwargs)
        
        self.object_scale   = float(object_scale)
        self.no_object_scale    = float(no_object_scale)
        self.coord_scale    = float(coord_scale)
        self.class_scale    = float(class_scale)
    
    @property
    def output_names(self):
        return ['loss', 'loss_xy', 'loss_wh', 'loss_conf', 'loss_class']
    
    def compute_loss(self, y_true, y_pred, mask, batch_size, nb_box, criterion):
        if criterion == 'mse':
            loss = K.square(y_true - y_pred)
        elif criterion == 'categorical':
            loss = keras.losses.categorical_crossentropy(y_true, y_pred, from_logits = True)
        elif criterion == 'binary':
            loss = keras.losses.binary_crossentropy(y_true, y_pred, from_logits = True)
        else:
            raise ValueError('Unknown loss : {}'.format(criterion))
        
        return K.divide_no_nan(
            K.sum(K.reshape(loss, [batch_size, -1]) * mask, axis = 1),
            nb_box
        )
    
    def average_loss(self, loss, mask, nb_box):
        return K.divide_no_nan(
            K.sum(K.reshape(loss, [K.shape(mask)[0], -1]) * mask, axis = 1), nb_box
        )
    
    def compute_loss_xy(self, true_xy, pred_xy, mask, nb_box):
        loss = K.mean(K.square(true_xy - pred_xy), axis = -1)
        return self.average_loss(loss, mask, nb_box) * self.coord_scale
    
    def compute_loss_wh(self, true_wh, pred_wh, mask, nb_box):
        loss = K.mean(K.square(K.sqrt(true_wh) - K.sqrt(pred_wh)), axis = -1)
        return self.average_loss(loss, mask, nb_box) * self.coord_scale

    def compute_loss_class(self, true_class, pred_class, mask, nb_box):
        loss = keras.losses.sparse_categorical_crossentropy(
            true_class, pred_class, from_logits = False
        )
        return self.average_loss(loss, mask, nb_box) * self.class_scale
    
    def compute_loss_conf(self, y_true, true_boxes, pred_xy, pred_wh, pred_iou, mask, nb_box):
        batch_size = K.shape(mask)[0]
        # true_boxes.shape == [batch_size, nb_true_objects, 4]
        # reshape pred_{xy / wh} to [batch_size, grid_h * grid_w * nb_box, 4] == mask.shape
        pred_xy = K.reshape(pred_xy, [batch_size, -1, 2])
        pred_wh = K.reshape(pred_wh, [batch_size, -1, 2])
        pred_iou    = K.reshape(pred_iou, [batch_size, -1])
        
        # we first compute the true IoU between the boxes and the predicted boxes
        # this will be used as target in the loss computation
        true_xywh   = K.reshape(y_true[..., :4], [batch_size, -1, 4])
        
        true_ious = _compute_iou(
            true_xywh[..., :2], true_xywh[..., 2:], pred_xy, pred_wh
        ) * mask
        
        # the secon step is to compute the confidence mask for both objects and no-objects
        # the object mask is equal to `mask` (i.e., where `y_true[..., 4] == 1`)
        # the no-object mask requires to compute the best IoU between the predicted boxes and all possible boxes (i.e., `true_boxes`)
        pred_xy     = pred_xy[:, :, None, :]
        pred_wh     = pred_wh[:, :, None, :]
        true_boxes  = true_boxes[:, None, :, :]
        
        # the computed IoU has shape [batch_size, grid_h * grid_w * nb_box, nb_true_objects]
        best_ious   = K.max(_compute_iou(
            true_boxes[:, :, :, :2], true_boxes[:, :, :, 2:], pred_xy, pred_wh
        ), axis = -1)
        # the no_object_mask is equal to 1 at each position where `mask == 0`
        # and the IoU of the predicted box is lower than 0.6
        # if the predicted box has an IoU greater than 0.6, we do not consider this prediction
        # in the no-object loss (even though no object was expected at this position)
        no_object_mask  = (1. - mask) * K.cast(true_ious < 0.6, mask.dtype)
        
        conf_mask = no_object_mask * self.no_object_scale + mask * self.object_scale
        
        # we can now compute the final loss as the square of
        # the predicted IoU (i.e., `y_pred[..., 4]`) minus the actual IoU
        # note that `true_ious` equals 0 where `mask = 0`
        # i.e., the expected IoU should be 0 where no-object is expected
        loss = K.square(pred_iou - true_ious)
        return K.divide_no_nan(
            K.sum(loss * conf_mask, axis = 1),
            K.cast(K.count_nonzero(conf_mask > 0, axis = 1), loss.dtype)
        )
    
    def call(self, y_true, y_pred):
        y_true, true_boxes = y_true
        
        mask    = K.cast(K.reshape(y_true[..., 4], [K.shape(y_pred)[0], -1]), y_pred.dtype)
        nb_box  = K.sum(mask, axis = 1)
        
        """ Adjust predictions """
        pred_xy     = y_pred[..., :2]
        pred_wh     = y_pred[..., 2:4]
        pred_conf   = y_pred[..., 4]
        pred_class  = y_pred[..., 5:]
        
        """ Get ground truth """
        true_xy     = y_true[..., :2]
        true_wh     = y_true[..., 2:4]
        true_class  = K.cast(y_true[..., 5], 'int32')
        
        """ Compute losses """
        loss_xy     = self.compute_loss_xy(true_xy, pred_xy, mask, nb_box)
        loss_wh     = self.compute_loss_wh(true_wh, pred_wh, mask, nb_box)
        loss_conf   = self.compute_loss_conf(
            y_true, true_boxes, pred_xy, pred_wh, pred_conf, mask, nb_box
        )
        loss_class  = self.compute_loss_class(true_class, pred_class, mask, nb_box)
        
        return {
            'loss'  : loss_xy + loss_wh + loss_conf + loss_class,
            'loss_xy'   : loss_xy,
            'loss_wh'   : loss_wh,
            'loss_conf' : loss_conf,
            'loss_class'    : loss_class
        }
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'object_scale'  : self.object_scale,
            'no_object_scale'   : self.no_object_scale,
            'coord_scale'   : self.coord_scale,
            'class_scale'   : self.class_scale
        })
        
        return config

def _compute_iou(true_xy, true_wh, pred_xy, pred_wh):
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half

    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half       

    intersect_mins  = K.maximum(pred_mins,  true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    return K.divide_no_nan(intersect_areas, union_areas)
