
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

import tensorflow as tf

@tf.function(input_signature = [
    tf.TensorSpec(shape = (None, None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (None, None, None), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.float32),
    tf.TensorSpec(shape = (), dtype = tf.bool)
])
def dice_coeff(y_true, y_pred, smoothing = 0.01, skip_empty = False):
    """
        Computes the Dice coefficient for possibly multi-label segmentation
        
        Arguments :
            - y_true    : 3-D Tensor ([batch_size, -1, n_labels]), the true segmentation
            - y_pred    : 3-D Tensor ([batch_size, -1, n_labels]), the predicted segmentation
            - smoothing : smoothing value
            - skip_empty    : whether to skip where `y_true == 0` in the whole segmentation
                /!\ WARNING : if the segmentation is empty for all the classes, the result will be 0
        Return :
            - dice_coeff    : score between [0 (bad), 1 (perfect)]
    """
    # shape = [batch_size, n_labels]
    intersect   = tf.reduce_sum(y_true * y_pred, axis = 1)
    union       = tf.reduce_sum(y_true, axis = 1) + tf.reduce_sum(y_pred, axis = 1)

    dice_coeff  = (2. * intersect + smoothing) / (union + smoothing)
    
    # shape = [batch_size]
    if skip_empty and tf.shape(y_pred)[-1] > 1:
        non_empty   = tf.reduce_any(tf.cast(y_true, tf.bool), axis = 1)
        
        dice_coeff  = tf.reduce_sum(tf.where(non_empty, dice_coeff, 0.), axis = -1)
        dice_coeff  = dice_coeff / tf.maximum(
            1., tf.reduce_sum(tf.cast(non_empty, tf.float32), axis = -1)
        )
    else:
        dice_coeff = tf.reduce_mean(dice_coeff, axis = -1)
    
    return dice_coeff

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smoothing = 0.01, reduction = 'none', name = 'DiceLoss', ** kwargs):
        super().__init__(name = name, reduction = 'none', ** kwargs)
        self.smoothing  = tf.cast(smoothing, tf.float32)
        self.skip_empty = tf.Variable(False, trainable = False, dtype = tf.bool, name = 'skip_empty')
    
    @property
    def metric_names(self):
        return ['loss', 'dice_coeff']
    
    def call(self, y_true, y_pred):
        batch_size, n_classes = tf.shape(y_pred)[0], tf.shape(y_pred)[-1]
        
        dice = dice_coeff(
            tf.reshape(y_true, [batch_size, -1, n_classes]),
            tf.reshape(y_pred, [batch_size, -1, n_classes]),
            self.smoothing,
            self.skip_empty
        )
        return 1. - dice, dice
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'smoothing' : self.smoothing
        })
        return config
    
class SparseDiceLoss(DiceLoss):
    def call(self, y_true, y_pred):
        n_labels    = tf.shape(y_pred)[-1]
        n_labels    = tf.reshape(tf.range(n_labels), [1, 1, 1, n_labels])
        
        y_true  = tf.cast(tf.expand_dims(y_true, axis = -1) == n_labels, tf.float32)
        return super().call(y_true, y_pred)
