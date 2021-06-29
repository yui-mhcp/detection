import numpy as np
import tensorflow as tf

class YoloLoss(tf.keras.losses.Loss):
    def __init__(self,
                 anchors,
                 
                 coord_scale        = 1.0,
                 object_scale       = 5.0,
                 no_object_scale    = 1.0,
                 class_scale        = 1.0,
                 
                 warmup_epochs  = 3,
                 
                 reduction  = 'none',
                 **kwargs
                ):
        super().__init__(reduction = 'none', **kwargs)
        
        assert len(anchors) % 2 == 0
        
        self.anchors    = tf.reshape(
            tf.cast(anchors, tf.float32), [1, 1, 1, len(anchors) // 2, 2]
        )
        
        self.object_scale    = tf.cast(object_scale,    tf.float32)
        self.no_object_scale = tf.cast(no_object_scale, tf.float32)
        self.coord_scale     = tf.cast(coord_scale,     tf.float32)
        self.class_scale     = tf.cast(class_scale,     tf.float32)
        
        self.seen   = 0
        self.warmup_epochs  = tf.cast(warmup_epochs,   tf.int32)
    
    @property
    def loss_names(self):
        return ['loss', 'loss_xy', 'loss_wh', 'loss_conf', 'loss_class']
    
    def call(self, y_true, y_pred):
        y_true, true_boxes = y_true
        
        shape = tf.shape(y_true)
        batch_size  = shape[0]
        grid_h      = shape[1]
        grid_w      = shape[2]
        nb_box      = shape[3]
        
        cell_x = tf.cast(tf.reshape(
            tf.tile(tf.range(grid_w), [grid_h]), 
            (1, grid_h, grid_w, 1, 1)
        ), tf.float32)
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(
            tf.concat([cell_x, cell_y], axis = -1), [batch_size, 1, 1, nb_box, 1]
        )
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * self.anchors
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = tf.nn.softmax(y_pred[..., 5:], axis = -1)
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.math.divide_no_nan(intersect_areas, union_areas)
        
        true_box_conf = y_true[..., 4] * iou_scores
        
        ### adjust class probabilities
        true_box_class = y_true[..., 5:]
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = true_boxes[..., 0:2]
        true_wh = true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.math.divide_no_nan(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis = 4)
        conf_mask = tf.cast(best_ious < 0.6, tf.float32) * (1 - y_true[..., 4]) * self.no_object_scale
        
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        #conf_mask = tf.reshape(
        #    y_true[..., 4] * self.object_scale + (1. - y_true[..., 4]) * self.no_object_scale, [-1, 1]
        #)
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * self.class_scale
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.cast(coord_mask == 0., tf.float32)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(
            self.seen < self.warmup_epochs,
            lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                     true_box_wh + tf.ones_like(true_box_wh) * self.anchors * no_boxes_mask, 
                     tf.ones_like(coord_mask)],
            lambda: [true_box_xy, true_box_wh, coord_mask]
        )
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0, tf.float32)) + 1e-6
        nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0, tf.float32)) + 1e-6
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0, tf.float32)) + 1e-6
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy)     * coord_mask) / nb_coord_box / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh)     * coord_mask) / nb_coord_box / 2.
        
        """loss_conf = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            tf.reshape(true_box_conf, [-1, 1]),
            tf.reshape(pred_box_conf, [-1, 1])
        ) * tf.reshape(conf_mask, [-1, 1]))"""
        
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask)  / nb_conf_box / 2.
        loss_class = tf.keras.losses.categorical_crossentropy(
            true_box_class, pred_box_class
        )
        loss_class = tf.reduce_sum(loss_class * class_mask) / nb_class_box
        
        loss = tf.cond(
            self.seen < self.warmup_epochs,
            lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
            lambda: loss_xy + loss_wh + loss_conf + loss_class
        )
        
        return loss, loss_xy, loss_wh, loss_conf, loss_class
    
    def get_config(self):
        config = super().get_config()
        config['anchors']   = np.reshape(self.anchors, [-1])
        
        config['object_scale']      = self.object_scale.numpy()
        config['no_object_scale']   = self.no_object_scale.numpy()
        config['coord_scale']       = self.coord_scale.numpy()
        config['class_scale']       = self.class_scale.numpy()
        
        config['warmup_epochs']     = self.warmup_epochs.numpy()
        
        return config
