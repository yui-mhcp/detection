import numpy as np
import tensorflow as tf

def truncate(tokens, max_length, keep_mode = 'start'):
    assert mode in ('random', 'start', 'end')
    
    if tf.shape(tokens)[0] > max_length:
        if keep_mode == 'random':
            start = tf.random.uniform(
                (), minval = 0, 
                maxval = tf.shape(tokens)[0] - max_length,
                dtype = tf.int32
            )
        elif keep_mode == 'end':
            start = tf.shape(tokens)[0] - max_length
        else:
            start = 0
                
    return tokens[start : start + max_length]

def pad_batch(batch, pad_value = 0, max_length = None, dtype = None):
    """
        Create a padded version of batch in a single np.ndarray
        Note that this function allows to have different shapes on different dimensions and will pad all of them. 
        However, all data must have the same rank
        
        Arguments : 
            - batch         : list of np.ndarray / tf.Tensor
            - pad_value     : the value to add as padding
            - max_length    : maximum length for each dimension. If not given, take the max length of datas 
            - dtype : dtype of the final output
        Return : 
            - padded_batch : np.ndarray of same rank as data
    """
    if not hasattr(batch[0], 'shape'): return np.array(batch)
    
    if dtype is None:
        b0 = batch[0] if not hasattr(batch[0], 'numpy') else batch[0].numpy()
        dtype = b0.dtype
    
    max_shape = batch[0].shape
    for b in batch:
        max_shape = [max(max_s, s) for max_s, s in zip(max_shape, b.shape)]
    if max_length is not None: max_shape[0] = min(max_shape[0], max_length)
    length = max_shape[0]
    max_shape = [len(batch)] + max_shape
    
    padded_batch = np.zeros(max_shape, dtype = dtype) + pad_value
    
    for i, b in enumerate(batch):
        if b.ndim == 1:
            padded_batch[i, :min(length, len(b))] = b[:length]
        elif b.ndim == 2:
            padded_batch[i, :min(length, len(b)), : b.shape[1]] = b[:length]
        elif b.ndim == 3:
            padded_batch[i, :min(length, len(b)), : b.shape[1], : b.shape[2]] = b[:length]
        elif b.ndim == 4:
            padded_batch[i, :min(length, len(b)), : b.shape[1], : b.shape[2], : b.shape[3]] = b[:length]
        
    return padded_batch

def concat_sequences(seq1, seq2, pad_value):
    """ Concat 2 batch of sequences where `seq1` and `seq2` have shape `(batch_size, seq_{1 / 2}_len)` """
    len_1, len_2 = tf.shape(seq1)[1], tf.shape(seq2)[1]

    if len_1 != len_2:
        padding = [(0,0), (0, tf.abs(len_1 - len_2))]
        if len_1 > len_2:
            seq2 = tf.pad(seq2, padding, constant_values = pad_value)
        else:
            seq1 = tf.pad(seq1, padding, constant_values = pad_value)
    
    return tf.concat([seq1, seq2], axis = 0)