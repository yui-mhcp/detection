import os

from unitest import Test, assert_function, assert_model_output, assert_equal

_filename = os.path.join('test', '__datas', 'lena.jpg')

def test_yolo(model_name):
    from utils.image import load_image
    from models.detection import YOLO
    from models.model_utils import is_model_name

    if not is_model_name(model_name):
        print("Model {} does not exist, skipping its consistency test !".format(model_name))
        return
    
    model = YOLO(nom = model_name)
    
    image = load_image(_filename, target_shape = model.input_size)
    
    assert_equal(model.get_input, image, _filename)
    
    model.detect(image)
    
    assert_model_output(model.detect, image, get_boxes = False, training = False)
    assert_function(model.detect, image, get_boxes = True)
    
    output = model.detect(image, get_boxes = False)[0]
    assert_function(model.decode_output, output)
    assert_equal(lambda: len(model.decode_output(output)), 2)
    assert_equal(lambda: len(model.decode_output(output, obj_threshold = 0.5)), 1)
    
    boxes = model.decode_output(output)
    
    assert_function(model.draw_prediction, image, boxes)
    assert_function(model.draw_prediction, image, boxes, as_mask = True)

@Test(sequential = True, model_dependant = 'yolo_faces')
def test_yolo_face():
    test_yolo('yolo_faces')