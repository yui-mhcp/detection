from models.detection.yolo import YOLO

def get_model(label = None, model = None, ** kwargs):
    assert label is not None or model is not None
    if model is not None:
        return YOLO(nom = model) if isinstance(model, str) else model
    elif label is not None:
        if label not in _pretrained:
            raise ValueError("No pretrained model for this object type !!\n  Supported : {}\n   Got : {}".format(list(_pretrained.keys()), label))
        
        return YOLO(nom = _pretrained[label])
    

def stream(label = None, model = None, ** kwargs):
    model = get_model(label = label, model = model)
    model.stream(** kwargs)

def detect(images, label = None, model = None, ** kwargs):
    model = get_model(label = label, model = model)
    return model.predict(images, ** kwargs)

_models = {
    'YOLO'    : YOLO
}

_pretrained = {
    'faces'     : 'yolo_faces'
}
