import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2 
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import streamlit as st
from gtts import gTTS
from IPython.display import Audio
from cvzone.HandTrackingModule import HandDetector
import pygame

pygame.mixer.init()

st.title("Capturing gesture")
frame_placeholder = st.empty()
stop_button_pressed = st.button('Stop')

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'
CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-16')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

# Setup capture
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
folders = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine','Aboard', 'All Gone', 'Baby', 'Beside', 'Book', 'Bowl', 'Bridge', 'Camp', 'Fond', 'Friend', 'High', 'House', 'How Many', 'I', 'I Love You', 'Marry', 'Medal', 'Mid Day', 'Middle', 'Money', 'Mother', 'Opposite', 'Rose', 'See', 'Short', 'Thank You', 'Write', 'yes', 'You']
prev = ""
while True: 
    ret, frame = cap.read()
    hands = detector.findHands(frame, draw=False)
    
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    label = np.argmax(detections['detection_multiclass_scores'])
    prediction = folders[label-1]
    # Get the first detection
    detection = detections['detection_boxes'][0, 0]

# Check if a bounding box exists
    
    if hands:
        if prediction != prev :
            prev = prediction
            print(prev)
            gesture = "{}.mp3".format(prev)
            if not os.path.exists(gesture):
                tts = gTTS(text=prev, lang='en')
                tts.save(gesture)
                Audio(gesture)
            pygame.mixer.music.load(gesture)  # Load the audio file
            pygame.mixer.music.play()
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)
    
    image = cv2.resize(image_np_with_detections, (800, 600))
    frame_placeholder.image(image , channels='RGB')

    #cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        break
