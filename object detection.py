import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import six.moves.urllib as urllib
import cv2

from collections import defaultdict
from PIL import Image

# Add object_detection module to path
sys.path.append("..")

# Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

# Video capture
cap = cv2.VideoCapture('C:\\path\\to\\your\\video.mp4')

# ---------------- Model Preparation ----------------

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

# Download model if not already present
if not os.path.exists(MODEL_NAME):
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

# Load TensorFlow model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# ---------------- Helper Function ----------------
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

# ---------------- Detection ----------------
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_np_expanded = np.expand_dims(frame, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # ---------------- Visualization ----------------
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=3,  # Reduced thickness
                min_score_thresh=0.5,
                skip_labels=True,  # Removed "POTHOLE" label
                skip_scores=False
            )

            # ---------------- Distance & Warning ----------------
            for i, b in enumerate(np.squeeze(boxes)):
                if np.squeeze(classes)[i] == 1:  # Class 1 = pothole in your setup
                    if np.squeeze(scores)[i] >= 0.9:
                        mid_x = (b[1] + b[3]) / 2
                        mid_y = (b[0] + b[2]) / 2
                        apx_distance = round((1 - (b[3] - b[1]))**4, 1)
                        cv2.putText(frame, '{}'.format(apx_distance),
                                    (int(mid_x * 800), int(mid_y * 450)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if apx_distance <= 0.5:
                            if 0.3 < mid_x < 0.7:
                                cv2.putText(frame, 'WARNING!!!',
                                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                            1.0, (0, 0, 255), 3)

            cv2.imshow('Pothole Detection', cv2.resize(frame, (800, 600)))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
