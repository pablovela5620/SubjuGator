#!/usr/bin/python
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import rospy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# To correctly import utils
sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

class classifier(object):
    def __init__(self):
        rospy.init_node('image_localizer')

        self.bridge = CvBridge()
        #/stereo/right/image_raw, /camera/seecam/image_raw
        self.sub1 = rospy.Subscriber('/camera/front/left/image_rect_color', Image, self.img_callback)

        self.pub1 = rospy.Publisher('path_label', Image, queue_size=1)
    def img_callback(self, data):
        try:
            print('working')
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            print(e)

        image_expanded = np.expand_dims(cv_image, axis = 0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            cv_image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)

        image = cv2.resize(cv_image, (720, 480))
        try:
            self.pub1.publish(self.bridge.cv2_to_imgmsg(image, 'bgr8'))
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    print('running')
    # Current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen model and labelmap
    WEIGHTS_DIR = 'Path_Inference/faster_rcnn_inception_v2_path/frozen_inference_graph.pb'

    PATH_TO_CKPT = WEIGHTS_DIR
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'Path_Inference', 'path_label_map.pbtxt')

    # Number of Classes
    NUM_CLASSES = 1

    # Load lable map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Loading pretrained weights graph
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.Session(graph=detection_graph)

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    classifier()
    rospy.spin()
