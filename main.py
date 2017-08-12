import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import shutil
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import object_detection

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 10

Detect_FPS = 2

if not os.path.isfile(MODEL_FILE):
    print('downloading pretrained model!')
    print(MODEL_FILE)
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def detect_Car(vehicles_list, tolerance):
    length = len(vehicles_list)
    for i in range(0, length):
        print("*************a new frame**************")
        print(i)
        if vehicles_list[i]:
            if i == 0:
                for list in vehicles_list[0]:
                    if list:
                        print("detecting a car")
            else:
                for list in vehicles_list[i]:
                    dis=1
                    found = False
                    while dis < 5 and found == False and (i - dis) >= 0:
                        for ll in vehicles_list[i - dis]:
                            if sameCar(list, ll, tolerance):
                                found = True
                                break
                        if found:
                            break
                        dis = dis + 1
                    if found == False:
                        print("detecting a car")

def sameCar(list1, list2, tolerance):

    print(list1)
    print("print***********************")
    print(list2)
    if list1[0] and list2[0]:
        for i in range(0,4):
        #print("compare")
            if (abs(list1[i] - list2[i]) >= tolerance):
                return False
        return True
    return False

def violation_judgement(people_list, vehicles_list, timeStamp):
    return
    print("================ timeStamp: " + str(timeStamp) + "s ================")
    print("----------    people_list   -----------")
    print(people_list)
    print("----------   vehicles_list  -----------")
    print(vehicles_list)
    print("================================================================\n\n")
    detect_Car(vehicles_list, tolerance = 0.02)

def process_frames(car_detect_threshold = 0.5,
                  people_detect_threshold = 0.2,
                  ):
    if os.path.exists('tmp2'):
        shutil.rmtree('tmp2')
    os.mkdir('tmp2')
    TEST_IMAGE_PATHS = 'tmp'
    IMAGE_SIZE = (6, 4)
    plt.figure(0)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            people_list = []
            vehicles_list = []
            timeStamp = 0.0;
            for image_path in sorted(os.listdir(TEST_IMAGE_PATHS)):
                image = Image.open('tmp/' + image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                ### get people and vehicles
                # print(boxes.shape)
                # print(scores.shape)
                # print(classes.shape)
                # print(num_detections)
                timeStamp += 1.0/Detect_FPS
                people = []
                vehicles = []
                _boxes, _scores, _classes = [[0,0,0,0],[0,0,0,0]],[0,0],[0,0]
                for box, score, c in zip(boxes[0], scores[0], classes[0]):
                    # print(box,score,c)
                    ok = False
                    if(c == 1 and score > people_detect_threshold):
                        people.append(box)
                        ok = True
                    if((c == 3 or c==6 or c == 8) and score > car_detect_threshold):
                        vehicles.append(box)
                        ok = True
                    if (ok):
                        _boxes.append(box)
                        _scores.append(score)
                        _classes.append(c)
                boxes, scores, classes = np.array(_boxes), np.array(_scores), np.array(_classes)
                people_list.append(people)
                vehicles_list.append(vehicles)
                violation_judgement(people_list, vehicles_list, timeStamp)

                # Visualization of the results of a detection.
                # print(np.squeeze(boxes).shape)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=3,
                    min_score_thresh=0.2)
                plt.imshow(image_np)
                plt.savefig('tmp2/' + image_path)


def get_frames(video_path,
               max_video_frames=100,):
    if not os.path.isfile(video_path):
       print('Can not find ' + video_path)
       sys.exit();
    if os.path.exists('tmp'):
       shutil.rmtree('tmp')
    os.mkdir('tmp')
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    inc_time = 1.0 / fps
    sample_time = 1.0 / Detect_FPS
    delay_time = 0.0
    cnt = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        if frame is None:
            break
        delay_time += inc_time
        if(delay_time >= sample_time):
            cnt += 1
            if cnt > max_video_frames:
                break
            delay_time -= sample_time
            cv2.imwrite('tmp/%04d.jpg' % cnt, frame)
    cap.release()

def generate_video():
    import skvideo
    import skvideo.io
    img = cv2.imread('tmp2/' + os.listdir('tmp2')[0])
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output.avi', fourcc, Detect_FPS, (img.shape[1], img.shape[0]), 1)
    for img_name in sorted(os.listdir('tmp2')):
        frame = cv2.imread('tmp2/' + img_name)
        try:
            out.write(frame.astype('uint8'))
        except:
            print("Error: video frame did not write")
    out.release()

if __name__ == '__main__':
    video_path = 'video/' + sys.argv[1]

    get_frames(video_path, max_video_frames=40)
    process_frames()
    generate_video()
