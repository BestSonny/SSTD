import numpy as np
import math
import matplotlib
import cv2
import glob
import shutil
import time
import sys
import os.path as osp
from matplotlib.patches import Polygon
from threading import Thread, Lock
from Queue import Queue
import threading
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
# Add lib to PYTHONPATH
this_dir = osp.dirname(__file__)
lib_path = osp.join(this_dir, 'text')
add_path(lib_path)
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

caffe_root = '../' # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
import caffe

from google.protobuf import text_format
from caffe.proto import caffe_pb2


def nms(dets, thresh, force_cpu=False, device_id=2):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    if force_cpu:
        return cpu_nms(dets, thresh)
    else:
        return gpu_nms(dets, thresh, device_id=device_id)

def xcycwh_angle_to_x1y1x2y2x3y3x4y4(box, scale=[1,1]):
    xc, yc, width, height, radians = box
    x0 = -0.5*width
    x1 = 0.5*width
    y0 = -0.5*height
    y1 = 0.5*height
    cos_radians = math.cos(radians)
    sin_radians = math.sin(radians)
    x0_r = scale[0]*(cos_radians*x0 - sin_radians*y0 + xc)
    x1_r = scale[0]*(cos_radians*x1 - sin_radians*y0 + xc)
    x2_r = scale[0]*(cos_radians*x1 - sin_radians*y1 + xc)
    x3_r = scale[0]*(cos_radians*x0 - sin_radians*y1 + xc)

    y0_r = scale[1]*(sin_radians*x0 + cos_radians*y0 + yc)
    y1_r = scale[1]*(sin_radians*x1 + cos_radians*y0 + yc)
    y2_r = scale[1]*(sin_radians*x1 + cos_radians*y1 + yc)
    y3_r = scale[1]*(sin_radians*x0 + cos_radians*y1 + yc)

    return [x0_r, x1_r, x2_r, x3_r, y0_r, y1_r, y2_r, y3_r]

def clip(x, min_value=0, max_value=float('inf')):
    return int(np.clip(x, min_value, max_value))

import os
cwd = os.getcwd()
print cwd
images = glob.glob("./examples/text/result/*.jpg")
results =  glob.glob("./examples/text/result/*.txt")
removes = images + results

for f in removes:
    os.remove(f)

# load PASCAL VOC labels
labelmap_file = "./examples/text/labelmap_text.prototxt"
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

image_dir = './examples/text/images'
image_list = glob.glob('{}/*.*'.format(image_dir))
image_list = sorted(image_list)

image_resizes = [512 + 128 +64]
threshold = 0.6
nets = []
device_id = 0
for image_resize in image_resizes:
    model_def = './examples/text/model/deploy.prototxt'
    model_weights = './examples/text/model/demo.caffemodel'
    model_modify = './examples/text/model/final_deploy.prototxt'
    lookup = 'step:'

    true_steps = ['    step: {}'.format(2**(2+i)) for i in range(1,5)]
    for i in range(1,4):
        step = image_resize / (image_resize / 64.0 - 2*i)
        true_steps.append('    step: {}'.format(step))
    print true_steps
    f =  open(model_modify, 'w')
    with open(model_def, 'r') as myFile:
        i = 0
        for num, line in enumerate(myFile, 1):
            if lookup in line:
                print 'found at line:', num
                f.write(true_steps[i]+'\r\n')
                i = i + 1
                continue
            f.write(line)
    f.close()

    caffe.set_device(device_id)
    caffe.set_mode_gpu()
    nets.append(caffe.Net(model_modify,      # defines the structure of the mode10
                    model_weights,  # contains the trained weights
                    caffe.TEST))     # use test mode (e.g., don't perform dropout)
    device_id = device_id + 1

t = 0
total_time = 0
for image_path in image_list:
    try:
        image = caffe.io.load_image(image_path)
        original_shape = image.shape
        original_image = image
    except:
        break
    height, width, channels = image.shape
    im_size_min = np.min(image.shape[0:2])
    im_size_max = np.max(image.shape[0:2])
    plt.imshow(original_image)
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    currentAxis = plt.gca()
    device_id = 0
    device_ids = range(len(image_resizes))
    original_images = [original_image] * len(image_resizes)
    my_queue = Queue()
    lock = Lock()
    params = zip(device_ids, image_resizes)
    for param in params:
        my_queue.put(param)
    detlist = []
    def worker():
        while True:
            global total_time
            global t
            #grabs host from queue
            id, resize = my_queue.get()
            image_resize_height = resize
            image_resize_width =  resize
            caffe.set_device(id)
            caffe.set_mode_gpu()
            transformer = caffe.io.Transformer({'data': (1,3,image_resize_height,image_resize_width)})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', np.array([104,117,123])) # mean pixel
            transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
            transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
            image = original_image
            start = time.clock()
            nets[id].blobs['data'].reshape(1,3,image_resize_height,image_resize_width)
            transformed_image = transformer.preprocess('data', image)
            nets[id].blobs['data'].data[...] = transformed_image
            detections = nets[id].forward()['detection_out']
            total_time = total_time + (time.clock() - start)*1000.0
            t = t + 1
            print 'avearage running time ' + str(total_time/t)
            print (image_path)
            det_label = detections[0,0,:,1]
            det_conf = detections[0,0,:,2]
            det_xmin = detections[0,0,:,3]
            det_ymin = detections[0,0,:,4]
            det_xmax = detections[0,0,:,5]
            det_ymax = detections[0,0,:,6]
            # Get detections with confidence higher than threshold
            top_indices = [i for i, conf in enumerate(det_conf) if conf >= threshold]
            top_conf = det_conf[top_indices]
            top_xmin = det_xmin[top_indices]
            top_ymin = det_ymin[top_indices]
            top_xmax = det_xmax[top_indices]
            top_ymax = det_ymax[top_indices]
            for i in xrange(top_conf.shape[0]):
                xmin = int(round(top_xmin[i] * image.shape[1]))
                ymin = int(round(top_ymin[i] * image.shape[0]))
                xmax = int(round(top_xmax[i] * image.shape[1]))
                ymax = int(round(top_ymax[i] * image.shape[0]))
                score = top_conf[i]

                xmin = max(0, int(round(top_xmin[i] * original_shape[1])))
                ymin = max(0, int(round(top_ymin[i] * original_shape[0])))
                xmax = min(original_shape[1]-1, int(round(top_xmax[i] * original_shape[1])))
                ymax = min(original_shape[0]-1, int(round(top_ymax[i] * original_shape[0])))
                coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1

                try:
                    assert xmin <= xmax and ymin <= ymax, 'left must less than right'
                except:
                    continue
                lock.acquire()
                detlist.append([xmin, ymin, xmax, ymax, score])
                lock.release()
            my_queue.task_done()

    for j in xrange(10):
        a = Thread(target=worker)
        a.daemon = True
        a.start()
    my_queue.join()

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    fp = open('./examples/text/result/res_{}.txt'.format(image_name),'w')
    if len(detlist) != 0:
        dets = np.array(detlist).astype(np.float32)
        #keep = nms(dets, 0.1)
        #dets = dets[keep, :]
        for j in range(dets.shape[0]):
            xmin, ymin, xmax, ymax, score = dets[j,:]
            color = colors[1]
            display_txt = '%.2f'%(score)
            coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))
            currentAxis.text(xmin, ymin, display_txt)
            fp.write('{},{},{},{}\r\n'.format(int(xmin), int(ymin), int(xmax), int(ymax)))

    plt.savefig('./examples/text/result/{}'.format(os.path.basename(image_path)))
    plt.close()
    fp.close()
