#!/usr/bin/env python
# coding: utf-8

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
import cv2
#tf.compat.v1
import scipy
import json
import json
import base64
from two_pass import count_patch
import shutil

tf.reset_default_graph()

LABEL_NAMES = np.asarray(["background", "pig"])


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = "ImageTensor:0"
    OUTPUT_TENSOR_NAME = "SemanticPredictions:0"
    INPUT_SIZE = 640
    FROZEN_GRAPH_NAME = "frozen_inference_graph"

    def __init__(self, modelname):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_def = None

        with open(modelname, "rb") as fd:
            graph_def = tf.GraphDef.FromString(fd.read())

        if graph_def is None:
            raise RuntimeError("Cannot find inference graph in tar archive.")

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name="")

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
        image: A PIL.Image object, raw input image.
        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        #target_size = (321, 201)
        resized_image = image.convert("RGB").resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]},
        )
        seg_map = batch_seg_map[0]
        return resized_image, seg_map


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


# 从 label 到 color_image
def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError("Expect 2-D input label")

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError("label value too large.")

    return colormap[label]


# 分割结果可视化
def vis_segmentation(image, seg_map,img_out, name):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    #image.save("./seg_map_result/" + name + "_1.png")
    plt.imshow(image)
    plt.axis("off")
    plt.title("input image")

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    #plt.imshow(seg_image)
    plt.imshow(img_out)
    plt.axis("off")
    plt.title("segmentation crop")

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis("off")
    plt.title("segmentation overlay")

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation="nearest")
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid("off")

    plt.savefig("./seg_map_result/" + name + ".png")
    # plt.show()


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


def main_test(filepath):
    # 加载模型
    modelname = "model/inference_graph-80000.pb"
    MODEL = DeepLabModel(modelname)
    print("model loaded successfully!")



    filelist = os.listdir(filepath)
    for item in filelist:
        print("process image of ", item)
        filepath_jpg=item.split("_")[0]
        name = item.split(".jpg", 1)[0]
        original_im = Image.open(filepath + item)
        resized_im, seg_map = MODEL.run(original_im)

        count_num = np.sum(seg_map > 0)
        print(count_num)
        if count_num> original_im.size[1]*original_im.size[0]/24:
            seg_map_clone=seg_map.copy()
            seg_map=count_patch(seg_map_clone,True)

            cur_json_dict = {
                "version": "5.0.1",
                "flags": {},
                "shapes": [
                ],
            }
            cur_json_dict["imageHeight"] = original_im.size[1]
            cur_json_dict["imageWidth"] = original_im.size[0]
            cur_json_dict["imagePath"] = "..\\"+filepath + item
            cur_json_dict["imageData"] = None #str(base64.b64encode(open(img_path, "rb").read()))
            point_data = []
            #print(seg_map.shape[0])
            #print(seg_map.shape[1])
            ck_list=-1
            num_list=0
            for y in range(1, seg_map.shape[0] - 1,2):
                for x in range(1, seg_map.shape[1] - 1,2):
                    if seg_map[y, x]==1:
                        if seg_map[y, x-1]==0 or seg_map[y, x+1]==0 or seg_map[y-1, x]==0 or seg_map[y+1, x]==0 :
                           point_data.append([x,y])
                           num_list=num_list+1
                           if ck_list==-1:
                             ck_list=num_list-1
            #print(len(point_data))
            #print(point_data[0])
            #print(point_data[1][0])
            start_point=point_data.pop(ck_list)
            test=[10,5]
            #print(start_point)
            #print(np.argmax([1,2,3,4,5,6,7], axis=0))

            np_min=1000
            point_list = []
            for i in range(0,len(point_data)):
               if np_min>10:
                  point_list.append(start_point)
               start_point_np = np.array(start_point)
               point_data_np = np.array(point_data)
               point_data_temp=point_data_np-start_point_np
               point_data_square=point_data_temp*point_data_temp
               point_data_sum=np.sum(point_data_square,axis=1)
               #print(point_data_sum)
               np_min=np.min(point_data_sum)
               #print(np_min)
               np_min_flag=np.argmin(point_data_sum)
               #print(np_min_flag)
               start_point = point_data.pop(np_min_flag)
               #print(np_min,np_min_flag)

            #print(point_data.shape[1])

            # delete 'b and '
           # cur_json_dict["imageData"] = cur_json_dict["imageData"][2:-1]
            cur_json_dict['shapes'].append(
                { "points": point_list,
                      "label": "pig",
                      "flags": {},
                      "shape_type": "polygon",
                      "group_id": None
                 })
            new_dict = cur_json_dict
            with open("jsons\\"+name + '.json', 'a+') as f:
                f.write(json.dumps(cur_json_dict))
            '''
            #r, g, b = resized_im.getpixel((1,1))
            r, g, b = resized_im.split()
            r_img = np.asarray(r)
            r_img = np.multiply(r_img, seg_map)
            r_t=Image.fromarray(np.uint8(r_img))

            g_img = np.asarray(g)
            g_img = np.multiply(g_img, seg_map)
            g_t=Image.fromarray(np.uint8(g_img))

            b_img = np.asarray(b)
            b_img = np.multiply(b_img, seg_map)
            b_t=Image.fromarray(np.uint8(b_img))

            img_out = Image.merge("RGB", (r_t, g_t, b_t))

            if not os.path.exists(filepath_jpg):
                 os.makedirs(filepath_jpg)

            img_out.save(filepath_jpg+"/"+ name + ".jpg")

            #print(resized_im)
            #print(seg_map)
            # 分割结果拼接
            vis_segmentation(resized_im, seg_map,img_out, name)
            '''
            # 单独保存分割结果
            # seg_map_name = name + '_seg.png'
            # resized_im_name = name + '_in.png'
            # path = './seg_map_result/'
            # scipy.misc.imsave(path + resized_im_name,resized_im)
            # scipy.misc.imsave(path + seg_map_name,seg_map)
        else :
            #os.remove(filepath + item)
            shutil.move(filepath + item, "notarget\\" )

if __name__ == "__main__":
    filepath = "stage_40\\"
    '''data={"imageWidth": 640}
    filename = 'a.json'
    with open(filename, 'w') as f:
        json.dump(data, f)
        json.dump(data, f)
        '''
    main_test(filepath)