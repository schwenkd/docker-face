import os
import cv2
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from icrawler.builtin import GoogleImageCrawler


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'ResizeBilinear_3:0'
    #The input size is an image with max dimensions 513x513
    INPUT_SIZE = 513
    #We are interested in humans which are label 15
    label = 15

    def __init__(self, path):
        #We initialize the graph
        self.graph = tf.Graph()
        #We get the graph definition from the pretrained model
        graph_def = tf.compat.v1.GraphDef.FromString(open(path, 'rb').read())
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def get_mask(self, image):
        #We get the image size
        height, width = image.shape[:2]
        #We compute the resize ratio to maintain the aspect ratio
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        #We compute the size of the image to input and resize the image
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        #We run the model and get the segmentation results
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        #We apply softmax on a pixel-wise basis
        seg_map = softmax(batch_seg_map[0][:target_size[1], :target_size[0]], axis=-1)
        #We return the channel corresponding to the human segmentation
        return seg_map[:, :, self.label]

    def transform(self, image, mask, query):
        #We resize the mask to match the original image
        mask = cv2.resize(mask, image.shape[:2][::-1])[:, :, np.newaxis]
        #We get the image size
        x0, y0, c0 = image.shape
        #If the query is "bokeh" we blur the background, if we get a different query we crawl google for an image
        if query != 'bokeh':
            #We create a folder to download the image. We need try-except if the folder already exists
            try:
                os.mkdir(query)
            except:
                pass
            #We run the crawler and download 1 image: https://pypi.org/project/icrawler/
            google_crawler = GoogleImageCrawler(storage={'root_dir': f'/tmp/{query}'})
            google_crawler.crawl(keyword=query, max_num=1)
            #We load the saved image
            background = cv2.imread(f'/tmp/{query}/000001.jpg')
            #We get the background size
            x, y, c = background.shape
            #We resize the background in order to match the original image but keeping aspect ratio
            new_x = x * y0 / y
            new_y = y * x0 / x
            if new_x > x0:
                new_y = y0
            else:
                new_x = x0
            background = cv2.resize(background, (int(new_y), int(new_x)))[:x0, :y0]
        else:
            #The background should be the same image but blurred. We blur the image
            background = cv2.blur(image.copy(), (x0 // 10, y0 // 10))
        #We blend both images using the segmentation mask
        new_img = image * mask + background * (1 - mask)
        #We return the transformed image
        return new_img
