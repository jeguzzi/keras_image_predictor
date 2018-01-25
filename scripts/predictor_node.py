#!/usr/bin/env python


from __future__ import division

from collections import deque
from datetime import datetime as dt
from diagnostic_msgs.msg import DiagnosticStatus
from diagnostic_updater import Updater
import keras.models
import numpy as np
import rospy
from sensor_msgs.msg import Image
from skimage.transform import resize
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
import tensorflow as tf


def input_from_image(msg, shape, normalize=True):
    # We assume 1 byte per channel
    channels = msg.step // msg.width
    values = np.array(msg.data).reshape(msg.height, msg.width, channels)
    values = resize(values, shape)
    if normalize:
        values = (values - np.mean(values)) / np.std(values)
    return values[np.newaxis, ...]


class Predictor:

    def __init__(self):
        model_path = rospy.get_param('~model')
        self.model = keras.models.load_model(model_path)
        self.input_shape = tuple([s.value for s in self.model.inputs[0].shape][1:])
        output_shape = tuple([s.value for s in self.model.outputs[0].shape][1:])

        strides = np.cumprod(output_shape[::-1])[::-1]
        dims = [MultiArrayDimension(size=dim, stride=stride)
                for stride, dim in zip(strides, output_shape)]
        layout = MultiArrayLayout(dim=dims, data_offset=0)
        self.msg = Float64MultiArray(layout=layout)

        self.graph = tf.get_default_graph()
        self.pub = rospy.Publisher('prediction', Float64MultiArray, queue_size=1)
        self.cts = deque(maxlen=5)

        rospy.Subscriber('input_image', Image, self.processImage, queue_size=1)
        rospy.loginfo('Loaded Keras model from {model_path}:'
                      '- input shape {self.input_shape}'
                      '- output shape {output_shape}'.format(**locals()))
        self.init_diagnostics()
        rospy.spin()

    def processImage(self, msg):
        input_values = input_from_image(msg, self.input_shape, normalize=True)
        with self.graph.as_default():
            t = dt.now()
            output_values = self.model.predict(input_values)[0]
            ct = (t - dt.now()).total_seconds()
            self.cts.append(ct)
        self.msg.data = output_values.flatten().tolist()
        self.pub.publish(output_values)

    def update_ct_diagnostic(self, stat):
        if self.cts:
            ct = np.mean(self.cts)
            stat.summary(DiagnosticStatus.OK, '{ct:.2f} s'.format(ct=ct))
        else:
            stat.summary(DiagnosticStatus.WARN, '?')
        return stat

    def init_diagnostics(self):
        updater = Updater()
        updater.add("Computation Time", self.update_ct_diagnostic)
        rospy.Timer(rospy.Duration(1), lambda evt: updater.update())


if __name__ == "__main__":
    rospy.init_node('predictor_node')
    Predictor()
