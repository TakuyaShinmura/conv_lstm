#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import conv_lstm_cell
import random
import cv2
import numpy as np
import math
from six.moves import xrange


tf.app.flags.DEFINE_string("data_dir","data/img/","Image directory.")
tf.app.flags.DEFINE_string("out_dir", "data/result/", "Output directory.")
tf.app.flags.DEFINE_integer("conv_channel", 32, "Size to convolution.")
tf.app.flags.DEFINE_boolean("use_peepholes", True, "Using peepholes or not.")
tf.app.flags.DEFINE_float("cell_clip", None, "Value of cell clipping.")
tf.app.flags.DEFINE_float("forget_bias", 1.0, "Value of forget bias.")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_integer("train_step", 10000, "Num to train.")
tf.app.flags.DEFINE_integer("batch_size", 10, "Num of batch size.")

FLAGS = tf.app.flags.FLAGS

IMG_SIZE = [64,64]
KERNEL_SIZE = [10, 10]
STRIDE = [1,1,1,1]


initializer = tf.truncated_normal_initializer(stddev=0.1)
activation = tf.nn.tanh



def inference(images):
    cell = conv_lstm_cell.ConvLSTMCell(FLAGS.conv_channel, img_size=IMG_SIZE, kernel_size=KERNEL_SIZE,
        stride= STRIDE, use_peepholes=FLAGS.use_peepholes, cell_clip=FLAGS.cell_clip, initializer=initializer,
        forget_bias=FLAGS.forget_bias, state_is_tuple=False, activation=activation)

    outputs, state = tf.nn.rnn(cell=cell, inputs=images, dtype=tf.float32)

    #最終時間での出力を取得
    last_output=outputs[-1]

    #結果を1×1で畳み込んで元画像と同じサイズに加工
    kernel = tf.Variable(tf.truncated_normal([1,1 ,FLAGS.conv_channel, 3],stddev=0.1))
    result = tf.nn.conv2d(last_output, kernel,[1,1,1,1], padding='SAME')
    result = tf.nn.sigmoid(result)

    return result


def loss(result, correct_image):
    loss = tf.reduce_mean(tf.abs(result-correct_image))
    return loss

def train(error):
    return tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(error)



def main(_):

    #入力データ(batch, width, height, channel)の4階テンソルの時系列リスト
    images = []
    for i in xrange(4):
        input_ph = tf.placeholder(tf.float32,[None, IMG_SIZE[0], IMG_SIZE[1], 3])
        tf.add_to_collection("input_ph", input_ph)
        images.append(input_ph)

    #正解データ(batch, width, height, channel)の4階テンソル
    y = tf.placeholder(tf.float32,[None, IMG_SIZE[0], IMG_SIZE[1], 3])


    result = inference(images)
    error = loss(result, y)
    train_step = train(error)

    init_op = tf.initialize_all_variables()

    with tf.Session() as sess:

        #テスト用のfeed_dictの作成
        test_feed = {}
        for i in xrange(4):
            img = cv2.imread(FLAGS.data_dir+str(109+i)+'.png')/255.0
            test_feed[tf.get_collection("input_ph")[i]] = [img]

        test_feed[y] = [cv2.imread(FLAGS.data_dir+str(113)+'.png')/255.0]



        sess.run(init_op)
        for step in xrange(FLAGS.train_step):

            feed_dict = {}

            # 訓練に使用する画像の最初のフレームをバッチサイズ分取得
            target = []
            for i in xrange(FLAGS.batch_size):
                target.append(random.randint(0,104))

            #入力画像のplaceholder用のfeed_dictを埋める
            for i in xrange(4):
                inputs = []
                for j in target:
                    file = FLAGS.data_dir+str(i+j)+'.png'
                    img = cv2.imread(file)/255.0
                    inputs.append(img)

                feed_dict[tf.get_collection("input_ph")[i]] = inputs


            #正解データのplaceholder用のfeed_dictを埋める
            correct = []
            for i in target:
                img = cv2.imread(FLAGS.data_dir+str(i+4)+'.png')/255.0
                correct.append(img)

            feed_dict[y] = correct

            print("step%d training"%step)
            sess.run(train_step, feed_dict=feed_dict)

            #10ステップごとにログを取り、テストデータに対する誤差と画像を出力
            if (step+1) % 10 == 0:

                created, error_val = sess.run([result, error], feed_dict=test_feed)
                print("step%d loss: %f" % (step, error_val))
                cv2.imwrite(FLAGS.out_dir+"step"+str(step)+".png", created[0] * 255.0)


if __name__ == "__main__":
    tf.app.run()












