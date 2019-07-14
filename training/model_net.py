import tensorflow as tf
import numpy as np
import os, os.path
import flowlib as imglib
from multiprocessing import Queue, Process
import argparse

"""
***To understand the code better, please read the third chapter of the thesis script.***
To run the code, please set the following arguments with suitable paths:
path: path of the training samples
network_details_path: the path in which you want to save the trained model and some other details about the training process  
"""
path = "~/patches16"
network_details_path = "~/networks_details/no_pad/"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')

DIR = path + "/original"
all_patches_original = []
all_patches_correct = []
all_patches_wrong = []
num_imgs_per_load = 0

filenames = [name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]
num_samples = len(filenames)

shuffle_counter_per_load = 0
bad_counter = 0

mean_R = 92.7982311707
std_R = 72.3253021888

mean_G = 88.0847809236
std_G = 71.2343988627

mean_B = 77.6385345249
std_B = 70.0788770213

subset_original = []
subset_correct = []
subset_wrong = []

def conv_relu(input, kernel_shape, bias_shape):
    if (type(input) != tf.Tensor):
        i = np.asarray(input).astype(np.float32)
    else:
        i = input
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(i, weights, strides=[1, 1, 1, 1], padding='VALID')

    return tf.nn.relu(tf.contrib.layers.batch_norm(conv + biases))

def variable_summaries(var):
    with tf.name_scope('summaries'):
        tf.summary.scalar('mean', tf.reduce_mean(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def fill_queue(queue):
    global filenames
    imgs_ref = []
    imgs_correct = []
    imgs_wrong = []
    for i in range(epoch_per_batch):

        for name in filenames:
            try:
                img_orig = imglib.read_image(path + "/original/" + name)
                img_correct = imglib.read_image(path + "/correct/" + name)
                img_wrong = imglib.read_image(path + "/wrong/" + name)

                imgs_ref.append(img_orig)
                imgs_correct.append(img_correct)
                imgs_wrong.append(img_wrong)
            except:
                continue
            if len(imgs_wrong) == 100:
                imgs_ref = np.asarray(imgs_ref, dtype= np.float32)
                imgs_correct = np.asarray(imgs_correct, dtype= np.float32)
                imgs_wrong = np.asarray(imgs_wrong, dtype= np.float32)

                imgs = (imgs_ref, imgs_correct, imgs_wrong)
                queue.put(imgs, block=True)
                imgs_ref = []
                imgs_correct = []
                imgs_wrong = []
    queue.put("done", block=True)


def model_network(filterSize, input_depth, input_image, num_filters, poolingOrder):
    num_hidden_layers = len(num_filters)
    hidden_conv = []
    hidden_pool = []
    poolCounter = 0
    for i in range(num_hidden_layers):

        if (i == 0):
                with tf.variable_scope("conv0"):
                    hidden_conv.append(
                        conv_relu(input_image, [filterSize, filterSize, input_depth, num_filters[i]], [num_filters[i]]))
                    if poolingOrder[i]==1:
                        hidden_pool.append(max_pool_2x2(hidden_conv[i]))
                        poolCounter += 1
        else:

                with tf.variable_scope("conv" + str(i)):
                    if poolingOrder[i-1] == 1:
                        hidden_conv.append(conv_relu(hidden_pool[-1], [filterSize, filterSize, num_filters[i - 1], num_filters[i]], [num_filters[i]]))
                    elif poolingOrder[i-1] == 0:
                        hidden_conv.append(conv_relu(hidden_conv[-1], [filterSize, filterSize, num_filters[i - 1], num_filters[i]],[num_filters[i]]))
                    else:
                        print("Poolingwise: arguments of the pooling order must be 0s or 1s.")
                    if poolingOrder[i]== 1:
                        hidden_pool.append(max_pool_2x2(hidden_conv[i]))

                    poolCounter += 1


    if poolingOrder[-1]== 1:
        lasthidden_flat = tf.layers.Flatten()(hidden_pool[-1])
    else:
        lasthidden_flat = tf.layers.Flatten()(hidden_conv[i])

    return lasthidden_flat

def read_validation_patches(path):
    ref_list = []
    cor_list = []
    wro_list = []
    folder_ref = os.path.join(path, "original")
    folder_cor = os.path.join(path, "correct")
    folder_wrong = os.path.join(path, "wrong")
    for file in os.listdir(folder_ref):

        ref_img = imglib.read_image(os.path.join(folder_ref, file))
        cor_img = imglib.read_image(os.path.join(folder_cor, file))
        wro_img = imglib.read_image(os.path.join(folder_wrong, file))

        ref_list.append(ref_img)
        cor_list.append(cor_img)
        wro_list.append(wro_img)
    ref_list = np.asarray(ref_list, dtype= np.float32)
    cor_list =  np.asarray(cor_list, dtype= np.float32)
    wro_list = np.asarray(wro_list, dtype= np.float32)

    return ref_list, cor_list, wro_list

patch_size = 16
x1 = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 3), name= "input_node")
x2 = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 3))
x3 = tf.placeholder(tf.float32, shape=(None, patch_size, patch_size, 3))
epoch_per_batch = 5
if (__name__ == "__main__"):

    parser = argparse.ArgumentParser.add_argument()

    channels_num = 3
    num_neurons = [10, 15, 20, 25, 30, 35, 40]
    pooling_order = [0,0,0,0,0,0,1]
    pooling_str = ""
    receptive_field_size = 3
    for i in pooling_order:
        pooling_str += str(i)
    if len(pooling_order)!=len(num_neurons):
        print("pooling order should have the same length as the number of layers.")
    with tf.variable_scope("NN_model") as scope:
        feature_original = model_network(receptive_field_size, 3, x1, num_neurons, pooling_order)
    with tf.variable_scope("NN_model", reuse=True):
        feature_correct = model_network(receptive_field_size, 3, x2, num_neurons, pooling_order)
        feature_wrong = model_network(receptive_field_size, 3, x3, num_neurons, pooling_order)
    with tf.name_scope("feature"):
        feature = feature_original
    global_step = tf.Variable(0, trainable=False)
    m = 10
    t = 0.02
    with tf.name_scope('loss'):
        matching_loss = tf.norm(feature_original - feature_correct, axis=1, ord=2)
        nonmatching_loss = tf.norm(feature_original - feature_wrong, axis=1, ord=2)
        nonmatching_loss_t = tf.maximum(0.0 , m - nonmatching_loss - t )
        loss = tf.reduce_mean( matching_loss + nonmatching_loss_t, axis=0)
    # # compute the validation loss:
    # with tf.name_scope('loss_valid'):
    #     matching_loss_eval = tf.norm(feature_original - feature_correct, axis=1, ord=2)
    #     matching_loss_eval_t =  tf.maximum(0.0 , matching_loss_eval - t )
    #     nonmatching_loss_eval = tf.norm(feature_original - feature_wrong, axis=1, ord=2)
    #     nonmatching_loss_eval_t = tf.maximum(0.0 , m - nonmatching_loss_eval - t)
    #     loss_eval = tf.reduce_mean(matching_loss_eval + nonmatching_loss_eval, axis=0)
    stepSize = 1e-4
    learning_rate = tf.train.exponential_decay(stepSize, global_step, 5000, 0.96, staircase=True)
    tf.summary.scalar("Loss", loss)
    with tf.name_scope("train_step"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

    queue = Queue(100)
    read_process = Process(target=fill_queue, args= (queue,))
    read_process.start()
    steps = []
    losses = []
    #in case you want to compute the validation loss during training:
    # validation_path = "/home/azin/eval_subset16/"
    s = ""
    description = "bn_ed5000"
    desc_INIT = "test_2"
    s += description
    for ind in range(len(num_neurons)):
        s = s + "_" + str(num_neurons[ind])
        desc_INIT = desc_INIT + "_" + str(num_neurons[ind])
    saver = tf.train.Saver()
    threshold = 100
    threat = 0
    run_level = 0
    batch_size = 100
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(
            network_details_path +'logs' + "/log_" + s  + "_" + pooling_str + "_"+ str(stepSize) + "_" + str(epoch_per_batch))
        # to save the initialization:
        #sess.run(tf.global_variables_initializer())
        #save_path = saver.save(sess, network_details_path + "inits/" + desc_INIT + "/init_values.ckpt")

        #restore the initialization:
        saver.restore(sess, network_details_path + "inits/" + desc_INIT + "/init_values.ckpt")
        j=0
        while True:
            steps.append(j)
            queue_result = queue.get(block=True)
            if queue_result == "done":
                break
            batch_original, batch_correct, batch_wrong = queue_result
            if len(batch_original) == 0:
                print("oops")
                break
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x1: batch_original, x2: batch_correct, x3: batch_wrong})
            train_writer.add_summary(summary, j)

            losses.append(sess.run(loss, feed_dict = {x1: batch_original, x2: batch_correct, x3: batch_wrong}))
            print("loss: ", losses[-1])
            if losses[-1] == 0.0:
                threat += 1
            if threat == threshold:
                break
            j += 1

        train_writer.add_graph(sess.graph)
        train_writer.flush()
        save_path = saver.save(sess, network_details_path +"trained_networks/" + s + "_"+ pooling_str + "_" + str(stepSize) + "_" + str(
            epoch_per_batch) + "/trained_variables.ckpt")
        print("Model saved in file: %s" % save_path)
