import tensorflow as tf
import numpy as np
import os, os.path
import freeze_graph

"""
To understand the code better, please read the 4th chapter of the thesis script.
To run this file, please set the following variables with suitable paths:
path_prefix: the path of the folder in which the trained network model is saved.
trained_model_name: the name of the trained model.
"""
path_prefix = "~/networks_details/"
trained_model_name = "bn_ed5000_10_15_20_25_30_35_40_0000001_1e-04_5"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def assure_path_exists(path):

    if not os.path.exists(path):
        print(path + " did not exist.")
        os.makedirs(path)
    else:
        print(path + " existed.")

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='VALID')
translate_10 = [0, -1]
translate_01 = [-1,0]
translate_11 = [-1 ,-1]

def do_multipooling_withDims(conv):
    pool_00 = max_pool_2x2(conv)
    conv_01 = tf.contrib.image.translate(conv, translate_01)
    pool_01 = max_pool_2x2(conv_01)

    conv_10 = tf.contrib.image.translate(conv, translate_10)
    pool_10 = max_pool_2x2(conv_10)

    conv_11 = tf.contrib.image.translate(conv, translate_11)
    pool_11 = max_pool_2x2(conv_11)

    output = tf.stack([pool_00, pool_01, pool_10, pool_11])
    output = tf.reshape(output, (4 * conv.shape[0], tf.shape(output)[2], tf.shape(output)[3], tf.shape(output)[4]))

    return output, output.shape[0]

def conv_relu(input, kernel_shape, bias_shape):
    if (type(input) != tf.Tensor):
        i = np.asarray(input).astype(np.float32)
    else:
        i = input

    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(i, weights,
                        strides=[1, 1, 1, 1], padding='VALID')

    return tf.nn.relu(tf.contrib.layers.batch_norm(conv + biases))

def variable_summaries(var):
    with tf.name_scope('summaries'):
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def unwarp(last_layer_output):
    last_layer_output = tf.transpose(last_layer_output, perm=[1, 2, 0, 3])
    shape = last_layer_output.get_shape().as_list()
    n = int(np.log2(shape[2]))

    shape_prepare_unwarp = []
    shape_prepare_unwarp.append(tf.shape(last_layer_output)[0])
    shape_prepare_unwarp.append(tf.shape(last_layer_output)[1])
    for i in range(n):
        shape_prepare_unwarp.append(2)
    shape_prepare_unwarp.append(tf.shape(last_layer_output)[3])
    last_layer_output = tf.reshape(last_layer_output, tuple(shape_prepare_unwarp))
    n = int(n / 2)

    for i in range(n):
        list_shape_reshape = []
        n_dim = len(last_layer_output.shape)
        list_shape_trans = list(range(n_dim))
        list_shape_trans[1], list_shape_trans[2] = list_shape_trans[2], list_shape_trans[1]
        last_layer_output = tf.transpose(last_layer_output, perm=list_shape_trans)
        list_shape_reshape.append(tf.shape(last_layer_output)[0] * tf.shape(last_layer_output)[1])
        list_shape_reshape.append(tf.shape(last_layer_output)[2] * tf.shape(last_layer_output)[3])
        for j in range(4, n_dim):
            list_shape_reshape.append(tf.shape(last_layer_output)[j])
        shape_for_reshape = tuple(list_shape_reshape)
        last_layer_output = tf.reshape(last_layer_output, shape_for_reshape)
    return last_layer_output

def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def model_network(filterSize, input_depth, x_image, num_imgs, num_filters, poolingOrder):
    num_hidden_layers = len(num_filters)
    hidden_conv = []
    hidden_pool = []
    for i in range(num_hidden_layers):

        if (i == 0):
                with tf.variable_scope("conv0"):
                    hidden_conv.append(
                        conv_relu(x_image, [filterSize, filterSize, input_depth, num_filters[i]], [num_filters[i]]))
                    variable_summaries(hidden_conv[i])
                    if poolingOrder[i]==1:
                        pooling_output = do_multipooling_withDims(hidden_conv[i])
                        hidden_pool.append(pooling_output)
        else:
                with tf.variable_scope("conv" + str(i)):
                    if poolingOrder[i-1] == 1:
                        hidden_conv.append(conv_relu(hidden_pool[-1], [filterSize, filterSize, num_filters[i - 1], num_filters[i]], [num_filters[i]]))
                    elif poolingOrder[i-1] == 0:
                        hidden_conv.append(conv_relu(hidden_conv[-1], [filterSize, filterSize, num_filters[i - 1], num_filters[i]],[num_filters[i]]))
                    else:
                        print("Poolingwise: pooling elements must be either zero or one.")
                    if poolingOrder[i]== 1:
                        pooling_output = do_multipooling_withDims(hidden_conv[i])
                        hidden_pool.append(pooling_output)
                    variable_summaries(hidden_conv[i])

    if poolingOrder[-1]== 1:
        output = unwarp(hidden_pool[-1])
    else:
        output = unwarp(hidden_conv[i])

    return output

x = tf.placeholder(tf.float32, shape=(1, None, None, None), name= "input_node")

last_layer_neurons = 0
if (__name__ == "__main__"):

    channels_num = 3
    num_neurons = [10, 15, 20, 25, 30, 35, 40]
    pooling_order = [0, 0, 0, 0, 0, 0, 1]
    last_layer_neurons = num_neurons [-1]
    receptive_field_size = 3
    #in case you want to low pass filter the input images:
    #gauss_kernel = gaussian_kernel(2.0, 0.0, 0.83333)
    #filter = tf.Session().run(gauss_kernel)
    #global kernel_toConv
    #kernel_toConv = np.zeros((5, 5, last_layer_neurons , last_layer_neurons ), dtype=np.float32)
    # for i in range(last_layer_neurons ):
    #     kernel_toConv[:, :, i, i] = filter
    # kernel_toConv = tf.convert_to_tensor(kernel_toConv)

    with tf.variable_scope("NN_model") as scope:
        feature_original = model_network(receptive_field_size, 3, x, num_neurons, pooling_order)
        feature_original = tf.identity(feature_original, name= "output_node")

    img_num = 1
    trained_vars_path = path_prefix + "trained_networks/" + trained_model_name
    description = ""
    if description != "":
        trained_model_name = trained_model_name + "_" + description

    assure_path_exists(path_prefix + "output_models/" + trained_model_name)
    assure_path_exists(path_prefix + 'graph_def/' + trained_model_name)
    with tf.Session() as sess:

        tf.train.write_graph(sess.graph.as_graph_def(), path_prefix + 'graph_def/'+ trained_model_name, "train.pbtxt")
        output_graph_name = "output_graph.pb"
        input_graph_path = path_prefix + 'graph_def/'+ trained_model_name + '/train.pbtxt'
        input_saver_def_path = ""
        input_binary = False
        input_checkpoint_path = trained_vars_path + "/trained_variables.ckpt"
        output_node_names = "NN_model/output_node"
        restore_op_name = trained_vars_path + "/trained_variables.ckpt"
        filename_tensor_name = path_prefix + "output_models/" + trained_model_name + "/Const:0"
        #path of the exported model. Ready to export to another program also in other languages! :)
        output_graph_path = path_prefix + "output_models/"+ trained_model_name +"/" + output_graph_name
        clear_devices = False
        freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                                  input_binary, input_checkpoint_path,
                                  output_node_names, restore_op_name,
                                  filename_tensor_name, output_graph_path,
                                  clear_devices, initializer_nodes= "")
