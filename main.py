import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import datetime

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, 
                layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    print("load_vgg ====================================================")
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    print("layers ====================================================")
    layer7_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, (1, 1), padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer4_1x1 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, (1, 1), padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    layer3_1x1 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, (1, 1), padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    layer7_out = tf.layers.conv2d_transpose(layer7_1x1, num_classes, 4, 2, padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    conv4sum = tf.add(layer7_out, layer4_1x1)
    conv_layer4_out = tf.layers.conv2d_transpose(conv4sum, num_classes, 4, 2, padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    conv3sum = tf.add(conv_layer4_out, layer3_1x1)
    final_layer = tf.layers.conv2d_transpose(conv3sum, num_classes, 16, 8, padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3), name='final_layer')
    # print(final_layer.get_shape())
    # tf.Print(final_layer, [tf.shape(final_layer)])
    return final_layer
# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    print("optimize ====================================================")
    # nn_last_layer = tf.Print(nn_last_layer, [tf.shape(nn_last_layer)[3]])
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")

    # logits = tf.Print(logits, [tf.shape(logits), tf.shape(correct_label)])
    # reshaped_labels = tf.reshape(correct_label, (-1, num_classes))
    # logits = tf.Print(logits, [tf.shape(logits), tf.shape(reshaped_labels)])
    # iou, iou_op = tf.metrics.mean_iou(reshaped_labels, logits, 2)

    # Tensorflow IOU
    # y_true_f = tf.reshape(reshaped_labels, [-1])
    # y_pred_f = tf.reshape(logits, [-1])
    # inter = tf.reduce_sum(tf.multiply(y_pred_f, y_true_f))
    # union = tf.reduce_sum(tf.subtract(tf.add(y_pred_f, y_true_f), tf.multiply(y_pred_f, y_true_f)))
    # loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), tf.div(inter, union))
    # intersection = tf.reduce_sum(y_true_f * y_pred_f)
    # loss = tf.constant(1.0)-(tf.constant(2.) * intersection) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=correct_label, logits=logits))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss
# tests.test_optimize(optimize)

def lr_generator(lr):
    original_rate = lr
    rate = lr
    max_deviation = 0.3
    change_rate = 0.1 * original_rate
    direction = 1
    while True:
        if rate == round(original_rate * (1 - max_deviation), 10):
            direction = 1
        elif rate == round(original_rate * (1+max_deviation), 10):
            direction = -1
        rate = rate + direction * change_rate
        rate = round(rate, 10)
        yield rate


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    keep_probability = 1.0
    lr_gen = lr_generator(0.001)

    # generator = get_batches_fn(batch_size)
    # image, label = next(generator)


    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for e in range(epochs):
        i = 0
        lr = next(lr_gen)
        print('=====running batch: {} with lr: {}'.format(e, lr))
        for image, label in get_batches_fn(batch_size):
        # generator = get_batches_fn(batch_size)
        # image, label = next(generator)
            feed_dict = {input_image: image,
                         correct_label: label,
                         keep_prob: keep_probability,
                         learning_rate: lr
                         }
            logits = tf.get_default_graph().get_tensor_by_name('logits:0')

            _, loss, out = sess.run([train_op, cross_entropy_loss, logits], feed_dict=feed_dict)

            i += 1
            if i % 5 == 0:
                print('loss:{}'.format(loss))

 

    # sess.run([train_op], {correct_label: np.arange(np.prod(shape)).reshape(shape), learning_rate: 10})
    # test, loss = sess.run([layers_output, cross_entropy_loss], {correct_label: np.arange(np.prod(shape)).reshape(shape)})

# tests.test_train_nn(train_nn)


def run():
    epochs = 50
    batch_size = 6
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes], name="correct_label")
        # learning_rate = tf.placeholder(tf.float32)
        # keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, optimizer, cross_entropy_loss = optimize(layers_output, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn,
                 optimizer, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate)
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    a = datetime.datetime.now()
    run()
    b = datetime.datetime.now()
    print('Time taken: {}'.format(b - a))
