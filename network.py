from ops import *

def RDB(inputs):
    G = 32
    with tf.variable_scope("conv1"):
        Fd_1 = tf.nn.relu(InstanceNorm(conv("conv1", inputs, G, 3, 1), "IN1"))
        temp = tf.concat([inputs, Fd_1], axis=3)
    with tf.variable_scope("conv2"):
        Fd_2 = tf.nn.relu(InstanceNorm(conv("conv2", temp, G, 3, 1), "IN2"))
        temp = tf.concat([inputs, Fd_1, Fd_2], axis=3)
    with tf.variable_scope("conv3"):
        Fd_3 = tf.nn.relu(InstanceNorm(conv("conv3", temp, G, 3, 1), "IN3"))
        temp = tf.concat([inputs, Fd_1, Fd_2, Fd_3], axis=3)
    with tf.variable_scope("conv4"):
        Fd_4 = tf.nn.relu(InstanceNorm(conv("conv4", temp, G, 3, 1), "IN4"))
        temp = tf.concat([inputs, Fd_1, Fd_2, Fd_3, Fd_4], axis=3)
    with tf.variable_scope("conv5"):
        Fd_5 = tf.nn.relu(InstanceNorm(conv("conv5", temp, G, 3, 1), "IN5"))
        temp = tf.concat([inputs, Fd_1, Fd_2, Fd_3, Fd_4, Fd_5], axis=3)
    with tf.variable_scope("conv6"):
        Fd_6 = tf.nn.relu(InstanceNorm(conv("conv6", temp, G, 3, 1), "IN6"))
    Fd_LF = (conv("conv7", tf.concat([inputs, Fd_1, Fd_2, Fd_3, Fd_4, Fd_5, Fd_6], axis=3), int(inputs.shape[-1]), 1, 1))
    return Fd_LF + inputs

class RDBG:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs_condition):
        inputs = inputs_condition
        with tf.variable_scope("generator"):
            with tf.variable_scope("c7s1-32"):
                inputs = tf.nn.relu((conv("conv", inputs, 32, 7, 1)))
            with tf.variable_scope("c5s2-64"):
                inputs = tf.nn.relu(tf.nn.dropout(InstanceNorm(conv("conv", inputs, 64, 5, 2), "IN1"), 0.5))
            for i in range(6):
                with tf.variable_scope("RDB" + str(i)):
                    inputs = RDB(inputs)
            with tf.variable_scope("c5s2-32"):
                inputs = tf.nn.relu(tf.nn.dropout(InstanceNorm(uconv("conv", inputs, 32, 5, 2), "IN2"), 0.5))
            with tf.variable_scope("c7s1-1"):
                inputs = tf.nn.tanh((uconv("conv", inputs, 1, 7, 1)))
            return (inputs + 1)*127.5

    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

class discriminator:
    def __init__(self, name):
        self.name = name

    def __call__(self, inputs, inputs_condition):
        inputs = tf.concat([inputs, inputs_condition], axis=3)
        inputs = tf.random_crop(inputs, [1, 70, 70, 2])
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("conv1"):
                inputs = leaky_relu(conv("conv1", inputs, 64, 5, 2))
            with tf.variable_scope("conv2"):
                inputs = leaky_relu(InstanceNorm(conv("conv2", inputs, 128, 5, 2), "IN1"))
            with tf.variable_scope("conv3"):
                inputs = leaky_relu(InstanceNorm(conv("conv3", inputs, 256, 5, 2), "IN2"))
            with tf.variable_scope("conv4"):
                inputs = leaky_relu(InstanceNorm(conv("conv4", inputs, 512, 5, 2), "IN3"))
            return tf.nn.sigmoid(conv("out", inputs, 1, int(inputs.shape[1]), 1, "VALID"))

    def var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)

