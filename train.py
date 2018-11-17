from network import RDBG, discriminator
import tensorflow as tf
from PIL import Image
import numpy as np
import os

class Denoise():
    def __init__(self, batch_size, img_h, img_w, img_c, lambd, epoch, clean_path, noised_path, save_path, epsilon, learning_rate, beta1, beta2):
        self.batch_size = batch_size
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.epoch = epoch
        self.save_path = save_path
        self.clean_path = clean_path
        self.noised_path = noised_path
        self.img_clean = tf.placeholder(tf.float32, [None, None, None, img_c])
        self.img_noised = tf.placeholder(tf.float32, [None, None, None, img_c])
        G = RDBG("generator")
        D = discriminator("discriminator")
        self.img_denoised = G(self.img_noised)
        self.logit_real = D(self.img_clean, self.img_noised)
        self.logit_fake = D(self.img_denoised, self.img_noised)
        self.d_loss = -tf.reduce_mean(tf.log(self.logit_real + epsilon)) - tf.reduce_mean(tf.log(1 - self.logit_fake + epsilon))
        self.l1_loss = tf.reduce_mean(tf.abs(self.img_denoised - self.img_clean))
        self.g_loss = -tf.reduce_mean(tf.log(self.logit_fake + epsilon)) + lambd * self.l1_loss
        self.Opt_D = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(self.d_loss, var_list=D.var())
        self.Opt_G = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2).minimize(self.g_loss, var_list=G.var())
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        clean_names = os.listdir(self.clean_path)
        saver = tf.train.Saver()
        step = 0
        for epoch in range(self.epoch):
            for i in range(clean_names.__len__()//self.batch_size):
                step = step + 1
                batch_clean = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_c])
                batch_noised = np.zeros([self.batch_size, self.img_h, self.img_w, self.img_c])
                for idx, name in enumerate(clean_names[i*self.batch_size:i*self.batch_size+self.batch_size]):
                    batch_clean[idx, :, :, 0] = np.array(Image.open(self.clean_path+name).resize([self.img_h, self.img_w]))
                    batch_noised[idx, :, :, 0] = np.array(Image.open(self.noised_path + name[:-3]+"png").resize([self.img_h, self.img_w]))
                self.sess.run(self.Opt_D, feed_dict={self.img_clean: batch_clean, self.img_noised: batch_noised})
                self.sess.run(self.Opt_G, feed_dict={self.img_clean: batch_clean, self.img_noised: batch_noised})
                if step % 10 == 0:
                    [d_loss, g_loss, l1_loss, denoised] = self.sess.run([self.d_loss, self.g_loss, self.l1_loss, self.img_denoised],
                                                                        feed_dict={self.img_clean: batch_clean, self.img_noised: batch_noised})
                    print("Step: %d, D_loss: %g, G_loss: %g, L1_loss: %g"%(step, d_loss, g_loss, l1_loss))
                    Image.fromarray(np.uint8(denoised[0, :, :, 0])).save("./results//"+str(step)+".jpg")
            saver.save(self.sess, self.save_path + "model.ckpt")

    def test(self, test_path, para_path):
        saver = tf.train.Saver()
        saver.restore(self.sess, para_path+".\\model.ckpt")
        img = np.float32(np.array(Image.open(test_path).convert("1")))*255
        [denoised] = self.sess.run([self.img_denoised],feed_dict={self.img_noised: img[np.newaxis, :, :, np.newaxis]})
        Image.fromarray(np.uint8(np.concatenate((img, denoised[0, :, :, 0]), axis=1))).show()

