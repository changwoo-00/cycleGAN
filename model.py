from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *

import cv2

class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet128
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        # test
        #vars_to_restore=[v for v in tf.global_variables() if "generatorB2A" in v.name]
        #self.saver = tf.train.Saver(vars_to_restore, max_to_keep=100)
        #
        self.saver = tf.train.Saver(max_to_keep=100)
        self.pool = ImagePool(args.max_size)

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.same_A = self.generator(self.real_A, self.options, True, name="generatorA2B")  # hcw used for identity loss
        self.same_B = self.generator(self.real_B, self.options, True, name="generatorB2A")  # hcw used for identity loss

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")
        
        ## Generator Loss
        ### _ = GAN loss + cycle loss(A->B->A) + cycle loss(B->A->B) ( + identity loss)
        self.g_loss_a2b = self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + 60 * abs_criterion(self.real_A, self.same_A)         # hcw identity loss

        ### _ = GAN loss + cycle loss(A->B->A) + cycle loss(B->A->B) ( + identity loss)
        self.g_loss_b2a = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + 60 * abs_criterion(self.real_B, self.same_B)         # hcw identity loss

        ### 
        self.g_loss = self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake)) \
            + self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake)) \
            + self.L1_lambda * abs_criterion(self.real_A, self.fake_A_) \
            + self.L1_lambda * abs_criterion(self.real_B, self.fake_B_) \
            + 60 * abs_criterion(self.real_A, self.same_A) + 60 * abs_criterion(self.real_B, self.same_B)         # hcw identity loss
        ##

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        ## Discriminator Loss
        self.db_loss_real = self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = self.criterionGAN(self.DB_fake_sample, tf.zeros_like(self.DB_fake_sample))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2
        self.da_loss_real = self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
        self.da_loss_fake = self.criterionGAN(self.DA_fake_sample, tf.zeros_like(self.DA_fake_sample))
        self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2
        self.d_loss = self.da_loss + self.db_loss
        ##

        self.g_loss_a2b_sum = tf.summary.scalar("g_loss_a2b", self.g_loss_a2b)
        self.g_loss_b2a_sum = tf.summary.scalar("g_loss_b2a", self.g_loss_b2a)
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.g_sum = tf.summary.merge([self.g_loss_a2b_sum, self.g_loss_b2a_sum, self.g_loss_sum])
        self.db_loss_sum = tf.summary.scalar("db_loss", self.db_loss)
        self.da_loss_sum = tf.summary.scalar("da_loss", self.da_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        self.db_loss_real_sum = tf.summary.scalar("db_loss_real", self.db_loss_real)
        self.db_loss_fake_sum = tf.summary.scalar("db_loss_fake", self.db_loss_fake)
        self.da_loss_real_sum = tf.summary.scalar("da_loss_real", self.da_loss_real)
        self.da_loss_fake_sum = tf.summary.scalar("da_loss_fake", self.da_loss_fake)
        self.d_sum = tf.summary.merge(
            [self.da_loss_sum, self.da_loss_real_sum, self.da_loss_fake_sum,
             self.db_loss_sum, self.db_loss_real_sum, self.db_loss_fake_sum,
             self.d_loss_sum]
        )

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def train(self, args):
        """Train cyclegan"""
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.d_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if args.continue_train:
            if self.load(args.checkpoint_dir): 
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

        for epoch in range(args.epoch):
            
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            #lr_s = tf.summary.scalar("learning_rate", lr)
            #self.writer.add_summary(lr_s, epoch)
            #lr = args.lr if epoch < args.epoch_step else args.lr*(300-epoch)/(300-args.epoch_step)
            #if epoch > 295:
            #    lr = args.lr if epoch < args.epoch_step else args.lr*(300-295)/(300-args.epoch_step)
            
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            #dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            dataB1 = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB/1'))
            dataB2 = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB/2'))
            dataB3 = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB/3'))
            dataB4 = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB/4'))
            dataB5 = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB/5'))
            dataB = []
            print(type(dataB1))
            np.random.shuffle(dataA)
            #np.random.shuffle(dataB)
            np.random.shuffle(dataB1)
            np.random.shuffle(dataB2)
            np.random.shuffle(dataB3)
            np.random.shuffle(dataB4)
            np.random.shuffle(dataB5)
            
            for i in range(len(dataB1)+len(dataB2)+len(dataB3)+len(dataB4)+len(dataB5)):
                r_i = np.random.randint(100)
                if r_i < 25:
                    dataB.append(dataB1[-1])
                    dataB1 = np.roll(dataB1,1)
                elif r_i < 50:
                    dataB.append(dataB2[-1])
                    dataB2 = np.roll(dataB2,1)
                elif r_i < 60:
                    dataB.append(dataB3[-1])
                    dataB3 = np.roll(dataB3,1)
                elif r_i < 90:
                    dataB.append(dataB4[-1])
                    dataB4 = np.roll(dataB4,1)
                else:
                    dataB.append(dataB5[-1])
                    dataB5 = np.roll(dataB5,1)

            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            
            #print(dataB)

            for idx in range(0, batch_idxs):

                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))

                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size, channel=args.input_nc) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                #if np.random.random() < 0.2:
                #    batch_images = add_sp_noise(batch_images, 's&p')
                #if np.random.random() < 0.2:
                #    batch_images = add_sp_noise(batch_images, 'gaussian')
                #if np.random.random() < 0.2:
                #    batch_images = add_black_square(batch_images)


                # Update G network and record fake outputs
                fake_A, fake_B, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.g_optim, self.g_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update D network
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                _, summary_str = self.sess.run(
                    [self.d_optim, self.d_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f" % (
                    epoch, idx, batch_idxs, time.time() - start_time)))

                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args, args.sample_dir, epoch, idx) # hcw added args

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name = "cyclegan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt)
            ckpt_paths = ckpt.all_model_checkpoint_paths    #hcw
            print(ckpt_paths)
            #ckpt_name = os.path.basename(ckpt_paths[-1])    #hcw # default [-1]
            #500ep : 'cyclegan.model-174002'
            #600ep : 'cyclegan.model-208002'
            #700ep : 'cyclegan.model-244002'
            #800ep : 'cyclegan.model-278002'

            #600ep : 'cyclegan.model-214002'
            #800ep : 'cyclegan.model-190002' , cyclegan.model-286002
            #1000ep : 'cyclegan.model-358002'
            temp_ckpt = 'cyclegan.model-174002'
            ckpt_name = os.path.basename(temp_ckpt)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self, args, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA')) 
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB')) 
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, args.load_size, args.fine_size, channel=args.input_nc) for batch_file in batch_files] # hcw load_train_data(batch_file, is_testing=True)
        sample_images = np.array(sample_images).astype(np.float32)

        #if np.random.random() < 0.2:
        #    sample_images = add_sp_noise(sample_images, 's&p')
        #if np.random.random() < 0.2:
        #    sample_images = add_sp_noise(sample_images, 'gaussian')
        #if np.random.random() < 0.2:
        #    sample_images = add_black_square(sample_images)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],

            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.bmp'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.bmp'.format(sample_dir, epoch, idx))

    def test(self, args):
        """Test cyclegan"""

        thresh_val_list = [25, 30, 35, 40, 45]
        #thresh_val_list = [35]
        #gamma_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        gamma_list = [1]
        #thresh_val_list = [25, 30]

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA')) # default /testA
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA')) # default /testB
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for thresh_val in thresh_val_list:
            for gamma in gamma_list:
                # write html for visual comparison
                index_path = os.path.join(args.test_dir, '{0}_{1}_{2}_index.html'.format(args.which_direction,thresh_val,gamma))
                index = open(index_path, "w")
                index.write("<html><body><table><tr>")
                #index.write("<th>name</th><th>input</th><th>output</th><th>difference</th><th>threshold {}</th><th>gamma {}</th></tr>".format(thresh_val, gamma))
                index.write("<th>name</th><th>input</th><th>output</th><th>difference</th><th>gamma{} threshold{} AND</th><th>gamma{} threshold{}</th><th>gamma{} threshold{}</th><th>diff cnt</th><th>threshold cnt</th></tr>".format(gamma, thresh_val,gamma, thresh_val,gamma, thresh_val))

                out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
                    self.testA, self.test_B)
                cnt = 0


                if not os.path.exists(os.path.join(args.test_dir,'input')):
                    os.mkdir(os.path.join(args.test_dir,'input'))
                if not os.path.exists(os.path.join(args.test_dir,'output')):
                    os.mkdir(os.path.join(args.test_dir,'output'))
                if not os.path.exists(os.path.join(args.test_dir,'difference')):
                    os.mkdir(os.path.join(args.test_dir,'difference'))
                #if not os.path.exists(os.path.join(args.test_dir,'threshold_'+str(thresh_val))):
                #    os.mkdir(os.path.join(args.test_dir,'threshold_'+str(thresh_val)))
                #if not os.path.exists(os.path.join(args.test_dir,'threshold_blur_'+str(thresh_val))):
                #    os.mkdir(os.path.join(args.test_dir,'threshold_blur_'+str(thresh_val)))
                if not os.path.exists(os.path.join(args.test_dir,'gamma_'+str(gamma)+'_threshold_'+str(thresh_val))):
                    os.mkdir(os.path.join(args.test_dir,'gamma_'+str(gamma)+'_threshold_'+str(thresh_val)))
                if not os.path.exists(os.path.join(args.test_dir,'shift_gamma_'+str(gamma)+'_threshold_'+str(thresh_val))):
                    os.mkdir(os.path.join(args.test_dir,'shift_gamma_'+str(gamma)+'_threshold_'+str(thresh_val)))
                if not os.path.exists(os.path.join(args.test_dir,'and_gamma_'+str(gamma)+'_threshold_'+str(thresh_val))):
                    os.mkdir(os.path.join(args.test_dir,'and_gamma_'+str(gamma)+'_threshold_'+str(thresh_val)))


                for input_file in sample_files:
                    print(input_file)
                    #if os.path.basename(input_file) != 'F7552F19C804710110001010_01normal.bmp':
                    #    continue

                    fake_file = os.path.basename(input_file)[:-4]+'_fake.bmp'
                    diff_file = os.path.basename(input_file)[:-4]+'_diff.bmp'
                    thresh_file = os.path.basename(input_file)[:-4]+'_thresh.bmp'
                    thresh_blur_file = os.path.basename(input_file)[:-4]+'_thresh_blur.bmp'
                    thresh_gamma_file = os.path.basename(input_file)[:-4]+'_gamma.bmp'

                    if cnt == 1:
                        start_time = time.time()
                    print('Processing image: ' + input_file)
                    sample_image = [load_test_data(input_file, args.load_size, args.fine_size, channel=args.input_nc),load_test_data2(input_file, args.load_size, args.fine_size, channel=args.input_nc)]
                    
                    sample_image = np.array(sample_image).astype(np.float32)

                    if args.input_nc!=3:
                        sample_image = np.expand_dims(sample_image, axis=3) # hcw

            
                    input_img_path = os.path.join(args.test_dir,'input',
                                              '{0}_{1}'.format(args.which_direction, os.path.basename(input_file)))
                    fake_img_path = os.path.join(args.test_dir,'output',
                                              '{0}_{1}'.format(args.which_direction, fake_file))
                    diff_img_path = os.path.join(args.test_dir,'difference',
                                              '{0}_{1}'.format(args.which_direction, diff_file))
                    #thresh_img_path = os.path.join(args.test_dir,'threshold_'+str(thresh_val),
                    #                          '{0}_{1}'.format(args.which_direction, thresh_file))
                    thresh_img_blur_path = os.path.join(args.test_dir,'threshold_blur_'+str(thresh_val),
                                              '{0}_{1}'.format(args.which_direction, thresh_blur_file))
                    thresh_img_gamma_path = os.path.join(args.test_dir,'gamma_'+str(gamma)+'_threshold_'+str(thresh_val),
                                              '{0}_{1}'.format(args.which_direction, thresh_gamma_file))
                    thresh_img_gamma_path_shift = os.path.join(args.test_dir,'shift_gamma_'+str(gamma)+'_threshold_'+str(thresh_val),
                                              '{0}_{1}'.format(args.which_direction, thresh_gamma_file))
                    thresh_img_gamma_path_and = os.path.join(args.test_dir,'and_gamma_'+str(gamma)+'_threshold_'+str(thresh_val),
                                              '{0}_{1}'.format(args.which_direction, thresh_gamma_file))
            

                    fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})


                    sample_image_r = np.array(restore_uint(sample_image)).astype(np.uint8)
                    fake_img_r = np.array(restore_uint(fake_img)).astype(np.uint8)

                    ## option : histogram shift ##

                    hist_table = hist_shift_table(sample_image_r[0,:,:,0])
                    sample_image_p = cv2.LUT(sample_image_r[0,:,:,0], hist_table)
                    hist_table = hist_shift_table(fake_img_r[0,:,:,0])
                    fake_img_p = cv2.LUT(fake_img_r[0,:,:,0], hist_table)

                    hist_table = hist_shift_table(sample_image_r[1,:,:,0])
                    sample_image_p2 = cv2.LUT(sample_image_r[1,:,:,0], hist_table)
                    hist_table = hist_shift_table(fake_img_r[1,:,:,0])
                    fake_img_p2 = cv2.LUT(fake_img_r[1,:,:,0], hist_table)

                    ####

                
                    diff_img = cv2.absdiff(sample_image_p, fake_img_p)
                    diff_img2 = cv2.absdiff(sample_image_p2, fake_img_p2)
            
                    _, thresh_img = cv2.threshold(diff_img, thresh_val, 255, cv2.THRESH_BINARY)
                    _, thresh_img2 = cv2.threshold(diff_img2, thresh_val, 255, cv2.THRESH_BINARY)

                    thresh_img_and = np.zeros(thresh_img.shape, dtype=np.uint8)
                    thresh_img_and[:-30,:-30] = cv2.bitwise_and(thresh_img2[30:,30:], thresh_img[:-30,:-30])

                    ## temp test
                    #sample_image_p_blur = cv2.blur(sample_image_p[0,:,:,0], (2,2))
                    #fake_img_p_blur = cv2.blur(fake_img_p[0,:,:,0], (2,2))
                    #diff_img_blur = cv2.absdiff(sample_image_p_blur, fake_img_p_blur)
                    #_, thresh_img_blur = cv2.threshold(diff_img_blur, thresh_val, 255, cv2.THRESH_BINARY)

                                                       # change the value here to get different result
                    #sample_image_gamma = self.adjust_gamma(sample_image_p, gamma=gamma)
                    #fake_img_gamma = self.adjust_gamma(fake_img_p, gamma=gamma)
                    #diff_img_gamma = cv2.absdiff(sample_image_gamma, fake_img_gamma)
                    #_, thresh_img_gamma = cv2.threshold(diff_img_gamma, thresh_val, 255, cv2.THRESH_BINARY)

                    ## temp erode&dilate ###
                    kernel = np.ones((3,3),dtype=np.uint8)
                    #thresh_img = cv2.erode(thresh_img, kernel)
                    #thresh_img = cv2.dilate(thresh_img, kernel)
                    #thresh_img = cv2.erode(thresh_img, np.ones((3,1),dtype=np.uint8))
                    #thresh_img = cv2.dilate(thresh_img, np.ones((3,1),dtype=np.uint8))

                    #thresh_img = cv2.erode(thresh_img, np.ones((1,3),dtype=np.uint8))
                    #thresh_img = cv2.dilate(thresh_img, np.ones((1,3),dtype=np.uint8))

                    pixel_cnt = np.sum(thresh_img_and)/255
                    diff_cnt = np.sum(thresh_img_and)
                    ##

            
                    #save_images(sample_image, [1, 1], input_img_path)   # save input image # image_path_i : image path in test_folder
                    #save_images(fake_img, [1, 1], fake_img_path)     # save fake image # fake_img_path : image path in test folder
                    
                    cv2.imwrite(input_img_path, sample_image_r[0,:,:,0])   # save input image # image_path_i : image path in test_folder
                    cv2.imwrite(fake_img_path, fake_img_r[0,:,:,0])     # save fake image # fake_img_path : image path in test folder
                    
                    
                    cv2.imwrite(diff_img_path, diff_img)
                    #cv2.imwrite(thresh_img_path, thresh_img)
                    #cv2.imwrite(thresh_img_blur_path, thresh_img_blur)
                    cv2.imwrite(thresh_img_gamma_path, thresh_img)
                    cv2.imwrite(thresh_img_gamma_path_shift, thresh_img2)
                    cv2.imwrite(thresh_img_gamma_path_and, thresh_img_and)

            


                    index.write("<td>%s</td>" % os.path.basename(input_img_path))

                    # save input, fake image as html
                    index.write("<td><img src='%s'></td>" % ('input/'+os.path.basename(input_img_path)))
                    index.write("<td><img src='%s'></td>" % ('output/'+os.path.basename(fake_img_path)))
                    index.write("<td><img src='%s'></td>" % ('difference/'+os.path.basename(diff_img_path)))
                    #index.write("<td><img src='%s'></td>" % ('threshold_'+str(thresh_val)+'/'+os.path.basename(thresh_img_path)))
                    #index.write("<td><img src='%s'></td>" % ('threshold_blur_'+str(thresh_val)+'/'+os.path.basename(thresh_img_blur_path)))
                    index.write("<td><img src='%s'></td>" % ('and_gamma_'+str(gamma)+'_threshold_'+str(thresh_val)+'/'+os.path.basename(thresh_img_gamma_path_and)))
                    index.write("<td><img src='%s'></td>" % ('gamma_'+str(gamma)+'_threshold_'+str(thresh_val)+'/'+os.path.basename(thresh_img_gamma_path)))
                    index.write("<td><img src='%s'></td>" % ('shift_gamma_'+str(gamma)+'_threshold_'+str(thresh_val)+'/'+os.path.basename(thresh_img_gamma_path_shift)))
                    index.write("<td>'%s'</td>" % (diff_cnt))
                    index.write("<td>'%s'</td>" % (pixel_cnt))

                    index.write("</tr>")
                    cnt = cnt+1
                index.close()

                print(time.time() - start_time)


    
    def test_d(self, args):
        """Test cyclegan discriminator"""

        thresh_val = 40

        start_time = time.time()
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_{1}_indexn.html'.format(args.which_direction,thresh_val))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th><th>difference</th><th>threshold {}</th></tr>".format(thresh_val))

        out_var, in_var = (self.testB, self.test_A) if args.which_direction == 'AtoB' else (
            self.testA, self.test_B)
        #out_var_d, in_var_d  = (self.DA_real, self.real_A)# if args.which_direction == 'AtoB' else (self.DB_real, self.real_B)
        out_var_d, in_var_d  = (self.DB_real, self.real_B)# if args.which_direction == 'AtoB' else (self.DB_real, self.real_B)
        cnt = 0


        if not os.path.exists(os.path.join(args.test_dir,'input')):
            os.mkdir(os.path.join(args.test_dir,'input'))
        if not os.path.exists(os.path.join(args.test_dir,'output')):
            os.mkdir(os.path.join(args.test_dir,'output'))
        if not os.path.exists(os.path.join(args.test_dir,'difference')):
            os.mkdir(os.path.join(args.test_dir,'difference'))
        if not os.path.exists(os.path.join(args.test_dir,'threshold_'+str(thresh_val))):
            os.mkdir(os.path.join(args.test_dir,'threshold_'+str(thresh_val)))


        for input_file in sample_files:

            fake_file = os.path.basename(input_file)[:-4]+'_fake.bmp'
            diff_file = os.path.basename(input_file)[:-4]+'_diff.bmp'
            thresh_file = os.path.basename(input_file)[:-4]+'_thresh.bmp'

            if cnt == 1:
                start_time = time.time()
            print('Processing image: ' + input_file)
            sample_image = [load_test_data(input_file, args.load_size, args.fine_size, channel=args.input_nc)]
            sample_image = np.array(sample_image).astype(np.float32)

            if args.input_nc!=3:
                sample_image = np.expand_dims(sample_image, axis=3) # hcw
            
            input_img_path = os.path.join(args.test_dir,'input',
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(input_file)))
            fake_img_path = os.path.join(args.test_dir,'output',
                                      '{0}_{1}'.format(args.which_direction, fake_file))
            diff_img_path = os.path.join(args.test_dir,'difference',
                                      '{0}_{1}'.format(args.which_direction, diff_file))
            thresh_img_path = os.path.join(args.test_dir,'threshold_'+str(thresh_val),
                                      '{0}_{1}'.format(args.which_direction, thresh_file))
            
            


            #fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            d_output = self.sess.run(out_var_d, feed_dict={in_var_d: sample_image})

            #d_output_avg = self.criterionGAN(d_output, tf.ones_like(d_output))

            #sample_image_p = np.array(restore_uint(sample_image)).astype(np.uint8)
            #fake_img_p = np.array(restore_uint(fake_img)).astype(np.uint8)
            #diff_img = cv2.absdiff(sample_image_p[0,:,:,0], fake_img_p[0,:,:,0])
            
            #_, thresh_img = cv2.threshold(diff_img, thresh_val, 255, cv2.THRESH_BINARY)

            
            save_images(sample_image, [1, 1], input_img_path)   # save input image # image_path_i : image path in test_folder
            #save_images(fake_img, [1, 1], fake_img_path)     # save fake image # fake_img_path : image path in test folder
            #cv2.imwrite(diff_img_path, diff_img)
            #cv2.imwrite(thresh_img_path, thresh_img)



            index.write("<td>%s</td>" % os.path.basename(input_img_path))

            # save input, fake image as html
            index.write("<td><img src='%s'></td>" % ('input/'+os.path.basename(input_img_path)))
            index.write("<td>%s</td>" % str(d_output))
            index.write("<td>%s</td>" % str(np.sum(d_output)))
            index.write("<td>%s</td>" % str(np.mean(d_output)))
            index.write("<td>%s</td>" % str(np.min(d_output)))
            index.write("<td>%s</td>" % str(np.min(np.abs(d_output))))
            #index.write("<td>%s</td>" % str(d_output_avg.eval()))
            #index.write("<td><img src='%s'></td>" % ('output/'+os.path.basename(fake_img_path)))
            #index.write("<td><img src='%s'></td>" % ('difference/'+os.path.basename(diff_img_path)))
            #index.write("<td><img src='%s'></td>" % ('threshold_'+str(thresh_val)+'/'+os.path.basename(thresh_img_path)))
            index.write("</tr>")
            cnt = cnt+1
        index.close()

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        lut_out = cv2.LUT(image, table)
        return lut_out