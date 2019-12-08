# -*- coding: utf-8 -*-

from __future__ import print_function
from random import randint
from keras.layers import Input
#from scipy.misc import imsave
from imageio import imwrite
from Backdoor.utils_tmp import *
from Backdoor.Model1 import Model1, generate_backdoor
from Backdoor.mnist_poison_generation import GenerateModel
from art.utils import load_mnist
from keract import get_activations
# from skimage.io import imread, imsave
import sys
import os
import time
import numpy as np
import pickle

# input image dimensions
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)

neuron_select_strategy = sys.argv[1]
threshold = float(sys.argv[2])
neuron_to_cover_num = int(sys.argv[3])
iteration_times = int(sys.argv[4])
backdoor_type = sys.argv[5]
train = False
source_target_desc = sys.argv[6]
sources = np.arange(10)
targets = (np.arange(10)+1)%10
model_name = "poisoned_mnist"+"_"+backdoor_type+"_"+source_target_desc
subdir = model_name+"_"+neuron_select_strategy+"_"+str(threshold)+"_"+str(neuron_to_cover_num)+"_"+str(iteration_times)

model = Model1(input_tensor=input_tensor,
               train=train,
               model_name=model_name,
               backdoor_type=backdoor_type,
               sources=sources,
               targets=targets
               ) #this works
# model = GenerateModel(backdoor_type="pattern", model_name=model_name, train=train)

print(model_name)

# model_layer_dict1 = init_coverage_tables(model)
model_layer_times1 = init_coverage_times(model)  # times of each neuron covered
model_layer_times2 = init_coverage_times(model)  # update when new image and adversarial images found
model_layer_value1 = init_coverage_value(model)

img_dir = './seeds'
img_paths = os.listdir(img_dir)
img_num = len(img_paths)

(x_raw, y_raw), (x_raw_test, y_raw_test), min_, max_ = load_mnist(raw=True)
# Poison training data
perc_poison = .33
(is_poison_train, x_poisoned_raw, y_poisoned_raw) = generate_backdoor(x_raw, y_raw, perc_poison, backdoor_type=backdoor_type)

difference_activation_block2_conv1 = []
difference_activation_block1_conv1 = []

if neuron_select_strategy == '[4]':
    x_raw_test = x_raw_test.reshape(x_raw_test.shape[0], 28, 28, 1)
    x_poisoned_raw = x_poisoned_raw.reshape(x_poisoned_raw.shape[0], 28, 28, 1)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    activations_clean_test = get_activations(model, x_raw_test, "block2_conv1")
    activation_rate_clean_test = np.mean(activations_clean_test['block2_conv1/Relu:0'] != 0, axis=(0, 3))
    activations_poisoned_train = get_activations(model, x_poisoned_raw, "block2_conv1")
    activation_rate_poisoned_train = np.mean(activations_poisoned_train['block2_conv1/Relu:0'] != 0, axis=(0, 3))
    difference_activation_block2_conv1 = np.subtract(activation_rate_poisoned_train, activation_rate_clean_test).clip(min=0)

    activations_clean_test = get_activations(model, x_raw_test, "block1_conv1")
    activation_rate_clean_test = np.mean(activations_clean_test['block1_conv1/Relu:0'] != 0, axis=(0, 3))
    activations_poisoned_train = get_activations(model, x_poisoned_raw, "block1_conv1")
    activation_rate_poisoned_train = np.mean(activations_poisoned_train['block1_conv1/Relu:0'] != 0, axis=(0, 3))
    difference_activation_block1_conv1 = np.subtract(activation_rate_poisoned_train, activation_rate_clean_test).clip(min=0)

filename = "activation_rate_clean_test"
outfile = open(filename,'wb')
pickle.dump(activation_rate_clean_test, outfile)
outfile.close()

filename = "activation_rate_poisoned_train"
outfile = open(filename,'wb')
pickle.dump(activation_rate_poisoned_train, outfile)
outfile.close()

neuron_to_cover_weight = 0.5
predict_weight = 0.5
learning_step = 0.02

save_dir = './generated_inputs/' + subdir + '/'

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# start = time.clock()
total_time = 0
total_norm = 0
adversial_num = 0
backdoor_activated = 0

total_perturb_adversial = 0

for i in range(img_num):

    start_time = time.clock()

    img_list = []

    img_path = os.path.join(img_dir,img_paths[i])

    img_name = img_paths[i].split('.')[0]

    mannual_label = int(img_name.split('_')[1])

    print(img_path)

    tmp_img = preprocess_image(img_path)

    orig_img = tmp_img.copy()

    img_list.append(tmp_img)

    update_coverage(tmp_img, model, model_layer_times2, threshold)

    while len(img_list) > 0:

        gen_img = img_list[0]

        img_list.remove(gen_img)

        # first check if input already induces differences
        pred1 = model.predict(gen_img)
        label1 = np.argmax(pred1[0])

        label_top5 = np.argsort(pred1[0])[-5:]

        update_coverage_value(gen_img, model, model_layer_value1)
        update_coverage(gen_img, model, model_layer_times1, threshold)

        orig_label = label1
        orig_pred = pred1

        loss_1 = K.mean(model.get_layer('before_softmax').output[..., orig_label])
        loss_2 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-2]])
        loss_3 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-3]])
        loss_4 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-4]])
        loss_5 = K.mean(model.get_layer('before_softmax').output[..., label_top5[-5]])

        layer_output = (predict_weight * (loss_2 + loss_3 + loss_4 + loss_5) - loss_1)
        print(layer_output)
        # neuron coverage loss
        loss_neuron = neuron_selection(model, model_layer_times1, model_layer_value1, neuron_select_strategy,
                                       neuron_to_cover_num, threshold,
                                       conv_layer_1_diff=difference_activation_block1_conv1,
                                       conv_layer_2_diff=difference_activation_block2_conv1)
        # loss_neuron = neuron_scale(loss_neuron) # useless, and negative result

        # extreme value means the activation value for a neuron can be as high as possible ...
        EXTREME_VALUE = False
        if EXTREME_VALUE:
            neuron_to_cover_weight = 2

        layer_output += neuron_to_cover_weight * K.sum(loss_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        grads_tensor_list = [loss_1, loss_2, loss_3, loss_4, loss_5]
        grads_tensor_list.extend(loss_neuron)
        grads_tensor_list.append(grads)
        # this function returns the loss and grads given the input picture

        iterate = K.function([input_tensor], grads_tensor_list)

        a, b = randint(0, 22), randint(0, 22)
        # a, b = 22, 22
        # a, b = 0, 0
        # we run gradient ascent for 3 steps
        for iters in range(iteration_times):

            loss_neuron_list = iterate([gen_img])

            # loss_neuron_list[-1] = constraint_occl(loss_neuron_list[-1], (a, b), (5, 5))

            perturb = loss_neuron_list[-1] * learning_step

            gen_img += perturb

            # previous accumulated neuron coverage
            previous_coverage = neuron_covered(model_layer_times1)[2]

            pred1 = model.predict(gen_img)
            label1 = np.argmax(pred1[0])

            update_coverage(gen_img, model, model_layer_times1, threshold) # for seed selection

            current_coverage = neuron_covered(model_layer_times1)[2]

            diff_img = gen_img - orig_img

            L2_norm = np.linalg.norm(diff_img)

            orig_L2_norm = np.linalg.norm(orig_img)

            perturb_adversial = L2_norm / orig_L2_norm

            if current_coverage - previous_coverage > 0.01 / (i + 1) and perturb_adversial < 0.02:
            # if perturb_adversial < 0.02:
                img_list.append(gen_img)
                # print('coverage diff = ', current_coverage - previous_coverage, 'perturb_adversial = ', perturb_adversial)

            if label1 != orig_label:
                update_coverage(gen_img, model, model_layer_times2, threshold)

                total_norm += L2_norm

                total_perturb_adversial += perturb_adversial

                # print('L2 norm : ' + str(L2_norm))
                # print('ratio perturb = ', perturb_adversial)

                gen_img_tmp = gen_img.copy()

                gen_img_deprocessed = deprocess_image(gen_img_tmp)

                save_img = save_dir + img_name + '_' + str(label1) + '_' + str(get_signature()) + '.png'
                
                
                #imsave(save_img, gen_img_deprocessed)
                imwrite(save_img, gen_img_deprocessed)

                adversial_num += 1

                if label1 == (orig_label + 1)%10:
                    backdoor_activated += 1


    end_time = time.clock()

    print('covered neurons percentage %d neurons %.3f'
          % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

    duration = end_time - start_time

    print('used time : ' + str(duration))

    total_time += duration

print('covered neurons percentage %d neurons %.3f'
      % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))

print('total_time = ' + str(total_time))
print('average_norm = ' + str(total_norm / adversial_num))
print('adversial num = ' + str(adversial_num))
print('average perb adversial = ' + str(total_perturb_adversial / adversial_num))
print('backdoor activation rate = ' + str(backdoor_activated / adversial_num))

results = open(save_dir+"{0}.txt".format("results"), "w")
results.write('covered neurons percentage %d neurons %.3f'
      % (len(model_layer_times2), neuron_covered(model_layer_times2)[2]))
results.write('\n average_norm = ' + str(total_norm / adversial_num))
results.write('\n adversial num = ' + str(adversial_num))
results.write('\n average perb adversial = ' + str(total_perturb_adversial / adversial_num))
results.write('\n backdoor activation rate = ' + str(backdoor_activated / adversial_num))
results.close()