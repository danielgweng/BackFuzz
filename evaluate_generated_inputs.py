import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from PIL import Image

imgs_list = {
    0: [],
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    6: [],
    7: [],
    8: [],
    9: []
}
seed_imgs = {}

test_dir = "poisoned_mnist_pixel_next_number_[1]_0.5_3_5_10x10"

inputs_path = "./generated_inputs/" + test_dir
seeds_path = "./seeds"
valid_images = [".png"]

total_perturbation = 0
total_perturbation_activations = 0
total_num_images = 0

save_dir = "./perturbations/"+test_dir+"/"

if os.path.exists(save_dir):
    for i in os.listdir(save_dir):
        path_file = os.path.join(save_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for f in os.listdir(seeds_path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue

    underscore_index = f.find("_")
    seed = int(f[0:underscore_index - 1])
    seed_imgs[seed] = imread(os.path.join(seeds_path, f))

for f in os.listdir(inputs_path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue

    underscore_index = f.find("_")
    num = int(f[underscore_index + 1])
    seed = int(f[0:underscore_index - 1])

    image = imread(os.path.join(inputs_path, f))
    seed_image = seed_imgs[seed]
    # perturbation = np.subtract(image, seed_image)
    # perturbation = image - seed_image
    perturbation = cv2.subtract(image, seed_image)

    # fig = plt.figure(figsize=(20, 8))
    # columns = 3
    # rows = 1
    # fig.add_subplot(rows, columns, 1)
    # plt.imshow(image)
    # fig.add_subplot(rows, columns, 2)
    # plt.imshow(seed_image)
    # fig.add_subplot(rows, columns, 3)
    # plt.imshow(perturbation)
    # plt.show()

    imgs_list[num].append(perturbation)
    imsave(save_dir+f, perturbation)

    total_perturbation += perturbation
    total_perturbation_activations += (perturbation != 0)
    total_num_images += 1

avg_perturbation = total_perturbation / float(total_num_images)
avg_perturbation_activations = total_perturbation_activations / float(total_num_images)
avg_activation_rate = np.mean(avg_perturbation)

width, height = avg_perturbation_activations.shape
distance = 2
if "pixel" in test_dir:
    backdoor_activation_rate = avg_perturbation_activations[width-distance, height-2]
else:
    backdoor_activation_rate = (avg_perturbation_activations[width-distance, height-distance]
                                + avg_perturbation_activations[width-distance-1, height-distance-1]
                                + avg_perturbation_activations[width-distance, height-distance-2]
                                + avg_perturbation_activations[width-distance-2, height-distance])/4


imsave(save_dir+"avg_perturbation.png", avg_perturbation.astype(np.uint8))
imsave(save_dir+"perturbation_activation_rate.png", (avg_perturbation_activations*255).astype(np.uint8))
# im = Image.fromarray(avg_perturbation_activations)
# im.save(save_dir+"perturbation_activation_rate.png")

results = open(save_dir+"{0}.txt".format("results"), "w")
results.write("backdoor pixel activation rate is {0}".format(str(backdoor_activation_rate)))
results.write("\navg perturbation pixel activation rate is {0}".format(str(avg_activation_rate)))
results.close()
