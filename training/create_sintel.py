import numpy as np
from PIL import Image
import math
import os
from random import randint
import flowlib
import argparse

def getCorrespodingPatch_biLinInterpolate(i, j, u, v, halfPatchSize, img2):
    patchSize = 2 * halfPatchSize
    patch = np.zeros((patchSize, patchSize, img2.shape[2]))
    u_int = math.floor(u)
    eps_u = u - u_int
    v_int = math.floor(v)
    eps_v = v - v_int
    countOutOfImg2 = 0

    for ii in range(i - halfPatchSize, i + halfPatchSize):
        for jj in range(j - halfPatchSize, j + halfPatchSize):
            try:
                patch[ii - (i - halfPatchSize), jj - (j - halfPatchSize), :] = (1 - eps_u) * (1 - eps_v)\
                * img2[ ii + v_int, jj + u_int, :] + (eps_u) * (1 - eps_v) * img2[ ii + v_int + 1,
                jj + u_int, :] + (1 - eps_u) * (eps_v) * img2[ii + v_int, jj + u_int + 1,:] + \
                (eps_u) * (eps_v) * img2[ii + v_int + 1,jj + u_int + 1, :]
            except:
                print(ii + v_int + 1, jj + u_int + 1)
                countOutOfImg2 += 1

    patch = patch.astype(np.uint8)

    return patch

def makeDirAndRemoveExistingFiles(path):
    if not os.path.exists(path):
        os.makedirs(path)
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", default= "~/sintel/training/final",
                        help="Path of the Sintel image folders")
    parser.add_argument("flow_path", default= "~/sintel/training",
                        help="Path of the Sintel flow folders")
    parser.add_argument("output_path", default="~/training_samples",
                        help="Path of the output, which is the path of the samples already made from KITTI")
    parser.add_argument("patch_creation_option", default = "gaussian",
                        help= "Option for the creation of the non-matching patches")

    imgPathPrefix = parser.image_path
    flowPathPrefix = parser.flow_path
    patchSize = 16
    img_seq = ["/alley_1", "/alley_2",
               "/ambush_2", "/ambush_4", "/ambush_5", "/ambush_6",
               "/bamboo_1",
               "/bandage_1", "/bandage_2",
               "/cave_2", "/cave_4",
               "/market_2", "/market_5", "/market_6",
               "/mountain_1",
               "/shaman_2", "/shaman_3",
               "/sleeping_1",
               "/temple_2"]
    num_flows_per_seq = [49, 49, 20, 32, 49, 19, 49, 49, 49, 49, 49, 49, 49, 39, 49, 49, 49, 49, 49]
    path = parser.output_path
    DIR = path + "/original"
    num_imgs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    if(num_imgs == 0):
        print("Please run the file to make the training samples for KITTI first.")
        exit()

    randLowerBound = -8
    randUpperBound = 8
    patchesNum = num_imgs
    halfPatchSize = int(patchSize / 2)
    bad_counter = 0

    non_matching_creation_option = parser.patch_creation_option

    std = 2 # standard deviation for the offset of the non-matching patches
    min_shift = 2 # minimum distance of the non-matching patches to the reference patch

    for seq in range(len(img_seq)):
        for imgNum in range(1, num_flows_per_seq[seq] + 1):
            img1Path = imgPathPrefix + str(img_seq[seq]) + "/frame_" + "0" + str(imgNum).zfill(3) + ".png"
            img2Path = imgPathPrefix + str(img_seq[seq]) + "/frame_" + "0" + str(imgNum + 1).zfill(3) + ".png"
            flowPath = flowPathPrefix + "/flow" + str(img_seq[seq]) + "/frame_" + "0" + str(imgNum).zfill(3) + ".flo"
            flowPath_invalid = flowPathPrefix + "/invalid" + str(img_seq[seq]) + "/frame_" + "0" + str(imgNum).zfill(
                3) + ".png"
            flowPath_occ = flowPathPrefix + "/occlusions" + str(img_seq[seq]) + "/frame_" + "0" + str(imgNum).zfill(
                3) + ".png"
            img1 = flowlib.read_image(img1Path)
            img2 = flowlib.read_image(img2Path)

            height = img1.shape[0]
            width = img1.shape[1]
            flow = flowlib.read_flo_file(flowPath)
            u = flow[:, :, 0]
            v = flow[:, :, 1]
            invalid, _ = flowlib.read_image(flowPath_invalid)
            occ, _ = flowlib.read_image(flowPath_occ)

            for i in range(halfPatchSize, height - halfPatchSize, 15):
                for j in range(halfPatchSize, width - halfPatchSize, 15):

                    if invalid[i, j] == 0 and occ[i, j] == 0:

                        if (i + int(v[i, j] - 1) - halfPatchSize < 0 or i + int(v[i, j] + 1) + halfPatchSize >= height):
                            continue

                        if (j + int(u[i, j] - 1) - halfPatchSize < 0 or j + int(u[i, j] + 1) + halfPatchSize >= width):
                            continue

                        if(non_matching_creation_option == "gaussian"):
                            shift = np.abs((np.random.normal(0, std, 1))) + min_shift
                            sign = np.random.choice([1 , -1])
                            rand_i = sign * int(shift)
                            shift = np.abs((np.random.normal(0, std, 1))) + min_shift
                            sign = np.random.choice([1, -1])
                            rand_j = sign * int(shift)
                        else:
                            rand_i = randint(randLowerBound, randUpperBound)
                            rand_j = randint(randLowerBound, randUpperBound)
                            while (rand_j + j + int(u[i, j] + 1) + halfPatchSize >= width or rand_j + j + int(u[i, j] - 1) - halfPatchSize < 0 or
                                   rand_j == 0 or rand_j == 1 or rand_j == -1 or rand_j == -2 or rand_j == 2):
                                rand_j = randint(randLowerBound, randUpperBound)

                            while (rand_i + i + halfPatchSize + int(v[i, j] + 1) >= height or rand_i + i + int(v[i, j] - 1) - halfPatchSize < 0 or
                                   rand_i == 0 or rand_i == 1 or rand_i == -1 or rand_i == -2 or rand_i == 2):
                                rand_i = randint(randLowerBound, randUpperBound)

                        patchesNum += 1
                        patch1 = img1[i - halfPatchSize:i + halfPatchSize, j - halfPatchSize:j + halfPatchSize,:]
                        patch2 = getCorrespodingPatch_biLinInterpolate(i, j, u[i, j], v[i, j], halfPatchSize, img2)
                        patch3 = img2[(i + int(v[i, j]) + rand_i) - halfPatchSize:(i + rand_i + int(
                            v[i, j])) + halfPatchSize, (j + rand_j + int(u[i, j])) - halfPatchSize:(j + rand_j + int(
                            u[i, j])) + halfPatchSize, :]

                        # To discard the uniform samples:
                        sub_wrong = np.asarray(patch2, dtype=np.int32) - np.asarray(patch3, dtype=np.int32)
                        if (np.std(sub_wrong) <= 1.0):
                            bad_counter += 1
                            continue

                        patch1 = Image.fromarray(patch1)
                        patch1.save(path + "/original" + "/0" + str(patchesNum).zfill(8) + ".png")
                        patch2 = Image.fromarray(patch2)
                        patch2.save(path + "/correct" + "/0" + str(patchesNum).zfill(8) + ".png")
                        patch3 = Image.fromarray(patch3)
                        patch3.save(path + "/wrong" + "/0" + str(patchesNum).zfill(8) + ".png")

            print(str(seq) + "done out of" + str(len(img_seq)), "img num: ", imgNum)

    print(bad_counter)
