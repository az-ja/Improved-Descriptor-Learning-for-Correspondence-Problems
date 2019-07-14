import numpy as np
from PIL import Image
import math
import flowlib
import os
from random import randint
import argparse

def getCorrespodingPatch_biLinInterpolate(i,j,u,v,halfPatchSize,img2):
    patchSize=2*halfPatchSize
    patch=np.zeros((patchSize,patchSize,img2.shape[2]))
    u_int=math.floor(u)
    eps_u = u - u_int
    v_int=math.floor(v)
    eps_v= v-v_int
    countOutOfImg2=0

    for ii in range(i-halfPatchSize,i+halfPatchSize):
        for jj in range(j-halfPatchSize,j+halfPatchSize):
            try:
                patch[ii-(i-halfPatchSize),jj-(j-halfPatchSize),:]=  (1-eps_u)*(1-eps_v)*img2[ii+v_int,jj+u_int , :] + \
                                (eps_u)*(1-eps_v)*img2[ii+v_int +1 ,jj+u_int , :] + \
                                (1 - eps_u) * ( eps_v) * img2[ii+v_int, jj+u_int +1, :] + \
                                (eps_u) * (eps_v) * img2[ii+v_int +1 , jj+u_int +1, :]
            except :
                print("out of range: " ,ii+v_int+1 , jj+u_int+1)
                countOutOfImg2+=1

    patch=patch.astype(np.uint8)

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
    parser.add_argument("image_path", default="~/kitti_2015/data_scene_flow/training/image_2",
                        help="Path of the KITTI images")
    parser.add_argument("flow_path", default="~/kitti_2015/data_scene_flow/training/flow_noc",
                        help="Path of the KITTI flow files")
    parser.add_argument("output_path", default="~/training_samples",
                        help="Path of the output training samples")
    parser.add_argument("patch_creation_option", default="gaussian",
                        help="Option for the creation of the non-matching patches")

    imgPathPrefix = "/home/azin/kitti_2015/data_scene_flow/training/image_2"
    flowPathPrefix="/home/azin/kitti_2015/data_scene_flow/training/flow_noc"
    patchSize= 16
    outputDirectory = "/home/azin/sample_test/"
    original= outputDirectory + "original"
    correct= outputDirectory + "correct"
    wrong= outputDirectory + "wrong"

    makeDirAndRemoveExistingFiles(original)
    makeDirAndRemoveExistingFiles(correct)
    makeDirAndRemoveExistingFiles(wrong)
    randLowerBound=-8
    randUpperBound=8
    patchesNum = 0
    halfPatchSize = int(patchSize / 2)

    non_matching_creation_option = parser.patch_creation_option

    std = 2  # standard deviation for the offset of the non-matching patches
    min_shift = 2  # minimum distance of the non-matching patches to the reference patch

    #from 180 till 199 (in range (180,200)) for testing. 160 to 180 for validation
    bad_counter = 0
    for imgNum in range(160):
        print(imgNum)
        img1Path=imgPathPrefix+"/0"+str(imgNum).zfill(5)+"_10.png"
        img2Path=imgPathPrefix+"/0"+str(imgNum).zfill(5)+"_11.png"
        flowPath=flowPathPrefix+"/0"+str(imgNum).zfill(5)+"_10.png"

        img1 = flowlib.read_image(img1Path)
        img2 = flowlib.read_image(img2Path)

        height = img1.shape[0]
        width = img1.shape[1]
        flow = flowlib.read_flow_png(flowPath)
        u = flow[:,:,0]
        v = flow[:,:,1]
        valid = flow[:,:,2]
        for i in range(halfPatchSize,height-halfPatchSize, 3):
            for j in range(halfPatchSize,width-halfPatchSize, 3):
                if valid[i,j]==1:

                    if (i + int(v[i,j]-1) - halfPatchSize < 0 or i + int(v[i,j]+1) + halfPatchSize >= height):
                        continue

                    if (j + int(u[i, j]-1) - halfPatchSize < 0 or j + int(u[i, j]+1) + halfPatchSize >= width):
                        continue

                    if (non_matching_creation_option == "gaussian"):
                        shift = np.abs((np.random.normal(0, std, 1))) + min_shift
                        sign = np.random.choice([1, -1])
                        rand_i = sign * int(shift)
                        shift = np.abs((np.random.normal(0, std, 1))) + min_shift
                        sign = np.random.choice([1, -1])
                        rand_j = sign * int(shift)
                    else:
                        rand_i = randint(randLowerBound, randUpperBound)
                        rand_j = randint(randLowerBound, randUpperBound)
                        while (rand_j + j + int(u[i, j]+1)+ halfPatchSize >= width or rand_j + j+int(u[i, j]-1) - halfPatchSize < 0 or rand_j == 0 or rand_j == 1 or rand_j == -1 or rand_j == -2 or rand_j == 2):
                            rand_j = randint(randLowerBound, randUpperBound)

                        while (rand_i + i + halfPatchSize+ int(v[i,j]+1) >= height or rand_i + i+ int(v[i,j]-1) - halfPatchSize < 0 or rand_i == 0 or rand_i == 1 or rand_i == -1 or rand_i == -2 or rand_i == 2):
                            rand_i = randint(randLowerBound, randUpperBound)


                    patch1 = img1[i-halfPatchSize:i+halfPatchSize, j-halfPatchSize:j+halfPatchSize,:]
                    patch2 = getCorrespodingPatch_biLinInterpolate(i,j,u[i,j],v[i,j],halfPatchSize,img2)
                    patch3 = img2[(i+int(v[i,j])+rand_i)-halfPatchSize:(i+rand_i+int(v[i,j]))+halfPatchSize, (j+rand_j+int(u[i,j]))-halfPatchSize:(j+rand_j+int(u[i,j]))+halfPatchSize,:]

                    # To discard the uniform patches
                    sub_wrong = np.asarray(patch2, dtype=np.int32) - np.asarray(patch3, dtype=np.int32)
                    if (np.std(sub_wrong) <= 1.0):

                        bad_counter += 1
                        continue
                    patchesNum += 1

                    patch1 = Image.fromarray(patch1)
                    patch2 = Image.fromarray(patch2)
                    patch3 = Image.fromarray(patch3)
                    patch1.save(original + "/0" + str(patchesNum).zfill(8) + ".png")
                    patch2.save(correct + "/0" + str(patchesNum).zfill(8) + ".png")
                    patch3.save(wrong + "/0" + str(patchesNum).zfill(8) + ".png")

        print(imgNum)
    print(bad_counter)
    print(patchesNum)