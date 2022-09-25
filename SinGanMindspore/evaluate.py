from __future__ import print_function
import os
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


method = 'DCP-CAP-MLC'

img_path = 'outputs/SOTS/fake'
gt_path = 'outputs/SOTS/gt'


txtFIle = './'  'a.txt'
avgFIle = './' + 'avg.txt'

ave_psnr = 0.0
ave_ssim = 0.0

for path, subdirs, files in os.walk(img_path):
    for i in range(len(files)):
        nameA = files[i]
        hazyName = img_path +"/"+ nameA
        print(hazyName)
        print(gt_path + "/" + nameA.split('_')[0] + '.jpg')
        if hazyName.endswith('.txt'):
            continue
        Ix = np.array(Image.open(hazyName).convert('RGB')) / 255.
        Jx = np.array(Image.open(gt_path + "/" + nameA.split('_')[0] + '.jpg').convert('RGB')) / 255.
        # Jx = np.array(Image.open(gt_path + "/" + nameA.split('_')[0] + '_Image_.jpg').convert('RGB')) / 255. # DHazy
        # Jx = np.array(
        #     Image.open(gt_path + "/" + nameA.split('_')[0] + '_' + nameA.split('_')[1] + '_GT.jpg').convert('RGB')) / 255. # OHaze

        W, H, C = Ix.shape

        Jx = cv2.resize(Jx, (H, W)).astype(np.float32)

        PSNR = compare_psnr(Ix, Jx, data_range=1)
        SSIM = compare_ssim(Ix, Jx, data_range=1, multichannel=True)
        ave_psnr += PSNR
        ave_ssim += SSIM

        print(nameA, PSNR, SSIM)

        with open(txtFIle, 'a') as f:
            f.write(str(i) + '\t' + nameA + '\t' + str(PSNR) + '\t' + str(SSIM) + '\n')


print(ave_psnr / len(files), ave_ssim / len(files))
with open(avgFIle, 'a') as f:
    f.write(method + '\t' + 'AVG' + '\t' + str(ave_psnr / len(files)) + '\t' + str(ave_ssim / len(files)) + '\n')
