import math
import cv2
import numpy as np
import os


def CIEDE2000(Lab_1, Lab_2):
    '''Calculates CIEDE2000 color distance between two CIE L*a*b* colors'''
    C_25_7 = 6103515625  # 25**7

    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7)))

    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2

    C1_ = math.sqrt(a1_ ** 2 + b1_ ** 2)
    C2_ = math.sqrt(a2_ ** 2 + b2_ ** 2)

    if b1_ == 0 and a1_ == 0:
        h1_ = 0
    elif a1_ >= 0:
        h1_ = math.atan2(b1_, a1_)
    else:
        h1_ = math.atan2(b1_, a1_) + 2 * math.pi

    if b2_ == 0 and a2_ == 0:
        h2_ = 0
    elif a2_ >= 0:
        h2_ = math.atan2(b2_, a2_)
    else:
        h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0:
        dh_ = 0
    elif dh_ > math.pi:
        dh_ -= 2 * math.pi
    elif dh_ < -math.pi:
        dh_ += 2 * math.pi
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)

    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2

    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_

    if _dh <= math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2
    elif _dh > math.pi and _sh < 2 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh > math.pi and _sh >= 2 * math.pi and C1C2 != 0:
        h_ave = (h1_ + h2_) / 2 - math.pi
    else:
        h_ave = h1_ + h2_

    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(
        3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)

    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0:
        h_ave_deg += 360
    elif h_ave_deg > 360:
        h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25) ** 2))

    R_C = 2 * math.sqrt(C_ave ** 7 / (C_ave ** 7 + C_25_7))
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T

    Lm50s = (L_ave - 50) ** 2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1

    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H

    dE_00 = math.sqrt(f_L ** 2 + f_C ** 2 + f_H ** 2 + R_T * f_C * f_H)
    return dE_00


## ----- 这一部分是测单张图片（a pair）的 CIEDE2000 value， 亲测可用， 这一段不要乱动了 -----
# img = cv2.imread("/home/chen/data/dehazing/MUNIT/res_10000/803_hazy.jpg")
# img = cv2.imread("./outputs/SOTS/fake/1352_0.8_0.08.jpg")
# img = np.float32(img)
# img *= 1./255
# Lab1 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
# L1, a1, b1 = cv2.split(Lab1)
# # print(L1, a1, b1)
# h, w = L1.shape
# print(h, w)
# # ref = cv2.imread("/home/chen/data/dehazing/dataset/NYUv2pics/nyu_images/803.jpg")
# ref = cv2.imread("./outputs/SOTS/gt/1352.jpg")
# ref = np.float32(ref)
# ref *= 1./255
# Lab2 = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
# L2, a2, b2 = cv2.split(Lab2)
#
# de = 0
# for i in range(0, h):
#     for j in range(0, w):
#         de += CIEDE2000(Lab1[i, j], Lab2[i, j])
# ciede = de / (w * h)
# print(ciede)
## --------------------------------------------------------------------------------

## ----- 以下写批量测量 CIEDE2000 取平均值作为结果值 -----

# list_dir = '/home/zzh/dev/csx/RDDN/results/d-hazy/dhazy_cyclegan/'
# ref_dir = '/home/zzh/dev/csx/dataset/D-hazy/testB/'

img_path = 'outputs/OHazy/fake'
gt_path = 'outputs/OHazy/gt'

txtFile = './outputs/' + 'ciede_all.txt'
avgFile = './outputs/' + 'CIEDE.txt'

# for epoch in range(3):
    # test_number = epoch * 2000
avg_de = 0.0 # avg_de 是批量ciede均值
for path, subdirs, files in os.walk(img_path):
    for num in range(len(files)):
        nameA = files[num]
        img_name = img_path + "/" + nameA
        # print(img_name)
        img = cv2.imread(img_name)
        img = np.float32(img) / 255
        Lab1 = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        L1, a1, b1 = cv2.split(Lab1)
        h, w = L1.shape
        # ref_name = ref_dir + nameA[:-9] + '_Image_.bmp'
        # ref_name = gt_path + "/" + nameA.split('_')[0] + '.jpg'  # SOTS & Hazy
        # ref_name = gt_path + "/" + nameA.split('_')[0] + '_Image_.jpg'  # DHazy
        ref_name = gt_path + "/" + nameA.split('_')[0] + '_' + nameA.split('_')[1] + '_GT.jpg'  # Ohazy
        print("ref:", ref_name)
        ref = cv2.imread(ref_name)
        ref = np.float32(ref) / 255
        Lab2 = cv2.cvtColor(ref, cv2.COLOR_BGR2Lab)
        de = 0 # de是每个pixel的ciede
        for i in range(0, h):
            for j in range(0, w):
                de += CIEDE2000(Lab1[i, j], Lab2[i, j])
        ciede = de / (w * h) # 这里这个ciede是一张图的ciede
        avg_de += ciede

        print(nameA, round(ciede, 4))

        with open(txtFile, 'a') as f:
            f.write(str(num) + '\t' + nameA + '\t' + str(round(ciede, 4)) + '\n')
    print(round(avg_de / len(files), 4))
    with open(avgFile, 'a') as f:
        # f.write('epoch ' + str(epoch) + ':' + '\t' + 'CIEDE2000: ' + str(round(avg_de / len(files), 4)) + '\n')
        f.write('iter ' + ':' + '\t' + 'CIEDE2000: ' + str(round(avg_de / len(files), 4)) + '\n')

## -----------------------------------------------------------------------------------------------------------------------







