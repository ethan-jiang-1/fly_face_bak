import cv2
import os
import numpy as np
from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug) #建立lambda表达式，
                                                #这里定义sometimes意味有时候做的操作
                                                # 实际上在深度学习的模型训练中，数据增强不能喧宾夺主，
                                                # 如果对每一张图片都加入高斯模糊的话实际上是毁坏了原来数据的特征，
                                                # 因此，我们需要“有时候”做，给这个操作加一个概率。

# 定义一组变换方法.
seq = iaa.Sequential([
    # 选择0到5种方法做变换
    iaa.SomeOf((0, 5),[
        iaa.Fliplr(0.5),  # 对50%的图片进行水平镜像翻转
        # iaa.Flipud(0.5),  # 对50%的图片进行垂直镜像翻转

        #将一些图像转换为它们的超像素表示，每幅图像采样20到200个超像素，
        # 但是不要用它们的平均值替换所有的超像素，
        # 只替换其中的一些(p_replace)。
        # sometimes(
        #             iaa.Superpixels(
        #                 p_replace=(0, 1.0),
        #                 n_segments=(20, 200)
        #                             )
        #          ),
        #用高斯模糊，均值模糊，中值模糊中的一种增强。注意OneOf的用法
        iaa.OneOf([
                    iaa.GaussianBlur((0, 3.0)),
                    iaa.AverageBlur(k=(2, 7)), # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                    iaa.MedianBlur(k=(3, 11)),
                ]),
        iaa.Fliplr(0.5),
        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
        # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        sometimes(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0, 0.7)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0, 0.7), direction=(0.0, 1.0)
                    ),
                ])),

        # sometimes(iaa.Affine(                          #对一部分图像做仿射变换
        #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},#图像缩放为80%到120%之间
        #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, #平移±20%之间
        #     rotate=(-45, 45),   #旋转±45度之间
        #     shear=(-16, 16),    #剪切变换±16度，（矩形变平行四边形）
        # )),
                    ]  ,  random_order=True)
                    ]  ,  random_order=True)


# 遍历要增强的文件夹，把所有的图片保存在imglist中
dir_this = os.path.dirname(__file__)
idx = '02'
path = '{}/../dataset_org_beard_styles/Beard Version 1.1/{}/'.format(dir_this, idx)
saved_path = '{}/../_output_beard_gen/{}/'.format(dir_this,idx)
imglist=[]
filelist = os.listdir(path)

for item in filelist:
    img = cv2.imdecode(np.fromfile(path+item,dtype=np.uint8),-1)

    # print('item is ', item)
    # print('img is ',img.shape)
    imglist.append(img)
    # print('imglist is ' ,imglist)
print('all the picture have been appent to imglist')

#对文件夹中的图片进行增强操作，循环5次
for count in range(10):
    images_aug = seq.augment_images(imglist)
    for index in range(len(images_aug)):
        filename = idx+'_'+str(count) + str(index) + '.jpg'
        # 保存图片
        # cv2.imwrite(savedpath + filename, images_aug[index])
        cv2.imencode(".jpg", images_aug[index])[1].tofile(saved_path+filename)
        # '.jpg'表示把当前图片img按照jpg格式编码，按照不同格式编码的结果不一样
        print('image of count%s index%s has been writen' % (count, index))
