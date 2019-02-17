import os
import numpy as np
import string, re
import csv
from PIL import Image
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
import imgaug as ia
import custom_datasets
import utils
import matplotlib as mpl

# https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=DBJWYcRiVknc

def get_index_of_label(CLASSES, label):
    return CLASSES.index(label)

def show_dataset(dataset,angle, n=6,mode ='edge'):
  # img = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
  #                  for i in range(len(dataset))))

  img = np.hstack((np.asarray(dataset[i][0])  for i in range(len(dataset))))

  plt.imshow(img)
  plt.axis('off')
  plt.savefig('sample_'+mode+str(angle)+'.png')
  # for i in range(len(dataset)):
  #     img = Image.fromarray(np.asarray(dataset[i][0]))
  #     img.save('%dth.png'%i)


CLASSES = string.digits + string.ascii_uppercase + string.ascii_lowercase

def get_class(filename):
    """Get the actual digit or character of the image"""
    return CLASSES[get_class_index(filename)]

def get_class_index(filename):
    return int(re.findall(r'.*img(\d+).*', filename)[0])-1

def make_csvfile(csvfile='train.csv',img_directory='data',label_file='datasplits/good_train'):
    imgs = list(map(lambda f:f.strip(),
                       open(label_file, 'r').readlines()))
    labels = np.array(list(map(get_class, imgs)))

    with open( img_directory+'/'+csvfile, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(list(zip(imgs,labels)))

class ImgAugTransform:
    def __init__(self,angle=30,mode='edge'):
        self.aug = iaa.Sequential([
            iaa.Scale((224, 224)),
            iaa.Affine(rotate=angle, mode=mode),
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

def augment_img(dataset,aug_dir,root_dir='data'):
    root_dir= os.path.join(root_dir,aug_dir)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
        print(root_dir,'genearted')
    imgs=[]
    labels=[]
    for i in range(len(dataset)):
        img_file='%dth.png'%i
        imgs.append(img_file)
        labels.append(dataset[i][1])

        img = Image.fromarray(np.asarray(dataset[i][0]))
        img.save(os.path.join(root_dir, img_file))

    with open( os.path.join(root_dir, aug_dir+'.csv'), 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(list(zip(imgs,labels)))
        print(os.path.join(root_dir, aug_dir+'.csv'),'saved')

if __name__ == "__main__":
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['image.interpolation'] = 'nearest'
    mpl.rcParams['figure.figsize'] = 15, 25

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.Resize((64, 64)),
        torchvision.transforms.RandomRotation(degrees=(-30, 30)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        # torchvision.transforms.Normalize((0.5,),(0.5,)),
        torchvision.transforms.ToPILImage(),
    ])
    # dataset = torchvision.datasets.ImageFolder('sampledata', transform=transforms)
    # show_dataset(dataset)

    # rotate mode experiment
    # modes=['constant', 'edge', 'symmetric', 'reflect', 'wrap']
    # for mode in modes:
    #     dataset = torchvision.datasets.ImageFolder('sampledata', transform=ImgAugTransform(mode=mode))
    #     show_dataset(dataset,mode=mode)

    # [-30,30] angle range generate
    CLASSES=string.digits + string.ascii_uppercase + string.ascii_lowercase
    angles=[ i for i in range(-30,31,10)]
    for angle in angles:
        if angle!=-30:
            break
        if angle==0:
            continue
        dataset=torch.utils.data.ConcatDataset([
            custom_datasets.Chars74k(csv_file='data/train.csv', root_dir='data', CLASSES=CLASSES,
                                     target_transform=None, transform=ImgAugTransform(angle=angle), ),
            custom_datasets.Chars74k(csv_file='data/validation.csv', root_dir='data', CLASSES=CLASSES,
                                     target_transform=None, transform=ImgAugTransform(angle=angle), ),
        ])

        # dataset = torchvision.datasets.ImageFolder('data/English/Img/GoodImg/Bmp', transform=ImgAugTransform(angle=angle))
        augment_img(dataset, aug_dir='augment_%d'%angle,)


    # make_csvfile('train.csv','data','datasplits/good_train')
    # make_csvfile('test.csv', 'data', 'datasplits/good_test')
    # make_csvfile('validation.csv', 'data', 'datasplits/good_validation')