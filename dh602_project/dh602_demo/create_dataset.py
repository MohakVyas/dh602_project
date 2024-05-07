import torch
import PIL 
PIL.PILLOW_VERSION = PIL.__version__
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
import pickle
import numpy as np
import pandas as pd
from vocab_build import Vocabulary
import nltk
import os
import matplotlib.pyplot as plt
from const import PAD
# # print(WORD)

# def get_vocab(vocab_path):
#     with open(vocab_path, 'rb') as f:
#         vocab = pickle.load(f)
#     return vocab


# if __name__ == "__main__":
#     # images = glob.glob( "C:\\Users\\Ruchika\\Downloads\\Compressed\\UCMerced_LandUse\\UCMerced_LandUse\\Images" + "\\*\\*")
#     # data = json.loads(open("C:\\Users\\Ruchika\\Downloads\\dataset_UCM.json", "r").read())
#     # captions = []
#     # vocab = get_vocab('E:\\ADC UCM\\vocab.pkl')
#     # for i in range(0, len(data['images'])):
#     #     sentence = data['images'][i]['sentences'][0]['raw'] 
#     #     tokens = nltk.tokenize.word_tokenize(str(sentence))
#     #     caption = []
#     #     caption.append(vocab('<start>'))
#     #     caption.extend([vocab(token.lower()) for token in tokens])
#     #     caption.append(vocab('<end>'))
#     #     captions.append(caption)
#     data=pd.read_csv('./clean_indiana_reports.csv')
#     projections=pd.read_csv('./indiana_projections.csv')
#     #merging to get images corresponding to uids
#     jdata=pd.merge(data,projections,on='uid')
#     captions = []
#     dataImages = []
#     # print(images)
#     vocab = get_vocab('vocab.pkl')
#     print(len(vocab))

### Copied from train.py
import argparse



import pickle
from vocab_build import Vocabulary
import glob 
import json
import nltk
from nltk import ngrams
from nltk.translate.bleu_score import modified_precision
import numpy as np
from utils import *
import pandas as pd
import os.path
from PIL import Image
import torch
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# import cv2
torch.manual_seed(1)
use_cuda = torch.cuda.is_available() 

if use_cuda:
    torch.cuda.manual_seed(1)

class Data_loader(object):
    def __init__(self, imgs, labels, max_len, batch_size, is_cuda, img_size=299, evaluation=False):
        self._imgs = imgs
        # self._labels = np.asarray(labels)
        self._labels=labels
        self._max_len = max_len
        self._is_cuda = is_cuda
        self.evaluation = evaluation
        self._step = 0
        self._batch_size = batch_size
        self.sents_size = len(imgs)
        self._stop_step = self.sents_size // batch_size
        # self.names = None
        self.names=imgs
        self._encode = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        # print('initialising imgs',self._imgs)

    def __iter__(self):
        return self

    def __next__(self):
        def img2variable(img_files):
            self.names = img_files
            # print('img_file',img_files)
            tensors = [self._encode(Image.open(img_name).convert('RGB')).unsqueeze(0) for img_name in img_files]
            # tensors = [self._encode(Image.open(img_name).convert('L')) for img_name in img_files]
            
            v = Variable(torch.cat(tensors, 0), volatile=self.evaluation)
            if self._is_cuda:
                v = v.cuda()
            return v

        def label2variable(labels):
            """maybe sth change between Pytorch versions, add func long() for compatibility
            """

            _labels = np.array(
                [l + [PAD] * (self._max_len - len(l)) for l in labels])

            _labels = Variable(torch.from_numpy(_labels),
                               volatile=self.evaluation).long()
            if self._is_cuda:
                _labels = _labels.cuda()
            return _labels

        if self._step == self._stop_step:
            self._step = 0
            raise StopIteration()

        _start = self._step * self._batch_size
        self._step += 1

        # print('checking',self._imgs)
        _imgs = img2variable(self._imgs[_start:_start + self._batch_size])
        # print('imgs after 168',_imgs)
        _labels = label2variable(
            self._labels[_start:_start + self._batch_size])
        
        
        plain_imgs=[Image.open(img).resize((32,32)) for img in self._imgs[_start:_start + self._batch_size]]#new line added (added plain_imgs)
        plain_img_paths=[img for img in self._imgs[_start:_start + self._batch_size]]
        return plain_img_paths,plain_imgs,_imgs, _labels 



# ##############################################################################
# Load datasets, While creating a function to load dataset in a different file, pickle shows an error. So to avoid that do this in train.py itself.
################################################################################
# from create_dataset import Data_loader

# vocab = get_vocab('vocab.pkl')
# args.vocab_size = len(vocab)
# # args.max_len = 30
# args.max_len=60

# # images = glob.glob(args.path+ "/*") 
# # data = json.loads(open(args.json_path, "r").read())['images']
# data=pd.read_csv('./clean_indiana_reports.csv')
# projections=pd.read_csv('./indiana_projections.csv')
# #merging to get images corresponding to uids
# jdata=pd.merge(data,projections,on='uid')
# captions = []
# dataImages = []
# # print(images)
# # vocab = get_vocab('vocab.pkl')
# ct = 0
# caption_lengths=[]
# # for i,row in jdata.iterrows():
# #     sentence = row['findings']
# #     tokens = nltk.tokenize.word_tokenize(str(sentence))

# #     caption = []
# #     caption.append(vocab('<start>'))
# #     caption.extend([vocab(token.lower()) for token in tokens])
# #     caption.append(vocab('<end>'))
# #     if(len(caption) <= 60):   #why?
# #         ct+=1
# #         #medical dataset has longer captions so might need to change this threshold to 50 or 60 (initially it was 30)
# #         try:
# #             captions.append(caption)
# #             # dataImages.append(images[i])
# #             dataImages.append("./images/"+row["filename"])
# #         except Exception as error:
# #             #most probably missing image file or incorrectly named
# #             print("error found: ",error)
# #             print("At: ",row)

# for i,row in jdata.iterrows():
#     sentence = row['findings']
#     new_string=""
#     flag=1
#     if(".dcm" in row['filename']):
#         new_string=row['filename'].replace(".dcm","")
#     else:
#         new_string=row['filename']

#     try:
#         img=Image.open(args.path+"/"+new_string)
#     except Exception as error:
#         print('error found: ',error)
#         print("At filename: ",new_string)
#         flag=0
#         continue
    
#     if(flag==0):
#         continue

#     tokens = nltk.tokenize.word_tokenize(str(sentence))
#     caption = []
#     caption.append(vocab('<start>'))
#     caption.extend([vocab(token.lower()) for token in tokens])
#     caption.append(vocab('<end>'))
#     caption_lengths.append(len(caption))
#     if(len(caption) <= 60):   #why?
#         ct+=1
#         #medical dataset has longer captions so might need to change this threshold to 50 or 60 (initially it was 30)
#         if(os.path.exists(args.path+"/"+new_string)): 
#             dataImages.append(args.path+"/"+new_string)
#             captions.append(caption)
# training_data = Data_loader(dataImages, captions, 20, batch_size=2, is_cuda=True)
# ######################
# # plt.scatter(list(range(len(caption_lengths))),caption_lengths)
# # plt.savefig("caption_lengths.png")
# # plt.show()

# # print(min(caption_lengths))
# ######################
# print(training_data.sents_size)
# print("ct is ",ct)
# plain_img_paths,plain_imgs,img, labels = next(training_data)

