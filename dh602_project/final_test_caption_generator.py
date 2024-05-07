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
from tqdm import tqdm
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from rouge import rouge_l, mask_score
from rouge_modified import rouge_l_score
import cv2
import models_modified
import models
from const import PAD
from optim import Optim, Policy_optim
from create_dataset import Data_loader

parser = argparse.ArgumentParser(description='Actor Dual-Critic Image Cationing')
parser.add_argument('--logdir', type=str, default='tb_logdir')
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--unuse_cuda', action='store_true')

parser.add_argument('--path', type=str, default='./images/images_normalized')
parser.add_argument('--json_path', type=str, default='data/data.json')
parser.add_argument('--save', type=str, default='imgcapt_v2_{}.pt')

parser.add_argument('--actor_pretrained', type=str, default='actor_pretrained.pth')
parser.add_argument('--critic_pretrained', type=str, default='critic_pretrained.pth')
# parser.add_argument('--enc_dec_pretrained', type=str, default='enc_dec.pth')
parser.add_argument('--actor_path', type=str, default='actor.pth')
parser.add_argument('--critic_path', type=str, default='critic.pth')
parser.add_argument('--enc_dec_path', type=str, default='enc_dec.pth')

parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--new_lr', type=float, default=5e-6)
parser.add_argument('--load_pretrained', type=bool, default=False)
# parser.add_argument('--load_pretrained', type=bool, default=True)

parser.add_argument('--actor_epochs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--iterations', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dec_hsz', type=int, default=256)
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--dropout', type=float, default=.5)
parser.add_argument('--grad_clip', type=float, default=1.)
parser.add_argument('--training_results',type=str, default='training-results.txt')
parser.add_argument('--pretrain_actor_logs',type=str,default='pretrain-actor-logs.txt')
parser.add_argument('--pretrain_critic_logs',type=str,default='pretrain-critic-logs.txt')
parser.add_argument('--pretrain_enc_dec_logs',type=str,default='pretrain-enc-dec-logs.txt')
parser.add_argument('--results_path',type=str,default='./resultsA2C')
parser.add_argument('--train_logs', type=str, default='actor_critic_train_logs.txt')
parser.add_argument('--train_reward_logs',type=str, default='actor_critic_reward_logs.txt')
parser.add_argument('--final_results_path',type=str,default='./')
args = parser.parse_args()

device=torch.device('cuda')
vocab = get_vocab('vocab.pkl')
vocab_size = len(vocab)
args.max_len=60

use_cuda = torch.cuda.is_available() and not args.unuse_cuda

actor_m = models_modified.Actor(vocab_size,
                    args.dec_hsz,
                    args.rnn_layers,
                    args.batch_size,
                    args.max_len,
                    args.dropout,
                    use_cuda)
actor_m.to(device)

actor= models.Actor(vocab_size,
                    args.dec_hsz,
                    args.rnn_layers,
                    args.batch_size,
                    args.max_len,
                    args.dropout,
                    use_cuda)
# print(torch.load('/home/ayushh/actor-critic/dh602_project/layer_norm_swin/actor.pth'))
checkpoint=torch.load('/home/ayushh/actor-critic/dh602_project/layer_norm_swin/actor.pth')
print(checkpoint.keys())
actor.load_state_dict(checkpoint['model'])
actor.to(device)
#input image

dataImages='./images/images_normalized/1_IM-0001-3001.png'

def testSentences_transformer(dataImages):
    
    fileName=dataImages.split('/')[-1]
    fileName=fileName.split('.')[0]+'.dcm.png'
    print('filename',fileName)
    data=pd.read_csv('./clean_indiana_reports.csv')
    projections=pd.read_csv('./indiana_projections.csv')
    reports=pd.merge(data,projections,on='uid')
    caption=reports['findings'][reports['filename']==fileName].iloc[0]
    tokens = nltk.tokenize.word_tokenize(str(caption))
    captions=[]
    captions.append(vocab('<start>'))
    captions.extend([vocab(token.lower()) for token in tokens])
    captions.append(vocab('<end>'))
    print('caption',captions)
    print('image',dataImages)
    training_data = Data_loader([dataImages], [captions], args.max_len, batch_size=args.batch_size, is_cuda=True, img_size=32)
    actor_m.eval()
    listPred = []
    listTrue = []
    idx=0
    for plain_img_paths,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Eval",
                             leave=False):
        print('plain_img_paths',plain_img_paths)
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc = actor_m.encode(plain_imgs)
        # enc = actor.encode(imgs)
        # hidden = actor.feed_enc(enc)
        # target, words = actor(hidden)
        target,words = actor_m(plain_img_paths,plain_imgs,labels)
        words=torch.IntTensor(words).to(device)
        words=words.view(1,-1)
        
        words = correctSentence(words)
        words = torch.tensor(words)
        words = words.unsqueeze(0)
        labels = correctSentence(labels)
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)

        predicted = getSentence(vocab, words)
        truth = getSentence(vocab, labels)
        rouge_score=rouge_l_score(predicted,truth)
        # dictPred = {"image_id": training_data.names[0].split("\\")[-1], "caption": predicted}
        # print(training_data.names)
        dictPred = {"image_id": training_data.names[0].split("/")[-1], "caption": predicted, "rouge": rouge_score}
        # dictTrue = {"image_id": training_data.names[0].split("\\")[-1], "caption": truth}
        dictTrue={"image_id": training_data.names[0].split("/")[-1], "caption": truth}

        listPred.append(dictPred)
        listTrue.append(dictTrue)
        idx+=1
        print('predicted',predicted)
        print('ground truth',truth)
        
    # with open(args.final_results_path+'/test_pred_transformer.json', 'w') as fout:
    #     json.dump(listPred , fout)
    # with open(args.final_results_path+'/test_ground_truth_transformer.json', 'w') as fout:
    #     json.dump(listTrue , fout)

# Writes predicted sentences in a json file
def testSentences(dataImages):
    
    fileName=dataImages.split('/')[-1]
    fileName=fileName.split('.')[0]+'.dcm.png'
    print('filename',fileName)
    data=pd.read_csv('./clean_indiana_reports.csv')
    projections=pd.read_csv('./indiana_projections.csv')
    reports=pd.merge(data,projections,on='uid')
    caption=reports['findings'][reports['filename']==fileName].iloc[0]
    tokens = nltk.tokenize.word_tokenize(str(caption))
    captions=[]
    captions.append(vocab('<start>'))
    captions.extend([vocab(token.lower()) for token in tokens])
    captions.append(vocab('<end>'))
    print('caption',captions)
    print('image',dataImages)
    # exit(0)
    training_data = Data_loader([dataImages], [captions], args.max_len, batch_size=args.batch_size, is_cuda=True, img_size=32)
    actor.eval()
    listPred = []
    listTrue = []
    idx=0
    for _,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Eval",
                             leave=False):
        
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc = actor.encode(plain_imgs)
        # enc = actor.encode(imgs)
        hidden = actor.feed_enc(enc)
        target, words = actor(hidden)
        
        words = correctSentence(words)
        words = torch.tensor(words)
        words = words.unsqueeze(0)
        labels = correctSentence(labels)
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)

        predicted = getSentence(vocab, words)
        truth = getSentence(vocab, labels)
        rouge_score=rouge_l_score(predicted,truth)
        # dictPred = {"image_id": training_data.names[0].split("\\")[-1], "caption": predicted}
        # print(training_data.names)
        dictPred = {"image_id": training_data.names[0].split("/")[-1], "caption": predicted, "rouge": rouge_score}
        # dictTrue = {"image_id": training_data.names[0].split("\\")[-1], "caption": truth}
        dictTrue={"image_id": training_data.names[0].split("/")[-1], "caption": truth}

        listPred.append(dictPred)
        listTrue.append(dictTrue)
        idx+=1
        print('predicted',predicted)
        print('ground truth',truth)

    # with open(args.final_results_path+'/test_pred.json', 'w') as fout:
    #     json.dump(listPred , fout)
    # with open(args.final_results_path+'/test_ground.json', 'w') as fout:
    #     json.dump(listTrue , fout)

# testSentences_transformer(dataImages)
testSentences(dataImages)

