import argparse

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

torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available() and not args.unuse_cuda

if use_cuda:
    torch.cuda.manual_seed(args.seed)



# ##############################################################################
# Load datasets, While creating a function to load dataset in a different file, pickle shows an error. So to avoid that do this in train.py itself.
################################################################################
from create_dataset import Data_loader
vocab = get_vocab('vocab.pkl')
args.vocab_size = len(vocab)
# args.max_len = 30
args.max_len=60

# images = glob.glob(args.path+ "/*") 
# data = json.loads(open(args.json_path, "r").read())['images']
data=pd.read_csv('./clean_indiana_reports.csv')
projections=pd.read_csv('./indiana_projections.csv')
augmented_projections=pd.read_csv('./augmented_project.csv')
#merging to get images corresponding to uids
projections['filename']=projections['filename'].apply(lambda x: f'{args.path}/{x}')
augmented_projections['filename']=augmented_projections['filename'].apply(lambda x: f'./augmen_images/{x}')
jdata_1=pd.merge(data,projections,on='uid')
jdata_2=pd.merge(data,augmented_projections,on='uid')
captions = []
dataImages = []
# print(images)
vocab = get_vocab('vocab.pkl')
# for i in range(0, len(data)):
#     sentence = data[i]['sentences'][0]['raw'] 
#     tokens = nltk.tokenize.word_tokenize(str(sentence))
#     caption = []
#     caption.append(vocab('<start>'))
#     caption.extend([vocab(token.lower()) for token in tokens])
#     caption.append(vocab('<end>'))
#     if(len(caption) <= 30):
#         captions.append(caption)
#         dataImages.append(images[i])

#arrange images in order of uids.
count_normal=0
for i,row in jdata_1.iterrows():
    sentence = str(row['findings'])
    new_string=""
    flag=1
    if(row['Problems']=='normal'):
        count_normal+=1
    if(row['Problems']=='normal' and count_normal>100):
        continue
    if(".dcm" in row['filename']):
        new_string=row['filename'].replace(".dcm","")
    else:
        new_string=row['filename']

    try:
        # img=Image.open(args.path+"/"+new_string)
        img=Image.open(new_string)
        # img.load()
    except:
        print('error:',sys.exc_info()[0])
        print("At filename: ",new_string)
        flag=0
        continue
    
    if(flag==0):
        continue

    tokens = nltk.tokenize.word_tokenize(str(sentence))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token.lower()) for token in tokens])
    caption.append(vocab('<end>'))
    if(len(caption) <= 60):   #why?
        #medical dataset has longer captions so might need to change this threshold to 50 or 60 (initially it was 30)
        if(os.path.exists(new_string)): 
            dataImages.append(new_string)
            captions.append(caption)

for i,row in jdata_2.iterrows():
    sentence = str(row['findings'])
    new_string=""
    flag=1
    if(row['Problems']=='normal'):
        count_normal+=1
    if(row['Problems']=='normal' and count_normal>100):
        continue
    if(".dcm" in row['filename']):
        new_string=row['filename'].replace(".dcm","")
    else:
        new_string=row['filename']

    try:
        # img=Image.open(args.path+"/"+new_string)
        img=Image.open(new_string)
        # img.load()
    except:
        print('error:',sys.exc_info()[0])
        print("At filename: ",new_string)
        flag=0
        continue
    
    if(flag==0):
        continue

    tokens = nltk.tokenize.word_tokenize(str(sentence))
    caption = []
    caption.append(vocab('<start>'))
    caption.extend([vocab(token.lower()) for token in tokens])
    caption.append(vocab('<end>'))
    if(len(caption) <= 60):   #why?
        #medical dataset has longer captions so might need to change this threshold to 50 or 60 (initially it was 30)
        if(os.path.exists(new_string)): 
            dataImages.append(new_string)
            captions.append(caption)

# captions_np=np.array(captions)
print('captions shape',len(captions))
training_data = Data_loader(dataImages, captions, args.max_len, batch_size=args.batch_size, is_cuda=True, img_size=32)
print("Dataset Loaded !")
# ##############################################################################
# Build model
# ##############################################################################
import models
from const import PAD
from optim import Optim, Policy_optim

actor = models.Actor(args.vocab_size,
                    args.dec_hsz,
                    args.rnn_layers,
                    args.batch_size,
                    args.max_len,
                    args.dropout,
                    use_cuda)

critic = models.Critic(args.vocab_size,
                      args.dec_hsz,
                      args.rnn_layers,
                      args.batch_size,
                      args.max_len,
                      args.dropout,
                      use_cuda)

EncoderDecoder = models.EncDec(args.dec_hsz, args.vocab_size).cuda()

if(args.load_pretrained):
    actor = load_checkpoint(actor,args.actor_pretrained)
    critic = load_checkpoint(critic, args.critic_pretrained)
    EncoderDecoder=load_checkpoint(EncoderDecoder,args.enc_dec_path)

# self, params, lr, grad_clip, new_lr
# self, params, lr, is_pre,
                #  grad_clip, new_lr=0.0, weight_decay=0.
optim_pre_A = Optim(actor.get_trainable_parameters(),
                    args.lr, True, args.grad_clip)
optim_pre_C = Optim(critic.parameters(), args.lr, True,
                    args.grad_clip, weight_decay=0.5)
optim_pre_ED = torch.optim.RMSprop(EncoderDecoder.parameters(), lr=0.0005)


# optim_A = Policy_optim(actor.get_trainable_parameters(), args.lr,
#                        args.new_lr, args.grad_clip)
# optim_C = Optim(critic.parameters(), args.lr,
                # False, args.new_lr, args.grad_clip)
optim_A = Policy_optim(actor.get_trainable_parameters(), args.lr,
                       args.grad_clip, args.new_lr)
optim_C = Optim(critic.parameters(), args.lr,
                False, args.grad_clip, args.new_lr)

optim_ED = torch.optim.RMSprop(EncoderDecoder.parameters(), lr=0.0005)

criterion_A = torch.nn.CrossEntropyLoss(ignore_index=PAD)
criterion_C = torch.nn.MSELoss()

if use_cuda:
    actor = actor.cuda()
    critic = critic.cuda()

# ##############################################################################
# Training
# ##############################################################################
from tqdm import tqdm

from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
from rouge import rouge_l, mask_score
from rouge_modified import rouge_l_score
import cv2

device=torch.device('cuda')

def pre_train_actor(epoch):
    index = 0
    b1, b2, b3, b4, rouge = 0.0,0.0,0.0,0.0, 0.0
    for _,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Pre-train Actor",
                             leave=False):
        optim_pre_A.zero_grad()
        actor.zero_grad()

        index+=1
        # plain_imgs=cv2.resize(plain_imgs,(32,32))
        # enc = actor.encode(imgs)
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # print("plain_imgs_shape",plain_imgs.shape)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc=actor.encode(plain_imgs)
        hidden = actor.feed_enc(enc)
        target = actor(hidden, labels)
        _, words = actor(hidden)
        loss = criterion_A(target.view(-1, target.size(2)), labels.view(-1))
        loss.backward()
        # optim_pre_A.clip_grad_norm()
        optim_pre_A.step()
        print('epoch: ',epoch,"  Loss: ",loss,file=open(args.pretrain_actor_logs,'a'))

def pre_train_critic():
    iterations = 0
    actor.eval()
    critic.train()
    for _,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Pre-train Critic",
                             leave=False):
        optim_pre_C.zero_grad()
        critic.zero_grad()
        # plain_imgs=cv2.resize(plain_imgs,(32,32))
        # enc = actor.encode(imgs)
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc=actor.encode(plain_imgs)
        hidden_A = actor.feed_enc(enc)
        # we pre-train the critic network by feeding it with sampled actions from the fixed pre-trained actor.
        _, words = actor(hidden_A)
        # print('word shape',words.shape)
        policy_values = rouge_l(words, labels)

        hidden_C = critic.feed_enc(enc)
        estimated_values = critic(words, hidden_C)
        loss = criterion_C(estimated_values, policy_values)
        loss.backward()
        optim_pre_C.clip_grad_norm()
        optim_pre_C.step()
        print('Iteration: ',iterations,"  Loss: ",loss,file=open(args.pretrain_critic_logs,'a'))
        iterations += 1
   
    
def pre_train_enndec():
    actor.eval()
    b1, b2, b3, b4, rouge = 0.0,0.0,0.0,0.0, 0.0
    for _,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Pre-train EncoderDecoder",
                             leave=False):
        optim_pre_ED.zero_grad()
        EncoderDecoder.zero_grad()
        # enc = actor.encode(imgs)
        # plain_imgs=cv2.resize(plain_imgs,(32,32))
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc=actor.encode(plain_imgs)
        # we pre-train the critic network by feeding it with sampled actions from the fixed pre-trained actor.
        loss, acc = EncoderDecoder(enc, labels)

        loss.backward()
        optim_pre_ED.step()
        print("Loss: ",loss,file=open(args.pretrain_enc_dec_logs,'a'))
        

def train_actor_critic(GAMMA, epoch):
    actor.train()
    critic.train()
    EncoderDecoder.train()
    index = 0
    b1, b2, b3, b4, rouge = 0.0,0.0,0.0,0.0, 0.0
    for _,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Training",
                             leave=False):
        optim_A.zero_grad()
        optim_C.zero_grad()
        EncoderDecoder.zero_grad()
        index+=1
        # plain_imgs=cv2.resize(plain_imgs,(32,32))
        # enc = actor.encode(imgs)
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc=actor.encode(plain_imgs)
        hidden_A = actor.feed_enc(enc)
        target, words = actor(hidden_A) # actor gives out captions - words
        policy_values = rouge_l(words, labels) # policy values are calculated using rouge_l score

        # # log probabilities
        # logits = F.log_softmax(target, dim=-1)
        # log_probs = torch.gather(logits, dim=-1, index=words.unsqueeze(-1)).squeeze(-1)


        WriteInfiles(args.results_path, words, training_data.names, epoch, vocab, labels)
        # hidden_C = critic.feed_enc(enc)
        # estimated_values = critic(words, hidden_C) # critic gives out estimated values

        # loss_c = criterion_C(estimated_values, policy_values)
        # loss_c.backward()
        # optim_C.clip_grad_norm()
        # optim_C.step()

        # # reward = torch.mean(policy_values - estimated_values) # reward definition is flipped
        # # print('reward:',reward,file=open(args.train_reward_logs,'a'))
        # loss_a = criterion_A(target.view(-1, target.size(2)), labels.view(-1))
        # loss_a.backward()
        # # optim_A.clip_grad_norm()
        # # optim_A.step(reward) #? reward = loss? 

        # # Calculate advantage function
        # advantage = policy_values - estimated_values
        
        # # Weight the log probabilities by the advantage function
        # weighted_log_probs = log_probs * advantage
        # # Compute the mean of the weighted log probabilities as the loss
        # actor_loss = -torch.mean(weighted_log_probs)
        # print('reward:',-actor_loss,file=open(args.train_reward_logs,'a'))

        # optim_A.step(actor_loss)

#         actor_loss = torch.sum(-scaled_rewards)
    # actor_optimizer.zero_grad()
    # actor_loss.backward()
    # actor_optimizer.step()

        actor.zero_grad()
        EncoderDecoder.zero_grad()
        # enc = actor.encode(imgs)
        enc=actor.encode(plain_imgs)
        hidden_A = actor.feed_enc(enc)
        target, words = actor(hidden_A)
        #################################
        logits = F.log_softmax(target, dim=-1)
        log_probs = torch.gather(logits, dim=-1, index=words.unsqueeze(-1)).squeeze(-1)
        # print('log_probs:',log_probs.shape)
        #################################

        lossGen, accGen = EncoderDecoder(enc, words)
        lossReal, accReal = EncoderDecoder(enc, labels)
        loss_a = criterion_A(target.view(-1, target.size(2)), labels.view(-1))
        loss_a.backward()
        optim_A.clip_grad_norm()
        
        A = accReal - GAMMA*accGen
        A = A.view(-1)
        ################################
        # weighted_log_probs = log_probs * A
        # total_loss = torch.mean(weighted_log_probs)

        # total_loss = loss_a + A
        # total_loss = A

        # Update parameters
        # optim_A.train_step(total_loss)
        ################################
        optim_A.step(A)
        EncoderDecoder.zero_grad()
        lossReal.backward()
        optim_ED.step()

        WriteInfiles(args.results_path,words, training_data.names, epoch, vocab, labels)
        # bleuVals = getScores(words, labels)
        # b1 += bleuVals[0]
        # b2 += bleuVals[1]
        # b3 += bleuVals[2]
        # b4 += bleuVals[3]
        rouge += torch.mean(policy_values).item()
        print("LOSS", loss_a.item(), torch.mean(policy_values).item(),file=open(args.train_logs,'a'))

    # b1, b2, b3, b4, rouge =  (b1/index, b2/index, b3/index, b4/index, rouge/index)
    # print("AVERAGE SCORES B1:, B2:, B3:, B4:, ROUGE:", b1, b2, b3, b4, rouge,file=open(args.training_results,'a'))

    rouge = rouge/index
    print("AVERAGE ROUGE:", rouge,file=open(args.training_results,'a'))


def eval():
    actor.eval()
    b1, b2, b3, b4 = 0.0, 0.0, 0.0, 0.0
    rouge = 0.0
    index = 0
    for _,plain_imgs,imgs, labels in tqdm(training_data,
                             mininterval=1,
                             desc="Actor-Critic Eval",
                             leave=False):
        index += 1
        # enc = actor.encode(imgs)
        # plain_imgs=cv2.resize(plain_imgs,(32,32))
        transform=transforms.Compose([transforms.ToTensor()])
        plain_imgs=[transform(img) for img in plain_imgs]
        plain_imgs=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # plain_imgs=plain_imgs.unsqueeze(3)
        enc=actor.encode(plain_imgs)

        hidden = actor.feed_enc(enc)
        target, words = actor(hidden)
        
        words = correctSentence(words)
        words = torch.tensor(words)
        words = words.unsqueeze(0)
        labels = correctSentence(labels)
        labels = torch.tensor(labels)
        labels = labels.unsqueeze(0)
        # print(words)
        # print(labels)
        policy_values = rouge_l(words, labels)
        rouge+= policy_values
        b1+= getScores(words, labels)[0]
        b2+= getScores(words, labels)[1]
        b3+= getScores(words, labels)[2]
        b4+= getScores(words, labels)[3]

    b1, b2, b3, b4, rouge =  (b1/index, b2/index, b3/index, b4/index, rouge/index)
    print("AVERAGE SCORES B1:, B2:, B3:, B4:, ROUGE:", b1, b2, b3, b4, rouge,file=open('eval_results.txt','a'))
       
# Writes predicted sentences in a json file
def testSentences():
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

    with open(args.final_results_path+'/predicted.json', 'w') as fout:
        json.dump(listPred , fout)
    with open(args.final_results_path+'/ground_truth.json', 'w') as fout:
        json.dump(listTrue , fout)

try:
    if(not(args.load_pretrained)):
        actor.train()
        for step in range(args.actor_epochs):
            pre_train_actor(step)
            model_state_dict = actor.state_dict()
            model_source = {
                "model": model_state_dict,
            }
            torch.save(model_source, args.actor_pretrained)
        critic.train()
        for step in range(args.actor_epochs):
            pre_train_critic()
            model_state_dict = critic.state_dict()
            model_source = {
                "model": model_state_dict,
            }
            torch.save(model_source,args.critic_pretrained)


    # EncoderDecoder.train()
    # for step in range(0, 10):
    #     pre_train_enndec()
    #     model_state_dict = EncoderDecoder.state_dict()
    #     model_source = {
    #         "model": model_state_dict,
    #     }
    #     torch.save(model_source, args.enc_dec_path)

        EncoderDecoder.train()
        for step in range(args.actor_epochs):
            pre_train_enndec()
            model_state_dict = EncoderDecoder.state_dict()
            model_source = {
                "model": model_state_dict,
            }
            torch.save(model_source,args.enc_dec_path)

    GAMMA = 0.0
    for step in range(args.epochs):
        train_actor_critic(GAMMA, step)
        # GAMMA += 0.01
        GAMMA+=1/(args.epochs)
        model_state_dict = actor.state_dict()
        model_source = {
            "model": model_state_dict,
        }
        torch.save(model_source, args.actor_path)
        model_state_dict = critic.state_dict()
        model_source = {
            "model": model_state_dict,
        }
        torch.save(model_source,args.critic_path)
        model_state_dict = EncoderDecoder.state_dict()
        model_source = {
            "model": model_state_dict,
        }
        torch.save(model_source,args.enc_dec_path)
    # eval()
    testSentences()

except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early")
