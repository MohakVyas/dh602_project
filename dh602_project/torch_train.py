import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import reduce_dataset_dim, valid_test_split
from model_torch import TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from utility1 import save_tokenizer_pytorch
from settings import *
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from custom_schedule import CustomSchedule
import os
import torch
import torch.nn as nn
from nltk.lm import Vocabulary
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
torch.autograd.set_detect_anomaly(True)

class TextTokenizer(nn.Module):
    def __init__(self, max_tokens, output_sequence_length):
        super(TextTokenizer, self).__init__()
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        # self.standardize = standardize
        self.word_to_index_map = {}
        self.index_to_word_map = {}

    def forward(self, input_text):
        # Custom standardization
        # input_text = self.standardize(input_text)

        # Tokenization
        tokens = input_text
        tokens = tokens[:self.max_tokens]  # Truncate if longer than max_tokens

        # Convert tokens to integers
        token_indices = [self.word_to_index(token) for token in tokens]

        # Pad or truncate sequences to output_sequence_length
        if len(token_indices) < self.output_sequence_length:
            # Pad sequences if shorter than output_sequence_length
            token_indices += [0] * (self.output_sequence_length - len(token_indices))
        else:
            # Truncate sequences if longer than output_sequence_length
            token_indices = token_indices[:self.output_sequence_length]

        return torch.tensor(token_indices)

    def word_to_index(self, word):
        # If word not in vocabulary, add it
        if word not in self.word_to_index_map:
            index = len(self.word_to_index_map)
            self.word_to_index_map[word] = index
            self.index_to_word_map[index] = word
        return self.word_to_index_map.get(word, 0)  # Return 0 for unknown words

    def get_vocabulary(self):
        return list(self.word_to_index_map.keys())

# Constants
# MAX_VOCAB_SIZE = 10000
# SEQ_LENGTH = 50


# Define tokenizer
tokenizer = TextTokenizer(
    max_tokens=MAX_VOCAB_SIZE,
    output_sequence_length=SEQ_LENGTH
)

image_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    # Add more transformations as needed
])


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, image_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.keys=list(data.keys())
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name=self.keys[idx]
        caption=self.data[img_name]
      # img_name, caption = self.data[self.keys[idx]]['image'], self.data[self.keys[idx]]['caption']
        # print('files',os.listdir('/home/ayushh/Image-Captioning/COCO_dataset/train2014'))
        if(img_name in os.listdir('/home/ayushh/Image-Captioning/examples_img')):
            # Load image
            image = Image.open('/home/ayushh/Image-Captioning/examples_img/'+img_name).convert('L')
            # print('image found')
            # Apply image transformations if provided
            # if self.image_transform:
            #     image = self.image_transform(image)
            
            # Tokenize caption
            caption = self.tokenizer(caption)
            
            return image, caption

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]
    
    # Pad captions to ensure equal length
    max_length = max(len(caption) for caption in captions)
    padded_captions = []

    for caption in captions:
        padded_caption = torch.zeros(max_length, dtype=torch.long)
        padded_caption[:len(caption)] = caption.clone().detach()#torch.tensor(caption)
        padded_captions.append(padded_caption)
        # images=image_transforms(images)
    return images, torch.stack(padded_captions)
# Load dataset
# train_data = json.load(open('/home/ayushh/Image-Captioning/COCO_dataset/captions_mapping_train.json'))
# valid_data=json.load(open('/home/ayushh/Image-Captioning/COCO_dataset/captions_mapping_valid.json'))
# # For reducing the number of images in the dataset
# print("Number of training samples: ", len(train_data))
# print("Number of validation samples: ", len(valid_data))

# # Define tokenizer of Text Dataset
# tokenizer = TextVectorization(
#     max_tokens=MAX_VOCAB_SIZE,
#     output_mode="int",
#     output_sequence_length=SEQ_LENGTH,
#     standardize=custom_standardization,
# )

# # Adapt tokenizer to Text Dataset
# tokenizer.forward(text_data)

# # Define vocabulary size of Dataset
# VOCAB_SIZE = len(tokenizer.get_vocabulary())


# 20k images for validation set and 13432 images for test set
# valid_data, test_data = valid_test_split(valid_data)
# print("Number of validation samples after splitting with test set: ", len(valid_data))
# print("Number of test samples: ", len(test_data))

# Define transformations for images


# Create custom datasets and data loaders
# print(len(train_data))
# print(len(valid_data))
# print(train_data)
# exit(0)
# print(train_data.keys())
train_data={'2.jpg':'An airplane flying in the sky','4.jpg':'A wild animal eating grass','10.jpg':'A tennis player preparing to serve','14.jpg':'A train on the railway track','15.jpg':'A man snowboarding','20.jpg':'A boy on the waveboard'}
text_data=train_data.values()
text=""
for texts in text_data:
    text+=texts
tokenizer.forward(text)

# caps=[]
# for caption in train_data.values():
#     caps.append(caption)
# vectorizer=CountVectorizer()
# vectorizer.fit(caps)
train_dataset = CustomDataset(train_data,tokenizer, image_transforms)
valid_dataset = CustomDataset(train_data,tokenizer, image_transforms)
# print("train_data:",train_dataset[0])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,collate_fn=custom_collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,collate_fn=custom_collate_fn)
# if TEST_SET:
    # test_dataset = CustomDataset(test_data, tokenizer, image_transforms)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate_fn)

VOCAB_SIZE=len(tokenizer.get_vocabulary())
# print('vocab size:',VOCAB_SIZE)
from swin_transformer import *

model_swin=SwinModel(embed_dim, num_patch_x, num_patch_y, num_heads, num_mlp, window_size, shift_size, qkv_bias, num_classes)
# print(model_swin)

model_swin.load_state_dict(torch.load("/home/ayushh/MedMNIST_SWIN_Transformer/swin_classification_pytorch_model_weights.pth",map_location='cpu'))
device=torch.device('cpu')
model_swin.to(device)

total_params = sum(p.numel() for p in model_swin.parameters() if p.requires_grad)

# efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
# utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
# efficientnet.eval().to(device)
# print('swin model',model_swin)
# Define Model
cnn_model = model_swin
# cnn_model=efficientnet
# print('trainable params',total_params)
# cnn_model=model_swin
# self.linear = nn.Linear(self.alexnet.classifier[6].in_features, embed_size)
# linear=nn.Linear(cnn_model.fc.in_features,6)
# self.alexnet.classifier[6] = self.linear
# cnn_model.fc=linear
# cnn_model = model_swin
# layers = list(cnn_model.children())[:-1]
# layers.append(nn.Flatten())
# print("layers:",layers)
# print(cnn_model)
encoder = TransformerEncoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)
decoder = TransformerDecoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS, VOCAB_SIZE)
caption_model = ImageCaptioningModel(cnn_model, encoder, decoder)

# Define the loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is the index for padding

# Define the optimizer
optimizer = optim.Adam(caption_model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

# # Create a learning rate schedule
# lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, CustomSchedule(EMBED_DIM))


# Move model to appropriate device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device=torch.device('cpu')
caption_model.to(device)
# print(caption_model)

# Training loop
for epoch in range(EPOCHS):
    caption_model.train()
    epoch_loss=0
    for images, captions in train_loader:
        # print('training')
        captions = captions.to(device)
        plain_imgs=[image_transforms(img) for img in images]
        images=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
        # images=torch.stack(plain_imgs,dim=0).to(device)
        # images=images.to(device)
        optimizer.zero_grad()
        # print('input image shape',images.shape)
        # print('input caption shape',captions.shape)
        # print('image before model',images)
        outputs = caption_model(images,captions,True)
        # print("outputs_shape:", outputs.shape)
        # print("captions_shape:", captions.shape)
        # print("vocab_size:", VOCAB_SIZE)
        # print("outputs_view",(outputs.view(-1, VOCAB_SIZE)).shape)
        
        captions = captions[:, :-1]
        # print("captions_view",(captions.reshape(-1)).shape)
        # print('outputs',outputs)
        # print('outputs shape',outputs.shape)
        # print('captions',captions)
        # print('captions',captions.shape)
        loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.reshape(-1)) #captions.view(-1)
        epoch_loss+=loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(caption_model.parameters(), 0.1)
        optimizer.step()

    print('TRAINING LOSS PER EPOCH',epoch_loss/len(train_loader.dataset))
    
    # Evaluation
    caption_model.eval()
    # Validation set
    with torch.no_grad():
        valid_loss = 0.0
        for images, captions in valid_loader:
            captions = captions.to(device)
            plain_imgs=[image_transforms(img) for img in images]
            images=torch.stack(plain_imgs,dim=0).permute(0,2,3,1).to(device)
            # images=torch.stack(plain_imgs,dim=0).to(device)
            images=images.to(device)
            outputs = caption_model(images, captions, False)
            captions = captions[:, :-1]
            loss = criterion(outputs.view(-1, VOCAB_SIZE), captions.reshape(-1))
            valid_loss += loss.item() * images.size(0)
        valid_loss /= len(valid_loader.dataset)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Valid Loss: {valid_loss:.4f}")

# Save model weights
torch.save(caption_model.state_dict(), SAVE_DIR + 'model_weights_coco.pth')
print("model_saved")
# Save config model train
config_train = {
    "IMAGE_SIZE": IMAGE_SIZE,
    "MAX_VOCAB_SIZE": MAX_VOCAB_SIZE,
    "SEQ_LENGTH": SEQ_LENGTH,
    "EMBED_DIM": EMBED_DIM,
    "NUM_HEADS": NUM_HEADS,
    "FF_DIM": FF_DIM,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "VOCAB_SIZE": VOCAB_SIZE
}
json.dump(config_train, open(SAVE_DIR + 'config_train.json', 'w'))

# Save Tokenizer model
# save_tokenizer_pytorch(tokenizer, SAVE_DIR)
