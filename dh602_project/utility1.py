import torch
import torch.nn as nn
import torchvision.models as models
# from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
# from model_torch import TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
# from dataset import read_image_inf
import numpy as np
import json
import re
from settings import *

def save_tokenizer_pytorch(tokenizer, path_save):
    input = torch.nn.Sequential(nn.Linear(1, 1), nn.Flatten())
    output = tokenizer(input)
    torch.save(output, path_save + "tokenizer")
# import torch
# import torch.nn as nn
# import torchvision
# from model import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
# from dataset import read_image_inf
# import numpy as np
# import json
# import re
# from settings import *

# def save_tokenizer(tokenizer, path_save):
#     torch.save(tokenizer, path_save + "tokenizer")

# def get_inference_model(model_config_path):
#     with open(model_config_path) as json_file:
#         model_config = json.load(json_file)

#     EMBED_DIM = model_config["EMBED_DIM"]
#     FF_DIM = model_config["FF_DIM"]
#     NUM_HEADS = model_config["NUM_HEADS"]
#     VOCAB_SIZE = model_config["VOCAB_SIZE"]

#     cnn_model = get_cnn_model()
#     encoder = TransformerEncoderBlock(
#         embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS
#     )
#     decoder = TransformerDecoderBlock(
#         embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS, vocab_size=VOCAB_SIZE
#     )
#     caption_model = ImageCaptioningModel(
#         cnn_model=cnn_model, encoder=encoder, decoder=decoder
#     )

#     ##### It's necessary for init model -> without it, weights subclass model fails
#     cnn_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
#     training = False
#     decoder_input = torch.zeros(1, 1).long()
#     caption_model(cnn_input, training, decoder_input)
#     #####

#     return caption_model

# def generate_caption(image_path, caption_model, tokenizer, SEQ_LENGTH):
#     vocab = tokenizer.get_vocabulary()
#     index_lookup = dict(zip(range(len(vocab)), vocab))
#     max_decoded_sentence_length = SEQ_LENGTH - 1

#     # Read the image from the disk
#     img = read_image_inf(image_path)

#     # Pass the image to the CNN
#     img = caption_model.cnn_model(img)

#     # Pass the image features to the Transformer encoder
#     encoded_img = caption_model.encoder(img, training=False)

#     # Generate the caption using the Transformer decoder
#     decoded_caption = "sos "
#     for i in range(max_decoded_sentence_length):
#         tokenized_caption = tokenizer([decoded_caption])[:, :-1]
#         mask = tokenized_caption != 0
#         predictions = caption_model.decoder(
#             tokenized_caption, encoded_img, training=False, mask=mask
#         )
#         sampled_token_index = torch.argmax(predictions[0, i, :])
#         sampled_token = index_lookup[sampled_token_index.item()]
#         if sampled_token == "eos":
#             break
#         decoded_caption += " " + sampled_token

#     return decoded_caption.replace("sos ", "")
