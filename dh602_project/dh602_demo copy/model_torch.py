import torch
import torch.nn as nn
import torch.nn.functional as F
from settings import *

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.dense_proj = nn.Linear(256, embed_dim) #embed_dim, dense_dim
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        # print(self.layernorm_1)
        # self.inp_layer=nn.Linear(128,128)
    def forward(self, inputs, mask=None):
       
        # print("inputs_encoder_before_shape",inputs.shape)
        # inputs = self.dense_proj(inputs)
        # print("inputs_encoder_after_dense_shape",inputs.shape)
        # Reshape inputs to [batch_size, seq_len, embed_dim]
        inputs = inputs.unsqueeze(1)  # Add sequence length dimension
        # print("inputs_encoder_before_shape", inputs.shape)
        inputs = self.dense_proj(inputs)
        # print("inputs_encoder_after_dense_shape", inputs.shape)

        # Reshape inputs back to [batch_size, seq_len, embed_dim]
        # inputs = inputs.squeeze(1)
        # print("inputs_shape_after_squeeze:",inputs.shape)
        attention_output, _ = self.attention(
            inputs, inputs, inputs, attn_mask=None
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        return proj_input

# Example Usage
# block = TransformerEncoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)
# Assuming input tensor x of shape (batch_size, seq_length, embed_dim)
# x = torch.randn(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM)
# print("x_shape:",x.shape)
# output = block(x)
# print("output:",output.shape)

import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=sequence_length, embedding_dim=embed_dim
        )
        self.sequence_length = sequence_length
        # print("sequence_length:",self.sequence_length)
        self.vocab_size = vocab_size
        # print("vocab_size:",self.vocab_size)
        self.embed_dim = embed_dim
        # print("embed_dim:",self.embed_dim)

    def forward(self, inputs):
        batch_size, seq_length = inputs.size()
        positions = torch.arange(seq_length, device=inputs.device).unsqueeze(0).expand(batch_size, -1)
        # print("embedding_inputs:",inputs)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return inputs != 0


import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_heads, vocab_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        
        # Multi-Head Attention layers
        self.attention_1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Feed-forward network
        self.dense_proj = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.layernorm_1 = nn.LayerNorm(embed_dim)
        self.layernorm_2 = nn.LayerNorm(embed_dim)
        self.layernorm_3 = nn.LayerNorm(embed_dim)


        
        # Positional embedding
        self.embedding = PositionalEmbedding(SEQ_LENGTH, vocab_size, embed_dim)  # Adjusted
        
        # Output layer
        self.out = nn.Linear(embed_dim, vocab_size)
        
        # Dropout layers
        self.dropout_1 = nn.Dropout(0.1)
        self.dropout_2 = nn.Dropout(0.5)

    def get_causal_attention_mask(self, inputs):
        batch_size, sequence_length, embed_dim= inputs.size() #,_
        mask = torch.tril(torch.ones(sequence_length, sequence_length, device=inputs.device)) #sequence_length, sequence_length
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_length, seq_length]
        return mask

    def forward(self, inputs, encoder_outputs, training, mask=None):
        # print("dec_inputs_shape:",inputs.shape)
        inputs = self.embedding(inputs)
        # print('decoder input:',inputs)
        # print("after_self_embed:", inputs.shape)
        causal_mask = self.get_causal_attention_mask(inputs)
        inputs = self.dropout_1(inputs)
        # print('decoder input after attention and dropout',inputs)
        if(training):
            self.train()
        else: self.eval()

        if mask is not None:
            # padding_mask = mask.unsqueeze(1)
            # combined_mask = torch.minimum(padding_mask, causal_mask)
            padding_mask = mask.unsqueeze(2)

            # Add a new dimension to mask and calculate the element-wise minimum with causal_mask
            combined_mask = mask.unsqueeze(2)
            causal_mask = causal_mask
            # print("casual_mask:", causal_mask.shape)
            # print("combined_mask:", combined_mask.shape)
            combined_mask = torch.minimum(combined_mask, causal_mask)
            # print("combined_mask:", combined_mask.shape)
            # print("padding_mask:", padding_mask.shape)
        else:
            combined_mask = None
            padding_mask = None
        attention_output_1, _ = self.attention_1(inputs, inputs, inputs, attn_mask=combined_mask)
        # print('decoder attention_output_1',attention_output_1)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        # print('decoder out1',out_1)
        # print("inputs:",inputs.shape)
        # print("attention_output_1:",attention_output_1.shape)
        # print("out_1:",out_1.shape)
        padding_mask=padding_mask.float()
        # print('padding mask',padding_mask)
        # print('combined mask',combined_mask)
        # print("encoder_outputs:",encoder_outputs)
        attention_output_2, _ = self.attention_2(out_1, encoder_outputs, encoder_outputs, attn_mask=padding_mask)
        # print('decoder attention output2',attention_output_2)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        # print('decoder out2',out_2)

        proj_output = self.dense_proj(out_2)
        # print('decoder last dense proj',proj_output)
        proj_out = self.layernorm_3(out_2 + proj_output)
        # print('decoder after layernorm 3',proj_out)
        proj_out = self.dropout_2(proj_out)
        # print('decoder after dropout 2',proj_out)
        preds = self.out(proj_out)
        # print('decoder final pred',preds)
        return preds

    # def get_causal_attention_mask(self, inputs):
    #     batch_size, sequence_length, _ = inputs.size()
    #     mask = torch.tril(torch.ones(sequence_length, sequence_length, device=inputs.device))
    #     mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, seq_length, seq_length]
    #     return mask


import torch
import torch.nn as nn
import torch.nn.functional as F
# from swin_transformer import *

# model_swin=SwinModel(embed_dim, num_patch_x, num_patch_y, num_heads, num_mlp, window_size, shift_size, qkv_bias, num_classes)
# model_swin.load_state_dict(torch.load("/home/ayushh/MedMNIST_SWIN_Transformer/swin_classification_pytorch_model_weights.pth"))
# device=torch.device('cuda')
# model_swin.to(device)


class ImageCaptioningModel(nn.Module):
    def __init__(
        self,cnn_model, encoder, decoder, num_captions_per_image=1,    # cnn_model,
    ):
        super().__init__()
        self.cnn_model = cnn_model #cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.num_captions_per_image = num_captions_per_image
        self.loss = torch.nn.CrossEntropyLoss(reduction='none')

    def calculate_loss(self, y_true, y_pred, mask):
        # loss = F.cross_entropy(y_pred.transpose(1, 2), y_true, ignore_index=0, reduction='none')
        # loss *= mask
        # return loss.sum() / mask.sum()
        # print("now_calculate_loss_starts")
        loss = self.loss(y_pred.permute(0, 2, 1), y_true)
        mask = mask.to(loss.dtype)
        loss *= mask
        return torch.sum(loss) / torch.sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        # pred_ids = torch.argmax(y_pred, dim=-1)
        # correct = (pred_ids == y_true) & mask
        # return correct.sum().float() / mask.sum()
        _, predicted = torch.max(y_pred, 2)
        correct = torch.eq(predicted, y_true)
        accuracy = torch.logical_and(mask, correct)
        accuracy = accuracy.to(torch.float32)
        mask = mask.to(torch.float32)
        return torch.sum(accuracy) / torch.sum(mask)


    def forward(self, images, captions, train):
        # img_embed = self.cnn_model(inputs[0])
        # print("image_shape_before_swin:", image.shape)
        # img_embed=self.cnn_model(image)
        # print("img_embed_shape_after_swin",img_embed.shape)
        # encoder_out = self.encoder(img_embed, False)
        # print("encoder_out:", encoder_out.shape)
        # decoder_out = self.decoder(caption, encoder_out, training=train, mask=None)
        # return decoder_out
        if(train==True):
            self.train()
            batch_img=images
            batch_seq=captions
            batch_loss = 0
            batch_acc = 0
            # print('image',batch_img)
            # print("image_shape_before_swin:", batch_img.shape)
            img_embed = self.cnn_model(batch_img)
            # print('cnn model inside',self.cnn_model)
            # print("image_shape_after_swin:", img_embed.shape)
            # print('image embed',img_embed)
            encoder_out = self.encoder(img_embed)
            # print('encoder output', encoder_out)
            # print("encoder_output_shape:", encoder_out.shape)

            batch_seq_inp = batch_seq[:,:-1]
            batch_seq_true = batch_seq[:,1:]
            # print("batch_seq_inp:", batch_seq_inp.shape)
            # print("batch_seq_true:", batch_seq_true.shape)
            mask = batch_seq_true != 0

            batch_seq_pred = self.decoder(
                batch_seq_inp, encoder_out, training=True, mask=mask
            )
            # print("batch_seq_pred:", batch_seq_pred.shape)
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            batch_loss += caption_loss
            batch_acc += caption_acc

            loss = batch_loss
            acc = batch_acc / float(self.num_captions_per_image)
            
            # print({"loss": loss, "acc": acc})
            return batch_seq_pred
        
        else:
            # batch_img, batch_seq = batch_data
            self.eval()
            batch_img=images
            batch_seq=captions
            batch_loss = 0
            batch_acc = 0

            img_embed = self.cnn_model(batch_img)

            encoder_out = self.encoder(img_embed)

            batch_seq_inp = batch_seq[:, :-1]
            batch_seq_true = batch_seq[:, 1:]
            
            mask = batch_seq_true != 0

            batch_seq_pred = self.decoder(
                batch_seq_inp, encoder_out, training=False, mask=mask
            )
            
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            batch_loss += caption_loss
            batch_acc += caption_acc

            loss = batch_loss
            acc = batch_acc / float(self.num_captions_per_image)

            # print({"loss": loss, "acc": acc})
            return batch_seq_pred
        
    @property
    def metrics(self):
        return []  # Returning an empty list as PyTorch handles metrics differently
