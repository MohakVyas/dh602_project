import torch
import torch.nn as nn
import torch.nn.functional as F

# Desired image dimensions
IMAGE_SIZE = (299, 299)
# Max vocabulary size
MAX_VOCAB_SIZE = 2000000
# Fixed length allowed for any sequence
SEQ_LENGTH = 25
# Dimension for the image embeddings and token embeddings
EMBED_DIM = 516
# Number of self-attention heads
NUM_HEADS = 6
# Per-layer units in the feed-forward network
FF_DIM = 516 
# Shuffle dataset dim on tf.data.Dataset
SHUFFLE_DIM = 516
# Batch size
BATCH_SIZE = 64
# Numbers of training epochs
EPOCHS = 14

# Reduce Dataset
# If you want reduce number of train/valid images dataset, set 'REDUCE_DATASET=True'
# and set number of train/valid images that you want.
#### COCO dataset
# Max number train dataset images : 68363
# Max number valid dataset images : 33432
REDUCE_DATASET = False
# Number of train images -> it must be a value between [1, 68363]
NUM_TRAIN_IMG = 68363
# Number of valid images -> it must be a value between [1, 33432]
# N.B. -> IMPORTANT : the number of images of the test set is given by the difference between 33432 and NUM_VALID_IMG values.
# for instance, with NUM_VALID_IMG = 20000 -> valid set have 20000 images and test set have the last 13432 images.
NUM_VALID_IMG = 20000
# Data augmentation on train set
TRAIN_SET_AUG = True
# Data augmentation on valid set
VALID_SET_AUG = False
# If you want to calculate the performance on the test set.
TEST_SET = False

# Load train_data.json pathfile
train_data_json_path = "COCO_dataset/captions_mapping_train.json"
# Load valid_data.json pathfile
valid_data_json_path = "COCO_dataset/captions_mapping_valid.json"
# Load text_data.json pathfile
text_data_json_path  = "COCO_dataset/text_data.json"

# Save training files directory
SAVE_DIR = "save_train_dir/"

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.dense_proj = nn.Linear(embed_dim, dense_dim)
        self.layernorm_1 = nn.LayerNorm(embed_dim)

    def forward(self, inputs, mask=None):
        inputs = self.dense_proj(inputs)
        attention_output, _ = self.attention(
            inputs, inputs, inputs, attn_mask=None
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        return proj_input

# Example Usage
block = TransformerEncoderBlock(EMBED_DIM, FF_DIM, NUM_HEADS)
# Assuming input tensor x of shape (batch_size, seq_length, embed_dim)
x = torch.randn(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM)
output = block(x)
print(output.shape)
