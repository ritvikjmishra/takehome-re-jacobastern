from training.training import main as run_training
import random
import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

class MyArgs(object):
  def __init__(self):
    self.path_for_training_data = "./pdb_2021aug02_sample"
    self.path_for_outputs = "./content/test"
    self.previous_checkpoint = ""
    self.num_epochs = 2
    self.save_model_every_n_epochs = 5
    self.reload_data_every_n_epochs = 4
    self.num_examples_per_epoch = 200
    self.batch_size = 2000
    self.max_protein_length = 2000
    self.hidden_dim = 128
    self.num_encoder_layers = 3
    self.num_decoder_layers = 3
    self.num_neighbors = 32
    self.dropout = 0.1
    self.backbone_noise = 0.1
    self.rescut = 3.5
    self.debug = False
    self.gradient_norm = -1.0 #no norm
    self.decoder_use_full_cross_attention = True
    self.cross_attention_num_heads = 4


if __name__ == "__main__":
    args = MyArgs()
    run_training(args)