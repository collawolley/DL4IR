# #configs on cleverdon
# class DataConfig(object):
#   all_doc_path = "/home/mdehghani/DL4IR/Data/Robust04_Data/Robust04_docs/"
#   save_dir_data = "/home/mdehghani/DL4IR/Data/Robust04_Data/save/data"
#   save_dir_model = "/home/mdehghani/DL4IR/DataRobust04_Data/save/models"
#   qrel_path = '/home/mdehghani/DL4IR/Data/Robust04_Data/qrels.robust2004.txt'
#   run_path = '/home/mdehghani/DL4IR/Data/Robust04_Data/simple_klinfo.txt'
#   vocab_size = 100000
#   train_ratio = .7
#   valid_ratio = .15
#   test_ratio = .15


class DataConfig(object):
  all_doc_path = "/Users/Mosi/Desktop/DL4IR/Robust04_Data/Robust04_docs/"
  save_dir_data = "/Users/Mosi/Desktop/DL4IR/Robust04_Data/save/Data"
  save_dir_model = "/Users/Mosi/Desktop/DL4IR/Robust04_Data/save/LinReg"
  qrel_path = '/Users/Mosi/Desktop/DL4IR/Robust04_Data/qrels.robust2004.txt'
  run_path = '/Users/Mosi/Desktop/DL4IR/Robust04_Data/simple_klinfo.txt'
  vocab_size = 100000
  train_ratio = .7
  valid_ratio = .15
  test_ratio = .15

class LogRegConfig(object):
  batch_size = 100
  num_steps = 800
  summary_steps = 100
  decay_rate = 0.5

class fullyNNConfig(object):
  summary_steps = 100
  num_hidden_nodes = 1024
  batch_size = 100
  num_steps = 800
  beta_regul = 1e-3
  dropout_keep_prob_input = 1
  dropout_keep_prob_hidden = 0.5
  learning_rate = 0.5
  decay_steps = 1000
  decay_rate = 0.65
  regularization = False
  dropout = False
  learning_rate_decay = False
