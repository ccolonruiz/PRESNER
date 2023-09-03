import tensorflow as tf
import os, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.get_logger().setLevel('ERROR')
import transformers
transformers.logging.set_verbosity_error()
from transformers import BertTokenizer
import yaml

class NER_CBBERT():
    def __init__(self, cfg):
        with open(cfg, "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        self.cfg = cfg
        caller_path = os.path.dirname(os.path.abspath((inspect.stack()[1])[1]))
        self.tokenizer = BertTokenizer.from_pretrained(caller_path+cfg['path_bert_tokenizer'])
        self.model = tf.keras.models.load_model(caller_path+cfg['path_bert_model'])
