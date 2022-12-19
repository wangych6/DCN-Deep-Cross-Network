
import logging
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC

from config import parse_args
from model import DCN
from utils import setup_logging
from dataloader import create_virtual_data


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# If you have GPU, and the value is GPU serial number.
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def train_and_validate(args):
    model_params = {
        'hidden_units': [256, 128, 64],
        'dnn_dropout': 0.5,
        'embed_reg': 0.,
        'cross_w_reg': 0.,
        'cross_b_reg': 0.
    }
    
    feature_columns, train_data = create_virtual_data(args)
    
    mirrored_strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
    with mirrored_strategy.scope():
        model = DCN(feature_columns, **model_params)
        model.summary()
        model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=args.learning_rate), metrics=[AUC()])
    
    start = time.time()
    model.fit(x=train_data[0],
              y=train_data[1],
              epochs=1,
              batch_size=args.batch_size,
              )
    end = time.time()
    return end - start
    
    
def main():
    args = parse_args()
    setup_logging()
    
    logging.info("Star train, parameters: %s", args)
    total_time = train_and_validate(args)
    thoughout = args.batch_size * args.steps / total_time
    logging.info("############ Finish train, and thoughout: %s record/s, batch_size: %s", str(thoughout), str(args.batch_size))


if __name__ == "__main__":
    main()