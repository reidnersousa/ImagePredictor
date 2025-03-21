import tensorflow as tf
from tensorflow.keras import layers
import os

# Criar a mesma arquitetura do modelo original
def create_model():

    model = tf.keras.applications.EfficientNetB0(weights='imagenet')
    return model




### TODO  carrega o modelo junto com apifast e testa o modelo não terá fine-tuning