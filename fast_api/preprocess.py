import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from models import model
import os



def preprocess_image(image_file,img_size=(128,128)):
    try:

        #image = Image.open(io.BytesIO(image_file))
        image = Image.open(image_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize(img_size)
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image,axis=0)
        return image
    except Exception as e:
        raise ValueError(f"Erro ao processar imagem:{e}")

def predict(image_file,modelo):
    """
    :param image_file: Recebe a img
    :return: retorna a previsão do modelo
    """
    processed_image = preprocess_image(image_file)
    prediction = modelo.predict(processed_image)
    predicted_class = "Woman" if prediction >= 0.5 else "Man"

    print(f"Previsão: {predicted_class} (Probalidade: {prediction[0][0]:.4f})")



