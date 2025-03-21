from fastapi import FastAPI,UploadFile ,File , HTTPException
import numpy as np
from PIL import Image
import io
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.efficientnet import decode_predictions
from models import model

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

## app.py
from fast_api.menu import menu
from fast_api.labels import imagenet_class_index
app = FastAPI()

try:
    modelo = model.create_model()
except Exception as e:
    raise  HTTPException(status_code=500,detail=f"Erro ao carregar o modelo{e}")


def preprocess_image(image_file, img_size=(224, 224)):
    logging.info("Função preprocess_image chamada.")

    try:
        image = Image.open(io.BytesIO(image_file))

        # Verificar formato da imagem
        logging.info(f"Formato da imagem: {image.format}")

        if image.format not in ['JPEG', 'PNG']:
            raise ValueError("Formato de imagem inválido. Somente JPEG e PNG são permitidos.")

        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = image.resize(img_size)
        image = np.array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        return image
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erro ao processar imagem: {e}")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    print("Fui chamado")
    try:
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")

        # Lendo o conteúdo do arquivo
        contents = await file.read()

        # Verificando o tamanho do arquivo
        MAX_SIZE = 5 * 1024 * 1024  # Limite de 5MB
        if len(contents) > MAX_SIZE:
            raise HTTPException(status_code=400, detail="O arquivo excede o tamanho máximo permitido (5MB).")

        # Processando a imagem
        processed_image = preprocess_image(contents)

        # Fazendo a predição
        prediction = modelo.predict(processed_image)

        if prediction is None or len(prediction) == 0 or len(prediction[0]) == 0:
            raise HTTPException(status_code=500, detail="Erro ao gerar a previsão")

        # Interpretando a classe prevista

        decoded_preds = decode_predictions(prediction,top=1)[0]
        predicted_class = decoded_preds[0][1]
        predicted_probalility = float(decoded_preds[0][2])


        return {
            "filename": file.filename,
            "prediction": predicted_class,
            "probability": predicted_probalility
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no servidor: {e}")


@app.get("/",summary="Endpoint inicial",description=f"Erro interno no servidor")
def hello_world_root():
    return {"Hello": "World"}

@app.get("/menu")  # Definindo o endpoint corretamente
def get_menu():
    return {"menu": menu}
