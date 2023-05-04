from fastapi import FastAPI, File, UploadFile
import shutil
import os
import torch
import uvicorn
import asyncio
import update
from PIL import Image
import io

app = FastAPI()

#Instantly return a response to the client
@app.post("/update")
async def update_model(dataset: UploadFile = File(...)):
    
    temp_dataset_path = "/save_checkpoints/multi_resnet50_kd/temp-dataset.tar.gz"
    with open(temp_dataset_path, "wb") as buffer:
        buffer.write(await dataset.read())

    asyncio.create_task(update.update_model("/save_checkpoints/multi_resnet50_kd/"))
    return {"message": "Model is being updated in the background"} 

@app.post("/predict")
async def predict(image: UploadFile):
    contents = await image.read()
    pil_image = Image.open(io.BytesIO(contents))
    res = update.predict_class(pil_image)
    return {"class": res}


if __name__ == "__main__":
    uvicorn.run('main:app', host="0.0.0.0", port=8000, reload=True)