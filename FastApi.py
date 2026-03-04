from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
import torch.nn.functional as F
import cv2
import numpy as np
import model_2_architecher
from sigNet_pytorch_imlementation import SiameseNetwork
from image_processing import preprocessing_images_for_prediction, calculate_std_images
from preprocess import preprocess_signature

app = FastAPI()
model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@app.on_event("startup")
def load_model():
    global model

    # model = SiameseNetwork()
    model = model_2_architecher.SigNet()
    model.load_state_dict(torch.load("model/fine_tunining2.pth"))
    model.eval()
    model = model.to(device)





@app.post("/predict")
async def predict(image1: UploadFile = File(...),
                  image2: UploadFile=File(...)):


    content1 = await image1.read()
    content2 = await image2.read()

    np_array1 = np.frombuffer(content1, np.int8)
    np_array2 = np.frombuffer(content2, np.int8)

    img1 = cv2.imdecode(np_array1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np_array2, cv2.IMREAD_GRAYSCALE)



    std = calculate_std_images([img1, img2])


    with torch.no_grad():
        img1_processed = preprocess_signature(img1)
        img2_processed = preprocess_signature(img2)

        img1_processed = img1_processed.to(device)
        img2_processed = img2_processed.to(device)
        
        # emb1, emb2 = model(img1_processed,img2_processed)
        emb1 = model(img1_processed)
        emb2 = model(img2_processed)
        distance = F.pairwise_distance(emb1,emb2)

    result=""
    if distance > 0.27:
        result = {f"mismatch"}
    else:
        result = {f"Match"}

    confidence = (torch.exp(-distance)*100).item()
    confidence_result = f"{confidence:.1f}" + "%"

    return {
        "result": result,
        "confidence": confidence_result
        }


