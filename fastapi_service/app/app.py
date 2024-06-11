from fastapi import FastAPI, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Dict, Any
import base64
import json
import traceback


from model import load_autoencoder_model

model = None
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# app.mount("/", StaticFiles(directory="static", html=True), name="static")


class AutoencoderModelResponse(BaseModel):
    mse: float
    rmse: float


# create a route
# @app.get("/")
# def index():
#     return {"text": "Sentiment Analysis"}


# Register the function to run during startup
@app.on_event("startup")
def startup_event():
    global model
    model = load_autoencoder_model()


# # Your FastAPI route handlers go here
# @app.get("/predict")
# def predict_sentiment(text: str):
#     sentiment = model(text)

#     response = SentimentResponse(
#         text=text,
#         sentiment_label=sentiment.label,
#         sentiment_score=sentiment.score,
#     )

#     return response


@app.get("/")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_json/")
def create_upload_json(request: Request, file_upload: UploadFile):
    try:
        json_bytes = file_upload.file.read()
        dict_from_json: List[Dict[str, Any]] = json.loads(json_bytes.decode('utf-8'))  
    except Exception:
        return {"message": f"There was an error uploading the file - {traceback.format_exc()}"}
    
    response_model = model(dict_from_json)

    response = AutoencoderModelResponse(
        mse=response_model.mse,
        rmse=response_model.rmse
    )

    return templates.TemplateResponse("display.html",
                                      {"request": request,
                                       "mse": response.mse,
                                       "rmse": response.rmse})


# @app.post("/upload_json/")
# def create_upload_json(request: Request, file_upload: UploadFile):
#     try:
#         json_bytes = file_upload.file.read()
#         dict_from_json = json.loads(json_bytes.decode('utf-8'))
#         with open("uploaded_" + file_upload.filename, "wb") as f:
#             f.write(contents)
#     except Exception:
#         return {"message": "There was an error uploading the file"}
#     finally:
#         file_upload.file.close()

#     base64_encoded_image = base64.b64encode(contents).decode("utf-8")

#     return templates.TemplateResponse("display.html", {"request": request,  "myImage": base64_encoded_image})
