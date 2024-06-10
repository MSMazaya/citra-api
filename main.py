from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, Field
from skimage import io, color, feature, img_as_ubyte
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import DESCENDING, ASCENDING
from bson.binary import Binary, UuidRepresentation
from fastapi.encoders import jsonable_encoder
from typing import Optional

def compute_glcm_features(image, distances, angles):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)  # Convert to 8-bit image

    # Compute GLCM
    glcm = feature.graycomatrix(gray_image, distances, angles, symmetric=True, normed=True)

    # Extract features: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    features = {
        'contrast': feature.graycoprops(glcm, 'contrast'),
        'dissimilarity': feature.graycoprops(glcm, 'dissimilarity'),
        'homogeneity': feature.graycoprops(glcm, 'homogeneity'),
        'energy': feature.graycoprops(glcm, 'energy'),
        'correlation': feature.graycoprops(glcm, 'correlation'),
        'ASM': feature.graycoprops(glcm, 'ASM')
    }

    return gray_image, glcm, features

def get_features(url):
    image = io.imread('https://res.cloudinary.com/dw4bwn79m/image/upload/v1715842415/hbiec7fndiehrfguwbs7.jpg')

    distances = [0]
    angles = [0]

    # Compute GLCM features, GLCM matrix, and get the grayscale image
    gray_image, glcm, features = compute_glcm_features(image, distances, angles)

    print(features)

    return {
        'contrast': features["contrast"][0][0],
        'dissimilarity': features["dissimilarity"][0][0],
        'homogeneity': features["homogeneity"][0][0],
        'energy': features["energy"][0][0],
        'correlation': features["correlation"][0][0],
        'ASM': features["ASM"][0][0],
    }

origins = ["*"]


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_db_client():
    uri = "mongodb+srv://mmazaya:mvvWaAkr*3-7VU#@cluster0.f1ivbxc.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))
    app.mongodb_client = client
    app.database = app.mongodb_client['citra']
    print("Connected to the MongoDB database!")

@app.on_event("shutdown")
def shutdown_db_client():
    app.mongodb_client.close()


class Photo(BaseModel):
    id: str = Field(default_factory=uuid.uuid4, alias="_id")
    url: str
    file_name: str
    description: str
    setelah_pemakaian: bool
    waktu:str

@app.post("/")
def read_root(request: Request, photo: Photo):
    data = jsonable_encoder(photo)
    data["features"] = get_features(photo.url)
    data_inserted = request.app.database["tubes"].insert_one(data)
    created_data = request.app.database["tubes"].find_one(
        {"_id": data_inserted.inserted_id}
    )
    return created_data

@app.post("/get-data")
def read_item(request: Request, setelah_pemakaian: Optional[bool] = Query(None)):
    query_filter = {}
    if setelah_pemakaian is not None:
        query_filter["setelah_pemakaian"] = setelah_pemakaian
    return list(request.app.database["tubes"].find(query_filter).sort("waktu", ASCENDING))