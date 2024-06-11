from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, Field
from skimage import io, color, feature, img_as_ubyte
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import DESCENDING, ASCENDING
from bson.binary import Binary, UuidRepresentation
import math
from fastapi.encoders import jsonable_encoder
from typing import Optional
import certifi
ca = certifi.where()

def compute_glcm_features(image, distances, angles):
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)
    gray_image = img_as_ubyte(gray_image)  # Convert to 8-bit image

    # Compute GLCM
    glcm = feature.graycomatrix(gray_image, distances, angles, symmetric=True, normed=True)

    # Extract features: contrast, dissimilarity, homogeneity, energy, correlation, ASM
    features = {
        'contrast': feature.graycoprops(glcm, 'contrast').mean(),
        'dissimilarity': feature.graycoprops(glcm, 'dissimilarity').mean(),
        'homogeneity': feature.graycoprops(glcm, 'homogeneity').mean(),
        'energy': feature.graycoprops(glcm, 'energy').mean(),
        'correlation': feature.graycoprops(glcm, 'correlation').mean(),
        'ASM': feature.graycoprops(glcm, 'ASM').mean()
    }

    return features

def get_features(url):
    image = io.imread(url)

    distances = [1, 2, 3]
    angles = [0, math.pi/4, math.pi/2, 3*math.pi/4]

    # Compute GLCM features, GLCM matrix, and get the grayscale image
    features = compute_glcm_features(image, distances, angles)

    print(features)

    return {
        'contrast': features["contrast"],
        'dissimilarity': features["dissimilarity"],
        'homogeneity': features["homogeneity"],
        'energy': features["energy"],
        'correlation': features["correlation"],
        'ASM': features["ASM"],
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
    client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=ca)
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