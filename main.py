import io as cio
from fastapi import FastAPI, Request, Query
from pydantic import BaseModel, Field
from skimage import io, color, feature, img_as_ubyte
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import DESCENDING, ASCENDING
from fastapi.responses import FileResponse
from bson.binary import Binary, UuidRepresentation
import math
from fastapi.encoders import jsonable_encoder
from typing import Optional
import certifi
import matplotlib.pyplot as plt
import numpy as np

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


# Summary of comparison based on the theory of GLCM features
def summarize_comparison(comparison_results):
    summary = {
        'Moist (Lembab) Skin': [],
        'Dry (Kering) Skin': []
    }

    for feature, image in comparison_results.items():
        if feature in ['contrast', 'dissimilarity']:
            if image == "Image 2":
                summary['Dry (Kering) Skin'].append(f"{feature}: {image}")
            else:
                summary['Moist (Lembab) Skin'].append(f"{feature}: {image}")
        elif feature in ['homogeneity', 'energy', 'ASM']:
            if image == "Image 1":
                summary['Moist (Lembab) Skin'].append(f"{feature}: {image}")
            else:
                summary['Dry (Kering) Skin'].append(f"{feature}: {image}")
        elif feature == 'correlation':
            # correlation can indicate uniform texture which is moist skin
            if image == "Image 1":
                summary['Moist (Lembab) Skin'].append(f"{feature}: {image}")
            else:
                summary['Dry (Kering) Skin'].append(f"{feature}: {image}")

    return summary

def getComparison(url1, url2):
    image1 = io.imread(url1)
    image2 = io.imread(url2)

    # Define distances and angles for GLCM computation
    distances = [1, 2, 3]
    angles = [0, math.pi/4, math.pi/2, 3*math.pi/4]

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

        return gray_image, glcm, features
    # Compute GLCM features, GLCM matrix, and get the grayscale images
    gray_image1, glcm1, features1 = compute_glcm_features(image1, distances, angles)
    gray_image2, glcm2, features2 = compute_glcm_features(image2, distances, angles)

    def compare_textures(features1, features2):
        comparisons = {}
        for feature_name in features1:
            if features1[feature_name] > features2[feature_name]:
                comparisons[feature_name] = "Image 1"
            else:
                comparisons[feature_name] = "Image 2"
        return comparisons

    # Compare the features of the two images
    comparison_results = compare_textures(features1, features2)

    # Display the original and processed images side by side
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image 1
    axes[0, 0].imshow(image1)
    axes[0, 0].set_title('Original Image 1')
    axes[0, 0].axis('off')

    # Processed grayscale image 1
    axes[0, 1].imshow(gray_image1, cmap='gray')
    axes[0, 1].set_title('Processed Image 1')
    axes[0, 1].axis('off')

    # Display GLCM for image 1
    glcm_display1 = np.squeeze(glcm1[:, :, 0, 0])  # Select one GLCM matrix
    axes[0, 2].imshow(glcm_display1, cmap='gray')
    axes[0, 2].set_title('GLCM 1')
    axes[0, 2].axis('off')

    # Original image 2
    axes[1, 0].imshow(image2)
    axes[1, 0].set_title('Original Image 2')
    axes[1, 0].axis('off')

    # Processed grayscale image 2
    axes[1, 1].imshow(gray_image2, cmap='gray')
    axes[1, 1].set_title('Processed Image 2')
    axes[1, 1].axis('off')

    # Display GLCM for image 2
    glcm_display2 = np.squeeze(glcm2[:, :, 0, 0])  # Select one GLCM matrix
    axes[1, 2].imshow(glcm_display2, cmap='gray')
    axes[1, 2].set_title('GLCM 2')
    axes[1, 2].axis('off')

    tmp_path = "./comparison_plot.png"  # Change this to your desired temporary directory
    plt.savefig(tmp_path, format='png')
    plt.close(fig)

    return tmp_path


def summaryCompare(url1, url2):
    image1 = io.imread(url1)
    image2 = io.imread(url2)

    # Define distances and angles for GLCM computation
    distances = [1, 2, 3]
    angles = [0, math.pi/4, math.pi/2, 3*math.pi/4]

    # Compute GLCM features, GLCM matrix, and get the grayscale images
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

        return gray_image, glcm, features
    gray_image1, glcm1, features1 = compute_glcm_features(image1, distances, angles)
    gray_image2, glcm2, features2 = compute_glcm_features(image2, distances, angles)

    def compare_textures(features1, features2):
        comparisons = {}
        for feature_name in features1:
            if features1[feature_name] > features2[feature_name]:
                comparisons[feature_name] = "Image 1"
            else:
                comparisons[feature_name] = "Image 2"
        return comparisons

    # Compare the features of the two images
    comparison_results = compare_textures(features1, features2)

    # Generate the summary
    summary = summarize_comparison(comparison_results)

    return summary

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

class Compare(BaseModel):
    url1: str
    url2: str

@app.post("/compare-image")
def read_item(request: Request, c: Compare):
    image_path = getComparison(c.url1, c.url2)
    return FileResponse(image_path, media_type="image/png")

@app.post("/compare-stats")
def read_item(request: Request, c: Compare):
    return summaryCompare(c.url1, c.url2)
