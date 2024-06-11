import os
import sys
import requests
from PIL import Image
from datetime import datetime

def convert_and_upload(folder):
    if not os.path.isdir(folder):
        print(f"The path {folder} is not a valid directory.")
        return

    # Get list of .bmp files in the directory
    bmp_files = [f for f in os.listdir(folder) if f.endswith('.bmp')]

    if not bmp_files:
        print("No .bmp files found in the directory.")
        return

    for bmp_file in bmp_files:
        bmp_path = os.path.join(folder, bmp_file)
        jpg_file = bmp_file.replace('.bmp', '.jpg')
        jpg_path = os.path.join(folder, jpg_file)
        
        # Convert .bmp to .jpg
        with Image.open(bmp_path) as img:
            img = img.convert("RGB")
            img.save(jpg_path, "JPEG")
        
        # Upload to Cloudinary
        with open(jpg_path, "rb") as file:
            files = {"file": file}
            data = {"upload_preset": "h4oea9l0"}
            response = requests.post("https://api.cloudinary.com/v1_1/dncbtxucm/image/upload", files=files, data=data)
        
        if response.status_code != 200:
            print(f"Failed to upload {jpg_file}.")
            continue

        image_url = response.json().get("url")

        # Construct dictionary
        parts = jpg_file.replace('.jpg', '').split('_')
        description = " ".join(parts)
        # print("PARTS", parts)
        setelah_pemakaian = "after" in description
        day_str = parts[0][:2]  # Extract the first 2 letters as the day
        period = parts[1]
        
        # Determine the time based on period
        time_str = "06:00:00" if period == "morning" else "18:00:00"
        waktu = f"2024-06-{day_str} {time_str}"

        data_dict = {
            "url": image_url,
            "file_name": jpg_file,
            "description": description,
            "setelah_pemakaian": setelah_pemakaian,
            "waktu": waktu
        }

        # Send POST request
        post_response = requests.post("https://citra-api.onrender.com/", json=data_dict)
        # print(jpg_file)
        # print(setelah_pemakaian)
        if post_response.status_code != 200:
            print(f"Failed to send POST request for {jpg_file}.")
        else:
            print(f"Successfully processed and uploaded {jpg_file}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_path>")
    else:
        folder_path = sys.argv[1]
        convert_and_upload(folder_path)
