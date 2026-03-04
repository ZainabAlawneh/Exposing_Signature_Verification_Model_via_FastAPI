
import requests


url = "http://localhost:8000/predict"
image_path1 = "test_images/im1.png"
image_path2 = "test_images/im2.png"


with open(image_path1, "rb") as f1, open(image_path2, "rb") as f2:
    files = {"image1": f1,
             "image2": f2
             }
    response = requests.post(url, files=files)
    print("file is sent")
print("Status code:", response.status_code)
print("Response text:", response.content)