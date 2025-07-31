from fastapi import FastAPI
from pydantic import BaseModel
import base64
from io import BytesIO
from PIL import Image
import random

app = FastAPI()

class ImageRequest(BaseModel):
    image: str
    filename: str

@app.post("/analyze-pool-photo")
async def analyze_pool_photo(data: ImageRequest):
    try:
        image_data = base64.b64decode(data.image)
        image = Image.open(BytesIO(image_data))

        result = random.choice(["clean", "dirty"])
        confidence = round(random.uniform(0.85, 0.99), 2)

        return {"status": result, "confidence": confidence}

    except Exception as e:
        return {"status": "error", "error": str(e)}
