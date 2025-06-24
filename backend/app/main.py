from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io
from PIL import Image
import background_generation  # ton script

app = FastAPI()

@app.get("/")
def read_root():
    with open("index.html") as f:
        return f.read()

@app.post("/process")
async def process(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read()))

    # Appel de ton script pour générer l'image
    result_img = background_generation.process(img)  # adapte à ta fonction
    
    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")