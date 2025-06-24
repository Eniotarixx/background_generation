from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
import io
from PIL import Image
import background_generation  # mon script
import os

app = FastAPI()

@app.get("/")
def read_root():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "../../../frontend/index.html")
    with open(file_path) as f:
        return HTMLResponse(content=f.read())

@app.post("/process")
async def process(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read()))

    # Appel de ton script pour générer l'image
    result_img = background_generation.process(img)  # adapte à ta fonction

    buf = io.BytesIO()
    result_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")