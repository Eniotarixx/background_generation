from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()


# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Dossier des templates HTML
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})