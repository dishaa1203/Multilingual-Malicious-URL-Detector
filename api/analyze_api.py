from fastapi import FastAPI, UploadFile, Form, File
from modules.url_module import predict_url
from modules.text_module import predict_text
from modules.file_module import analyze_file
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
# Create FastAPI instance
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with ["chrome-extension://<extension-id>"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze(
    url: str = Form(None),
    text: str = Form(None),
    file: UploadFile = File(None)
):
    results = {}
    
    if url:
        score, expl = predict_url(url)
        results["url_check"] = {"score": score, "explanation": expl}

    if text:
        score, expl = predict_text(text)
        results["text_check"] = {"score": score, "explanation": expl}

    if file:
        path = f"temp_{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        score, expl = analyze_file(path)
        os.remove(path)
        results["file_check"] = {"score": score, "explanation": expl}

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
