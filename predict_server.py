# predict_server.py
from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
from predict_multimodal import predict_sample
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/analyze")
async def analyze(url: str = Form(...)):
    pred_class, pred_prob = predict_sample(url=url)
    result = {
        "url_check": {
            "explanation": ["Phishing" if pred_class==1 else "Benign"],
            "score": float(pred_prob[1])
        }
    }
    return {"results": result}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
