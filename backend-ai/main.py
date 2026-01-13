from fastapi import UploadFile, File
from services.predict import predict_image
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Static files (CSS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates (HTML)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def landing_page(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )
@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(
        "predict.html",
        {"request": request}
    )
@app.get("/result", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(
        "result.html",
        {"request": request}
    )
    
@app.post("/predict", response_class=HTMLResponse)
async def predict_result(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()

    prediction, confidence = predict_image(image_bytes)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prediction": prediction,
            "confidence": f"{confidence:.2f}"
        }
    )
