
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from telco_churn.constants import APP_HOST, APP_PORT
from telco_churn.pipeline.prediction_pipeline import TelcoData, TelcoClassifier
from telco_churn.pipeline.traning_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.SeniorCitizen: Optional[str] = None
        self.Dependents: Optional[str] = None
        self.tenure: Optional[str] = None
        self.MultipleLines: Optional[str] = None
        self.InternetService: Optional[str] = None
        self.OnlineSecurity: Optional[str] = None
        self.TechSupport: Optional[str] = None
        self.StreamingTV: Optional[str] = None
        self.StreamingMovies: Optional[str] = None
        self.Contract: Optional[str] = None
        self.PaperlessBilling: Optional[str] = None
        self.PaymentMethod: Optional[str] = None
        self.MonthlyCharges: Optional[str] = None
        self.TotalCharges: Optional[str] = None

    async def get_telco_data(self):
        form = await self.request.form()
        self.SeniorCitizen = form.get("SeniorCitizen")
        self.Dependents = form.get("Dependents")
        self.tenure = form.get("tenure")
        self.MultipleLines = form.get("MultipleLines")
        self.InternetService = form.get("InternetService")
        self.OnlineSecurity = form.get("OnlineSecurity")
        self.TechSupport = form.get("TechSupport")
        self.StreamingTV = form.get("StreamingTV")
        self.StreamingMovies = form.get("StreamingMovies")
        self.Contract = form.get("Contract")
        self.PaperlessBilling = form.get("PaperlessBilling")
        self.PaymentMethod = form.get("PaymentMethod")
        self.MonthlyCharges = form.get("MonthlyCharges")
        self.TotalCharges = form.get("TotalCharges")

@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "telco.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_telco_data()
        
        telco_data = TelcoData(
            SeniorCitizen=form.SeniorCitizen,
            Dependents=form.Dependents,
            tenure=form.tenure,
            MultipleLines=form.MultipleLines,
            InternetService=form.InternetService,
            OnlineSecurity=form.OnlineSecurity,
            TechSupport=form.TechSupport,
            StreamingTV=form.StreamingTV,
            StreamingMovies=form.StreamingMovies,
            Contract=form.Contract,
            PaperlessBilling=form.PaperlessBilling,
            PaymentMethod=form.PaymentMethod,
            MonthlyCharges=form.MonthlyCharges,
            TotalCharges=form.TotalCharges,
        )

        
        telco_df = telco_data.get_telco_input_data_frame()

        model_predictor = TelcoClassifier()

        value = model_predictor.predict(dataframe=telco_df)[0]

        status = None
        if value == 1:
            status = "Churned"
        else:
            status = "Not Churned"

        return templates.TemplateResponse(
            "telco.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)