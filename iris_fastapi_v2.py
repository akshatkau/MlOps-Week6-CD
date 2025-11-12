from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os
import time
import logging
import json

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

# -------------------------
# OpenTelemetry / Tracing
# -------------------------
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# -------------------------
# Structured logging (JSON)
# -------------------------
logger = logging.getLogger("iris-ml-service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(json.dumps({
    "severity": "%(levelname)s",
    "message": "%(message)s",
    "timestamp": "%(asctime)s"
}))
handler.setFormatter(formatter)
logger.addHandler(handler)

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="Iris Prediction API with Observability")

MODEL_PATH = "model.joblib"

# Train & save model if missing (keeps your original behaviour)
def train_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump((model, iris.target_names), MODEL_PATH)
    print("✅ Model trained and saved to", MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    train_model()

model, target_names = joblib.load(MODEL_PATH)

# Input schema (same fields you had)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# App readiness/liveness state
app_state = {"is_ready": False, "is_alive": True}

@app.on_event("startup")
async def startup_event():
    # simulate model loading or any warm-up
    time.sleep(1)
    app_state["is_ready"] = True

@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=500)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=503)

# middleware to add process time header
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

# Generic exception handler that logs trace_id
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

# Dummy model wrapper (we use real model but keep small simulated compute)
def inference_fn(data_array: np.ndarray):
    # small sleep to simulate heavier model if you want (adjustable)
    time.sleep(0.05)
    preds = model.predict(data_array)
    return preds

@app.post("/predict")
async def predict(iris: IrisInput, request: Request):
    with tracer.start_as_current_span("model_inference") as span:
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")

        try:
            data = np.array([[iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]])
            preds = inference_fn(data)
            class_name = target_names[preds[0]]
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                "input": {
                    "sepal_length": iris.sepal_length,
                    "sepal_width": iris.sepal_width,
                    "petal_length": iris.petal_length,
                    "petal_width": iris.petal_width
                },
                "result": class_name,
                "latency_ms": latency,
                "status": "success"
            }))

            return {"prediction": class_name}

        except Exception as e:
            logger.exception(json.dumps({
                "event": "prediction_error",
                "trace_id": trace_id,
                "error": str(e)
            }))
            raise HTTPException(status_code=500, detail="Prediction failed")
