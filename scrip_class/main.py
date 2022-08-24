from fastapi import FastAPI
import scrip
import importlib
importlib.reload(scrip)

app = FastAPI()

@app.get("/")
def home(text:str):
    pred = scrip.Predicted(text)
    return {'TIPO :': pred}
