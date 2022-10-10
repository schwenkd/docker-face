from starlette.responses import StreamingResponse
from fastapi import FastAPI, File, UploadFile
import requests

#We generate a new FastAPI app in the Prod environment
#https://fastapi.tiangolo.com/
app = FastAPI(title='Health Check')



#Call your get function for a health Check
#to receive both (face-bokeh and face-emotion)
@app.get("/", tags=["Health Check"])
async def root():
    r1 = requests.get("http://34.201.243.175:8001/")
    r2 = requests.get("http://34.201.243.175:8002/")

    return {"face-bokeh-status": r1.reason, "face-emotion-status": r2.reason}