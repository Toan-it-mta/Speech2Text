from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from pymongo import MongoClient
import random
from typing import List

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
    "http://api.aiacademy.edu.vn",
    "192.168.1.58:8002",
    "192.168.1.58"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(filename='./v_osint_sentiment.log',
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )


class Record(BaseModel):
    path: str
    trueScript: str

class Input(BaseModel):
    humman: str
    records: List[Record]

client = MongoClient("192.168.1.60", 27017)
collection = client['ASR']['ASR'] 


def on_success(data=None, message="success"):
    if data is not None:
        return {
            "message": message,
            "result": data
        }
    return {
        "message": message,
    }


def on_fail(message="fail"):
    return {
        "message": message,
    }
    
@app.get("/getAudioScript")
def getAudioScript(numberSample: int):
    try:
        cursor = collection.find({"compare": False, "visitted": False}, {'_id':0})
        samples = list(cursor)
        random_samples = random.sample(samples, min(numberSample, len(samples)))
        
        #Update sample is visitted
        for sample in random_samples:
            myquery = { "path": sample['path']}
            update_data = { "$set": { "visitted": True } }
            collection.update_one(myquery, update_data)
        return random_samples
    except Exception as e:
        return e

@app.post("/submitAudioScript")
def getAudioScript(input: Input):
    try:
        records = input.records
        for record in records:
            myquery = { "path": record.path }
            update_data = { "$set": {
                            "gold_sentence": record.trueScript,
                            "labeled": True,
                            "humman": input.humman
                            }}
    
            collection.update_many(myquery,update_data)
    except Exception as e:
        return e
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
