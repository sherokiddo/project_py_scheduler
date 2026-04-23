from fastapi import FastAPI
from handlers import router_v1

app = FastAPI(title="Schedule API")

app.include_router(router_v1)

@app.get("/")
async def root():
    return {"message": "Schedule API is running"}
