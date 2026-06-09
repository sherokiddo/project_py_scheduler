from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from handlers import router_v1

app = FastAPI(title="Schedule API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_v1)

@app.get("/")
async def root():
    return {"message": "Schedule API is running"}
