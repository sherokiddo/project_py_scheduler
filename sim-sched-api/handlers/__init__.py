from fastapi import APIRouter, status
from pydantic import BaseModel

from core.imports import *

router_v1 = APIRouter(prefix="/api/v1", tags=["schedule"])

from .v1 import *