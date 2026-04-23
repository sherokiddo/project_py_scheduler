from pydantic import BaseModel
from typing import Any, Dict, List

class BaseScheme_Response(BaseModel):
    status:  str
    code:    int
    data:    Dict[str, Any] # json string 