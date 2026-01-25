from pydantic import BaseModel, Field
from typing import Literal

class RouteDecision(BaseModel):
    """
    LLM outputs this to decide where to route the user.
    """
    route: Literal["IT","Finance"] = Field(
        description="The category of the user's request"
    )
