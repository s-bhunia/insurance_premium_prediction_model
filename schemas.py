from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated

class Customer(BaseModel):
    age : Annotated[int, Field(..., ge=0, le=120, description = 'The age of the customer')]
    sex: Annotated[Literal['male', 'female'], Field(..., description="the sex of the customer(male/female)")]
    height : Annotated[float, Field(..., gt=0, le=3, description="The height of the customer in mtr")]
    weight : Annotated[float, Field(..., gt=0, le=300, description="The weight of the customer in kgs")]
    children : Annotated[int, Field(..., ge=0, le=10, description = "Number of children customer have")]
    smoker : Annotated[Literal["yes", "no"], Field(..., description = "Do the customer smoke")]
    region : Annotated[Literal['southeast', 'southwest', 'northeast', 'northwest'], Field(..., description = "the region from the customer belongs to")]

    @computed_field
    @property
    def bmi(self) -> float:
        bmi = round(self.weight / (self.height**2), 1)
        if bmi <= 0 or bmi > 100: 
            raise ValueError("Invalid BMI calculation")
        return bmi
    
