from typing import Optional

from pydantic import BaseModel, Field


class AttributeSpec(BaseModel):
    """Relevant Meta Class to collect Attribute Information"""

    name: str = Field(..., description="Attribute Name")
    description: str = Field(...,
                             description="Specification of an attribute.")


class ProductCategorySpec(BaseModel):
    """Relevant Meta Class to collect Product Information"""

    name: str = Field(..., description="Product Category Name")
    description: str = Field(..., description="Explains why a product belongs to this category.")
    attributes: list[AttributeSpec] = Field(..., description="List of potential attributes of a product in this category")


class Attribute(BaseModel):
    """Relevant Meta Class to collect Attribute Information"""

    name: str = Field(..., description="Attribute Name")
    description: str = Field(...,
                             description="Specification of an attribute.")
    examples: Optional[list[str]] = Field(description="Example values of this attribute.")


class ProductCategory(BaseModel):
    """Relevant Meta Class to collect Product Information"""

    name: str = Field(..., description="Product Category Name")
    description: str = Field(..., description="Explains why a product belongs to this category.")
    attributes: list[Attribute] = Field(..., description="List of potential attributes of a product in this category")