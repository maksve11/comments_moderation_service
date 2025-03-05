from pydantic import BaseModel

class CommentRequest(BaseModel):
    comment: str

    class Config:
        schema_extra = {
            "example": {
                "comment": "This is a sample comment text."
            }
        }

class CommentResponse(BaseModel):
    toxic: bool
    message: str

    class Config:
        schema_extra = {
            "example": {
                "toxic": False,
                "message": "This comment is safe."
            }
        }