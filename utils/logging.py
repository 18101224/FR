import uuid 

def get_id():
    return str(uuid.uuid4())[:10]

__all__ = ['get_id']