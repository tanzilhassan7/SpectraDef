from fastapi import APIRouter, HTTPException, Response
from typing import List

from app.storage import storage
from app.schemas import FixtureItem

router = APIRouter(prefix="/api/fixtures", tags=["fixtures"])

@router.get("", response_model=List[FixtureItem])
async def list_fixtures():
    return storage.list_fixtures()

@router.get("/{fixture_id}")
async def get_fixture_image(fixture_id: str):
    fixture = storage.get_fixture(fixture_id)
    if not fixture:
        raise HTTPException(status_code=404, detail=f"Fixture '{fixture_id}' not found")
    return Response(content=fixture["image_bytes"], media_type=fixture["content_type"])
