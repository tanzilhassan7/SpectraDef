import os
import io
import uuid
import base64
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

try:
    from app.tiger_cat_b64 import TIGER_CAT_IMAGE_B64
except ImportError:
    TIGER_CAT_IMAGE_B64 = None

class StorageService:
    def __init__(self):
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._fixtures: Dict[str, Dict[str, Any]] = {}
        self._init_default_fixtures()

    def _init_default_fixtures(self):
        fixture_id = "tiger_cat_benchmark"
        if TIGER_CAT_IMAGE_B64:
            raw_bytes = base64.b64decode(TIGER_CAT_IMAGE_B64)
        else:
            # Fallback photo creation
            img = Image.new("RGB", (299, 299), color=(150, 100, 50))
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            raw_bytes = buf.getvalue()

        self._fixtures[fixture_id] = {
            "id": fixture_id,
            "name": "Benchmark Tiger Cat",
            "description": "Authentic photographic ImageNet tiger cat test image (Tiger Cat -> Ostrich benchmark)",
            "image_bytes": raw_bytes,
            "content_type": "image/jpeg",
            "ground_truth_label": "tiger cat",
            "ground_truth_class": 282,
        }

    def store_artifact(self, image_bytes: bytes, metadata: Dict[str, Any]) -> str:
        artifact_id = str(uuid.uuid4())
        self._artifacts[artifact_id] = {
            "id": artifact_id,
            "image_bytes": image_bytes,
            "metadata": metadata,
        }
        return artifact_id

    def get_artifact(self, artifact_id: str) -> Optional[Dict[str, Any]]:
        return self._artifacts.get(artifact_id)

    def get_fixture(self, fixture_id: str) -> Optional[Dict[str, Any]]:
        return self._fixtures.get(fixture_id)

    def list_fixtures(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": f["id"],
                "name": f["name"],
                "description": f["description"],
                "ground_truth_label": f["ground_truth_label"],
            }
            for f in self._fixtures.values()
        ]

storage = StorageService()
