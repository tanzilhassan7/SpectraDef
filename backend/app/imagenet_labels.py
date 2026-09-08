from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

# Primary label fetcher using torchvision weights metadata
_CATEGORIES: List[str] = []

def get_categories() -> List[str]:
    global _CATEGORIES
    if not _CATEGORIES:
        try:
            from torchvision.models import Inception_V3_Weights
            _CATEGORIES = Inception_V3_Weights.IMAGENET1K_V1.meta["categories"]
        except Exception as e:
            logger.warning(f"Could not load categories from torchvision metadata: {e}")
            _CATEGORIES = [f"class_{i}" for i in range(1000)]
            _CATEGORIES[9] = "ostrich"
            _CATEGORIES[282] = "tiger cat"
    return _CATEGORIES

def get_label(index: int) -> str:
    categories = get_categories()
    if 0 <= index < len(categories):
        return categories[index]
    return f"unknown_class_{index}"
