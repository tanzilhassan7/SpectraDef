from PIL import Image
from app.services.transformation_service import transformation_service

def test_transformations():
    img = Image.new("RGB", (299, 299), color=(200, 100, 50))
    transforms = ["blur", "jpeg", "resize", "brightness", "contrast", "noise", "sharpen"]

    for name in transforms:
        transformed = transformation_service.apply_transform(img, name, strength=0.5)
        assert isinstance(transformed, Image.Image)
        assert transformed.size == (299, 299)

def test_multi_view_inference():
    img = Image.new("RGB", (299, 299), color=(200, 100, 50))
    views = transformation_service.run_multi_view_inference(img, strength=0.5)
    assert len(views) == 7
    for v in views:
        assert "name" in v
        assert "prediction" in v
        assert "confidence" in v
