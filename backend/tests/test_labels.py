from app.imagenet_labels import get_label, get_categories

def test_imagenet_labels():
    categories = get_categories()
    assert len(categories) == 1000
    assert get_label(9).lower().startswith("ostrich") or "ostrich" in get_label(9).lower()
    assert get_label(282).lower().startswith("tiger cat") or "tiger cat" in get_label(282).lower()
    assert get_label(9999) == "unknown_class_9999"
