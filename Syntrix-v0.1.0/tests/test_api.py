def test_import_api():
    from syntrix.inference.api import app
    assert app.title=="Syntrix Inference API"
