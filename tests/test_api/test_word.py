from fastapi.testclient import TestClient
from web.main import app
import pytest
import json

client = TestClient(app)

def test_one():
    name = "Murakami"
    response = client.get(f'classify?name={name}')
    assert response.status_code == 200
    response_content  = json.loads(response.content)
    assert isinstance(response_content, dict)


def test_bulk():
    names = ["Herman", "Franz", "Carl", "Fredrick"]
    params = ("&").join([f"names={name}" for name in names])
    response = client.post(f"bulk_classify?{params}")
    response_content  = json.loads(response.content)
    assert isinstance(response_content,  list)
