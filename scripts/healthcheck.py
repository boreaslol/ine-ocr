import json
import os
import urllib.request


request = urllib.request.Request(
    "http://127.0.0.1:8100/readyz",
    headers={"Authorization": "Bearer " + os.environ["INE_OCR_API_BEARER_TOKEN"]},
)
with urllib.request.urlopen(request, timeout=20) as response:
    assert response.status == 200 and json.load(response)["ok"] is True
