"""
Dummy webhook receiver for local development.

Prints every incoming payload so you can see what the streamer would send
to a real downstream system (Drive-Thru Order Orchestrator, gantry controller, etc.)

Run alongside the main server and streamer:
  python webhook_dummy.py

Listens on: http://localhost:9000
Endpoint:   POST /gantry/webhook
"""

from datetime import datetime

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI(title="Dummy Webhook Receiver")


@app.post("/gantry/webhook")
async def receive(request: Request):
    payload = await request.json()

    plate      = payload.get("plate", "?")
    confidence = payload.get("confidence", "?")
    review     = payload.get("requiresReview", False)

    # Pretty-print to terminal so it's easy to see during development
    divider = "─" * 52
    flag    = "  ⚑ REQUIRES STAFF REVIEW" if review else ""
    print(f"\n{divider}")
    print(f"  PLATE DETECTED{flag}")
    print(f"{divider}")
    print(f"  Plate       : {payload.get('formatted', plate)}")
    print(f"  Confidence  : {confidence.upper()}")
    print(f"  Checksum OK : {payload.get('checksumValid')}")
    print(f"  Format      : {payload.get('format')}")
    print(f"  OCR score   : {payload.get('ocrConfidence')}")
    print(f"  Timestamp   : {payload.get('timestamp')}")
    print(f"{divider}\n")

    return JSONResponse({"received": True, "plate": plate})


@app.get("/")
def health():
    return {"status": "ok", "service": "dummy-webhook"}


if __name__ == "__main__":
    print("Dummy webhook receiver listening on http://localhost:9000")
    print("Waiting for plates...\n")
    uvicorn.run("webhook_dummy:app", host="0.0.0.0", port=9000, reload=False)