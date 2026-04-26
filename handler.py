import runpod
import requests
import base64
import sys
from io import BytesIO
from PIL import Image
from fashn_vton import TryOnPipeline

print("[BOOT] Loading TryOnPipeline...", flush=True)
pipeline = TryOnPipeline(weights_dir="/app/weights")
print("[BOOT] Pipeline ready", flush=True)


def download_image(url):
    print(f"[DOWNLOAD] Fetching {url[:80]}...", flush=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    print(f"[DOWNLOAD] OK ({len(response.content)} bytes)", flush=True)
    return Image.open(BytesIO(response.content)).convert("RGB")


def handler(event):
    try:
        print(f"[HANDLER] New job: {event.get('id', 'unknown')}", flush=True)
        input_data = event["input"]

        person_image_url = input_data["person_image_url"]
        garment_image_url = input_data["garment_image_url"]
        category = input_data.get("category", "tops")

        print(f"[HANDLER] Category: {category}", flush=True)
        print("[HANDLER] Downloading person image...", flush=True)
        person_image = download_image(person_image_url)
        print("[HANDLER] Downloading garment image...", flush=True)
        garment_image = download_image(garment_image_url)

        print("[HANDLER] Running inference...", flush=True)
        result = pipeline(person_image, garment_image, category=category)
        print("[HANDLER] Inference done, encoding output...", flush=True)

        output_image = result.images[0]
        buffered = BytesIO()
        output_image.save(buffered, format="JPEG", quality=95)
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        print(f"[HANDLER] Done, returning {len(result_base64)} chars of base64", flush=True)
        return {"status": "success", "image_base64": result_base64}

    except Exception as e:
        print(f"[HANDLER] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return {"status": "error", "error": str(e)}


runpod.serverless.start({"handler": handler})
