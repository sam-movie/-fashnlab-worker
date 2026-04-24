import runpod
import requests
import base64
from io import BytesIO
from PIL import Image
from fashn_vton import TryOnPipeline

# Load model at cold start
pipeline = TryOnPipeline(weights_dir="./weights")

def download_image(url):
    """Download image from URL and return PIL Image."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def handler(event):
    """RunPod serverless handler for virtual try-on."""
    try:
        input_data = event["input"]
        
        person_image_url = input_data["person_image_url"]
        garment_image_url = input_data["garment_image_url"]
        category = input_data.get("category", "tops")
        
        # Download images
        person_image = download_image(person_image_url)
        garment_image = download_image(garment_image_url)
        
        # Run inference
        result = pipeline(
            person_image,
            garment_image,
            category=category,
            num_inference_steps=30,
            guidance_scale=2.5,
        )
        
        # Convert result to base64
        output_image = result.images[0]
        buffered = BytesIO()
        output_image.save(buffered, format="JPEG", quality=95)
        result_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return {"status": "success", "image_base64": result_base64}
        
    except Exception as e:
        return {"status": "error", "error": str(e)}

runpod.serverless.start({"handler": handler})
