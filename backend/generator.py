import torch
import requests
from io import BytesIO
from PIL import Image
from typing import List, Union , Optional

# HuggingFace Diffusers for the GenAI pipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler

class DesignCreatorIPAdapter:
    """
    A specialized Vision Model wrapper that uses IP-Adapter to blend 
    multiple reference images (e.g., Pinterest finds) into a single new design.
    """
    
    def __init__(self, base_model_id: str = "runwayml/stable-diffusion-v1-5"):
        """
        Initializes the Stable Diffusion pipeline and loads the IP-Adapter weights.
        Includes production-standard GPU memory management.
        """
        print("[*] Initializing Vision Engine...")
        
        # 1. Device Management: Automatically use NVIDIA GPU, Mac M1/M2 (MPS), or CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            self.dtype = torch.float16 # Half-precision saves massive VRAM
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.dtype = torch.float32
        else:
            self.device = "cpu"
            self.dtype = torch.float32
            print("[!] Warning: No GPU found. Generation will be very slow.")

        # 2. Load Base Pipeline
        # We use DDIM scheduler as it works very well with IP-Adapters
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            base_model_id, 
            torch_dtype=self.dtype,
            scheduler=noise_scheduler,
            safety_checker=None # Disabled for faster local prototyping
        ).to(self.device)

        # 3. Load IP-Adapter (The magic that allows images as prompts)
        print("[*] Loading IP-Adapter weights...")
        self.pipeline.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="models", 
            weight_name="ip-adapter_sd15.bin"
        )
        
        # How much the reference images influence the final design (0.0 to 1.0)
        self.pipeline.set_ip_adapter_scale(0.8) 
        print(f"[+] Vision Engine ready on {self.device.upper()}")

    def download_image(self, url: str) -> Optional[Image.Image]:
        """Helper method to convert URLs from the search agent into PIL Images."""
        try:
            # Add Browser Headers to bypass basic bot-protection firewalls
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGB")
            # Resize to standard SD resolution
            return img.resize((512, 512))
        except Exception as e:
            print(f"[-] Failed to load image from {url}: {e}")
            return None

    def generate_blended_design(
        self, 
        image_urls: List[str], 
        text_prompt: str = "A highly detailed, intricate traditional deity poshak or custom card design, vivid colors, rich embroidery or patterns, professional studio photography, flat lay",
        negative_prompt: str = "human, face, body, blurry, low resolution, deformed, text, watermark",
        num_outputs: int = 1
    ) -> List[Image.Image]:
        """
        Takes multiple image URLs, mathematically averages their visual styles, 
        and generates a new blended design.
        """
        print(f"[*] Processing {len(image_urls)} reference images...")
        
        # Convert URLs to PIL Images
        reference_images = []
        for url in image_urls:
            img = self.download_image(url)
            if img:
                reference_images.append(img)
                
        if not reference_images:
            raise ValueError("No valid reference images could be loaded.")

        print(f"[*] Generating {num_outputs} new design(s)...")
        # The pipeline automatically handles lists of images by combining their embeddings
        generated_images = self.pipeline(
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            ip_adapter_image=[reference_images],
            num_inference_steps=30,
            num_images_per_prompt=num_outputs,
            guidance_scale=7.5
        ).images

        return generated_images

# --- Quick Testing Block ---
if __name__ == "__main__":
    # Test execution (Note: Requires downloading model weights ~4GB on first run)
    try:
        creator = DesignCreatorIPAdapter()
        
        # Example: Mock URLs (In reality, these come from your agent_search.py)
        test_urls = [
            "https://placehold.co/512x512/ea580c/ffffff.png?text=Orange+Poshak",
            "https://placehold.co/512x512/ca8a04/ffffff.png?text=Gold+Zari"
        ]
        
        results = creator.generate_blended_design(test_urls, num_outputs=2)
        
        for i, img in enumerate(results):
            filename = f"generated_design_{i}.png"
            img.save(filename)
            print(f"[+] Saved {filename}")
            
    except Exception as e:
        print(f"[!] Engine Error: {e}")