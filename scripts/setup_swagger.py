import requests
import os
from pathlib import Path

def download_swagger_files():
    # Create directories if they don't exist
    static_dir = Path("app/static/swagger-ui")
    static_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Swagger UI bundle
    bundle_url = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js"
    bundle_path = static_dir / "swagger-ui-bundle.js"
    
    if not bundle_path.exists():
        print("Downloading swagger-ui-bundle.js...")
        response = requests.get(bundle_url)
        if response.status_code == 200:
            bundle_path.write_bytes(response.content)
            print("Successfully downloaded swagger-ui-bundle.js")
        else:
            print("Failed to download swagger-ui-bundle.js")
    
    # Download favicon
    favicon_url = "https://fastapi.tiangolo.com/img/favicon.png"
    favicon_path = static_dir / "favicon.png"
    
    if not favicon_path.exists():
        print("Downloading favicon.png...")
        response = requests.get(favicon_url)
        if response.status_code == 200:
            favicon_path.write_bytes(response.content)
            print("Successfully downloaded favicon.png")
        else:
            print("Failed to download favicon.png")

if __name__ == "__main__":
    download_swagger_files()