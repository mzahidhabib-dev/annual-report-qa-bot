import os
from dotenv import load_dotenv
from src.ingestion.image_extractor import extract_images, describe_image

load_dotenv()

def run_test():
    pdf_path = os.path.join("sample_docs", "Northbridge_Dynamics_2025_Annual_Report.pdf")
    output_dir = "temp_images"
    
    print(f"==================================================")
    print(f"PHASE 3: IMAGE EXTRACTION (Isolated Test)")
    print(f"==================================================")
    
    print(f"Step 1: Extracting images from {pdf_path}")
    images = extract_images(pdf_path, output_dir)
    print(f"[SUCCESS] Found {len(images)} pages containing images.")
    
    if not images:
        print("No images found to process. Exiting.")
        return
        
    print(f"\n[INFO] For testing speed, we will only analyze the first 3 image pages found.")
    
    print("\nStep 2: Describing images with Gemini Vision")
    for img_data in images[:3]:
        page_num = img_data["page_number"]
        path = img_data["image_path"]
        print(f"\n--- Analyzing Page {page_num} ({path}) ---")
        desc = describe_image(path)
        print(f"Vision Output:\n{desc}\n")

if __name__ == "__main__":
    run_test()
