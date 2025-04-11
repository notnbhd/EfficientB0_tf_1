import os
import requests
import time
import json
import io
import hashlib
import argparse
import logging
import random
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_directories(base_dir, query):
    """Create necessary directories for the dataset"""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, "raw", query), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "train", query), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", query), exist_ok=True)
    logging.info(f"Created directories in {base_dir}")

def search_images_google_api(query, num_images=100, api_key=None, search_engine_id=None):
    """Search for images using Google Custom Search API"""
    api_key = api_key or "***"
    search_engine_id = search_engine_id or "***"
    GOOGLE_SEARCH_API_URL = "https://www.googleapis.com/customsearch/v1"
    
    image_urls = []
    
    # Google API returns max 10 results per query, so we need to paginate
    items_per_page = 10
    num_pages = min(10, (num_images + items_per_page - 1) // items_per_page)
    
    for page in range(1, num_pages + 1):
        start_index = (page - 1) * items_per_page + 1
        
        params = {
            'key': api_key,
            'cx': search_engine_id,
            'q': query,
            'searchType': 'image',
            'num': items_per_page,
            'start': start_index,
            'imgSize': 'large',
            'safe': 'active',
        }
        
        try:
            response = requests.get(GOOGLE_SEARCH_API_URL, params=params)
            response.raise_for_status()
            
            search_results = response.json()
            
            if 'items' in search_results:
                for item in search_results['items']:
                    if 'link' in item:
                        image_urls.append(item['link'])
            
            logging.info(f"Retrieved {len(image_urls)} images so far (page {page}/{num_pages})")
            time.sleep(1)
            
        except Exception as e:
            logging.error(f"Error searching Google API (page {page}): {str(e)}")
    
    return image_urls[:num_images]

def download_image(url, save_path):
    """Download an image from a URL and save it"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check if the content is actually an image
        content_type = response.headers.get('Content-Type', '')
        if 'image' not in content_type:
            logging.warning(f"URL does not contain an image: {url}")
            return None
            
        # Open the image to verify it's valid
        img = Image.open(io.BytesIO(response.content))
        img.verify()
        
        # Save the image
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        return save_path
    except Exception as e:
        logging.error(f"Error downloading {url}: {str(e)}")
        return None

def get_image_hash(image_path):
    """Generate a hash of an image to detect duplicates"""
    try:
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        logging.error(f"Error hashing {image_path}: {str(e)}")
        return None

def download_and_process_images(image_urls, raw_dir, min_width=256, min_height=256):
    """Download and perform basic processing on images"""
    downloaded_count = 0
    valid_images = []
    hashes = set()
    
    for i, url in enumerate(image_urls):
        save_path = os.path.join(raw_dir, f"image_{i:04d}.jpg")
        
        # Try to download the image
        if not download_image(url, save_path):
            continue
            
        # Check for duplicates using hash
        img_hash = get_image_hash(save_path)
        if img_hash in hashes:
            logging.info(f"Skipping duplicate image: {os.path.basename(save_path)}")
            if os.path.exists(save_path):
                os.remove(save_path)
            continue
        hashes.add(img_hash)
        
        try:
            # Verify image dimensions
            img = Image.open(save_path)
            
            # Filter by size
            if img.width < min_width or img.height < min_height:
                logging.info(f"Skipping small image: {os.path.basename(save_path)}")
                if os.path.exists(save_path):
                    os.remove(save_path)
                continue
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(save_path, "JPEG", quality=90)
            
            valid_images.append(save_path)
            downloaded_count += 1
            logging.info(f"Downloaded and verified image {downloaded_count}: {os.path.basename(save_path)}")
            
        except Exception as e:
            logging.error(f"Error processing {save_path}: {str(e)}")
            if os.path.exists(save_path):
                os.remove(save_path)
    
    return downloaded_count, valid_images

def split_dataset(query, images, base_dir, train_ratio=0.8):
    """Split images into train and test sets"""
    # Shuffle the images
    random.shuffle(images)
    
    # Calculate split point
    split_idx = int(len(images) * train_ratio)
    
    # Split into train and test
    train_images = images[:split_idx]
    test_images = images[split_idx:]
    
    # Copy images to respective directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    
    # Process train images
    for i, img_path in enumerate(train_images):
        try:
            img = Image.open(img_path)
            train_path = os.path.join(train_dir, query ,f"train_{i:04d}.jpg")
            img.save(train_path, "JPEG", quality=90)
            logging.info(f"Added to training set: {os.path.basename(train_path)}")
        except Exception as e:
            logging.error(f"Error copying to train set: {str(e)}")
    
    # Process test images
    for i, img_path in enumerate(test_images):
        try:
            img = Image.open(img_path)
            test_path = os.path.join(test_dir, query, f"test_{i:04d}.jpg")
            img.save(test_path, "JPEG", quality=90)
            logging.info(f"Added to test set: {os.path.basename(test_path)}")
        except Exception as e:
            logging.error(f"Error copying to test set: {str(e)}")
    
    return len(train_images), len(test_images)

def create_dataset(query, base_dir="./dataset", num_images=100, train_ratio=0.8, 
                  api_key=None, search_engine_id=None):
    """Main function to create and split an image dataset"""
    # Set up directories
    setup_directories(base_dir, query)
    raw_dir = os.path.join(base_dir, "raw", query)
    
    # Search for images
    image_urls = search_images_google_api(query, num_images, api_key, search_engine_id)
    logging.info(f"Found {len(image_urls)} image URLs")
    
    # Save URLs for reference
    with open(os.path.join(base_dir, "image_urls.txt"), "w") as f:
        for url in image_urls:
            f.write(f"{url}\n")
    
    # Download and process images
    downloaded_count, valid_images = download_and_process_images(image_urls, raw_dir)
    logging.info(f"Successfully downloaded and processed {downloaded_count} images")
    
    # Split the dataset
    train_count, test_count = split_dataset(query, valid_images, base_dir, train_ratio)
    
    # Create a summary file
    with open(os.path.join(base_dir, "dataset_summary.txt"), "w") as f:
        f.write(f"Dataset Summary\n")
        f.write(f"==============\n\n")
        f.write(f"Search Query: {query}\n")
        f.write(f"Date Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Images Downloaded: {downloaded_count}\n")
        f.write(f"Training Set: {train_count} images ({train_ratio*100:.1f}%)\n")
        f.write(f"Test Set: {test_count} images ({(1-train_ratio)*100:.1f}%)\n")
    
    return {
        "total": downloaded_count,
        "train": train_count,
        "test": test_count
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and split an image dataset for ML")
    parser.add_argument("query", help="Search query for images")
    parser.add_argument("--dir", default="./dataset", help="Base directory for the dataset")
    parser.add_argument("--num", type=int, default=100, help="Maximum number of images to download")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data (0-1)")
    parser.add_argument("--api-key", help="Google API Key")
    parser.add_argument("--cx", help="Custom Search Engine ID")
    
    args = parser.parse_args()
    
    print(f"Creating dataset for: {args.query}")
    print(f"Base directory: {args.dir}")
    print(f"Train/test split: {args.train_ratio*100:.1f}%/{(1-args.train_ratio)*100:.1f}%")
    
    result = create_dataset(
        query=args.query,
        base_dir=args.dir,
        num_images=args.num,
        train_ratio=args.train_ratio,
        api_key=args.api_key,  
        search_engine_id=args.cx
    )
    
    print("\nDataset creation complete!")
    print(f"Total images: {result['total']}")
    print(f"Training set: {result['train']} images")
    print(f"Test set: {result['test']} images")
    print(f"Dataset stored in: {os.path.abspath(args.dir)}")
