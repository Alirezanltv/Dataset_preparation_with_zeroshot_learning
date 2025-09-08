import streamlit as st
import concurrent.futures
import hashlib
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException
except ImportError:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import RatelimitException

def initialize_yolo_world():
    """Initialize YOLO-World model using Ultralytics and store in session state"""
    with st.status("Initializing YOLO-World model..."):
        try:
            # Download and load YOLO-World model
            model = YOLO("yolov8l-world.pt")  # or yolov8m-world.pt, yolov8l-world.pt
            st.session_state.yolo_model = model
            st.success("✅ YOLO-World model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to load YOLO-World: {e}")
            return False

def get_yolo_model():
    """Get YOLO model from session state"""
    if 'yolo_model' not in st.session_state:
        st.error("YOLO model not found in session state")
        return None
    return st.session_state.yolo_model

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def apply_nms_simple(boxes, scores, labels, iou_threshold=0.3):
    """Simple Non-Maximum Suppression implementation"""
    if len(boxes) == 0:
        return [], [], []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Get indices sorted by score (highest first)
    indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(indices) > 0:
        # Keep the box with highest score
        current_idx = indices[0]
        keep_indices.append(current_idx)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current_idx]
        remaining_indices = indices[1:]
        
        # Calculate IoU for all remaining boxes
        ious = []
        for idx in remaining_indices:
            iou = calculate_iou(current_box, boxes[idx])
            ious.append(iou)
        
        ious = np.array(ious)
        
        # Keep only boxes with IoU below threshold
        indices = remaining_indices[ious < iou_threshold]
    
    # Return filtered results
    filtered_boxes = [boxes[i].tolist() for i in keep_indices]
    filtered_scores = [scores[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]
    
    return filtered_boxes, filtered_scores, filtered_labels

def test_yolo_world():
    """Test YOLO-World with known image"""
    with st.status("Testing YOLO-World with sample image..."):
        try:
            # Get model from session state
            model = get_yolo_model()
            if model is None:
                st.error("YOLO model not available for testing")
                return False
            
            # Test with cats image
            test_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            test_image_pil = Image.open(requests.get(test_url, stream=True).raw)
            test_image_np = np.array(test_image_pil)
            
            # Display test image
            st.image(test_image_pil, caption="Test image", width=400)
            
            # Set test classes for YOLO-World
            model.set_classes(["cat", "remote control", "couch", "person"])
            
            # Run inference
            results = model(test_image_np, conf=0.1, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                # Show results in a table
                result_data = []
                for i, (box, conf, cls) in enumerate(zip(boxes.xyxy, boxes.conf, boxes.cls)):
                    class_name = model.names[int(cls)]
                    result_data.append({"Class": class_name, "Confidence": f"{conf:.3f}"})
                
                st.table(result_data)
                st.success("✅ YOLO-World is working correctly!")
                return True
            else:
                st.error("❌ YOLO-World test failed - no detections")
                return False
                
        except Exception as e:
            st.error(f"❌ YOLO-World test failed: {e}")
            return False

def detect_objects_yolo(image_path: Path, target_classes: List[str]) -> Tuple[Dict, np.ndarray]:
    """Detect objects using YOLO-World"""
    try:
        # Get model from session state
        model = get_yolo_model()
        if model is None:
            st.error("YOLO model not available for detection")
            return {
                "image_path": str(image_path),
                "image_size": [0, 0],
                "boxes": [],
                "scores": [],
                "labels": [],
                "target_classes": target_classes
            }, None
        
        # Load image
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        
        # Set classes for YOLO-World
        model.set_classes(target_classes)
        
        # Try different confidence thresholds
        best_boxes = []
        best_scores = []
        best_labels = []
        best_count = 0
        
        confidence_levels = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        
        for conf in confidence_levels:
            try:
                # Run inference
                results = model(image_np, conf=conf, verbose=False)
                
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes_tensor = results[0].boxes
                    
                    # Extract data
                    boxes = boxes_tensor.xyxy.cpu().numpy().tolist()
                    scores = boxes_tensor.conf.cpu().numpy().tolist()
                    class_ids = boxes_tensor.cls.cpu().numpy().astype(int).tolist()
                    labels = [target_classes[class_id] for class_id in class_ids]
                    
                    # Apply NMS
                    nms_boxes, nms_scores, nms_labels = apply_nms_simple(
                        boxes, scores, labels, iou_threshold=0.3
                    )
                    
                    if len(nms_boxes) > best_count:
                        best_boxes = nms_boxes
                        best_scores = nms_scores
                        best_labels = nms_labels
                        best_count = len(nms_boxes)
                        
                    if len(nms_boxes) > 0:
                        break  # Found something good
                    
            except Exception:
                continue
        
        # Convert to our format
        detection_data = {
            "image_path": str(image_path),
            "image_size": list(image_np.shape[:2]),
            "boxes": best_boxes,
            "scores": best_scores,
            "labels": best_labels,
            "target_classes": target_classes
        }
        
        return detection_data, image_np
        
    except Exception as e:
        st.error(f"Error detecting objects in {image_path}: {e}")
        return {
            "image_path": str(image_path),
            "image_size": [0, 0],
            "boxes": [],
            "scores": [],
            "labels": [],
            "target_classes": target_classes
        }, None

def create_visualization(image, detections: Dict):
    """Create visualization using OpenCV and return as PIL Image"""
    try:
        if image is None:
            return None
        
        boxes = detections["boxes"]
        scores = detections["scores"]
        labels = detections["labels"]
        
        # Convert to BGR for OpenCV (we'll convert back later)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image.copy()
        
        # Colors for different classes
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 128),  # Purple
            (255, 165, 0)   # Orange
        ]
        
        if len(boxes) > 0:
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                x1, y1, x2, y2 = [int(coord) for coord in box]
                color = colors[i % len(colors)]
                
                # Draw thick rectangle
                thickness = 3
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, thickness)
                
                # Prepare label text
                label_text = f"{label}: {score:.3f}"
                
                # Calculate text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                text_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, font, font_scale, text_thickness
                )
                
                # Draw label background
                label_bg_x1 = x1
                label_bg_y1 = max(0, y1 - text_height - 10)
                label_bg_x2 = x1 + text_width + 10
                label_bg_y2 = y1
                
                cv2.rectangle(image_bgr, 
                             (label_bg_x1, label_bg_y1), 
                             (label_bg_x2, label_bg_y2), 
                             color, -1)
                
                # Draw label text
                cv2.putText(image_bgr, label_text, 
                           (x1 + 5, y1 - 5), 
                           font, font_scale, (255, 255, 255), text_thickness)
        else:
            # Add "No detections" text
            cv2.putText(image_bgr, "NO OBJECTS DETECTED", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add title
        title = f"YOLO-World: {len(boxes)} objects detected"
        cv2.putText(image_bgr, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(image_bgr, title, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
        
        # Convert back to RGB for display
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
        
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None

def download_one_image(index_and_result: Tuple[int, Dict], out_dir: Path, timeout: int = 15) -> Optional[Path]:
    """Download single image"""
    idx, result = index_and_result
    
    url = result.get("image")
    if not url:
        return None
    
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        if response.status_code != 200:
            return None
        
        # Create filename
        hsh = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        w = result.get("width", 0)
        h = result.get("height", 0)
        filename = f"{idx:04d}_{w}x{h}_{hsh}.jpg"
        filepath = out_dir / filename
        
        # Save image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        # Verify it's a valid image
        try:
            Image.open(filepath).verify()
            return filepath
        except:
            if filepath.exists():
                filepath.unlink()  # Delete corrupted image
            return None
            
    except Exception:
        return None

def download_images_for_topic(topic: str, query: str, limit: int, base_dir: Path) -> List[Path]:
    """Download images for a specific topic"""
    topic_dir = base_dir / topic
    topic_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    
    with st.status(f"Searching for {topic} images...") as status:
        # Search for images
        try:
            with DDGS() as ddgs:
                results = []
                for r in ddgs.images(
                    query=query,
                    region="wt-wt",
                    safesearch="off",
                    max_results=limit * 2,
                ):
                    results.append(r)
                    if len(results) >= limit:
                        break
                        
            status.update(label=f"Found {len(results)} image candidates")
            
            if not results:
                st.warning(f"No results found for '{topic}'")
                return []
            
            # Download images in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                jobs = list(enumerate(results))
                
                download_progress = st.progress(0)
                for i, file_path in enumerate(executor.map(
                    lambda x: download_one_image(x, topic_dir), jobs
                )):
                    if file_path:
                        downloaded_files.append(file_path)
                        status.update(label=f"Downloaded {len(downloaded_files)}/{limit} images")
                    
                    # Update progress
                    download_progress.progress((i + 1) / len(jobs))
            
            status.update(label=f"Successfully downloaded {len(downloaded_files)} images", state="complete")
            
        except Exception as e:
            st.error(f"Search failed: {e}")
            return []
    
    return downloaded_files

def save_detections(detections, image, topic_dir, image_name):
    """Save detection results to disk"""
    # Create output directories
    detections_dir = topic_dir / "detections"
    visualizations_dir = topic_dir / "visualizations"
    detections_dir.mkdir(exist_ok=True)
    visualizations_dir.mkdir(exist_ok=True)
    
    # Save detection results as JSON
    detection_file = detections_dir / f"{image_name}_detection.json"
    with open(detection_file, 'w') as f:
        json.dump(detections, f, indent=2)
    
    # Save visualization image
    if image is not None:
        visualization = create_visualization(image, detections)
        if visualization is not None:
            viz_path = visualizations_dir / f"{image_name}_viz.jpg"
            cv2.imwrite(str(viz_path), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            return str(viz_path)
    
    return None

def get_clean_folder_name(topic):
    """Clean topic name for folder use"""
    folder_name = "".join(c if c.isalnum() or c in "_ " else "_" for c in topic)
    folder_name = folder_name.replace(" ", "_").lower()
    return folder_name

def main():
    st.set_page_config(
        page_title="YOLO-World Object Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 YOLO-World Object Detection")
    st.markdown("This app lets you search for images and apply YOLO-World object detection on them.")
    
    # Create base directory
    base_dir = Path("YOLO_World_UI_Data")
    base_dir.mkdir(exist_ok=True)
    
    # Sidebar for app info
    with st.sidebar:
        st.header("About")
        st.info(
            "This app uses YOLO-World for zero-shot object detection. "
            "It can detect objects based on text prompts without explicit training."
        )
        st.header("Instructions")
        st.markdown(
            "1. Enter a search topic\n"
            "2. Set the number of images\n"
            "3. Download images\n"
            "4. Enter object classes to detect\n"
            "5. Run detection"
        )
        
        # Advanced options
        st.header("Advanced Options")
        iou_threshold = st.slider("IoU Threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.1)
    
    # Step 1: Initialize YOLO-World
    if "model_initialized" not in st.session_state:
        st.session_state.model_initialized = False
    
    if not st.session_state.model_initialized:
        if st.button("Initialize YOLO-World Model"):
            if initialize_yolo_world():
                st.session_state.model_initialized = True
                
                # Test the model
                test_yolo_world()
    else:
        st.success("✅ YOLO-World model is loaded and ready!")
    
    # Main app flow - only continue if model is initialized
    if st.session_state.model_initialized:
        # Step 2: Get search topic
        st.header("Step 1: Search for Images")
        
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input("Enter a topic to search for images:", placeholder="e.g., construction site, forest animals")
        with col2:
            query = st.text_input("Enter a more specific search query (optional):", placeholder="Will use topic if empty")
        
        if not query and topic:
            query = topic
            
        col1, col2 = st.columns(2)
        with col1:
            num_images = st.slider("Number of images to download:", min_value=1, max_value=20, value=6)
        with col2:
            download_button = st.button("Download Images", disabled=not topic)
        
        # Step 3: Download images when button clicked
        if "downloaded_files" not in st.session_state:
            st.session_state.downloaded_files = []
            st.session_state.topic_dir = None
            st.session_state.folder_name = ""
        
        if download_button and topic:
            folder_name = get_clean_folder_name(topic)
            st.session_state.folder_name = folder_name
            
            downloaded_files = download_images_for_topic(
                topic=folder_name,
                query=query or topic,
                limit=num_images,
                base_dir=base_dir
            )
            
            if downloaded_files:
                st.session_state.downloaded_files = downloaded_files
                st.session_state.topic_dir = base_dir / folder_name
                st.success(f"✅ Downloaded {len(downloaded_files)} images for '{topic}'!")
            else:
                st.error(f"❌ Failed to download images for '{topic}'")
        
        # Step 4: Show downloaded images and get detection classes
        if st.session_state.downloaded_files:
            st.header("Step 2: Detect Objects")
            
            # Show a sample of downloaded images
            st.subheader("Downloaded Images")
            image_cols = st.columns(3)
            for i, img_path in enumerate(st.session_state.downloaded_files[:6]):
                with image_cols[i % 3]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=img_path.name, use_column_width=True)
                    except:
                        st.error(f"Failed to load {img_path.name}")
            
            # Get detection classes
            st.subheader("Object Detection")
            
            detection_prompt = st.text_input(
                "Enter objects to detect (comma-separated):",
                placeholder="e.g., person, car, dog"
            )
            
            if detection_prompt:
                classes = [cls.strip() for cls in detection_prompt.split(",")]
                
                if st.button("Run Detection"):
                    # Verify model is available
                    model = get_yolo_model()
                    if model is None:
                        st.error("❌ YOLO-World model not available. Please initialize the model first.")
                        st.stop()
                
                    with st.status("Running object detection...") as status:
                        all_detections = []
                        total_objects = 0
                        
                        # Create columns for results
                        st.subheader("Detection Results")
                        result_container = st.container()
                        
                        # Process each image
                        for i, img_path in enumerate(st.session_state.downloaded_files):
                            status.update(label=f"Processing image {i+1}/{len(st.session_state.downloaded_files)}: {img_path.name}")
                            
                            # Run detection
                            detections, image = detect_objects_yolo(img_path, classes)
                            all_detections.append(detections)
                            
                            # Count objects
                            num_objects = len(detections["labels"])
                            total_objects += num_objects
                            
                            # Save results
                            if num_objects > 0:
                                viz_path = save_detections(detections, image, st.session_state.topic_dir, img_path.stem)
                                
                                # Display results in UI
                                with result_container:
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.image(img_path, caption=f"Original: {img_path.name}", use_column_width=True)
                                    with col2:
                                        visualization = create_visualization(image, detections)
                                        if visualization is not None:
                                            st.image(visualization, caption=f"Detected: {num_objects} objects", use_column_width=True)
                                        
                                        # Show detected objects table
                                        if num_objects > 0:
                                            result_data = []
                                            for label, score in zip(detections["labels"], detections["scores"]):
                                                result_data.append({"Class": label, "Confidence": f"{score:.3f}"})
                                            st.table(result_data)
                        
                        # Save summary
                        summary = {
                            "topic": st.session_state.folder_name,
                            "target_classes": classes,
                            "total_images": len(st.session_state.downloaded_files),
                            "images_with_detections": sum(1 for d in all_detections if len(d["labels"]) > 0),
                            "total_objects_detected": total_objects,
                            "model_used": "YOLO-World (Ultralytics)"
                        }
                        
                        summary_file = st.session_state.topic_dir / f"{st.session_state.folder_name}_summary.json"
                        with open(summary_file, 'w') as f:
                            json.dump(summary, f, indent=2)
                        
                        # Final summary
                        status.update(label=f"Detection complete!", state="complete")
                        
                        st.success(f"✅ Detection complete! Found {total_objects} objects across {summary['images_with_detections']} images.")
                        
                        # Show download link for results folder
                        st.info(f"Results saved to {st.session_state.topic_dir}")

if __name__ == "__main__":
    main()