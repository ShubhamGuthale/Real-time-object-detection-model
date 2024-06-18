#run using ==> streamlit run object detection.py

import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
from yolo_predictions import YOLO_Pred  # Import custom YOLO class

# Load YOLO model and configuration
@st.cache_resource
def load_model():
    yolo = YOLO_Pred('./Model/weights/best.onnx', 'data.yaml')  # Adjust path to your onnx model and data file
    return yolo

model = load_model()

def make_prediction(img):
    # Convert PIL image to numpy array if not already
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Make predictions using YOLO model
    prediction = model.predictions(img)

    # Initialize the result dictionary
    result = {"boxes": [], "labels": [], "scores": []}

    for item in prediction:
        if 'bbox' in item and 'class' in item and 'confidence' in item:
            result["boxes"].append(item["bbox"])
            result["labels"].append(item["class"])
            result["scores"].append(item["confidence"])

    # Convert lists to numpy arrays if not empty
    if result["boxes"]:
        result["boxes"] = np.array(result["boxes"])

    return result

def create_image_with_bboxes(img, prediction):
    # Ensure img is a numpy array
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Convert to tensor
    img_tensor = torch.tensor(img).permute(2, 0, 1).float()  # Convert (H, W, 3) -> (3, H, W)
    img_tensor_uint8 = img_tensor.to(torch.uint8)  # Convert to uint8

    boxes = torch.tensor(prediction["boxes"], dtype=torch.float)
    labels = prediction["labels"]

    # Convert (x, y, width, height) to (xmin, ymin, xmax, ymax)
    boxes = box_convert(boxes, in_fmt="xywh", out_fmt="xyxy")

    # Use a consistent color for all bounding boxes
    img_with_bboxes = draw_bounding_boxes(
        img_tensor_uint8,
        boxes=boxes,
        labels=[str(label) for label in labels],  # Ensure labels are strings
        colors=["blue"] * len(labels),  # Use blue for all bounding boxes
        width=2
    )

    img_with_bboxes_np = img_with_bboxes.permute(1, 2, 0).cpu().numpy()  # Convert back to (H, W, 3)
    return img_with_bboxes_np

# Streamlit Dashboard
st.title("Custom Object Detector :tea: :coffee:")
upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload).convert("RGB")  # Convert to RGB

    try:
        prediction = make_prediction(img)  # Make prediction

        if "boxes" in prediction and len(prediction["boxes"]) > 0:
            img_with_bbox = create_image_with_bboxes(np.array(img), prediction)  # Create image with bounding boxes

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(111)
            plt.imshow(img_with_bbox)
            plt.xticks([])
            plt.yticks([])
            ax.spines[['top', 'bottom', 'right', 'left']].set_visible(False)

            st.pyplot(fig, use_container_width=True)

            st.header("Predicted Probabilities")
            # Display predicted classes and confidences
            pred_data = [
                {"Class": label, "Confidence": f"{score:.2f}"}
                for label, score in zip(prediction["labels"], prediction["scores"])
            ]
            st.write(pred_data)

        else:
            st.error("No objects detected in the image.")

    except Exception as e:
        st.error(f"Error processing prediction: {e}")
