import collections
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from openvino.tools.mo.front import tf as ov_tf_front
from openvino.tools import mo
import streamlit as st

# A directory where the model will be located.
base_model_dir = Path("model")

# The name of the model from Open Model Zoo
model_name = "ssdlite_mobilenet_v2"

# Paths for the model files (assumed to be already downloaded and extracted)
archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"

# Convert the Model
precision = "FP16"
converted_model_path = Path("model") / f"{model_name}_{precision.lower()}.xml"
trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
if not converted_model_path.exists():
    ov_model = mo.convert_model(
        tf_model_path,
        compress_to_fp16=(precision == "FP16"),
        transformations_config=trans_config_path,
        tensorflow_object_detection_api_pipeline_config=tf_model_path.parent / "pipeline.config",
        reverse_input_channels=True,
    )
    ov.save_model(ov_model, converted_model_path)
    del ov_model

# Load the Model
core = ov.Core()
device = "CPU"  # Set to CPU, you can modify this if needed
model = core.read_model(model=converted_model_path)
compiled_model = core.compile_model(model=model, device_name=device)

# Get the input and output nodes.
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
height, width = list(input_layer.shape)[1:3]

# Define the specific classes you're interested in detecting (based on COCO dataset)
classes_to_detect = {
    0: "background",
    1: "person",
    45: "bottle",  # Bottle class index 45
    67: "cell phone",
    62: "tv",
    73: "laptop",
    63: "mouse",
    27: "backpack",
    31: "handbag",
    25: "glasses"
}

# Colors for the class labels
colors = cv2.applyColorMap(
    src=np.arange(0, 255, 255 / len(classes_to_detect), dtype=np.float32).astype(np.uint8),
    colormap=cv2.COLORMAP_RAINBOW,
).squeeze()

def process_results(frame, results, thresh=0.2):  # Keep the threshold lower for more bottle detections
    h, w = frame.shape[:2]
    results = results.squeeze()
    boxes, labels, scores = [], [], []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        if int(label) in classes_to_detect and score > thresh:
            boxes.append(tuple(map(int, (xmin * w, ymin * h, (xmax - xmin) * w, (ymax - ymin) * h))))
            labels.append(int(label))
            scores.append(float(score))
        if int(label) == 45:  # Log bottle detections
            print(f"Bottle detected with confidence score: {score}")
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=thresh, nms_threshold=0.6)
    if len(indices) == 0:
        return []
    return [(labels[idx], scores[idx], boxes[idx]) for idx in indices.flatten()]

def draw_boxes(frame, boxes):
    for label, score, box in boxes:
        # Use modulo to avoid out-of-bounds index error
        color = tuple(map(int, colors[label % len(colors)]))
        
        # Draw the box
        x2 = box[0] + box[2]
        y2 = box[1] + box[3]
        cv2.rectangle(img=frame, pt1=box[:2], pt2=(x2, y2), color=color, thickness=3)

        # Draw the label inside the box
        cv2.putText(
            img=frame,
            text=f"{classes_to_detect[label]} {score:.2f}",
            org=(box[0] + 10, box[1] + 30),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=frame.shape[1] / 1000,
            color=color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )
    return frame

# Streamlit UI
st.title("Object Detection with OpenVINO")
st.text("This app performs real-time object detection on a video stream.")

# Video input options (file upload or webcam)
use_webcam = st.checkbox("Use Webcam")
if not use_webcam:
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

# Main processing function to handle video stream
def run_object_detection(video_source):
    processing_times = collections.deque()
    video_capture = cv2.VideoCapture(video_source)
    
    # Define the codec and create VideoWriter object for display
    stframe = st.empty()  # Placeholder to stream video frames in Streamlit

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            st.write("Video has ended.")
            break

        # Resize frame if necessary for performance improvement
        input_img = cv2.resize(frame, (width, height))
        input_img = input_img[np.newaxis, ...]

        start_time = time.time()
        results = compiled_model([input_img])[output_layer]
        stop_time = time.time()
        boxes = process_results(frame, results)

        # Draw bounding boxes on the frame
        frame = draw_boxes(frame, boxes)

        # Calculate processing time
        processing_times.append(stop_time - start_time)
        if len(processing_times) > 200:
            processing_times.popleft()

        processing_time = np.mean(processing_times) * 1000
        fps = 1000 / processing_time

        # Add FPS and inference time text to the frame
        cv2.putText(
            img=frame,
            text=f"Inference time: {processing_time:.1f}ms, FPS: {fps:.1f}",
            org=(20, 40),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Display video frame in Streamlit
        stframe.image(frame, channels="BGR")

    video_capture.release()

# Run the object detection if a file is uploaded or webcam is selected
if use_webcam:
    run_object_detection(0)  # 0 is the device ID for webcam
elif uploaded_file:
    run_object_detection(uploaded_file)