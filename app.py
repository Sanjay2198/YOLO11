import spaces
import gradio as gr
from pathlib import Path
import uuid
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
VIDEO_RUNS_DIR = APP_DIR / "runs"

EXAMPLE_CONFIGS = [
    ("zidane.jpg", "yolo11s.pt", 0.25, 0.45, 300),
    ("bus.jpg", "yolo11m.pt", 0.25, 0.45, 300),
    ("yolo_vision.jpg", "yolo11x.pt", 0.25, 0.45, 300),
    ("Tricycle.jpg", "yolo11x-cls.pt", 0.25, 0.45, 300),
    ("tcganadolu.jpg", "yolo11m-obb.pt", 0.25, 0.45, 300),
    ("San Diego Airport.jpg", "yolo11x-seg.pt", 0.25, 0.45, 300),
    ("Theodore_Roosevelt.png", "yolo11l-pose.pt", 0.25, 0.45, 300),
    ("bike and car.jpg", "yolo11s.pt", 0.25, 0.45, 300),
]

EXAMPLES = [
    [str(APP_DIR / image_name), model, conf, iou, max_detection]
    for image_name, model, conf, iou, max_detection in EXAMPLE_CONFIGS
    if (APP_DIR / image_name).exists()
]

def _find_video_output(run_dir, input_video_path):
    source = Path(input_video_path)
    preferred_suffixes = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    for suffix in preferred_suffixes:
        candidate = run_dir / f"{source.stem}{suffix}"
        if candidate.exists():
            return str(candidate)

    for candidate in run_dir.iterdir():
        if candidate.is_file() and candidate.suffix.lower() in preferred_suffixes:
            return str(candidate)

    return None

def _toggle_input_type(input_type):
    is_image = input_type == "Image"
    return (
        gr.update(visible=is_image),
        gr.update(visible=not is_image),
        gr.update(visible=is_image),
        gr.update(visible=not is_image),
    )

@spaces.GPU
def yolo_inference(input_type, image, video, model_id, conf_threshold, iou_threshold, max_detection):
    """
    Performs object detection, instance segmentation, pose estimation,
    oriented object detection, or classification using a YOLOv11 model
    on an image or video.
    """
    model = YOLO(model_id)

    if input_type == "Image":
        if image is None:
            width, height = 640, 480
            blank_image = Image.new("RGB", (width, height), color="white")
            draw = ImageDraw.Draw(blank_image)
            message = "No image provided"
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), message, font=font)
            text_x = (width - (bbox[2] - bbox[0])) / 2
            text_y = (height - (bbox[3] - bbox[1])) / 2
            draw.text((text_x, text_y), message, fill="black", font=font)
            return blank_image, None

        results = model.predict(
            source=image,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            max_det=max_detection,
            show_labels=True,
            show_conf=True,
        )

        # Initialize annotated_image with the first result
        if len(results) > 0:
            image_array = results[0].plot()
            annotated_image = Image.fromarray(image_array[..., ::-1])
            return annotated_image, None
        else:
            # Return the original image if no detections
            return image, None

    if input_type == "Video":
        if video is None:
            return None, None

        video_path = video["path"] if isinstance(video, dict) and "path" in video else str(video)
        run_name = f"predict_{uuid.uuid4().hex[:8]}"
        run_dir = VIDEO_RUNS_DIR / run_name

        model.predict(
            source=video_path,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=640,
            max_det=max_detection,
            show_labels=True,
            show_conf=True,
            save=True,
            project=str(VIDEO_RUNS_DIR),
            name=run_name,
            exist_ok=True,
        )

        output_video = _find_video_output(run_dir, video_path)
        return None, output_video

    return None, None

def yolo_inference_for_examples(image, model_id, conf_threshold, iou_threshold, max_detection):
    """Wrapper for Gradio examples (image-only)."""
    annotated_image, _ = yolo_inference(
        input_type="Image",
        image=image,
        video=None,
        model_id=model_id,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detection=max_detection
    )
    return annotated_image

with gr.Blocks() as app:
    gr.Markdown("# YOLOv11: Image and Video Inference")
    gr.Markdown("Upload images or videos for object detection, segmentation, pose, OBB, or classification using YOLOv11 models.")

    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Input Image")
            video = gr.Video(label="Input Video", visible=False, sources=["upload"])
            input_type = gr.Radio(
                choices=["Image", "Video"],
                value="Image",
                label="Input Type",
            )
            model_id = gr.Dropdown(
                label="Model Name",
                choices=[
                    'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt', 
                    'yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt', 'yolo11l-seg.pt', 'yolo11x-seg.pt',
                    'yolo11n-pose.pt', 'yolo11s-pose.pt', 'yolo11m-pose.pt', 'yolo11l-pose.pt', 'yolo11x-pose.pt',
                    'yolo11n-obb.pt', 'yolo11s-obb.pt', 'yolo11m-obb.pt', 'yolo11l-obb.pt', 'yolo11x-obb.pt',
                    'yolo11n-cls.pt', 'yolo11s-cls.pt', 'yolo11m-cls.pt', 'yolo11l-cls.pt', 'yolo11x-cls.pt'
                ],
                value="yolo11n.pt",
            )
            conf_threshold = gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence Threshold")
            iou_threshold = gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU Threshold")
            max_detection = gr.Slider(minimum=1, maximum=300, step=1, value=300, label="Max Detections")
            infer_button = gr.Button("Run Detection")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Output")
            output_video = gr.Video(label="Output Video", visible=False)

    input_type.change(
        fn=_toggle_input_type,
        inputs=[input_type],
        outputs=[image, video, output_image, output_video],
    )

    # Main inference
    infer_button.click(
        fn=yolo_inference,
        inputs=[input_type, image, video, model_id, conf_threshold, iou_threshold, max_detection],
        outputs=[output_image, output_video],
    )

    # Examples
    if EXAMPLES:
        gr.Examples(
            examples=EXAMPLES,
            fn=yolo_inference_for_examples,
            inputs=[image, model_id, conf_threshold, iou_threshold, max_detection],
            outputs=[output_image],
            label="Examples",
        )
    else:
        gr.Markdown("No local example images found in the app directory.")

if __name__ == '__main__':
    app.launch(allowed_paths=[str(APP_DIR)])
