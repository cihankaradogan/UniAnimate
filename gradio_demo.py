import gradio as gr
import subprocess
import os
from PIL import Image

def generate_video(ref_image, source_video):
    # Save the uploaded files
    ref_image_path = "data/images/ref_image.jpg"
    source_video_path = "data/videos/source_video.mp4"
    os.makedirs(os.path.dirname(ref_image_path), exist_ok=True)
    os.makedirs(os.path.dirname(source_video_path), exist_ok=True)
    ref_image.save(ref_image_path)
    source_video.save(source_video_path)
    
    # Run the pose alignment script
    saved_pose_dir = "data/saved_pose/ref_image"
    os.makedirs(saved_pose_dir, exist_ok=True)
    subprocess.run([
        "python", "run_align_pose.py",
        "--ref_name", ref_image_path,
        "--source_video_paths", source_video_path,
        "--saved_pose_dir", saved_pose_dir
    ])
    
    # Run the UniAnimate model to generate the video
    output_video_path = "output/generated_video.mp4"
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    subprocess.run([
        "python", "inference.py",
        "--cfg", "configs/UniAnimate_infer.yaml"
    ])
    
    return output_video_path

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.inputs.Image(type="file", label="Reference Image"),
        gr.inputs.Video(type="file", label="Source Video")
    ],
    outputs=gr.outputs.Video(label="Generated Video"),
    title="UniAnimate Video Generation",
    description="Upload a reference image and a source video to generate an animated video using UniAnimate."
)

# Launch the interface
iface.launch()