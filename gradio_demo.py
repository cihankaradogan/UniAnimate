import gradio as gr
import subprocess
import os
from PIL import Image
import torch
from swapper import getFaceSwapModel, getFaceAnalyser, get_many_faces, swap_face
from codeformer.app import inference_app
import onnxruntime
import numpy as np
import copy  # Import the copy module
import imageio

def generate_video(ref_image, source_video_path, apply_faceswap, apply_codeformer):
    gr.Info("Saving the uploaded files...")
    ref_image_path = "data/images/ref_image.jpg"
    os.makedirs(os.path.dirname(ref_image_path), exist_ok=True)
    ref_image.save(ref_image_path)
    
    gr.Info("Running the pose alignment script...")
    saved_pose_dir = "data/saved_pose/ref_image"
    os.makedirs(saved_pose_dir, exist_ok=True)
    subprocess.run([
        "python", "run_align_pose.py",
        "--ref_name", ref_image_path,
        "--source_video_paths", source_video_path,
        "--saved_pose_dir", saved_pose_dir
    ])
    
    gr.Info("Running the UniAnimate model to generate the video...")
    subprocess.run([
        "python", "inference.py",
        "--cfg", "configs/UniAnimate_infer.yaml"
    ])
    
    generated_video_path = "./outputs/UniAnimate_infer/rank_01_00_00_seed_11_image_local_image_dwpose_ref_image_768x512.mp4"
    
    if not apply_faceswap and not apply_codeformer:
        gr.Info("Returning the original generated video.")
        return generated_video_path
    
    gr.Info("Loading the generated video...")
    reader = imageio.get_reader(generated_video_path, 'ffmpeg')
    fps = reader.get_meta_data()['fps']
    
    if apply_faceswap:
        gr.Info("Initializing InSwapper...")
        model_path = "./checkpoints/inswapper_128.onnx"
        face_swapper = getFaceSwapModel(model_path)
        providers = onnxruntime.get_available_providers()
        face_analyser = getFaceAnalyser(model_path, providers)
    
    if apply_faceswap:
        gr.Info("Loading the reference image for face swapping...")
        ref_image_cv2 = np.array(ref_image.convert("RGB"))
        ref_faces = get_many_faces(face_analyser, ref_image_cv2)
    
    gr.Info("Processing each frame...")
    processed_frames = []
    for frame in reader:
        frame = np.array(frame)
        
        if apply_faceswap:
            target_faces = get_many_faces(face_analyser, frame)
            if target_faces is not None:
                temp_frame = copy.deepcopy(frame)
                for i in range(len(target_faces)):
                    if ref_faces is None:
                        raise gr.Error("No reference faces found!")
                    temp_frame = swap_face(
                        face_swapper,
                        ref_faces,
                        target_faces,
                        0,  # Assuming single reference face
                        i,
                        temp_frame
                    )
                frame = temp_frame
        
        if apply_codeformer:
            frame_pil = Image.fromarray(frame)
            enhanced_frame_pil = inference_app(
                image=frame_pil,
                background_enhance=True,
                face_upsample=True,
                upscale=1,
                codeformer_fidelity=0.5
            )
            frame = np.array(enhanced_frame_pil)
        
        processed_frames.append(frame)
    
    reader.close()
    
    gr.Info("Saving the processed frames as a new video...")
    output_processed_video_path = "outputs/processed_generated_video.mp4"
    writer = imageio.get_writer(output_processed_video_path, fps=fps, codec='libx264', quality=8)
    for frame in processed_frames:
        writer.append_data(frame)
    writer.close()
    
    gr.Info("Video processing complete.")
    return output_processed_video_path

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.Image(type="pil", label="Reference Image"),
        gr.Video(label="Source Video"),
        gr.Checkbox(label="Apply Face Swap"),
        gr.Checkbox(label="Apply CodeFormer Enhancement")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="UniAnimate Video Generation",
    description="Reference Image and Source Video should be portrait size and resolution should be divisable by 64!"
)

# Launch the interface
iface.launch(share=True, debug=True)