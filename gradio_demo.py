import gradio as gr
import subprocess
import os
from PIL import Image
import cv2
import torch
from swapper import getFaceSwapModel, getFaceAnalyser, get_many_faces, swap_face

# Ensure the correct import path for CodeFormer
from codeformer.app import CodeFormer


def generate_video(ref_image, source_video, apply_faceswap, apply_codeformer):
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
    
    # Load the generated video
    cap = cv2.VideoCapture(output_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize InSwapper and CodeFormer if needed
    if apply_faceswap:
        model_path = "./checkpoints/inswapper_128.onnx"
        face_swapper = getFaceSwapModel(model_path)
        providers = onnxruntime.get_available_providers()
        face_analyser = getFaceAnalyser(model_path, providers)
    
    if apply_codeformer:
        codeformer = CodeFormer()
    
    # Load the reference image for face swapping if needed
    if apply_faceswap:
        ref_image_cv2 = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
        ref_faces = get_many_faces(face_analyser, ref_image_cv2)
    
    # Process each frame
    processed_frames = []
    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        
        if apply_faceswap:
            # Detect faces in the current frame
            target_faces = get_many_faces(face_analyser, frame)
            
            if target_faces is not None:
                temp_frame = copy.deepcopy(frame)
                for i in range(len(target_faces)):
                    if ref_faces is None:
                        raise Exception("No reference faces found!")
                    
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
            # Enhance the frame using CodeFormer
            frame = codeformer.enhance(frame)
        
        processed_frames.append(frame)
    
    cap.release()
    
    # Save the processed frames as a new video
    output_processed_video_path = "output/processed_generated_video.mp4"
    out = cv2.VideoWriter(output_processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    for frame in processed_frames:
        out.write(frame)
    out.release()
    
    return output_processed_video_path

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_video,
    inputs=[
        gr.inputs.Image(type="file", label="Reference Image"),
        gr.inputs.Video(type="file", label="Source Video"),
        gr.inputs.Checkbox(label="Apply Face Swap"),
        gr.inputs.Checkbox(label="Apply CodeFormer Enhancement")
    ],
    outputs=gr.outputs.Video(label="Generated Video"),
    title="UniAnimate Video Generation",
    description="Upload a reference image and a source video to generate an animated video using UniAnimate."
)

# Launch the interface
iface.launch(share=True, debug=True)