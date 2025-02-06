# http_api.py
import os
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.background import BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from server.avatar import get_or_create_avatar
from server.config import DEFAULT_VIDEO_PATH, DEFAULT_BBOX_SHIFT, DEFAULT_BATCH_SIZE, DEFAULT_FPS, TEMP_DIR
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

app = FastAPI(title="MuseTalk HTTP Lipsync API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Adjust allowed origins for production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files(file_paths: list):
    """Remove temporary files."""
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

@app.post("/lipsync")
async def lipsync_endpoint(
    avatar_id: str = Form(...),
    chunk: int = Form(0),
    bbox_shift: int = Form(DEFAULT_BBOX_SHIFT),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Process lipsync by receiving form data (including an audio file) and metadata.
    Returns the processed video (MP4) as a streaming response.
    """
    try:
        if not avatar_id or not audio_file:
            raise HTTPException(status_code=400, detail="'avatar_id' and 'audio_file' are required.")
        
        # Save the uploaded audio file asynchronously.
        audio_temp_path = os.path.join(TEMP_DIR, f"{avatar_id}_{chunk}_audio.wav")
        async with aiofiles.open(audio_temp_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Assume the avatar video is stored in data/video as <avatar_id>.mp4.
        video_path = "data/video/" + avatar_id + ".mp4"
        
        # Create or get the avatar (this may include preparation if needed).
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size)
        
        def run_inference():
            mp4_path = avatar.inference(
                audio_path=audio_temp_path,
                out_vid_name=f"{avatar_id}_result_chunk{chunk}",
                fps=DEFAULT_FPS,
                skip_save_images=False,
            )
            return mp4_path
        
        mp4_file_path = await run_in_threadpool(run_inference)
        
        if not os.path.exists(mp4_file_path):
            raise HTTPException(status_code=500, detail="Generated video file not found.")
        
        background_tasks.add_task(cleanup_temp_files, [audio_temp_path, mp4_file_path])
        # Stream the generated video as the response
        async def video_stream():
            async with aiofiles.open(mp4_file_path, "rb") as vid_file:
                while True:
                    chunk = await vid_file.read(8192)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(video_stream(), media_type="video/mp4")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/create_avatar")
async def create_avatar_endpoint(
    avatar_id: str = Form(...),
    video_file: UploadFile = File(...),
    bbox_shift: int = Form(DEFAULT_BBOX_SHIFT),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
):
    """
    Upload a video to the data/video folder and create an avatar from it.
    The video is saved to the 'data/video' folder and then used to create the avatar.
    """
    try:
        # Ensure the destination folder exists.
        video_folder = "data/video"
        os.makedirs(video_folder, exist_ok=True)
        
        # Save the uploaded video file
        video_path = os.path.join(video_folder, avatar_id + Path(video_file.filename).suffix)
        async with aiofiles.open(video_path, "wb") as f:
            content = await video_file.read()
            await f.write(content)
        
        # Force preparation mode so the avatar is (re)created from the new video.
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size, True)
        
        return {"message": f"Avatar '{avatar_id}' created successfully using video '{video_file.filename}'."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")