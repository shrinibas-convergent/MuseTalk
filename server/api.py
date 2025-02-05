# http_api.py
import os
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.background import BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from server.avatar import get_or_create_avatar
from server.config import DEFAULT_VIDEO_PATH, DEFAULT_BBOX_SHIFT, DEFAULT_BATCH_SIZE, DEFAULT_FPS, TEMP_DIR

app = FastAPI(title="MuseTalk HTTP Lipsync API")

os.makedirs(TEMP_DIR, exist_ok=True)

def cleanup_temp_files(file_paths: list):
    for path in file_paths:
        if os.path.exists(path):
            os.remove(path)

@app.post("/lipsync")
async def lipsync_endpoint(
    avatar_id: str = Form(...),
    bbox_shift: int = Form(DEFAULT_BBOX_SHIFT),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Process lipsync by receiving JSON metadata and audio in one HTTP POST request.
    """
    try:
        if not avatar_id or not audio_file:
            raise HTTPException(status_code=400, detail="'avatar_id' and 'audio_file' are required.")
        
        # Save the audio file temporarily
        audio_temp_path = os.path.join(TEMP_DIR, f"{avatar_id}_audio.wav")
        with open(audio_temp_path, "wb") as f:
            f.write(await audio_file.read())
        video_path = "data/video/" + avatar_id
        # Create or get the avatar
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size)

        # Perform inference in a separate thread
        def run_inference():
            mp4_path = avatar.inference(
                audio_path=audio_temp_path,
                out_vid_name=f"{avatar_id}_result",
                fps=DEFAULT_FPS,
                skip_save_images=False,
            )
            return mp4_path

        mp4_file_path = await run_in_threadpool(run_inference)

        if not os.path.exists(mp4_file_path):
            raise HTTPException(status_code=500, detail="Generated video file not found.")
        
        background_tasks.add_task(cleanup_temp_files, [audio_temp_path, mp4_file_path])
        # Stream the generated video as the response
        def video_stream():
            with open(mp4_file_path, "rb") as vid_file:
                while chunk := vid_file.read(8192):
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
        video_path = os.path.join(video_folder, avatar_id)
        with open(video_path, "wb") as f:
            f.write(await video_file.read())
        
        # Force preparation mode so the avatar is (re)created from the new video.
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size, True)
        
        return {"message": f"Avatar '{avatar_id}' created successfully using video '{video_file.filename}'."}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")