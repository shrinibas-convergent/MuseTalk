# http_api.py
import os
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.background import BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from server.avatar import get_or_create_avatar
from server.config import DEFAULT_VIDEO_PATH, DEFAULT_BBOX_SHIFT, DEFAULT_BATCH_SIZE, DEFAULT_FPS, TEMP_DIR, RESULTS_DIR
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import shutil
import time

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
    chunk: str = Form(...),
    bbox_shift: int = Form(DEFAULT_BBOX_SHIFT),
    batch_size: int = Form(DEFAULT_BATCH_SIZE),
    audio_file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Process lipsync by receiving form data and an audio file.
    Generates DASH segments and returns the URL of the manifest.
    This single endpoint supports multiple requests for the same avatar by isolating output using a unique identifier.
    """
    try:
        if not avatar_id or not audio_file or not chunk:
            raise HTTPException(status_code=400, detail="'avatar_id', 'chunk', and 'audio_file' are required.")

        # Generate a unique identifier using the chunk parameter and current timestamp.
        unique_id = f"{chunk}_{int(time.time())}"

        # Save the uploaded audio file with a unique name.
        audio_temp_path = Path(TEMP_DIR) / f"{avatar_id}_{unique_id}_audio.wav"
        async with aiofiles.open(audio_temp_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)

        # Assume the avatar video is stored in data/video/<avatar_id>.mp4.
        video_path = Path("data/video") / f"{avatar_id}.mp4"
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size)

        # Run the DASH inference, which writes DASH segments and an MPD manifest into a unique subfolder.
        manifest_path = await run_in_threadpool(avatar.inference_dash, str(audio_temp_path), DEFAULT_FPS, unique_id)

        background_tasks.add_task(cleanup_temp_files, [str(audio_temp_path)])

        # Construct a URL for the manifest. This assumes the manifest is served via the /dash endpoint.
        manifest_url = f"/dash/{avatar_id}/dash_output/{unique_id}/manifest.mpd"
        return JSONResponse(content={"manifest_url": manifest_url})
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
    Upload a video to data/video and create an avatar.
    """
    try:
        video_folder = Path("data/video")
        video_folder.mkdir(parents=True, exist_ok=True)
        video_path = video_folder / f"{avatar_id}{Path(video_file.filename).suffix}"
        async with aiofiles.open(video_path, "wb") as f:
            content = await video_file.read()
            await f.write(content)
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size, True)
        return JSONResponse(content={"message": f"Avatar '{avatar_id}' created successfully using video '{video_file.filename}'."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Endpoint to serve DASH files from the avatar's dash_output folder.
@app.get("/dash/{avatar_id}/{file_path:path}")
async def serve_dash_files(avatar_id: str, file_path: str):
    dash_dir = Path(RESULTS_DIR) / avatar_id / "dash_output"
    full_path = dash_dir / file_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(full_path))
