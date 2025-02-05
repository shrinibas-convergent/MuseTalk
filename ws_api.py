# ws_api.py
import os
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from avatar import get_or_create_avatar
from config import DEFAULT_VIDEO_PATH, DEFAULT_BBOX_SHIFT, DEFAULT_BATCH_SIZE, DEFAULT_FPS, TEMP_DIR

app = FastAPI(title="MuseTalk WebSocket Lipsync API (Combined Message)")

os.makedirs(TEMP_DIR, exist_ok=True)

@app.websocket("/ws/lipsync_combined")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # Receive a single JSON message that contains both metadata and base64-encoded audio data.
        data = await websocket.receive_json()
        avatar_id = data.get("avatar_id")
        video_path = data.get("video_path", DEFAULT_VIDEO_PATH)
        bbox_shift = data.get("bbox_shift", DEFAULT_BBOX_SHIFT)
        batch_size = data.get("batch_size", DEFAULT_BATCH_SIZE)
        audio_data_b64 = data.get("audio_data")  # Base64 encoded audio file
        
        if not avatar_id or not audio_data_b64:
            await websocket.send_text("Error: 'avatar_id' and 'audio_data' are required in the message.")
            await websocket.close()
            return

        # Decode the base64 audio data and save to a temporary file.
        audio_bytes = base64.b64decode(audio_data_b64)
        audio_temp_path = os.path.join(TEMP_DIR, f"{avatar_id}_audio.wav")
        with open(audio_temp_path, "wb") as f:
            f.write(audio_bytes)

        # Create (or get) the avatar instance.
        avatar = await run_in_threadpool(get_or_create_avatar, avatar_id, video_path, bbox_shift, batch_size)

        # Run inference.
        def run_inference():
            mp4_path = avatar.inference(audio_path=audio_temp_path,
                                        out_vid_name=f"{avatar_id}_result",
                                        fps=DEFAULT_FPS,
                                        skip_save_images=False)
            return mp4_path

        mp4_file_path = await run_in_threadpool(run_inference)

        # Stream the MP4 file in chunks.
        if not os.path.exists(mp4_file_path):
            await websocket.send_text("Error: Generated video file not found.")
        else:
            with open(mp4_file_path, "rb") as vid_file:
                chunk_size = 8192
                while True:
                    chunk = vid_file.read(chunk_size)
                    if not chunk:
                        break
                    await websocket.send_bytes(chunk)
            await websocket.send_text("END_OF_VIDEO")

        # Clean up temporary files.
        if os.path.exists(audio_temp_path):
            os.remove(audio_temp_path)
        if os.path.exists(mp4_file_path):
            os.remove(mp4_file_path)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        await websocket.send_text(f"Server error: {e}")
        await websocket.close()
