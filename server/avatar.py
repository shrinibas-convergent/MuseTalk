# avatar.py
import os
import cv2
import json
import time
import glob
import torch
import queue
import pickle
import shutil
import threading
import subprocess
import numpy as np
from tqdm import tqdm
import copy

# Import helper functions from the musetalk package
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
from musetalk.utils.blending import get_image_blending
from musetalk.utils.utils import datagen

# Import global models from model.py
from server.model import audio_processor, vae, unet, pe, device, timesteps

# Import configuration defaults
from server.config import DEFAULT_BATCH_SIZE, RESULTS_DIR

# Global semaphore to limit concurrent inference requests.
inference_semaphore = threading.Semaphore(1)

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)

class Avatar:
    def __init__(self, avatar_id, video_path, bbox_shift, batch_size, preparation):
        self.avatar_id = avatar_id
        self.video_path = str(video_path)
        self.bbox_shift = bbox_shift
        self.avatar_path = os.path.join(RESULTS_DIR, avatar_id)
        self.full_imgs_path = os.path.join(self.avatar_path, "full_imgs")
        self.coords_path = os.path.join(self.avatar_path, "coords.pkl")
        self.latents_out_path = os.path.join(self.avatar_path, "latents.pt")
        self.video_out_path = os.path.join(self.avatar_path, "vid_output")
        self.mask_out_path = os.path.join(self.avatar_path, "mask")
        self.mask_coords_path = os.path.join(self.avatar_path, "mask_coords.pkl")
        self.avatar_info_path = os.path.join(self.avatar_path, "avatar_info.json")
        self.avatar_info = {
            "avatar_id": self.avatar_id,
            "video_path": self.video_path,
            "bbox_shift": self.bbox_shift
        }
        self.preparation = preparation
        self.batch_size = batch_size
        self.init()

    def init(self):
        if self.preparation:
            if os.path.exists(self.avatar_path):
                print(f"Avatar {self.avatar_id} exists. Using existing data.")
                try:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = sorted(
                        glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                    )
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = sorted(
                        glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                    )
                    self.mask_list_cycle = read_imgs(input_mask_list)
                    return
                except Exception as e:
                    print(f"Error loading existing avatar, recreating it: {e}")
                    shutil.rmtree(self.avatar_path)
            print(f"Creating avatar: {self.avatar_id}")
            osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
            self.prepare_material()
        else:
            if not os.path.exists(self.avatar_path):
                print(f"Avatar {self.avatar_id} not found. Switching to preparation mode.")
                osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                self.prepare_material()
            else:
                with open(self.avatar_info_path, "r") as f:
                    avatar_info = json.load(f)
                if avatar_info.get('bbox_shift', None) != self.avatar_info['bbox_shift']:
                    print("bbox_shift changed, recreating avatar.")
                    shutil.rmtree(self.avatar_path)
                    osmakedirs([self.avatar_path, self.full_imgs_path, self.video_out_path, self.mask_out_path])
                    self.prepare_material()
                else:
                    self.input_latent_list_cycle = torch.load(self.latents_out_path)
                    with open(self.coords_path, 'rb') as f:
                        self.coord_list_cycle = pickle.load(f)
                    input_img_list = sorted(
                        glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                    )
                    self.frame_list_cycle = read_imgs(input_img_list)
                    with open(self.mask_coords_path, 'rb') as f:
                        self.mask_coords_list_cycle = pickle.load(f)
                    input_mask_list = sorted(
                        glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]')),
                        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
                    )
                    self.mask_list_cycle = read_imgs(input_mask_list)

    def prepare_material(self):
        print("Preparing avatar data materials ...")
        with open(self.avatar_info_path, "w") as f:
            json.dump(self.avatar_info, f)
        if os.path.isfile(self.video_path):
            cap = cv2.VideoCapture(self.video_path)
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(os.path.join(self.full_imgs_path, f"{count:08d}.png"), frame)
                count += 1
            cap.release()
        else:
            print(f"Copying images from folder: {self.video_path}")
            files = sorted([file for file in os.listdir(self.video_path) if file.endswith("png")])
            for filename in files:
                shutil.copyfile(
                    os.path.join(self.video_path, filename),
                    os.path.join(self.full_imgs_path, filename)
                )
        input_img_list = sorted(glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
        print("Extracting landmarks ...")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, self.bbox_shift)
        input_latent_list = []
        placeholder = coord_placeholder
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == placeholder:
                continue
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents = vae.get_latents_for_unet(resized_crop_frame)
            input_latent_list.append(latents)
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []
        for i, frame in enumerate(frame_list + frame_list[::-1]):
            cv2.imwrite(os.path.join(self.full_imgs_path, f"{i:08d}.png"), frame)
            mask = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop_box = (0, 0, frame.shape[1], frame.shape[0])
            cv2.imwrite(os.path.join(self.mask_out_path, f"{i:08d}.png"), mask)
            self.mask_coords_list_cycle.append(crop_box)
            self.mask_list_cycle.append(mask)
        with open(self.mask_coords_path, 'wb') as f:
            pickle.dump(self.mask_coords_list_cycle, f)
        with open(self.coords_path, 'wb') as f:
            pickle.dump(self.coord_list_cycle, f)
        torch.save(self.input_latent_list_cycle, self.latents_out_path)
        print("Avatar preparation complete.")

    def inference_continuous_sliding_realtime_stream(self, audio_path, fps, unique_id):
        """
        Implements a continuous, realtime inference pipeline using a sliding window,
        with a dynamic manifest (live DASH). The pipeline works as follows:
          - Compute full audio features.
          - Build overlapping sliding windows (e.g., 5-second window with a 1-second step).
          - A producer thread processes each window and enqueues only new frames (all for the first window, then the last 'step' frames).
          - FFmpeg is launched in live mode (-live 1) to mux these frames with the complete audio into DASH segments,
            dynamically updating the MPD manifest.
          - The function returns immediately (after the first window is processed) so that the manifest is available,
            while the background threads continue processing.
        Returns the manifest path.
        """
        with inference_semaphore:
            print("Starting continuous sliding realtime inference ...")
            start_time = time.time()

            # Compute full audio features and base chunks.
            whisper_feature = audio_processor.audio2feat(audio_path)
            base_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
            total_chunks = len(base_chunks)
            # Sliding window parameters.
            window_size = 125   # 5 seconds * 25 FPS.
            step = 25           # 1 second * 25 FPS.
            sliding_windows = []
            for start in range(0, total_chunks - window_size + 1, step):
                sliding_windows.append(base_chunks[start: start + window_size])
            num_windows = len(sliding_windows)
            print(f"Total sliding windows: {num_windows}")

            # Create a thread-safe queue for new frames.
            frames_queue = queue.Queue()

            # Producer thread: process each window and enqueue only new frames.
            def producer():
                first_window = True
                for w in tqdm(range(num_windows), desc="Processing sliding windows"):
                    window_chunks = sliding_windows[w]
                    window_frames = []
                    try:
                        for _, (whisper_batch, latent_batch) in enumerate(
                            datagen(window_chunks, self.input_latent_list_cycle, self.batch_size)
                        ):
                            with torch.no_grad():
                                audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
                                audio_feature_batch = pe(audio_feature_batch)
                                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                                recon = vae.decode_latents(pred_latents)
                                for frame in recon:
                                    window_frames.append(frame)
                    except Exception as e:
                        print(f"Error processing sliding window {w}: {e}")
                        continue
                    if first_window:
                        for frame in window_frames:
                            frames_queue.put(frame)
                        first_window = False
                    else:
                        for frame in window_frames[-step:]:
                            frames_queue.put(frame)
                print("Producer thread finished.")

            producer_thread = threading.Thread(target=producer, daemon=True)
            producer_thread.start()

            # Wait until the first window's frames are available.
            while frames_queue.empty():
                time.sleep(0.1)

            # Start FFmpeg in live mode for dynamic DASH output.
            first_frame = self.frame_list_cycle[0]
            height, width, _ = first_frame.shape
            dash_dir = os.path.join(self.avatar_path, "dash_output", unique_id)
            os.makedirs(dash_dir, exist_ok=True)
            manifest_path = os.path.join(dash_dir, "manifest.mpd")
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-live", "1",  # Enable live mode for dynamic manifest updates.
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{width}x{height}",
                "-framerate", str(fps),
                "-i", "pipe:0",
                "-i", audio_path,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "veryfast",
                "-crf", "23",
                "-c:a", "aac",
                "-shortest",
                "-f", "dash",
                "-use_template", "1",
                "-use_timeline", "1",
                "-seg_duration", "2",
                manifest_path
            ]
            print("Running ffmpeg for live DASH output:", " ".join(ffmpeg_cmd))
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            def writer(ffmpeg_stdin):
                while True:
                    try:
                        frame = frames_queue.get(timeout=1)
                        try:
                            ffmpeg_stdin.write(frame.astype(np.uint8).tobytes())
                            ffmpeg_stdin.flush()
                        except Exception as e:
                            print("Error writing frame to ffmpeg:", e)
                    except queue.Empty:
                        if not producer_thread.is_alive():
                            break
                try:
                    ffmpeg_stdin.close()
                except Exception as e:
                    print("Error closing ffmpeg stdin:", e)
                print("Writer thread finished.")

            writer_thread = threading.Thread(target=writer, args=(ffmpeg_process.stdin,), daemon=True)
            writer_thread.start()

            # Return the manifest path immediately once the first window is processed.
            total_time_initial = time.time() - start_time
            print("Initial window processed. Returning manifest URL. (Initial processing time:", total_time_initial, "seconds)")
            # Note: Background threads continue writing frames and updating the DASH output.
            return manifest_path

def get_or_create_avatar(avatar_id, video_path, bbox_shift, batch_size=DEFAULT_BATCH_SIZE, preparation=True):
    """
    Returns an instance of Avatar based on the provided avatar_id.
    If the avatar directory exists, it is reused; otherwise, it is created.
    """
    avatar_dir = os.path.join(RESULTS_DIR, avatar_id)
    if os.path.exists(avatar_dir):
        avatar_instance = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=batch_size,
            preparation=False
        )
    else:
        avatar_instance = Avatar(
            avatar_id=avatar_id,
            video_path=video_path,
            bbox_shift=bbox_shift,
            batch_size=batch_size,
            preparation=True
        )
    return avatar_instance
