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
# Adjust the value as needed (here 1 means only one inference at a time).
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
        # If preparation is requested and the avatar already exists, try to load it.
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

        # If video_path is a file, extract frames.
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

        # Create a cycle by mirroring the lists.
        self.frame_list_cycle = frame_list + frame_list[::-1]
        self.coord_list_cycle = coord_list + coord_list[::-1]
        self.input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        self.mask_coords_list_cycle = []
        self.mask_list_cycle = []

        # For simplicity, we use a grayscale version of the frame as the mask.
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

    def inference_dash(self, audio_path, fps, unique_id):
        """
        Process the given audio file and generate MPEG-DASH output.
        DASH segments and an MPD manifest are written to a unique subfolder under the avatar's dash_output folder.
        Returns the path to the manifest file.
        """
        # Limit concurrent inferences using a global semaphore.
        with inference_semaphore:
            print("Starting DASH streaming inference ...")
            start_time = time.time()

            # Run inference as before to populate frames.
            whisper_feature = audio_processor.audio2feat(audio_path)
            whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
            total_chunks = len(whisper_chunks)
            res_frame_queue = queue.Queue()
            # Use a local variable for concurrency isolation.
            local_idx = 0
            raw_frame_queue = queue.Queue()

            def process_frames():
                nonlocal local_idx
                count = 0
                timeout_counter = 0
                while count < total_chunks:
                    try:
                        res_frame = res_frame_queue.get(timeout=1)
                        timeout_counter = 0
                    except queue.Empty:
                        timeout_counter += 1
                        if timeout_counter >= 3:
                            print("No frames received for 3 seconds, breaking process_frames loop.")
                            break
                        continue
                    bbox = self.coord_list_cycle[local_idx % len(self.coord_list_cycle)]
                    ori_frame = self.frame_list_cycle[local_idx % len(self.frame_list_cycle)].copy()
                    x1, y1, x2, y2 = bbox
                    try:
                        res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                    except Exception as e:
                        print(f"Error resizing frame: {e}")
                        continue
                    mask = self.mask_list_cycle[local_idx % len(self.mask_list_cycle)]
                    mask_crop_box = self.mask_coords_list_cycle[local_idx % len(self.mask_coords_list_cycle)]
                    combined_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                    raw_frame_queue.put(combined_frame)
                    local_idx += 1
                    count += 1

            process_thread = threading.Thread(target=process_frames)
            process_thread.start()

            try:
                # Wrap the model inference in torch.no_grad() to avoid storing gradients.
                with torch.no_grad():
                    for _, (whisper_batch, latent_batch) in enumerate(
                        tqdm(gen := datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size),
                             total=int((total_chunks + self.batch_size - 1) / self.batch_size))
                    ):
                        try:
                            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
                            audio_feature_batch = pe(audio_feature_batch)
                            latent_batch = latent_batch.to(dtype=unet.model.dtype)
                            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                            recon = vae.decode_latents(pred_latents)
                            for res_frame in recon:
                                res_frame_queue.put(res_frame)
                        except Exception as e:
                            print("Error during inference batch:", e)
            except Exception as e:
                print("Error iterating through datagen:", e)
            process_thread.join()
            print(f"Inference processing complete. Total time: {time.time() - start_time:.2f}s")

            first_frame = self.frame_list_cycle[0]
            height, width, _ = first_frame.shape

            # Create a unique dash output directory for this request.
            dash_dir = os.path.join(self.avatar_path, "dash_output", unique_id)
            os.makedirs(dash_dir, exist_ok=True)
            manifest_path = os.path.join(dash_dir, "manifest.mpd")

            # Use ffprobe to get the duration of the provided audio file.
            try:
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        audio_path
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True
                )
                audio_duration = float(result.stdout.strip())
            except Exception as e:
                print("Error getting audio duration:", e)
                audio_duration = 0

            # Build the ffmpeg command.
            # Adding the '-shortest' option ensures that encoding stops when the audio ends.
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
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
            print("Running ffmpeg for DASH output:", " ".join(ffmpeg_cmd))
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            def write_frames():
                last_frame = None
                frame_count = 0
                while True:
                    try:
                        frame = raw_frame_queue.get(timeout=1)
                        last_frame = frame
                        frame_count += 1
                        ffmpeg_process.stdin.write(frame.tobytes())
                        ffmpeg_process.stdin.flush()
                    except queue.Empty:
                        break
                try:
                    ffmpeg_process.stdin.close()
                except Exception as e:
                    print("Error closing ffmpeg stdin:", e)
                print("Writer thread finished.")

            writer_thread = threading.Thread(target=write_frames)
            writer_thread.start()
            writer_thread.join()
            ffmpeg_process.wait()
            stderr_output = ffmpeg_process.stderr.read()
            if stderr_output:
                print("FFmpeg error output:", stderr_output.decode())
            print("DASH streaming inference complete.")

            # Free GPU cache to reduce fragmentation after inference.
            torch.cuda.empty_cache()

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
