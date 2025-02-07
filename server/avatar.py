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
        self.idx = 0
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

    def inference_mse(self, audio_path, fps):
        """
        Process the given audio file and generate a fragmented MP4 stream (suitable for Media Source Extensions).
        Yields chunks of the MP4 stream.
        """
        print("Starting MSE streaming inference ...")
        start_time = time.time()
        # Extract audio features and divide audio into chunks.
        whisper_feature = audio_processor.audio2feat(audio_path)
        whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature, fps=fps)
        total_chunks = len(whisper_chunks)
        res_frame_queue = queue.Queue()
        self.idx = 0

        # This queue will hold the final blended frames to be sent to ffmpeg.
        raw_frame_queue = queue.Queue()

        # Thread to process frames (get landmarks, blend, etc.).
        def process_frames():
            count = 0
            while count < total_chunks:
                try:
                    res_frame = res_frame_queue.get(timeout=1)
                except queue.Empty:
                    continue
                bbox = self.coord_list_cycle[self.idx % len(self.coord_list_cycle)]
                ori_frame = self.frame_list_cycle[self.idx % len(self.frame_list_cycle)].copy()
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception as e:
                    print(f"Error resizing frame: {e}")
                    continue
                mask = self.mask_list_cycle[self.idx % len(self.mask_list_cycle)]
                mask_crop_box = self.mask_coords_list_cycle[self.idx % len(self.mask_coords_list_cycle)]
                combined_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                raw_frame_queue.put(combined_frame)
                self.idx += 1
                count += 1

        process_thread = threading.Thread(target=process_frames)
        process_thread.start()

        # Run inference in batches (populate res_frame_queue).
        gen = datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
        for _, (whisper_batch, latent_batch) in enumerate(
            tqdm(gen, total=int((total_chunks + self.batch_size - 1) / self.batch_size))
        ):
            audio_feature_batch = torch.from_numpy(whisper_batch).to(device=unet.device, dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            recon = vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_queue.put(res_frame)
        process_thread.join()
        print(f"Inference processing complete. Total time: {time.time() - start_time:.2f}s")

        # Determine frame dimensions from the first frame.
        first_frame = self.frame_list_cycle[0]
        height, width, _ = first_frame.shape

        # Setup ffmpeg command to encode raw frames to a fragmented MP4 stream.
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-f", "mp4",
            "pipe:1"
        ]
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

        # Thread to write raw frames into ffmpeg's stdin.
        def write_frames():
            while True:
                try:
                    frame = raw_frame_queue.get(timeout=1)
                except queue.Empty:
                    break
                try:
                    # Write the raw bytes (BGR order) of the frame.
                    ffmpeg_process.stdin.write(frame.tobytes())
                except Exception as e:
                    print("Error writing frame to ffmpeg:", e)
                    break
            ffmpeg_process.stdin.close()

        writer_thread = threading.Thread(target=write_frames)
        writer_thread.start()

        # Yield the ffmpeg output in chunks.
        while True:
            chunk = ffmpeg_process.stdout.read(8192)
            if not chunk:
                break
            yield chunk

        writer_thread.join()
        ffmpeg_process.wait()
        print("MSE streaming inference complete.")
        
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
