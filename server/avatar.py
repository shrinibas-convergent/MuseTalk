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
DEFAULT_CHUNK_DURATION = 3

# Global semaphore to limit concurrent inference requests.
inference_semaphore = threading.Semaphore(1)

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path, exist_ok=True)

def wait_for_file(filepath, timeout=10):
    """Wait until the file exists and its size remains stable."""
    start_time = time.time()
    while not os.path.exists(filepath):
        if time.time() - start_time > timeout:
            raise Exception(f"Timeout waiting for file {filepath}")
        time.sleep(0.1)
    initial_size = os.path.getsize(filepath)
    time.sleep(0.5)
    while True:
        new_size = os.path.getsize(filepath)
        if new_size == initial_size:
            break
        initial_size = new_size
        if time.time() - start_time > timeout:
            raise Exception(f"Timeout waiting for file {filepath} to stabilize")
        time.sleep(0.5)

# --- Persistent DASH Packaging using FIFO ---
def start_dash_packager(fifo_path, manifest_path, chunk_duration):
    """
    Launch a persistent ffmpeg process that reads from the FIFO (fifo_path) and continuously produces a live MPD.
    """
    # The command reads from the FIFO as a live input and outputs DASH segments and MPD.
    packager_cmd = [
        "ffmpeg",
        "-re",
        "-i", fifo_path,
        "-c", "copy",
        "-f", "dash",
        "-window_size", "5",
        "-live", "1",
        "-update_period", "2",
        "-seg_duration", str(chunk_duration),
        manifest_path
    ]
    print("Starting persistent DASH packager with command:")
    print(" ".join(packager_cmd))
    # Launch ffmpeg persistently.
    return subprocess.Popen(packager_cmd)

def append_segment_to_fifo(segment_path, fifo_path, timeout=5):
    """
    Wait for the segment to be stable, then open it and append its bytes to the FIFO.
    Convert the segment (which is in MP4) to a transport stream (TS) to allow concatenation.
    """
    wait_for_file(segment_path, timeout=timeout)
    # Convert the segment to TS on the fly.
    ts_temp = segment_path + ".ts"
    convert_cmd = [
        "ffmpeg",
        "-y",
        "-i", segment_path,
        "-c", "copy",
        "-f", "mpegts",
        ts_temp
    ]
    subprocess.run(convert_cmd, check=True)
    # Append the TS file contents to the FIFO.
    with open(ts_temp, "rb") as ts_file, open(fifo_path, "ab") as fifo:
        shutil.copyfileobj(ts_file, fifo)
    os.remove(ts_temp)
    print(f"Appended segment {os.path.basename(segment_path)} to FIFO.")

# --- End of Persistent Packaging using FIFO ---

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

    def inference_dash(self, audio_path, fps, unique_id, chunk_duration=DEFAULT_CHUNK_DURATION):
        """
        Process the given audio file in chunks.
        For each audio chunk (of chunk_duration seconds), run inference to generate a video chunk,
        mux that video with the corresponding audio chunk, and feed the segment into a persistent
        ffmpeg dash muxer running on a FIFO.
        Returns the manifest path once the first chunk is processed.
        """
        with inference_semaphore:
            print("Starting chunked DASH streaming inference ...")
            start_time = time.time()

            # Create base directory for this request.
            base_dir = os.path.join(self.avatar_path, "dash_output", unique_id)
            osmakedirs([base_dir])
            audio_chunks_dir = os.path.join(base_dir, "audio_chunks")
            video_chunks_dir = os.path.join(base_dir, "video_chunks")
            segments_dir = os.path.join(base_dir, "segments")
            osmakedirs([audio_chunks_dir, video_chunks_dir, segments_dir])

            # Create FIFO for persistent dash packaging.
            fifo_path = os.path.join(base_dir, "fifo.ts")
            if os.path.exists(fifo_path):
                os.remove(fifo_path)
            os.mkfifo(fifo_path)
            print(f"Created FIFO at {fifo_path}")

            # Start persistent dash muxer process using ffmpeg.
            manifest_path = os.path.join(base_dir, "manifest.mpd")
            dash_cmd = [
                "ffmpeg",
                "-re",
                "-i", fifo_path,
                "-c", "copy",
                "-f", "dash",
                "-window_size", "5",
                "-live", "1",
                "-update_period", "2",
                "-seg_duration", str(chunk_duration),
                manifest_path
            ]
            print("Starting persistent dash muxer with command:")
            print(" ".join(dash_cmd))
            dash_proc = subprocess.Popen(dash_cmd)
            # We'll feed segments into the FIFO as they are produced.

            # Split the input audio into chunks.
            split_cmd = [
                "ffmpeg",
                "-y",
                "-i", audio_path,
                "-f", "segment",
                "-segment_time", str(chunk_duration),
                "-c", "copy",
                os.path.join(audio_chunks_dir, "chunk_%03d.wav")
            ]
            subprocess.run(split_cmd, check=True)
            audio_chunks = sorted(glob.glob(os.path.join(audio_chunks_dir, "chunk_*.wav")))
            if not audio_chunks:
                raise Exception("No audio chunks produced.")
            print(f"Created {len(audio_chunks)} audio chunks.")

            # Events for synchronization.
            first_chunk_event = threading.Event()
            all_segments_event = threading.Event()

            def process_chunk(i, audio_chunk):
                print(f"Processing chunk {i+1}/{len(audio_chunks)}: {audio_chunk}")
                # Compute features for the current chunk.
                whisper_feature = audio_processor.audio2feat(audio_chunk)
                whisper_chunks = audio_processor.feature2chunks(whisper_feature, fps)
                total_frames = len(whisper_chunks)
                res_frame_queue = queue.Queue()
                raw_frame_queue = queue.Queue()
                local_idx = 0

                def inference_worker():
                    with torch.no_grad():
                        for _, (whisper_batch, latent_batch) in enumerate(
                            datagen(whisper_chunks, self.input_latent_list_cycle, self.batch_size)
                        ):
                            try:
                                audio_feature_batch = torch.from_numpy(whisper_batch).to(
                                    device=unet.device, dtype=unet.model.dtype
                                )
                                audio_feature_batch = pe(audio_feature_batch)
                                latent_batch = latent_batch.to(dtype=unet.model.dtype)
                                pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
                                recon = vae.decode_latents(pred_latents)
                                for res_frame in recon:
                                    res_frame_queue.put(res_frame)
                            except Exception as e:
                                print("Error during inference batch:", e)
                    # Signal that inference is done.
                    res_frame_queue.put(None)

                def processing_worker():
                    nonlocal local_idx
                    while True:
                        res_frame = res_frame_queue.get()
                        if res_frame is None:
                            break
                        # Use the next available original frame and mask for blending.
                        bbox = self.coord_list_cycle[local_idx % len(self.coord_list_cycle)]
                        ori_frame = self.frame_list_cycle[local_idx % len(self.frame_list_cycle)].copy()
                        x1, y1, x2, y2 = bbox
                        try:
                            res_frame_resized = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                        except Exception as e:
                            print(f"Error resizing frame: {e}")
                            continue
                        mask = self.mask_list_cycle[local_idx % len(self.mask_list_cycle)]
                        mask_crop_box = self.mask_coords_list_cycle[local_idx % len(self.mask_coords_list_cycle)]
                        combined_frame = get_image_blending(ori_frame, res_frame_resized, bbox, mask, mask_crop_box)
                        raw_frame_queue.put(combined_frame)
                        local_idx += 1

                inf_thread = threading.Thread(target=inference_worker)
                proc_thread = threading.Thread(target=processing_worker)
                inf_thread.start()
                proc_thread.start()
                inf_thread.join()
                proc_thread.join()
                print(f"Chunk {i} generated {local_idx} frames.")

                if i == 0 and local_idx < 5:
                    print("First chunk has insufficient frames; waiting extra 2 seconds.")
                    time.sleep(2)

                first_frame = self.frame_list_cycle[0]
                height, width, _ = first_frame.shape
                video_chunk_path = os.path.join(video_chunks_dir, f"video_chunk_{i:03d}.mp4")
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "rawvideo",
                    "-pixel_format", "bgr24",
                    "-video_size", f"{width}x{height}",
                    "-framerate", str(fps),
                    "-i", "pipe:0",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-preset", "veryfast",
                    "-crf", "23",
                    video_chunk_path
                ]
                ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                while True:
                    try:
                        frame = raw_frame_queue.get(timeout=10)
                        ffmpeg_process.stdin.write(frame.tobytes())
                    except queue.Empty:
                        break
                try:
                    ffmpeg_process.stdin.close()
                except Exception as e:
                    print("Error closing ffmpeg stdin:", e)
                ffmpeg_process.wait()

                wait_for_file(video_chunk_path)
                wait_for_file(audio_chunk)

                final_segment_path = os.path.join(segments_dir, f"segment_{i:03d}.mp4")
                mux_cmd = [
                    "ffmpeg",
                    "-y",
                    "-fflags", "+genpts",
                    "-i", video_chunk_path,
                    "-i", audio_chunk,
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                    "-reset_timestamps", "1",
                    "-shortest",
                    "-f", "mp4",
                    final_segment_path
                ]
                subprocess.run(mux_cmd, check=True)
                print(f"Segment {i:03d} created.")
                # Append the segment to the FIFO (convert to TS on the fly)
                append_segment_to_fifo(final_segment_path, fifo_path)
                if i == 0:
                    first_chunk_event.set()

            # Process audio chunks sequentially.
            audio_chunks = sorted(glob.glob(os.path.join(audio_chunks_dir, "chunk_*.wav")))
            if not audio_chunks:
                raise Exception("No audio chunks produced.")
            print(f"Total {len(audio_chunks)} audio chunks created.")

            # Process the first chunk synchronously.
            process_chunk(0, audio_chunks[0])
            first_chunk_event.wait()

            # Process remaining chunks in background.
            def process_remaining():
                for i in range(1, len(audio_chunks)):
                    process_chunk(i, audio_chunks[i])
                print("All chunks processed.")
                all_segments_event.set()
            remaining_thread = threading.Thread(target=process_remaining, daemon=True)
            remaining_thread.start()

            remaining_thread.join()
            # After processing all chunks, we close the FIFO by opening it in write mode and closing immediately.
            with open(fifo_path, "wb") as fifo:
                pass
            # Wait for persistent dash packager to finish.
            dash_proc.wait()
            print(f"Inference processing complete (all chunks). Total time: {time.time() - start_time:.2f}s")
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
