#!/bin/bash
# Ensure the necessary directories exist
mkdir -p ./models/dwpose
mkdir -p ./models/face-parse-bisent
mkdir -p ./models/musetalk
mkdir -p ./models/sd-vae-ft-mse
mkdir -p ./models/whisper

download_if_not_exists() {
  local file_id="$1"
  local target_file="$2"

  if [ ! -f "$target_file" ]; then
    echo "Downloading $target_file ..."
    gdown "$file_id" -O "$target_file"
  else
    echo "File $target_file already exists. Skipping download."
  fi
}

# dwpose downloads
download_if_not_exists "1LbyYfkmbA4VZAjqNtsRqb9yt1HnuSm0r" "./models/dwpose/dw-ll_ucoco.pth"
download_if_not_exists "1pSmcttaDpaJzDrgQnYHclwA0Qr0Yxj8p" "./models/dwpose/dw-ll_ucoco_384.onnx"
download_if_not_exists "1tsKT_5m9ZOA0DEGrwZ21o5I-YvJL4GFe" "./models/dwpose/dw-ll_ucoco_384.pth"
download_if_not_exists "18g7-KXfl_mmq7iXl6jVu00IFM30iUdnF" "./models/dwpose/dw-mm_ucoco.pth"
download_if_not_exists "1IegxzLrJoM-0bpsTgSGsbbRIVoVO7m1f" "./models/dwpose/dw-ss_ucoco.pth"
download_if_not_exists "1tUNzL6yCpQC7XsL_OmMjsEnDksfyBvE9" "./models/dwpose/dw-tt_ucoco.pth"
download_if_not_exists "1pNAaT94ukcjrCjYya0OEyj3w5kxJGd1h" "./models/dwpose/rtm-l_ucoco_256-95bb32f5_20230822.pth"
download_if_not_exists "1tyookCwD3_6bZRrcAvMQiSoN4czla077" "./models/dwpose/rtm-x_ucoco_256-05f5bcb7_20230822.pth"
download_if_not_exists "1VlG4F2JLCi5brytFTIbxTKl7CEsCexJX" "./models/dwpose/rtm-x_ucoco_384-f5b50679_20230822.pth"
download_if_not_exists "15yEYI9eW0FdB8N_SsgiaFq-rLPhvQ1z7" "./models/dwpose/yolox_l.onnx"

# face-parse-bisent downloads
download_if_not_exists "1lA10dzSls3TgbU-lSbmGEAicXvkiFi0d" "./models/face-parse-bisent/79999_iter.pth"
download_if_not_exists "1PJPpUp104L3Fiz4DVEdC8zaJspCx0zty" "./models/face-parse-bisent/resnet18-5c106cde.pth"

# musetalk downloads
download_if_not_exists "1P7I8_eMQuVvNlYEG5TYeP-oJI0THryDu" "./models/musetalk/musetalk.json"
download_if_not_exists "1Qh_xzn9QsyWShHHGumQcp9CHoolY0UND" "./models/musetalk/pytorch_model.bin"

# sd-vae-ft-mse downloads
download_if_not_exists "1DGQZiUn1qviHAkcb2I0mXM9jUBtqsgd-" "./models/sd-vae-ft-mse/config.json"
download_if_not_exists "1efO4svHes87kTKVdIDjHmqFFqMWT5Nf_" "./models/sd-vae-ft-mse/diffusion_pytorch_model.bin"
download_if_not_exists "1IfnNrDNMViLjc6HqBxS3bJB5CDU7LHD1" "./models/sd-vae-ft-mse/diffusion_pytorch_model.safetensors"

# whisper download
download_if_not_exists "1VWkmayMeB0IgaYAkKuhhjzYfO6v7_-aB" "./models/whisper/tiny.pt"
