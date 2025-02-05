from PIL import Image
import numpy as np
import cv2
import copy
from face_parsing import FaceParsing

fp = FaceParsing()

def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s

def face_seg(image):
    seg_image = fp(image)
    if seg_image is None:
        print("error, no person_segment")
        return None

    seg_image = seg_image.resize(image.size)
    return seg_image

def get_image(image,face,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    #print(image.shape)
    #print(face.shape)
    
    body = Image.fromarray(image[:,:,::-1])
    face = Image.fromarray(face[:,:,::-1])

    x, y, x1, y1 = face_box 
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box
    face_position = (x, y)

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    mask_image = Image.fromarray(mask_array)
    
    face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
    body.paste(face_large, crop_box[:2], mask_image)
    body = np.array(body)
    return body[:,:,::-1]

def get_image_prepare_material(image,face_box,upper_boundary_ratio = 0.5,expand=1.2):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array,crop_box

def get_image_blending(image, face, face_box, mask_array, crop_box):
    """
    Blends the 'face' into the 'image' using a mask.
    
    Parameters:
      image: The original image.
      face: The face image to be inserted.
      face_box: A tuple (x, y, x1, y1) defining the region in the image where the face should go.
      mask_array: The mask used for blending. Can be either a 3-channel image or already grayscale.
      crop_box: A tuple (x_s, y_s, x_e, y_e) defining the region to crop from the image.
      
    Returns:
      Blended image.
    """
    # Make a copy of the original image
    body = image.copy()
    
    # Unpack the face and crop boxes
    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box
    
    # Extract the region from the original image where blending will occur
    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    
    # Determine the region in face_large where the face will be inserted
    face_region_y1 = y - y_s
    face_region_y2 = y1 - y_s
    face_region_x1 = x - x_s
    face_region_x2 = x1 - x_s
    face_large[face_region_y1:face_region_y2, face_region_x1:face_region_x2] = face

    # Check if the mask is already single-channel. If not, convert it.
    if len(mask_array.shape) == 2:
        mask_image = mask_array.astype(np.float32) / 255.0
    else:
        mask_image = cv2.cvtColor(mask_array, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Ensure the mask has 3 channels to blend with a color image.
    mask_image_3ch = np.stack([mask_image]*3, axis=-1)

    # Blend the face_large region with the original image region using the mask.
    # For each pixel: output = alpha * face_large + (1 - alpha) * original
    blended_region = (mask_image_3ch * face_large + (1 - mask_image_3ch) * body[y_s:y_e, x_s:x_e]).astype(np.uint8)
    
    # Replace the blended region back into the original image
    body[y_s:y_e, x_s:x_e] = blended_region

    return body