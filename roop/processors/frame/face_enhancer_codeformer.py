from typing import Any, List, Callable
import torch
import threading
import cv2

from torchvision.transforms.functional import normalize
from codeformer.facelib.utils.face_restoration_helper import FaceRestoreHelper
from codeformer.basicsr.utils.registry import ARCH_REGISTRY
from codeformer.basicsr.archs.rrdbnet_arch import RRDBNet
from codeformer.basicsr.utils.realesrgan_utils import RealESRGANer
from codeformer.basicsr.utils import img2tensor, tensor2img

import roop.globals
import roop.processors.frame.core
from roop.core import update_status
from roop.typing import Frame, Face
from roop.utilities import conditional_download, resolve_relative_path, is_image, is_video


UPSCALE_FACTOR = 2  # integer, higher value needs more VRAM
FIDELITY = 0.8  # float, 1 = better fidelity
BACKGROUND_ENHANCE = True  # enable = slower + more VRAM
FACE_UPSAMPLE = True  # enable = slower + more VRAM


NAME = 'ROOP.FACE-ENHANCER.CODEFORMER'
CODE_FORMER = None
FACE_ENHANCER = None
UPSAMPLER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def pre_check() -> bool:
    download_directory_path = resolve_relative_path('../models')
    conditional_download(download_directory_path, [
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    ])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.target_path) and not is_video(roop.globals.target_path):
        update_status('Select an image or video for target path.', NAME)
        return False
    return True


def post_process() -> None:
    global CODE_FORMER, FACE_ENHANCER, UPSAMPLER
    CODE_FORMER = None
    FACE_ENHANCER = None
    UPSAMPLER = None


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:
    with THREAD_SEMAPHORE:

        face_helper = get_face_enhancer()
        face_helper.read_image(temp_frame)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()

        cropped_faces = face_helper.cropped_faces
        faces_enhanced = enhance_face_in_frame(cropped_faces)
        for face_enhanced in faces_enhanced:
            face_helper.add_restored_face(face_enhanced)
        face_helper.get_inverse_affine()

        result = face_helper.paste_faces_to_input_image(
            upsample_img=get_upsampler().enhance(temp_frame, outscale=UPSCALE_FACTOR)[0] if BACKGROUND_ENHANCE else None,
            face_upsampler=get_upsampler() if FACE_UPSAMPLE else None
        )
        face_helper.clean_all()

    return result


def process_frames(source_path: str, frame_paths: List[str], update: Callable[[], None]) -> None:
    for frame_path in frame_paths:
        try:
            frame = cv2.imread(frame_path)
            result = process_frame(None, frame)
            cv2.imwrite(frame_path, result)
        except Exception as e:
            print(e)
            continue
        if update:
            update()


def process_image(source_path: str, image_path: str, output_file: str) -> None:
    image = cv2.imread(image_path)
    result = process_frame(None, image)
    cv2.imwrite(output_file, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)


def get_code_former() -> Any:
    global CODE_FORMER
    with THREAD_LOCK:
        model_path = resolve_relative_path('../models/codeformer.pth')
        if CODE_FORMER is None:
            model = torch.load(model_path)['params_ema']
            CODE_FORMER = ARCH_REGISTRY.get('CodeFormer')(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=['32', '64', '128', '256'],
            ).to(DEVICE)
            CODE_FORMER.load_state_dict(model)
            CODE_FORMER.eval()
        return CODE_FORMER


def get_face_enhancer() -> Any:
    global FACE_ENHANCER
    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            FACE_ENHANCER = FaceRestoreHelper(
                upscale_factor=UPSCALE_FACTOR,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=DEVICE
            )
        return FACE_ENHANCER


def get_upsampler() -> Any:
    global UPSAMPLER
    with THREAD_LOCK:
        if UPSAMPLER is None:
            half: bool = DEVICE != "cpu"
            weight_path = resolve_relative_path('../models/RealESRGAN_x2plus.pth')
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
            )
            UPSAMPLER = RealESRGANer(
                scale=2,
                model_path=weight_path,
                model=model,
                tile=400,
                tile_pad=40,
                pre_pad=0,
                half=half,
                device=DEVICE,
            )
        return UPSAMPLER


def enhance_face_in_frame(cropped_faces):
    try:
        faces_enhanced = []
        for _, cropped_face in enumerate(cropped_faces):
            face_in_tensor = normalize_face(cropped_face)
            face_enhanced = restore_face(face_in_tensor)
            faces_enhanced.append(face_enhanced)
        return faces_enhanced
    except RuntimeError as e:
        print(f'Failed inference for CodeFormer-code: {e}')
        return cropped_faces


def restore_face(face_in_tensor):
    with torch.no_grad():
        enhanced_face_in_tensor = get_code_former()(face_in_tensor, w=FIDELITY, adain=True)[0]
    try:
        restored_face = tensor2img(enhanced_face_in_tensor, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
        del enhanced_face_in_tensor
    except RuntimeError as e:
        print(f'Failed inference for CodeFormer-tensor: {e}')
        restored_face = tensor2img(face_in_tensor, rgb2bgr=True, min_max=(-1, 1)).astype('uint8')
    return restored_face


def normalize_face(face):
    face_in_tensor = img2tensor(face / 255.0, bgr2rgb=True, float32=True)
    normalize(face_in_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    return face_in_tensor.unsqueeze(0).to(DEVICE)
