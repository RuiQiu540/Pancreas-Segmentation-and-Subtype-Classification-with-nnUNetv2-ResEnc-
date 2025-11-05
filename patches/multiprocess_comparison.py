from pathlib import Path
import time
from multiprocessing import Pool, set_start_method
import numpy as np
import SimpleITK as sitk

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_spacing

RAW_ROOT = Path(r"D:\project\nnUNet_reproducing\sample\nnUNet_raw")
OUT_ROOT = Path(r"D:\project\nnUNet_reproducing\sample\nnUNet_preprocessed\resampled_1mm")
DATASET = "Dataset111_Reproduce"

IMAGES_TR = RAW_ROOT / DATASET / "imagesTr"
LABELS_TR = RAW_ROOT / DATASET / "labelsTr"
OUT_IMAGES_TR = OUT_ROOT / DATASET / "imagesTr"
OUT_LABELS_TR = OUT_ROOT / DATASET / "labelsTr"

TARGET_SPACING = (1.0, 1.0, 1.0)

io = SimpleITKIO()
#----------------------------Configuration------------------------------------------------
def save_image(array_cxyz:np.array,props:dict,out_path:Path):
    array = array_cxyz[0]
    img = sitk.GetImageFromArray(array.astype(np.float32))
    img.SetSpacing(props["sitk_stuff"]["spacing"])
    img.SetOrigin(props["sitk_stuff"]["origin"])
    img.SetDirection(props["sitk_stuff"]["direction"])
    sitk.WriteImage(img,str(out_path))

def process_one(img_path:Path):
    stem = img_path.name[:-7]
    if stem.endswith("0000"):
        case_id = stem[:-5]
    else:
        case_id = stem

    img, img_props = io.read_images((str(img_path),))
    img_rs = resample_data_or_seg_to_spacing(
        img,
        current_spacing = img_props["spacing"],
        new_spacing = TARGET_SPACING,
        is_seg = False,
        order = 3,
        order_z = 0,
        force_separate_z = False
    )
    OUT_IMAGES_TR.mkdir(parents = True, exist_ok = True)
    save_image(img_rs,img_props, OUT_IMAGES_TR / img_path.name)
    # resample training dataset
    # if there exists labels
    
    label_path = LABELS_TR / f"{case_id}.nii.gz"

    if label_path.exists():
        label, label_props = io.read_seg(str(label_path))
        label_rs = resample_data_or_seg_to_spacing(
            label,
            current_spacing = label_props["spacing"],
            new_spacing = TARGET_SPACING,
            is_seg = True,
            order = 0,
            order_z = 0,
            force_separate_z = False
        )
        OUT_LABELS_TR.mkdir(parents = True, exist_ok = True)
        io.write_seg(label_rs[0],str(OUT_LABELS_TR/f"{case_id}.nii.gz"), label_props)
    # resample labels if exists
    return img_path.name

def main():
    images = sorted(IMAGES_TR.glob("*.nii.gz"))
    print(f"found {len(images)} samples")

    t0 = time.perf_counter()
    for p in images:
        process_one(p)
    t1 = time.perf_counter()

    print("native for loop completed!")
    print(f"time spent on this progress is {t1-t0}")
    t2 = time.perf_counter()
    with Pool(4) as pool:
        for i in pool.imap_unordered(process_one,images):
            pass
    t3 = time.perf_counter()
    
    # we can add a chunksize to imap_unordered(...) to reduce the dispatch overhead.
    # however, after testing with chunksize=2, it actually became slower.
    # this is because our samples vary a lot in size, so bundling multiple tasks together makes the slow ones block the fast ones.
    # from the tests (and discussion with GPT), it seems chunksize only helps when tasks are fairly uniform and the total number of tasks is large.
    # the actual gain in this case is negligible, so I keep chunksize=1 here.

    
    print("multi processor completed!")
    print(f"time spent on this progress is {t3-t2}")
    
    print(f"multi process is {(t1-t0)/(t3-t2)}x faster")
    print("task completed")

if __name__ == "__main__":

    main()
