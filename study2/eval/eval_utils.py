import os
from pathlib import Path
from study2.static import *

def get_unprocessed_fnames(dir_a=RAW_DIR, dir_b=PROCESSED_DIR,dir_c=PROCESSED_REJECT_DIR, full_path=False):
    """
    Returns a list of file paths (relative to dir_a) that exist in dir_a (and its subdirectories)
    but NOT in dir_b (and its subdirectories).

    Args:
        dir_a (str or Path): Source directory to compare from.
        dir_b (str or Path): Directory to compare against.

    Returns:
        list[str]: List of file paths (relative to dir_a) not present in dir_b.
    """
    dir_a = Path(dir_a).resolve()
    dir_b = Path(dir_b).resolve()
    dir_c = Path(dir_c).resolve()

    # Get all files in each directory (recursively)
    files_a = {f.relative_to(dir_a) for f in dir_a.rglob('*') if f.is_file()}
    files_b = {f.relative_to(dir_b) for f in dir_b.rglob('*') if f.is_file()}
    files_c = {f.relative_to(dir_c) for f in dir_c.rglob('*') if f.is_file()}

    # remove "REJECTED_" prefix from files_c
    files_c = {Path(str(f).replace("REJECTED_","")) for f in files_c}

    # Compute difference
    unique_files = files_a - files_b
    unique_files = unique_files - files_c
    unique_files = [str(f).split('\\')[-1] for f in unique_files]  # only keep filenames

    # Return as full paths if you prefer
    if full_path:
        return [str(dir_a + f) for f in unique_files]

    else:
        return [str(f) for f in unique_files]

    pass



def get_processed_fnames(full_path=False):
    dirs = {'cond0':PROCESSED_COND0_DIR,
            'cond1':PROCESSED_COND1_DIR,
            'rejected':PROCESSED_REJECT_DIR}
    fname_dict = {}

    all_fnames = []
    for dir_key, dir_path in dirs.items():
        dir_path = Path(dir_path).resolve()
        files = [f.relative_to(dir_path) for f in dir_path.rglob('*') if f.is_file()]
        if full_path:
            files = [str(dir_path + f) for f in files]
        else:
            files = [str(f) for f in files]
        all_fnames.extend(files)
        fname_dict[dir_key] = files
    fname_dict['all'] = all_fnames

    return fname_dict


