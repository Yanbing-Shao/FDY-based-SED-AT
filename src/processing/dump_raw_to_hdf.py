#!/usr/bin/env python3
import argparse
import librosa
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import soundfile as sf
import h5py
import os
import multiprocessing as mp


parser = argparse.ArgumentParser()
parser.add_argument("input_csv")
parser.add_argument("-o", "--output", type=str, required=True, help="Output data hdf5")
parser.add_argument(
    "-g", "--group", required=True, type=str, help="Group to create in the hdf5 file"
)
parser.add_argument(
    "-f", "--folder", type=str, required=True, help="Path to audio folder"
)
parser.add_argument("-sr", type=int, default=32000)
parser.add_argument("-sep", default=",", type=str)
args = parser.parse_args()


@logger.catch
def worker(idx_filename, q):
    try:
        idx = idx_filename[0]
        fname = idx_filename[1]
        y, sr = sf.read(args.folder + fname, dtype="float32")
        if y.ndim > 1:
        # Merge channels
            y = y.mean(-1)
        if sr != args.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
        res = (idx, y, fname)
        q.put(res)
        return res
    except Exception as e:
        print(f"Error processing {idx_filename[1]}: {e}")
        return None


@logger.catch
def listener(q):
    """listens for messages on the q, writes to file."""
    if os.path.exists(args.output):
        hf = h5py.File(args.output, "r+")
    else:
        hf = h5py.File(args.output, "w")

    dname = f"{args.group}/audio_32k"
    if dname in hf:
        dset = hf[dname]
    else:
        dset = hf.create_dataset(dname, shape=(len(filenames), args.sr * 10))

    # Add flienames to h5 dataset
    fnameCol = f"{args.group}/filenames"
    if fnameCol not in hf:
        fname = hf.create_dataset(fnameCol, data=np.array(filenames,dtype=h5py.string_dtype()))
    else:
        fname = hf[fnameCol]

    while 1:
        m = q.get()
        if m == "kill":
            logger.success("Killing listener")
            hf.close()
            break
        idx = m[0]
        audio = m[1]
        dset[idx] = audio
        fname[idx] = m[2]


logger.add("somefile.log", enqueue=True)
df = pd.read_csv(args.input_csv, sep=args.sep)
assert "filename" in df.columns, "Header needs to contain 'filename'"

filenames = df["filename"].to_list()
MAX_FILES = 30001
# maybe all jobs[] are stored in memory-so this script can only handle files < 65gb(my computer mem)
if len(filenames) > MAX_FILES:
    filenames = filenames[:MAX_FILES]
# must use Manager queue here, or will not work
# when process sgp_unlabelled face brokenPipeError
manager = mp.Manager()
q = manager.Queue(maxsize=10000)
pool = mp.Pool(mp.cpu_count())

# put listener to work first
watcher = pool.apply_async(listener, (q,))

try:
    # fire off workers
    jobs = []
    for i, fname in enumerate(filenames):
        job = pool.apply_async(worker, ((i, fname), q))
        jobs.append(job)

    # 使用 tqdm 监控任务进度
    for job in tqdm(jobs, desc="Processing audio"):
        job.get()  # 等待所有任务完成

except Exception as e:
    logger.error(f"Main process error: {e}")
    pool.terminate()
finally:
    q.put("kill")  # 确保发送终止信号
    pool.close()
    pool.join()
