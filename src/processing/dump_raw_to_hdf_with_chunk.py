#!/usr/bin/env python3
import argparse
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import h5py
import os
import gc
import time
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock

class AudioProcessor:
    def __init__(self, args):
        self.args = args
        self.lock = Lock()  # HDF5写入锁
        self.chunk_size = self.calculate_chunk_size()
        
    def calculate_chunk_size(self):
        """动态计算分块大小"""
        df = pd.read_csv(self.args.input_csv, sep=self.args.sep)
        total_files = len(df)
        base_chunk = 2000  # 基础分块大小
        return max(base_chunk, total_files // 10 + 1)

    def process_file(self, file_info):
        """处理单个音频文件"""
        idx, rel_path = file_info
        full_path = Path(self.args.folder) / rel_path
        error = None
        y_processed = None

        try:
            # 读取原始音频
            y, sr = sf.read(full_path, dtype="float32")
            
            # 多声道合并
            if y.ndim > 1:
                y = y.mean(axis=-1)
            
            # 重采样
            if sr != self.args.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.args.sr)
            
            # 标准化长度
            target_length = self.args.sr * 10
            if len(y) > target_length:
                y = y[:target_length]
            elif len(y) < target_length:
                y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
            return (idx, y, rel_path, None)
        
        except Exception as e:
            return (idx, None, rel_path, str(e))

    def write_to_hdf5(self, hf, data):
        """线程安全的HDF5写入"""
        with self.lock:
            audio_dset = hf.require_dataset(
                name=f"{self.args.group}/audio_32k",
                shape=(self.total_files, self.args.sr * 10),
                dtype=np.float32,
                maxshape=(None, self.args.sr * 10),
                chunks=(100, self.args.sr * 10),
                fillvalue=0.0
            )

            name_dset = hf.require_dataset(
                name=f"{self.args.group}/filenames",
                shape=(self.total_files,),
                dtype=h5py.string_dtype(encoding='utf-8'),
                fillvalue="ERROR"
            )

            for idx, y, fname, error in data:
                if error is None:
                    audio_dset[idx] = y
                    name_dset[idx] = fname
                else:
                    audio_dset[idx] = np.zeros(self.args.sr * 10)
                    name_dset[idx] = f"ERROR: {error}"

    def process_chunk(self, chunk_files, hdf5_file):
        """处理单个数据块"""
        with Pool(processes=self.args.workers) as pool:
            results = []
            with tqdm(total=len(chunk_files), desc=f"处理分块 {self.current_chunk}/{self.total_chunks}") as pbar:
                for result in pool.imap_unordered(self.process_file, chunk_files):
                    results.append(result)
                    pbar.update(1)
            
            # 写入HDF5并清理内存
            with h5py.File(hdf5_file, 'a') as hf:
                self.write_to_hdf5(hf, results)
            
            del results
            gc.collect()

    def run(self):
        """主处理流程"""
        # 初始化文件
        df = pd.read_csv(self.args.input_csv, sep=self.args.sep)
        self.total_files = len(df)
        filenames = list(enumerate(df["filename"]))
        
        # 计算分块
        self.total_chunks = (self.total_files + self.chunk_size - 1) // self.chunk_size
        chunks = [filenames[i*self.chunk_size:(i+1)*self.chunk_size] 
                for i in range(self.total_chunks)]

        # 创建空HDF5文件
        with h5py.File(self.args.output, 'w') as f:
            pass

        # 分块处理
        for self.current_chunk, chunk in enumerate(chunks, 1):
            start_time = time.time()
            self.process_chunk(chunk, self.args.output)
            
            # 打印内存状态
            chunk_time = time.time() - start_time
            print(f"分块 {self.current_chunk} 完成 | "
                f"耗时: {chunk_time:.1f}s | "
                f"剩余内存: {self.get_memory_usage()} MB")

    def get_memory_usage(self):
        """获取当前进程内存使用（仅限Linux）"""
        if os.name != 'posix':
            return "N/A"
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) // 1024
        return "N/A"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="audio2hdf5")
    parser.add_argument("input_csv", help="输入CSV文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出HDF5文件路径")
    parser.add_argument("-g", "--group", required=True, help="HDF5组名称")
    parser.add_argument("-f", "--folder", required=True, help="音频根目录") 
    parser.add_argument("-sr", type=int, default=32000, help="目标采样率")
    parser.add_argument("-sep", default=",", type=str)
    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(), 
                      help="并发工作进程数")
    parser.add_argument("--chunk-multiplier", type=float, default=1.5,
                      help="动态分块大小乘数（基于系统内存）")
    args = parser.parse_args()

    processor = AudioProcessor(args)
    try:
        processor.run()
        print("处理完成！输出文件：", args.output)
    except KeyboardInterrupt:
        print("\n用户中断，正在清理...")
    except Exception as e:
        print(f"致命错误: {str(e)}")
        exit(1)