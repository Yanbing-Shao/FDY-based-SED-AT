import numpy as np
import pandas as pd

class ManyHotEncoderNumpy:
    """
    NumPy版本的ManyHotEncoder，不依赖PyTorch
    """
    
    def __init__(
        self,
        taxonomy,
        use_taxo_fine,
        audio_len,
        frame_len,
        frame_hop,
        net_pooling=1,
        fs=16000,
    ):

        self.taxonomy_coarse = taxonomy["coarse"]
        self.taxonomy_fine = taxonomy["fine"]

        if use_taxo_fine:
            self.taxonomy = self.taxonomy_fine
            labels = self.taxonomy["class_labels"]
        else:
            self.taxonomy = self.taxonomy_coarse
            labels = self.taxonomy["class_labels"]

        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()

        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.fs = fs
        self.net_pooling = net_pooling
        n_frames = self.audio_len * self.fs
        self.n_frames = int(int((n_frames / self.frame_hop)) / self.net_pooling)
        
        # 使用NumPy数组而不是torch.Tensor
        self.ftc_matrix = self.compute_fine_to_coarse_matrix()

    def compute_fine_to_coarse_matrix(self):
        """计算细粒度到粗粒度的映射矩阵（NumPy版本）"""
        classes_fine = self.taxonomy_fine["class_labels"]
        classes_coarse = self.taxonomy_coarse["class_labels"]
        t_matrix = np.zeros((len(classes_fine), len(classes_coarse)))

        for k in self.taxonomy_fine["SONYC"].keys():
            c_fine = self.taxonomy_fine["SONYC"][k]
            if c_fine == "no-annotation":
                continue
            c_coarse = self.taxonomy_coarse["SONYC"][k]
            idx_fine = classes_fine.index(c_fine)
            idx_coarse = classes_coarse.index(c_coarse)
            t_matrix[idx_fine, idx_coarse] = 1

        for k in self.taxonomy_fine["SINGA-PURA"].keys():
            c_fine = self.taxonomy_fine["SINGA-PURA"][k]
            c_coarse = self.taxonomy_coarse["SINGA-PURA"][k]
            if c_fine == "no-annotation":
                continue
            idx_fine = classes_fine.index(c_fine)
            idx_coarse = classes_coarse.index(c_coarse)
            t_matrix[idx_fine, idx_coarse] = 1

        return t_matrix

    def fine_to_coarse(self, prob):
        """
        NumPy版本的细粒度到粗粒度转换
        专门针对 batch=1 的推理场景优化
    
        Args:
            prob: numpy数组，形状为:
                - 强标签: (1, n_classes, time) 
                - 弱标签: (1, n_classes)
    
        Returns:
            numpy数组，粗粒度的预测概率，形状对应:
                  - 强标签: (1, n_coarse_classes, time)
                  - 弱标签: (1, n_coarse_classes)
        """
        if prob.ndim == 3:
            # 强标签情况: (1, n_fine_classes, time)
            # 转置为 (1, time, n_fine_classes)
            prob_transposed = np.transpose(prob, (0, 2, 1))
        
            # 与ftc_matrix广播相乘: (1, time, n_fine_classes, n_coarse_classes)
            # ftc_matrix形状: (n_fine_classes, n_coarse_classes)
            prob_expanded = prob_transposed[:, :, :, np.newaxis]
            expanded_ftc = self.ftc_matrix[np.newaxis, np.newaxis, :, :]
            multiplied = prob_expanded * expanded_ftc  # 转置以匹配维度
        
            # 沿细粒度通道维度取最大值: (1, time, n_coarse_classes)
            max_values = np.max(multiplied, axis=2, keepdims=False)
        
            # 转置回原始形状: (1, n_coarse_classes, time)
            result = np.transpose(max_values, (0, 2, 1))
            
            return result
        
        elif prob.ndim == 2:
            # 弱标签情况: (1, n_fine_classes)
            # 扩展维度: (1, n_fine_classes, 1)
            prob_expanded = prob[:, :, np.newaxis]
        
            # 与ftc_matrix广播相乘: (1, n_fine_classes, n_coarse_classes)
            multiplied = prob_expanded * self.ftc_matrix
        
            # 沿细粒度通道维度取最大值: (1, n_coarse_classes)
            result = np.max(multiplied, axis=1)
        
            return result

    def convert_label_fine_to_coarse(self, label):
        """将细粒度标签转换为粗粒度标签（NumPy版本）"""
        idx_fine = self.taxonomy_fine["class_labels"].index(label)
        coarse_idx = np.where(self.ftc_matrix[idx_fine])[0][0]
        return self.taxonomy_coarse["class_labels"][coarse_idx]

    def encode_weak(self, labels, dset):
        """编码弱标签（与原始版本相同）"""
        labels = list(map(lambda label: self.taxonomy[dset][label], labels))
        y = np.zeros(len(self.labels))
        for label in labels:
            if not pd.isna(label) and label != "no-annotation":
                i = self.labels.index(label)
                y[i] = 1
        return y

    def _time_to_frame(self, time):
        """时间到帧的转换"""
        samples = time * self.fs
        frame = samples / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        """帧到时间的转换"""
        frame = frame * self.net_pooling / (self.fs / self.frame_hop)
        return np.clip(frame, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, label_df, dset):
        """编码强标签DataFrame（与原始版本相同）"""
        assert any([x is not None for x in [self.audio_len, self.frame_len, self.frame_hop]])

        samples_len = self.n_frames
        y = np.zeros((samples_len, len(self.labels)))
        
        if type(label_df) is pd.DataFrame:
            if {"onset", "offset", "event_label"}.issubset(label_df.columns):
                for _, row in label_df.iterrows():
                    unified_label = self.taxonomy[dset][row["event_label"]]
                    if (not pd.isna(row["event_label"]) and 
                        unified_label != "no-annotation"):
                        i = self.labels.index(unified_label)
                        onset = int(self._time_to_frame(row["onset"]))
                        offset = int(np.ceil(self._time_to_frame(row["offset"])))
                        y[onset:offset, i] = 1
        else:
            raise NotImplementedError(
                f"To encode_strong, type is pandas.Dataframe with onset, offset and event_label "
                f"columns, type given: {type(label_df)}"
            )
        return y

    def decode_weak(self, labels):
        """解码弱标签（与原始版本相同）"""
        result_labels = []
        for i, value in enumerate(labels):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def decode_strong(self, labels, taxo_level=None):
        """解码强标签（与原始版本相同）"""
        if len(labels) == len(self.taxonomy_coarse["class_labels"]):
            class_labels = self.taxonomy_coarse["class_labels"]
        elif len(labels) == len(self.taxonomy_fine["class_labels"]):
            class_labels = self.taxonomy_fine["class_labels"]
        elif taxo_level == "coarse":
            class_labels = self.taxonomy_coarse["class_labels"]
        elif taxo_level == "fine":
            class_labels = self.taxonomy_fine["class_labels"]
        else:
            class_labels = self.labels
            
        result_labels = []
        for i, label_column in enumerate(labels):
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                result_labels.append([
                    class_labels[i],
                    self._frame_to_time(row[0]),
                    self._frame_to_time(row[1])
                ])
        return result_labels

    def find_contiguous_regions(self, activity_array):
        """查找连续区域（与原始版本相同）"""
        change_indices = np.logical_xor(
            activity_array[1:], activity_array[:-1]
        ).nonzero()[0]
        change_indices += 1

        if activity_array[0]:
            change_indices = np.r_[0, change_indices]
        if activity_array[-1]:
            change_indices = np.r_[change_indices, activity_array.size]

        return change_indices.reshape((-1, 2))

    def state_dict(self):
        """返回状态字典（与原始版本相同）"""
        return {
            "labels": self.labels,
            "audio_len": self.audio_len,
            "frame_len": self.frame_len,
            "frame_hop": self.frame_hop,
            "net_pooling": self.net_pooling,
            "fs": self.fs,
        }

    @classmethod
    def load_state_dict(cls, state_dict):
        """从状态字典加载（与原始版本相同）"""
        # 注意：这个方法需要taxonomy信息，所以可能需要调整
        # 这里保持与原始版本相同的接口
        labels = state_dict["labels"]
        audio_len = state_dict["audio_len"]
        frame_len = state_dict["frame_len"]
        frame_hop = state_dict["frame_hop"]
        net_pooling = state_dict["net_pooling"]
        fs = state_dict["fs"]
        # 这里需要提供taxonomy，可能需要额外的参数
        return cls(labels, audio_len, frame_len, frame_hop, net_pooling, fs)