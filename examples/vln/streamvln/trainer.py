# Copyright (c) Alibaba, Inc. and its affiliates.
"""
StreamVLN Training Entry Point

Custom SFT trainer for StreamVLN that:
1. Uses StreamVLNTrainArguments for VLN-specific parameters
2. Creates StreamVLNDataset when detecting VLN data path
3. Integrates with ms-swift's standard training pipeline

Usage:
    python examples/vln/streamvln/trainer.py --custom_register_path examples/vln/streamvln ...
"""

import os
import sys
from typing import Optional, List, Union

# Add paths for direct script execution
if __name__ == '__main__':
    _current_dir = os.path.dirname(os.path.abspath(__file__))
    # Add ms-swift root to path (for swift.llm imports)
    _msswift_root = os.path.dirname(os.path.dirname(os.path.dirname(_current_dir)))
    if _msswift_root not in sys.path:
        sys.path.insert(0, _msswift_root)
    # Add current directory to path (for local imports)
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)

from swift.llm.train.sft import SwiftSft
from swift.utils import get_logger

# Use imports that work both as module and direct script
try:
    from .arguments import StreamVLNTrainArguments
    from .dataset import StreamVLNDataset
except ImportError:
    # Direct script execution - use absolute imports
    from arguments import StreamVLNTrainArguments
    from dataset import StreamVLNDataset

logger = get_logger()


class StreamVLNSft(SwiftSft):
    """
    StreamVLN 自定义 SFT 流程
    
    使用 ms-swift 标准数据格式，让框架自动处理多模态输入
    """
    args_class = StreamVLNTrainArguments
    args: StreamVLNTrainArguments

    def _get_dataset(self):
        """
        重写数据集获取逻辑，检测 StreamVLN 数据集并创建
        
        支持以下格式：
        - 单路径: --dataset /path/to/dataset
        - 多路径(列表): --dataset /path/to/dataset1 /path/to/dataset2
        - 多路径(逗号分隔): --dataset /path/to/dataset1,/path/to/dataset2
        """
        if self.args.dataset:
            # 处理多路径：统一为逗号分隔的字符串
            if isinstance(self.args.dataset, list):
                data_path = ','.join(self.args.dataset)
            else:
                data_path = self.args.dataset
            
            # 检查是否至少有一个路径包含 annotations.json
            paths = [p.strip() for p in data_path.split(',')]
            has_streamvln = any(
                os.path.isdir(p) and os.path.exists(os.path.join(p, 'annotations.json'))
                for p in paths
            )
            
            if has_streamvln:
                logger.info(f"Detected StreamVLN dataset(s), creating StreamVLNDataset (paths={paths})")
                train_dataset = StreamVLNDataset(
                    data_path=data_path,
                    num_frames=self.args.num_frames,
                    num_history=self.args.num_history,
                    num_future_steps=self.args.num_future_steps,
                    use_random=self.args.use_random,
                    max_samples=self.args.vln_max_samples,
                )
                logger.info(f"StreamVLN Dataset created: {len(train_dataset)} samples")
                return train_dataset, None
        
        # 调用父类获取数据集
        return super()._get_dataset()

    def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
        """
        重写编码逻辑，跳过 StreamVLNDataset 的 HuggingFace 预处理
        """
        # StreamVLNDataset 是 PyTorch Dataset，不需要 HuggingFace 预处理
        if isinstance(train_dataset, StreamVLNDataset):
            logger.info("Skipping HuggingFace preprocessing for StreamVLNDataset")
            return train_dataset, val_dataset
        
        # 其他数据集使用父类逻辑
        return super()._encode_dataset(train_dataset, val_dataset, pre_process=pre_process)

    def _post_process_datasets(self, datasets):
        """
        重写后处理逻辑，为 StreamVLNDataset 创建 LazyLLMDataset 包装
        """
        from swift.llm.dataset import LazyLLMDataset
        
        args = self.args
        template = self.template
        
        for i, dataset in enumerate(datasets):
            if dataset is None:
                continue
            
            if isinstance(dataset, StreamVLNDataset):
                # StreamVLNDataset 需要用 LazyLLMDataset 包装
                logger.info(f"Wrapping StreamVLNDataset with LazyLLMDataset")
                datasets[i] = LazyLLMDataset(
                    dataset, 
                    template.encode, 
                    strict=args.strict, 
                    random_state=args.data_seed
                )
            else:
                # 其他数据集使用父类逻辑
                pass
        
        # 对非 StreamVLNDataset 调用父类处理
        has_non_streamvln = any(
            d is not None and not isinstance(d, (StreamVLNDataset, LazyLLMDataset)) 
            for d in datasets
        )
        if has_non_streamvln:
            datasets = super()._post_process_datasets(datasets)
        
        return datasets

    def _show_dataset(self, train_dataset, val_dataset):
        """
        重写展示逻辑
        """
        # 检查是否是 LazyLLMDataset 包装的 StreamVLNDataset
        from swift.llm.dataset import LazyLLMDataset
        
        inner_dataset = train_dataset
        if isinstance(train_dataset, LazyLLMDataset):
            inner_dataset = train_dataset.dataset
        
        if isinstance(inner_dataset, StreamVLNDataset):
            logger.info(f"StreamVLN Dataset: {len(inner_dataset)} samples")
            # 展示一个样本
            if len(inner_dataset) > 0:
                sample = inner_dataset[0]
                logger.info(f"Sample keys: {sample.keys()}")
                logger.info(f"Number of messages: {len(sample.get('messages', []))}")
                logger.info(f"Number of images: {len(sample.get('images', []))}")
            return
        super()._show_dataset(train_dataset, val_dataset)


def train_main(args: Optional[Union[List[str], StreamVLNTrainArguments]] = None):
    """Main entry point for StreamVLN training."""
    return StreamVLNSft(args).main()


if __name__ == '__main__':
    train_main()
