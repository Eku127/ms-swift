# Copyright (c) Alibaba, Inc. and its affiliates.
"""自定义数据集注册：Caption3o-LongCap-v4"""
from typing import Dict, Any, Optional
from swift.llm import DatasetMeta, register_dataset
from swift.llm.dataset.preprocessor.core import ResponsePreprocessor


class Caption3oLongCapPreprocessor(ResponsePreprocessor):
    """Caption3o-LongCap-v4 数据集预处理器
    
    数据集格式: {image: PIL.Image, caption: str}
    转换为: {images: [image], messages: [{"role": "user", "content": "请描述这张图片。"}, {"role": "assistant", "content": caption}]}
    
    参考 COCO2014Preprocess 的实现方式
    """

    def preprocess(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # 获取图像和描述
        caption = row.get('caption', '')
        
        # 如果 caption 为空，跳过这条数据
        if not caption or not isinstance(caption, str) or len(caption.strip()) == 0:
            return None
        
        # 设置 query 和 response（ResponsePreprocessor 会将其转换为 messages）
        row['query'] = '请描述这张图片。'  # 中文提示词
        row['response'] = caption
        
        # 调用父类方法转换为 messages 格式
        # 注意：image 字段会通过 columns 映射自动转换为 images
        return super().preprocess(row)


# 注册数据集
# 使用 columns 参数将 'image' 字段映射为 'images'
register_dataset(
    DatasetMeta(
        ms_dataset_id='prithivMLmods/Caption3o-LongCap-v4',
        preprocess_func=Caption3oLongCapPreprocessor(columns={'image': 'images'}),
        split=['train'],
        tags=['multi-modal', 'vision', 'caption', 'zh'],
        huge_dataset=True,
    )
)

