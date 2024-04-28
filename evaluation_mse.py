""" Evaluate accuracy on ImageNet dataset of PipeEdge """
import os
import argparse
import time
import torch
from typing import List
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from transformers import DeiTFeatureExtractor, ViTFeatureExtractor
from runtime import forward_hook_quant_encode, forward_pre_hook_quant_decode
from utils.data import ViTFeatureExtractorTransforms
import model_cfg
from evaluation_tools.evaluation_quant_test import *

def _make_shard(model_name, model_file, stage_layers, stage):
    shard = model_cfg.module_shard_factory(model_name, model_file, stage_layers[stage],
                                           stage_layers[stage], stage)
    # shard.register_buffer('quant_bits', q_bits)
    shard.eval()
    return shard

def _forward_model(input_tensor, model_shards, batch_idx):
    num_shards = len(model_shards)
    print(f"num_shards: {num_shards}")
    temp_tensor = input_tensor

    for idx in range(num_shards):
        print(f"layer_id: {idx + 1}")
        shard = model_shards[idx]
        temp_tensor = shard(temp_tensor)
        print(f"the type of temp_tensor: {type(temp_tensor)}")

        if isinstance(temp_tensor, torch.Tensor):
            temp_tensor_st = temp_tensor
        elif isinstance(temp_tensor, tuple) and isinstance(temp_tensor[0], torch.Tensor):
            temp_tensor_st = temp_tensor[0]
        print(f"the shape of output: {temp_tensor_st.shape}")
        
        numpy_array = temp_tensor_st.cpu().numpy()
        file_name = f"result/original_data/batch_{batch_idx + 1}/layer_{idx + 1}_x.npy"
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.save(file_name, numpy_array)

    return temp_tensor

def evaluation(args):
    # localize parameters
    dataset_path = args.dataset_root
    dataset_split = args.dataset_split
    batch_size = args.batch_size
    ubatch_size = args.ubatch_size
    num_workers = args.num_workers
    partition = args.partition
    quant = args.quant
    output_dir = args.output_dir
    model_name = args.model_name
    model_file = args.model_file
    num_stop_batch = args.stop_at_batch
    is_clamp = True

    # load dataset
    if model_name in ['facebook/deit-base-distilled-patch16-224',
                        'facebook/deit-small-distilled-patch16-224',
                        'facebook/deit-tiny-distilled-patch16-224']:
        feature_extractor = DeiTFeatureExtractor.from_pretrained(model_name)
    else:
        feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    val_transform = ViTFeatureExtractorTransforms(feature_extractor)
    
    val_dataset = ImageFolder(os.path.join(dataset_path, dataset_split),
                            transform = val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        num_workers = num_workers,
        shuffle=True,
        pin_memory=True
    )

    # model config
    parts = [int(i) for i in range(1, int(partition) + 1)]
    # assert len(parts) % 2 == 0
    num_shards = len(parts)
    stage_layers = [i for i in range(1, len(parts) + 1)]

    # model construct
    model_shards = []
    # q_bits = []
    for stage in range(num_shards):
        model_shards.append(_make_shard(model_name, model_file, stage_layers, stage))
        
     # run inference
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(val_loader):
            if batch_idx == num_stop_batch and num_stop_batch:
                break
            if batch_idx > 9:
                break
            output = _forward_model(input, model_shards, batch_idx)
    end_time = time.time()
    print(f"total time = {end_time - start_time}")

if __name__ == "__main__":
    """Main function."""
    parser = argparse.ArgumentParser(description="Pipeline Parallelism Evaluation on Single GPU",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Eval configs
    parser.add_argument("-q", "--quant", type=str,
                        help="comma-delimited list of quantization bits to use after each stage")
    parser.add_argument("-pt", "--partition", type=str, default= '48',
                        help="comma-delimited list of start/end layer pairs, e.g.: '1,24,25,48'; "
                             "single-node default: all layers in the model")
    parser.add_argument("-o", "--output-dir", type=str, default="result/")
    parser.add_argument("-st", "--stop-at-batch", type=int, default=None, help="the # of batch to stop evaluation")
    
    # Device options
    parser.add_argument("-d", "--device", type=str, default='cuda',
                        help="compute device type to use, with optional ordinal, "
                             "e.g.: 'cpu', 'cuda', 'cuda:1'")
    parser.add_argument("-n", "--num-workers", default=16, type=int,
                        help="the number of worker threads for the dataloder")
    # Model options
    parser.add_argument("-m", "--model-name", type=str, default="google/vit-base-patch16-224",
                        choices=model_cfg.get_model_names(),
                        help="the neural network model for loading")
    parser.add_argument("-M", "--model-file", type=str,
                        help="the model file, if not in working directory")
    # Dataset options
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
    parser.add_argument("-u", "--ubatch-size", default=8, type=int, help="microbatch size")

    dset = parser.add_argument_group('Dataset arguments')
    dset.add_argument("--dataset-name", type=str, default='ImageNet', choices=['CoLA', 'ImageNet'],
                      help="dataset to use")
    dset.add_argument("--dataset-root", type=str, default= "ImageNet/",
                      help="dataset root directory (e.g., for 'ImageNet', must contain "
                           "'ILSVRC2012_devkit_t12.tar.gz' and at least one of: "
                           "'ILSVRC2012_img_train.tar', 'ILSVRC2012_img_val.tar'")
    dset.add_argument("--dataset-split", default='ILSVRC2012_img_val/', type=str,
                      help="dataset split (depends on dataset), e.g.: train, val, validation, test")
    dset.add_argument("--dataset-indices-file", default=None, type=str,
                      help="PyTorch or NumPy file with precomputed dataset index sequence")
    dset.add_argument("--dataset-shuffle", type=bool, nargs='?', const=True, default=False,
                      help="dataset shuffle")
    args = parser.parse_args()

    evaluation(args)