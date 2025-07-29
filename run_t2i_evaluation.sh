自动修复 sample() 方法参数错误
==========================================
正在自动修复代码中的参数错误...
已创建 far_t2i.py 的备份文件
✅ 已修复 far_t2i.py 中的 sample() 方法调用
设置目录结构和评估文件...
文件创建成功！

开始T2I专用评估...
配置总结:
- GPU配置: 1,2,3,4
- 批次大小: 32
- 模型路径: pretrained_models/far/far_t2i
- 架构配置: diffloss_d=3, diffloss_w=1024

==========================================
开始T2I模型评估
==========================================

运行 1: simple prompts, 50 steps

==========================================
运行T2I评估
采样步数: 50
提示词类型: simple
架构配置: diffloss_d=3, diffloss_w=1024
==========================================
开始时间: 2025-07-29 17:39:01
[2025-07-29 17:39:02,527] torch.distributed.run: [WARNING] 
[2025-07-29 17:39:02,527] torch.distributed.run: [WARNING] *****************************************
[2025-07-29 17:39:02,527] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2025-07-29 17:39:02,527] torch.distributed.run: [WARNING] *****************************************
| distributed init (rank 1): env://, gpu 1
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
| distributed init (rank 2): env://, gpu 2
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
| distributed init (rank 3): env://, gpu 3
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
| distributed init (rank 0): env://, gpu 0
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
dsw95381-5f7b687f66-w5sjz:87930:87930 [0] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87930:87930 [0] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:87930:87930 [0] NCCL INFO cudaDriverVersion 12040
NCCL version 2.19.3+cuda11.8
dsw95381-5f7b687f66-w5sjz:87931:87931 [1] NCCL INFO cudaDriverVersion 12040
dsw95381-5f7b687f66-w5sjz:87933:87933 [3] NCCL INFO cudaDriverVersion 12040
dsw95381-5f7b687f66-w5sjz:87932:87932 [2] NCCL INFO cudaDriverVersion 12040
dsw95381-5f7b687f66-w5sjz:87932:87932 [2] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87931:87931 [1] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87933:87933 [3] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87932:87932 [2] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:87931:87931 [1] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:87933:87933 [3] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO comm 0x562444655b00 rank 1 nranks 4 cudaDev 1 nvmlDev 2 busId 4a000 commId 0xcba2cd4cb01d45ee - Init START
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO comm 0x557d47c553f0 rank 0 nranks 4 cudaDev 0 nvmlDev 1 busId 1e000 commId 0xcba2cd4cb01d45ee - Init START
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO comm 0x565414857e30 rank 3 nranks 4 cudaDev 3 nvmlDev 4 busId 89000 commId 0xcba2cd4cb01d45ee - Init START
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO comm 0x5567c9e55dc0 rank 2 nranks 4 cudaDev 2 nvmlDev 3 busId 4f000 commId 0xcba2cd4cb01d45ee - Init START
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Setting affinity for GPU 2 to ffffffff,00000000,ffffffff
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Setting affinity for GPU 4 to ffffffff,00000000,ffffffff,00000000
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Setting affinity for GPU 1 to ffffffff,00000000,ffffffff
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Channel 00/02 :    0   1   2   3
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Channel 01/02 :    0   1   2   3
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Channel 00/0 : 1[2] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Channel 00/0 : 2[3] -> 3[4] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Channel 01/0 : 1[2] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Channel 01/0 : 2[3] -> 3[4] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Channel 00/0 : 0[1] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Channel 00/0 : 3[4] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Channel 01/0 : 3[4] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Channel 01/0 : 0[1] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Channel 00/0 : 1[2] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Channel 01/0 : 1[2] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Channel 00/0 : 3[4] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Channel 01/0 : 3[4] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Channel 00/0 : 2[3] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Channel 01/0 : 2[3] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:87932:88589 [2] NCCL INFO comm 0x5567c9e55dc0 rank 2 nranks 4 cudaDev 2 nvmlDev 3 busId 4f000 commId 0xcba2cd4cb01d45ee - Init COMPLETE
dsw95381-5f7b687f66-w5sjz:87930:88590 [0] NCCL INFO comm 0x557d47c553f0 rank 0 nranks 4 cudaDev 0 nvmlDev 1 busId 1e000 commId 0xcba2cd4cb01d45ee - Init COMPLETE
dsw95381-5f7b687f66-w5sjz:87931:88592 [1] NCCL INFO comm 0x562444655b00 rank 1 nranks 4 cudaDev 1 nvmlDev 2 busId 4a000 commId 0xcba2cd4cb01d45ee - Init COMPLETE
dsw95381-5f7b687f66-w5sjz:87933:88591 [3] NCCL INFO comm 0x565414857e30 rank 3 nranks 4 cudaDev 3 nvmlDev 4 busId 89000 commId 0xcba2cd4cb01d45ee - Init COMPLETE
[17:39:41.365172] job dir: /mnt/nas-data-1/chenhao/FAR
[17:39:41.365351] Namespace(attn_dropout=0.1,
batch_size=16,
blr=0.0001,
buffer_size=0,
cached_path='',
cfg=3.0,
cfg_schedule='linear',
class_num=1000,
data_path='./prompts',
device='cuda',
diffloss_d=3,
diffloss_w=1024,
diffusion_batch_mul=1,
dist_backend='nccl',
dist_on_itp=False,
dist_url='env://',
distributed=True,
ema=False,
ema_rate=0.999,
epochs=400,
eval_bsz=32,
eval_freq=2,
evaluate=True,
gpu=0,
grad_clip=1.0,
img_size=256,
label_drop_prob=0.1,
local_rank=-1,
log_dir='./t2i_evaluation_results/t2i_steps50_simple',
loss_weight=False,
lr=None,
lr_schedule='constant',
mask_ratio_min=0.7,
min_lr=0.0,
model='far_t2i',
num_gpus_permachine=2,
num_iter=10,
num_machine=1,
num_sampling_steps='50',
num_workers=5,
online_eval=False,
output_dir='./t2i_evaluation_results/t2i_steps50_simple',
patch_size=1,
pin_mem=True,
proj_dropout=0.1,
rank=0,
resume='pretrained_models/far/far_t2i',
save_last_freq=5,
seed=1,
speed_test=True,
speed_test_steps=5,
start_epoch=0,
temperature=1.0,
text_model_path='pretrained/Qwen2-VL-1.5B-Instruct',
use_cached=False,
use_unictrol=False,
vae_embed_dim=16,
vae_path='pretrained/vae/kl16.ckpt',
vae_stride=16,
warmup_epochs=100,
weight_decay=0.02,
world_size=4)
[17:39:41.892647] Working with z of shape (1, 16, 16, 16) = 4096 dimensions.
[17:39:42.470532] Loading pre-trained KL-VAE
[17:39:42.470597] Missing keys:
[17:39:42.470609] []
[17:39:42.470620] Unexpected keys:
[17:39:42.470630] []
[17:39:42.470640] Restored from pretrained/vae/kl16.ckpt
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.78s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.10s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.12s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.11s/it]
[17:40:00.777694] Model = FAR_T2I(
  (context_embed): Linear(in_features=1536, out_features=1024, bias=True)
  (z_proj): Linear(in_features=16, out_features=1024, bias=True)
  (z_proj_ln): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (encoder_blocks): ModuleList(
    (0-15): 16 x BasicTransformerBlock(
      (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn1): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn2): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (ff): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=4096, out_features=1024, bias=True)
        )
      )
    )
  )
  (encoder_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (decoder_embed): Linear(in_features=1024, out_features=1024, bias=True)
  (decoder_blocks): ModuleList(
    (0-15): 16 x BasicTransformerBlock(
      (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn1): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn2): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (ff): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=4096, out_features=1024, bias=True)
        )
      )
    )
  )
  (decoder_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (diffloss): DiffLoss(
    (net): SimpleMLPAdaLN(
      (time_embed): TimestepEmbedder(
        (mlp): Sequential(
          (0): Linear(in_features=256, out_features=1024, bias=True)
          (1): SiLU()
          (2): Linear(in_features=1024, out_features=1024, bias=True)
        )
      )
      (cond_embed): Linear(in_features=1024, out_features=1024, bias=True)
      (index_cond_embed): Linear(in_features=1, out_features=1024, bias=True)
      (input_proj): Linear(in_features=16, out_features=1024, bias=True)
      (res_blocks): ModuleList(
        (0-2): 3 x ResBlock(
          (in_ln): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (mlp): Sequential(
            (0): Linear(in_features=1024, out_features=1024, bias=True)
            (1): SiLU()
            (2): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (adaLN_modulation): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1024, out_features=3072, bias=True)
          )
        )
      )
      (final_layer): FinalLayer(
        (norm_final): LayerNorm((1024,), eps=1e-06, elementwise_affine=False)
        (linear): Linear(in_features=1024, out_features=32, bias=True)
        (adaLN_modulation): Sequential(
          (0): SiLU()
          (1): Linear(in_features=1024, out_features=2048, bias=True)
        )
      )
    )
  )
)
[17:40:00.780127] Number of trainable parameters: 561.30256M
[17:40:03.347899] base lr: 1.00e-04
[17:40:03.347945] actual lr: 2.50e-05
[17:40:03.347958] effective batch size: 64
[rank3]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[rank0]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[rank1]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[rank2]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[17:40:04.156755] AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.95)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 2.5e-05
    maximize: False
    weight_decay: 0.0

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.95)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 2.5e-05
    maximize: False
    weight_decay: 0.02
)
[17:40:12.178672] Resume checkpoint pretrained_models/far/far_t2i
[17:40:14.931742] With optim & sched!
Traceback (most recent call last):
  File "main_far_t2i.py", line 351, in <module>
    main(args)
  File "main_far_t2i.py", line 290, in main
    evaluate(text_tokenizer, text_model, llm_system_prompt, model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer, cfg=args.cfg, use_ema=True)
  File "/mnt/nas-data-1/chenhao/FAR/engine_far_t2i.py", line 187, in evaluate
    sampled_images = model_without_ddp.sample_tokens(vae, bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
  File "/mnt/nas-data-1/chenhao/FAR/models/far_t2i.py", line 357, in sample_tokens
    sampled_token_latent = self.diffloss.sample(z, temperature_iter, cfg_iter, index)     # torch.Size([512, 16])
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 40, in sample
    sampled_token_latent = self.gen_diffusion.p_sample_loop(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 459, in p_sample_loop
[17:40:15.651188] Switch to ema
[17:40:15.668672] Generation step 1/5
    for sample in self.p_sample_loop_progressive(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 510, in p_sample_loop_progressive
    out = self.p_sample(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 407, in p_sample
    out = self.p_mean_variance(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 92, in p_mean_variance
    return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 280, in p_mean_variance
    model_output = model(x, t, **model_kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 129, in __call__
    return self.model(x, new_ts, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 235, in forward_with_cfg
    model_out = self.forward(combined, t, c, index)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 222, in forward
    index = self.index_cond_embed(index)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Half
Traceback (most recent call last):
  File "main_far_t2i.py", line 351, in <module>
    main(args)
  File "main_far_t2i.py", line 290, in main
    evaluate(text_tokenizer, text_model, llm_system_prompt, model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer, cfg=args.cfg, use_ema=True)
  File "/mnt/nas-data-1/chenhao/FAR/engine_far_t2i.py", line 187, in evaluate
    sampled_images = model_without_ddp.sample_tokens(vae, bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
  File "/mnt/nas-data-1/chenhao/FAR/models/far_t2i.py", line 357, in sample_tokens
    sampled_token_latent = self.diffloss.sample(z, temperature_iter, cfg_iter, index)     # torch.Size([512, 16])
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 40, in sample
    sampled_token_latent = self.gen_diffusion.p_sample_loop(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 459, in p_sample_loop
    for sample in self.p_sample_loop_progressive(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 510, in p_sample_loop_progressive
    out = self.p_sample(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 407, in p_sample
    out = self.p_mean_variance(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 92, in p_mean_variance
    return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 280, in p_mean_variance
    model_output = model(x, t, **model_kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 129, in __call__
    return self.model(x, new_ts, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 235, in forward_with_cfg
    model_out = self.forward(combined, t, c, index)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 222, in forward
    index = self.index_cond_embed(index)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Half
dsw95381-5f7b687f66-w5sjz:87930:88713 [0] NCCL INFO [Service thread] Connection closed by localRank 1
dsw95381-5f7b687f66-w5sjz:87932:88712 [2] NCCL INFO [Service thread] Connection closed by localRank 1
Traceback (most recent call last):
  File "main_far_t2i.py", line 351, in <module>
Traceback (most recent call last):
  File "main_far_t2i.py", line 351, in <module>
dsw95381-5f7b687f66-w5sjz:87933:88714 [3] NCCL INFO [Service thread] Connection closed by localRank 2
    main(args)
  File "main_far_t2i.py", line 290, in main
    main(args)
  File "main_far_t2i.py", line 290, in main
    evaluate(text_tokenizer, text_model, llm_system_prompt, model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer, cfg=args.cfg, use_ema=True)
  File "/mnt/nas-data-1/chenhao/FAR/engine_far_t2i.py", line 187, in evaluate
    evaluate(text_tokenizer, text_model, llm_system_prompt, model_without_ddp, vae, ema_params, args, 0, batch_size=args.eval_bsz, log_writer=log_writer, cfg=args.cfg, use_ema=True)
  File "/mnt/nas-data-1/chenhao/FAR/engine_far_t2i.py", line 187, in evaluate
    sampled_images = model_without_ddp.sample_tokens(vae, bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
  File "/mnt/nas-data-1/chenhao/FAR/models/far_t2i.py", line 357, in sample_tokens
    sampled_images = model_without_ddp.sample_tokens(vae, bsz=batch_size, num_iter=args.num_iter, cfg=cfg,
  File "/mnt/nas-data-1/chenhao/FAR/models/far_t2i.py", line 357, in sample_tokens
    sampled_token_latent = self.diffloss.sample(z, temperature_iter, cfg_iter, index)     # torch.Size([512, 16])
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 40, in sample
    sampled_token_latent = self.diffloss.sample(z, temperature_iter, cfg_iter, index)     # torch.Size([512, 16])
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 40, in sample
    sampled_token_latent = self.gen_diffusion.p_sample_loop(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 459, in p_sample_loop
    sampled_token_latent = self.gen_diffusion.p_sample_loop(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 459, in p_sample_loop
    for sample in self.p_sample_loop_progressive(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 510, in p_sample_loop_progressive
    for sample in self.p_sample_loop_progressive(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 510, in p_sample_loop_progressive
    out = self.p_sample(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 407, in p_sample
    out = self.p_sample(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 407, in p_sample
    out = self.p_mean_variance(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 92, in p_mean_variance
    out = self.p_mean_variance(
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 92, in p_mean_variance
    return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 280, in p_mean_variance
    return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/gaussian_diffusion.py", line 280, in p_mean_variance
    model_output = model(x, t, **model_kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 129, in __call__
    model_output = model(x, t, **model_kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/diffusion/respace.py", line 129, in __call__
    return self.model(x, new_ts, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 235, in forward_with_cfg
    return self.model(x, new_ts, **kwargs)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 235, in forward_with_cfg
    model_out = self.forward(combined, t, c, index)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 222, in forward
    model_out = self.forward(combined, t, c, index)
  File "/mnt/nas-data-1/chenhao/FAR/models/diffloss.py", line 222, in forward
    index = self.index_cond_embed(index)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Half
    index = self.index_cond_embed(index)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/nn/modules/linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
RuntimeError: mat1 and mat2 must have the same dtype, but got Long and Half
[2025-07-29 17:40:22,639] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: 1) local_rank: 0 (pid: 87930) of binary: /home/pai/envs/far/bin/python
Traceback (most recent call last):
  File "/home/pai/envs/far/bin/torchrun", line 33, in <module>
    sys.exit(load_entry_point('torch==2.2.2', 'console_scripts', 'torchrun')())
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/distributed/run.py", line 812, in main
    run(args)
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/distributed/run.py", line 803, in run
    elastic_launch(
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 135, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/pai/envs/far/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 268, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
============================================================
main_far_t2i.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2025-07-29_17:40:22
  host      : dsw95381-5f7b687f66-w5sjz
  rank      : 1 (local_rank: 1)
  exitcode  : 1 (pid: 87931)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[2]:
  time      : 2025-07-29_17:40:22
  host      : dsw95381-5f7b687f66-w5sjz
  rank      : 2 (local_rank: 2)
  exitcode  : 1 (pid: 87932)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
[3]:
  time      : 2025-07-29_17:40:22
  host      : dsw95381-5f7b687f66-w5sjz
  rank      : 3 (local_rank: 3)
  exitcode  : 1 (pid: 87933)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
------------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-07-29_17:40:22
  host      : dsw95381-5f7b687f66-w5sjz
  rank      : 0 (local_rank: 0)
  exitcode  : 1 (pid: 87930)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
============================================================
结束时间: 2025-07-29 17:40:22
错误：T2I评估失败，退出代码: 1
请查看上方的错误信息进行调试
❌ 运行 1 失败

运行 2: medium prompts, 50 steps

==========================================
运行T2I评估
采样步数: 50
提示词类型: medium
架构配置: diffloss_d=3, diffloss_w=1024
==========================================
开始时间: 2025-07-29 17:40:23
[2025-07-29 17:40:24,708] torch.distributed.run: [WARNING] 
[2025-07-29 17:40:24,708] torch.distributed.run: [WARNING] *****************************************
[2025-07-29 17:40:24,708] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2025-07-29 17:40:24,708] torch.distributed.run: [WARNING] *****************************************
| distributed init (rank 1): env://, gpu 1
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
| distributed init (rank 3): env://, gpu 3
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
| distributed init (rank 0): env://, gpu 0
| distributed init (rank 2): env://, gpu 2
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
[W Utils.hpp:133] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarInt)
dsw95381-5f7b687f66-w5sjz:90170:90170 [0] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90170:90170 [0] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:90170:90170 [0] NCCL INFO cudaDriverVersion 12040
NCCL version 2.19.3+cuda11.8
dsw95381-5f7b687f66-w5sjz:90171:90171 [1] NCCL INFO cudaDriverVersion 12040
dsw95381-5f7b687f66-w5sjz:90171:90171 [1] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90171:90171 [1] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:90173:90173 [3] NCCL INFO cudaDriverVersion 12040
dsw95381-5f7b687f66-w5sjz:90173:90173 [3] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90173:90173 [3] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:90172:90172 [2] NCCL INFO cudaDriverVersion 12040
dsw95381-5f7b687f66-w5sjz:90172:90172 [2] NCCL INFO Bootstrap : Using eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90172:90172 [2] NCCL INFO NET/Plugin : dlerror=libnccl-net.so: cannot open shared object file: No such file or directory No plugin found (libnccl-net.so), using internal implementation
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO NET/IB : Using [0]mlx5_2:1/RoCE [1]mlx5_3:1/RoCE [2]mlx5_6:1/RoCE [3]mlx5_9:1/RoCE [RO]; OOB eth0:33.104.62.6<0>
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Using non-device net plugin version 0
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Using network IB
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO comm 0x55de0f055830 rank 2 nranks 4 cudaDev 2 nvmlDev 3 busId 4f000 commId 0xbd25713708758e27 - Init START
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO comm 0x55e8886560b0 rank 3 nranks 4 cudaDev 3 nvmlDev 4 busId 89000 commId 0xbd25713708758e27 - Init START
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO comm 0x560d866570c0 rank 1 nranks 4 cudaDev 1 nvmlDev 2 busId 4a000 commId 0xbd25713708758e27 - Init START
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO comm 0x557d9f6581c0 rank 0 nranks 4 cudaDev 0 nvmlDev 1 busId 1e000 commId 0xbd25713708758e27 - Init START
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Setting affinity for GPU 3 to ffffffff,00000000,ffffffff
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Setting affinity for GPU 1 to ffffffff,00000000,ffffffff
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Setting affinity for GPU 4 to ffffffff,00000000,ffffffff,00000000
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Setting affinity for GPU 2 to ffffffff,00000000,ffffffff
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Channel 00/02 :    0   1   2   3
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Channel 01/02 :    0   1   2   3
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO NCCL_MAX_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO NCCL_MIN_NCHANNELS set by environment to 2.
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO P2P Chunksize set to 524288
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Channel 00/0 : 0[1] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Channel 00/0 : 2[3] -> 3[4] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Channel 01/0 : 0[1] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Channel 01/0 : 2[3] -> 3[4] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Channel 00/0 : 3[4] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Channel 01/0 : 3[4] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Channel 00/0 : 1[2] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Channel 01/0 : 1[2] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Channel 00/0 : 2[3] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Channel 01/0 : 2[3] -> 1[2] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Connected all rings
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Channel 00/0 : 3[4] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Channel 01/0 : 3[4] -> 2[3] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Channel 00/0 : 1[2] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Channel 01/0 : 1[2] -> 0[1] via P2P/CUMEM/read
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO Connected all trees
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO threadThresholds 8/8/64 | 32/8/64 | 512 | 512
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO 2 coll channels, 0 nvls channels, 2 p2p channels, 2 p2p channels per peer
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO NCCL_LAUNCH_MODE set by environment to PARALLEL
dsw95381-5f7b687f66-w5sjz:90173:90482 [3] NCCL INFO comm 0x55e8886560b0 rank 3 nranks 4 cudaDev 3 nvmlDev 4 busId 89000 commId 0xbd25713708758e27 - Init COMPLETE
dsw95381-5f7b687f66-w5sjz:90170:90480 [0] NCCL INFO comm 0x557d9f6581c0 rank 0 nranks 4 cudaDev 0 nvmlDev 1 busId 1e000 commId 0xbd25713708758e27 - Init COMPLETE
dsw95381-5f7b687f66-w5sjz:90171:90481 [1] NCCL INFO comm 0x560d866570c0 rank 1 nranks 4 cudaDev 1 nvmlDev 2 busId 4a000 commId 0xbd25713708758e27 - Init COMPLETE
dsw95381-5f7b687f66-w5sjz:90172:90483 [2] NCCL INFO comm 0x55de0f055830 rank 2 nranks 4 cudaDev 2 nvmlDev 3 busId 4f000 commId 0xbd25713708758e27 - Init COMPLETE
[17:40:48.764720] job dir: /mnt/nas-data-1/chenhao/FAR
[17:40:48.764900] Namespace(attn_dropout=0.1,
batch_size=16,
blr=0.0001,
buffer_size=0,
cached_path='',
cfg=3.0,
cfg_schedule='linear',
class_num=1000,
data_path='./prompts',
device='cuda',
diffloss_d=3,
diffloss_w=1024,
diffusion_batch_mul=1,
dist_backend='nccl',
dist_on_itp=False,
dist_url='env://',
distributed=True,
ema=False,
ema_rate=0.999,
epochs=400,
eval_bsz=32,
eval_freq=2,
evaluate=True,
gpu=0,
grad_clip=1.0,
img_size=256,
label_drop_prob=0.1,
local_rank=-1,
log_dir='./t2i_evaluation_results/t2i_steps50_medium',
loss_weight=False,
lr=None,
lr_schedule='constant',
mask_ratio_min=0.7,
min_lr=0.0,
model='far_t2i',
num_gpus_permachine=2,
num_iter=10,
num_machine=1,
num_sampling_steps='50',
num_workers=5,
online_eval=False,
output_dir='./t2i_evaluation_results/t2i_steps50_medium',
patch_size=1,
pin_mem=True,
proj_dropout=0.1,
rank=0,
resume='pretrained_models/far/far_t2i',
save_last_freq=5,
seed=1,
speed_test=True,
speed_test_steps=5,
start_epoch=0,
temperature=1.0,
text_model_path='pretrained/Qwen2-VL-1.5B-Instruct',
use_cached=False,
use_unictrol=False,
vae_embed_dim=16,
vae_path='pretrained/vae/kl16.ckpt',
vae_stride=16,
warmup_epochs=100,
weight_decay=0.02,
world_size=4)
[17:40:49.154771] Working with z of shape (1, 16, 16, 16) = 4096 dimensions.
[17:40:49.771710] Loading pre-trained KL-VAE
[17:40:49.771782] Missing keys:
[17:40:49.771795] []
[17:40:49.771805] Unexpected keys:
[17:40:49.771816] []
[17:40:49.771828] Restored from pretrained/vae/kl16.ckpt
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.96s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:03<00:00,  1.96s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.02s/it]
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:04<00:00,  2.02s/it]
[17:41:04.127682] Model = FAR_T2I(
  (context_embed): Linear(in_features=1536, out_features=1024, bias=True)
  (z_proj): Linear(in_features=16, out_features=1024, bias=True)
  (z_proj_ln): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (encoder_blocks): ModuleList(
    (0-15): 16 x BasicTransformerBlock(
      (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn1): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn2): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (ff): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=4096, out_features=1024, bias=True)
        )
      )
    )
  )
  (encoder_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (decoder_embed): Linear(in_features=1024, out_features=1024, bias=True)
  (decoder_blocks): ModuleList(
    (0-15): 16 x BasicTransformerBlock(
      (norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn1): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (attn2): Attention(
        (to_q): Linear(in_features=1024, out_features=1024, bias=True)
        (to_k): Linear(in_features=1024, out_features=1024, bias=True)
        (to_v): Linear(in_features=1024, out_features=1024, bias=True)
        (to_out): ModuleList(
          (0): Linear(in_features=1024, out_features=1024, bias=True)
          (1): Dropout(p=0.0, inplace=False)
        )
      )
      (norm3): LayerNorm((1024,), eps=1e-05, elementwise_affine=False)
      (ff): FeedForward(
        (net): ModuleList(
          (0): GELU(
            (proj): Linear(in_features=1024, out_features=4096, bias=True)
          )
          (1): Dropout(p=0.0, inplace=False)
          (2): Linear(in_features=4096, out_features=1024, bias=True)
        )
      )
    )
  )
  (decoder_norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
  (diffloss): DiffLoss(
    (net): SimpleMLPAdaLN(
      (time_embed): TimestepEmbedder(
        (mlp): Sequential(
          (0): Linear(in_features=256, out_features=1024, bias=True)
          (1): SiLU()
          (2): Linear(in_features=1024, out_features=1024, bias=True)
        )
      )
      (cond_embed): Linear(in_features=1024, out_features=1024, bias=True)
      (index_cond_embed): Linear(in_features=1, out_features=1024, bias=True)
      (input_proj): Linear(in_features=16, out_features=1024, bias=True)
      (res_blocks): ModuleList(
        (0-2): 3 x ResBlock(
          (in_ln): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (mlp): Sequential(
            (0): Linear(in_features=1024, out_features=1024, bias=True)
            (1): SiLU()
            (2): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (adaLN_modulation): Sequential(
            (0): SiLU()
            (1): Linear(in_features=1024, out_features=3072, bias=True)
          )
        )
      )
      (final_layer): FinalLayer(
        (norm_final): LayerNorm((1024,), eps=1e-06, elementwise_affine=False)
        (linear): Linear(in_features=1024, out_features=32, bias=True)
        (adaLN_modulation): Sequential(
          (0): SiLU()
          (1): Linear(in_features=1024, out_features=2048, bias=True)
        )
      )
    )
  )
)
[17:41:04.129630] Number of trainable parameters: 561.30256M
[17:41:05.318686] base lr: 1.00e-04
[17:41:05.318761] actual lr: 2.50e-05
[17:41:05.318774] effective batch size: 64
[rank1]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[rank2]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[rank0]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[rank3]:[W Utils.hpp:106] Warning: Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated; use TORCH_NCCL_ASYNC_ERROR_HANDLING instead (function getCvarString)
[17:41:05.620813] AdamW (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.95)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 2.5e-05
    maximize: False
    weight_decay: 0.0

Parameter Group 1
    amsgrad: False
    betas: (0.9, 0.95)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 2.5e-05
    maximize: False
    weight_decay: 0.02
)
