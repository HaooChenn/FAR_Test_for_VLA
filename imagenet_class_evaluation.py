import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path
import json

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models.vae import AutoencoderKL
from models import far
import copy
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from torchvision.utils import make_grid
from typing import Optional
from PIL import Image

def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None, format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    """Save generated images in a grid format with proper normalization"""
    images = images * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
    grid = make_grid(images, nrow=nrow, **kwargs)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid

def update_ema(target_params, source_params, rate=0.99):
    """Update exponential moving average parameters"""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

def evaluate_imagenet_classes(model_without_ddp, vae, ema_params, args, epoch, 
                             target_classes, batch_size=16, log_writer=None, 
                             cfg=1.0, use_ema=True):
    """
    Evaluate ImageNet class-conditional generation for specific classes
    
    这个函数现在使用了正确的ImageNet模型推理接口，
    通过分析engine_far.py，我们发现需要使用sample_tokens_nomask或sample_tokens_mask方法
    """
    model_without_ddp.eval()
    
    # 设置每个类别生成的图像数量
    images_per_class = 5  # 每个类别生成5张图片用于评估
    num_steps = len(target_classes)  # 每个类别一个步骤
    
    # 创建详细的时间和生成日志目录
    speed_results_dir = os.path.join(args.output_dir, "speed_results")
    generated_images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(speed_results_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # 如果请求使用EMA参数，则切换到EMA参数
    if use_ema:
        print("切换到EMA参数进行评估...")
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        model_without_ddp.load_state_dict(ema_state_dict)

    # 存储详细的时间统计和质量指标
    detailed_times = []
    total_generation_time = 0
    total_images_generated = 0
    warmup_steps = 1  # 跳过第一步以获得准确的时间统计
    
    print(f"开始评估 {len(target_classes)} 个ImageNet类别...")
    print(f"每个类别生成 {images_per_class} 张图片")
    
    # 遍历每个目标类别
    for step, (class_name, class_id) in enumerate(target_classes.items()):
        print(f"\n步骤 {step+1}/{num_steps}: 生成 {class_name} (类别 {class_id})")
        
        # 创建类别标签张量 - 为批次中的所有图片重复相同的类别ID
        batch_size_current = min(batch_size, images_per_class)
        class_labels = torch.full((batch_size_current,), class_id, dtype=torch.long, device='cuda')
        
        # 记录此类别生成的时间
        torch.cuda.synchronize()
        step_start_time = time.time()
        
        # 使用FAR模型生成图片
        # 关键修复：使用正确的ImageNet推理方法
        device = torch.device("cuda")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # 根据engine_far.py的实现，我们需要使用正确的方法名
                if hasattr(args, 'mask') and args.mask:
                    # 如果启用掩码，使用掩码采样方法
                    sampled_tokens = model_without_ddp.sample_tokens_mask(
                        bsz=batch_size_current, 
                        num_iter=args.num_iter, 
                        cfg=cfg,
                        cfg_schedule=args.cfg_schedule, 
                        labels=class_labels,
                        temperature=args.temperature
                    )
                else:
                    # 标准采样方法（推荐用于评估）
                    sampled_tokens = model_without_ddp.sample_tokens_nomask(
                        bsz=batch_size_current, 
                        num_iter=args.num_iter, 
                        cfg=cfg,
                        cfg_schedule=args.cfg_schedule, 
                        labels=class_labels,
                        temperature=args.temperature
                    )
                
                # 使用VAE解码器将潜在表示转换为图像
                sampled_images = vae.decode(sampled_tokens / 0.2325)

        torch.cuda.synchronize()
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        
        # 记录详细的时间信息（跳过预热步骤）
        if step >= warmup_steps:
            detailed_times.append({
                'step': step + 1,
                'class_name': class_name,
                'class_id': class_id,
                'batch_size': batch_size_current,
                'total_time': step_time,
                'time_per_image': step_time / batch_size_current,
                'images_per_second': batch_size_current / step_time
            })
            total_generation_time += step_time
            total_images_generated += batch_size_current
        
        print(f"为 {class_name} 生成了 {batch_size_current} 张图片，用时 {step_time:.3f}秒 "
              f"(每张图片 {step_time/batch_size_current:.3f}秒, "
              f"{batch_size_current/step_time:.2f} 张图片/秒)")
        
        # 保存此类别生成的图片
        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        
        # 保存单个类别的结果
        class_output_path = os.path.join(generated_images_dir, f"{class_name}_class{class_id}.png")
        save_image(sampled_images, nrow=min(4, batch_size_current), show=False, 
                  path=class_output_path, to_grayscale=False)
        
        # 同时保存该类别生成的第一张图片作为独立文件
        if sampled_images.shape[0] > 0:
            single_image_path = os.path.join(generated_images_dir, f"{class_name}_class{class_id}_sample1.png")
            save_image(sampled_images[0:1], nrow=1, show=False, path=single_image_path, to_grayscale=False)

    # 计算综合统计数据（排除预热步骤）
    if len(detailed_times) > 0:
        avg_time_per_image = total_generation_time / total_images_generated
        avg_images_per_second = total_images_generated / total_generation_time
        
        # 创建综合的评估报告
        evaluation_report = {
            'model_configuration': {
                'model_name': args.model,
                'model_path': args.resume,
                'use_ema': use_ema,
                'timestamp': datetime.datetime.now().isoformat()
            },
            'generation_parameters': {
                'batch_size': batch_size,
                'num_iter': args.num_iter,
                'num_sampling_steps': args.num_sampling_steps,
                'cfg': cfg,
                'cfg_schedule': args.cfg_schedule,
                'temperature': args.temperature,
                'img_size': args.img_size,
                'use_mask': getattr(args, 'mask', False)
            },
            'evaluation_setup': {
                'target_classes': target_classes,
                'images_per_class': images_per_class,
                'total_classes_evaluated': len(target_classes),
                'warmup_steps': warmup_steps
            },
            'performance_metrics': {
                'total_images_generated': total_images_generated,
                'total_generation_time': total_generation_time,
                'average_time_per_image': avg_time_per_image,
                'average_images_per_second': avg_images_per_second,
                'min_time_per_image': min(t['time_per_image'] for t in detailed_times),
                'max_time_per_image': max(t['time_per_image'] for t in detailed_times),
                'std_time_per_image': np.std([t['time_per_image'] for t in detailed_times])
            },
            'detailed_class_results': detailed_times
        }
        
        # 保存综合JSON报告
        report_path = os.path.join(speed_results_dir, f"imagenet_evaluation_report_cfg{cfg}.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        # 保存CSV摘要以便于分析
        csv_path = os.path.join(speed_results_dir, f"imagenet_summary_cfg{cfg}.csv")
        with open(csv_path, 'w') as f:
            f.write("Class_Name,Class_ID,Images_Generated,Time_Per_Image,Images_Per_Second\n")
            for result in detailed_times:
                f.write(f"{result['class_name']},{result['class_id']},{result['batch_size']},"
                       f"{result['time_per_image']:.5f},{result['images_per_second']:.2f}\n")
        
        # 显示综合结果
        print("\n" + "="*60)
        print("IMAGENET 类别条件评估结果")
        print("="*60)
        print(f"模型: {args.model}")
        print(f"评估的类别数: {len(target_classes)}")
        print(f"生成的图片总数: {total_images_generated}")
        print(f"总生成时间: {total_generation_time:.3f} 秒")
        print(f"平均每张图片时间: {avg_time_per_image:.3f} 秒")
        print(f"平均每秒图片数: {avg_images_per_second:.2f}")
        print(f"结果保存至: {report_path}")
        print(f"生成的图片保存至: {generated_images_dir}")
        print("="*60 + "\n")

    # 恢复原始模型参数
    if use_ema:
        print("恢复原始模型参数...")
        model_without_ddp.load_state_dict(model_state_dict)

def get_args_parser():
    """解析ImageNet类别评估的命令行参数"""
    parser = argparse.ArgumentParser('FAR ImageNet类别评估', add_help=False)
    
    # 模型参数
    parser.add_argument('--model', default='far_large', type=str, metavar='MODEL',
                        help='要评估的模型名称')
    
    # VAE参数  
    parser.add_argument('--img_size', default=256, type=int,
                        help='图片输入尺寸')
    parser.add_argument('--vae_path', default="pretrained/vae/kl16.ckpt", type=str,
                        help='VAE检查点路径')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='VAE输出嵌入维度')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='分词器步长，默认使用KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='作为一个补丁分组的token数量')

    # 生成参数
    parser.add_argument('--num_iter', default=10, type=int,
                        help='生成图片的自回归迭代次数')
    parser.add_argument('--cfg', default=3.0, type=float, help="分类器自由指导")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--temperature', default=1.0, type=float, help='采样温度')
    parser.add_argument('--eval_bsz', type=int, default=8, help='生成批次大小')
    parser.add_argument('--mask', action='store_true', help='使用掩码采样增加多样性')

    # 模型架构参数
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='最小掩码比例')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='注意力丢弃率')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='投影丢弃率')
    parser.add_argument('--buffer_size', type=int, default=64)
    parser.add_argument('--class_num', default=1000, type=int)

    # 扩散损失参数
    parser.add_argument('--diffloss_d', type=int, default=3)
    parser.add_argument('--diffloss_w', type=int, default=1024)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)

    # 系统参数
    parser.add_argument('--output_dir', default='./output_dir',
                        help='保存结果的路径')
    parser.add_argument('--device', default='cuda',
                        help='用于评估的设备')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', required=True,
                        help='模型检查点路径')

    # 分布式参数
    parser.add_argument('--world_size', default=1, type=int,
                        help='分布式进程数')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='用于设置分布式训练的URL')

    return parser

def main(args):
    """ImageNet类别条件生成的主评估函数"""
    
    # 如果需要，初始化分布式训练
    misc.init_distributed_mode(args)

    print('评估目录: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("配置:")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # 设置种子以确保可重现性
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # 定义我们要评估的目标ImageNet类别
    # 这些代表了多样化的对象、动物和场景集合
    target_classes = {
        'golden_retriever': 207,       # 动物 - 狗品种
        'tabby_cat': 281,             # 动物 - 猫品种  
        'red_fox': 277,               # 动物 - 野生动物
        'monarch_butterfly': 323,      # 动物 - 昆虫
        'daisy': 985,                 # 植物 - 花朵
        'rose': 973,                  # 植物 - 花朵
        'lighthouse': 437,            # 建筑 - 建筑物
        'castle': 483,                # 建筑 - 历史建筑
        'cottage': 500,               # 建筑 - 房屋
        'sports_car': 817,            # 车辆 - 汽车
        'steam_locomotive': 820,       # 车辆 - 火车
        'sailboat': 554,              # 车辆 - 船只
        'aircraft_carrier': 403,       # 车辆 - 舰船
        'mountain_bike': 671,         # 车辆 - 自行车
        'pizza': 963,                 # 食物 - 预制食品
        'strawberry': 949,            # 食物 - 水果
        'coffee_mug': 504,            # 物品 - 容器
        'violin': 889,                # 物品 - 乐器
        'backpack': 414,              # 物品 - 包
        'umbrella': 879               # 物品 - 工具
    }

    print(f"\n评估 {len(target_classes)} 个精心选择的ImageNet类别:")
    for class_name, class_id in target_classes.items():
        print(f"  {class_name}: {class_id}")

    # 初始化VAE模型用于解码潜在表示
    print("\n初始化VAE...")
    vae = AutoencoderKL(
        embed_dim=args.vae_embed_dim, 
        ch_mult=(1, 1, 2, 2, 4), 
        ckpt_path=args.vae_path
    ).cuda().eval()
    
    # 冻结VAE参数
    for param in vae.parameters():
        param.requires_grad = False
    
    # 使用指定架构初始化FAR模型
    print(f"初始化FAR模型: {args.model}")
    model = far.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask=args.mask,  # 评估期间的掩码设置
        mask_ratio_min=args.mask_ratio_min,
        label_drop_prob=args.label_drop_prob,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=args.num_sampling_steps,
        diffusion_batch_mul=args.diffusion_batch_mul,
    )

    model.to(device)
    model_without_ddp = model
    
    print("模型 = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("可训练参数数量: {:.2f}M".format(n_params / 1e6))

    # 如果需要，处理分布式训练
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # 加载模型检查点
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        print(f"从 {args.resume} 加载检查点")
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
        
        # 加载模型状态字典
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        
        # 如果可用，加载EMA参数
        if 'model_ema' in checkpoint:
            ema_state_dict = checkpoint['model_ema']
            ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
            print("EMA参数加载成功")
        else:
            ema_params = copy.deepcopy(model_params)
            print("未找到EMA参数，使用常规参数")
            
        print("检查点加载成功")
        del checkpoint
    else:
        raise FileNotFoundError(f"在 {args.resume} 找不到检查点")

    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 运行评估
    print("\n开始ImageNet类别条件评估...")
    torch.cuda.empty_cache()
    
    evaluate_imagenet_classes(
        model_without_ddp=model_without_ddp,
        vae=vae,
        ema_params=ema_params,
        args=args,
        epoch=0,  # 对于评估不相关
        target_classes=target_classes,
        batch_size=args.eval_bsz,
        log_writer=None,
        cfg=args.cfg,
        use_ema=True
    )
    
    print("评估成功完成！")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
