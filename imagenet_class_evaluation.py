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
    
    This function generates images for predefined ImageNet classes and measures
    generation speed and quality. Unlike the full ImageNet evaluation, this focuses
    on a curated set of classes that represent different categories of objects.
    """
    model_without_ddp.eval()
    
    # Define the number of images to generate per class
    images_per_class = 5  # Generate 5 images per class for evaluation
    num_steps = len(target_classes)  # One step per class
    
    # Create detailed timing and generation logs
    speed_results_dir = os.path.join(args.output_dir, "speed_results")
    generated_images_dir = os.path.join(args.output_dir, "generated_images")
    os.makedirs(speed_results_dir, exist_ok=True)
    os.makedirs(generated_images_dir, exist_ok=True)
    
    # Switch to EMA parameters if requested
    if use_ema:
        print("Switching to EMA parameters for evaluation...")
        model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
        for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
            assert name in ema_state_dict
            ema_state_dict[name] = ema_params[i]
        model_without_ddp.load_state_dict(ema_state_dict)

    # Storage for detailed timing and quality metrics
    detailed_times = []
    total_generation_time = 0
    total_images_generated = 0
    warmup_steps = 1  # Skip first step for timing accuracy
    
    print(f"Starting evaluation of {len(target_classes)} ImageNet classes...")
    print(f"Generating {images_per_class} images per class")
    
    # Iterate through each target class
    for step, (class_name, class_id) in enumerate(target_classes.items()):
        print(f"\nStep {step+1}/{num_steps}: Generating {class_name} (class {class_id})")
        
        # Create class labels tensor - repeat the same class ID for all images in batch
        batch_size_current = min(batch_size, images_per_class)
        class_labels = torch.full((batch_size_current,), class_id, dtype=torch.long, device='cuda')
        
        # Record timing for this class generation
        torch.cuda.synchronize()
        step_start_time = time.time()
        
        # Generate images using the FAR model
        device = torch.device("cuda")
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # The FAR model's sample_tokens method handles the iterative generation
                sampled_images = model_without_ddp.sample_tokens(
                    vae=vae, 
                    bsz=batch_size_current, 
                    num_iter=args.num_iter, 
                    cfg=cfg,
                    cfg_schedule=args.cfg_schedule, 
                    labels=class_labels, 
                    device=device,
                    temperature=args.temperature
                )

        torch.cuda.synchronize()
        step_end_time = time.time()
        step_time = step_end_time - step_start_time
        
        # Record detailed timing information (skip warmup)
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
        
        print(f"Generated {batch_size_current} images of {class_name} in {step_time:.3f}s "
              f"({step_time/batch_size_current:.3f}s per image, "
              f"{batch_size_current/step_time:.2f} images/sec)")
        
        # Save generated images for this class
        torch.distributed.barrier()
        sampled_images = sampled_images.detach().cpu()
        
        # Save individual class results
        class_output_path = os.path.join(generated_images_dir, f"{class_name}_class{class_id}.png")
        save_image(sampled_images, nrow=min(4, batch_size_current), show=False, 
                  path=class_output_path, to_grayscale=False)
        
        # Also save the first image generated for this class as a standalone file
        if sampled_images.shape[0] > 0:
            single_image_path = os.path.join(generated_images_dir, f"{class_name}_class{class_id}_sample1.png")
            save_image(sampled_images[0:1], nrow=1, show=False, path=single_image_path, to_grayscale=False)

    # Calculate comprehensive statistics (excluding warmup)
    if len(detailed_times) > 0:
        avg_time_per_image = total_generation_time / total_images_generated
        avg_images_per_second = total_images_generated / total_generation_time
        
        # Create comprehensive evaluation report
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
                'img_size': args.img_size
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
        
        # Save comprehensive JSON report
        report_path = os.path.join(speed_results_dir, f"imagenet_evaluation_report_cfg{cfg}.json")
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2)
        
        # Save CSV summary for easy analysis
        csv_path = os.path.join(speed_results_dir, f"imagenet_summary_cfg{cfg}.csv")
        with open(csv_path, 'w') as f:
            f.write("Class_Name,Class_ID,Images_Generated,Time_Per_Image,Images_Per_Second\n")
            for result in detailed_times:
                f.write(f"{result['class_name']},{result['class_id']},{result['batch_size']},"
                       f"{result['time_per_image']:.5f},{result['images_per_second']:.2f}\n")
        
        # Display comprehensive results
        print("\n" + "="*60)
        print("IMAGENET CLASS-CONDITIONAL EVALUATION RESULTS")
        print("="*60)
        print(f"Model: {args.model}")
        print(f"Classes evaluated: {len(target_classes)}")
        print(f"Total images generated: {total_images_generated}")
        print(f"Total generation time: {total_generation_time:.3f} seconds")
        print(f"Average time per image: {avg_time_per_image:.3f} seconds")
        print(f"Average images per second: {avg_images_per_second:.2f}")
        print(f"Results saved to: {report_path}")
        print(f"Generated images saved to: {generated_images_dir}")
        print("="*60 + "\n")

    # Restore original model parameters
    if use_ema:
        print("Restoring original model parameters...")
        model_without_ddp.load_state_dict(model_state_dict)

def get_args_parser():
    """Parse command line arguments for ImageNet class evaluation"""
    parser = argparse.ArgumentParser('FAR ImageNet Class Evaluation', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='far_large', type=str, metavar='MODEL',
                        help='Name of model to evaluate')
    
    # VAE parameters  
    parser.add_argument('--img_size', default=256, type=int,
                        help='images input size')
    parser.add_argument('--vae_path', default="pretrained/vae/kl16.ckpt", type=str,
                        help='path to VAE checkpoint')
    parser.add_argument('--vae_embed_dim', default=16, type=int,
                        help='vae output embedding dimension')
    parser.add_argument('--vae_stride', default=16, type=int,
                        help='tokenizer stride, default use KL16')
    parser.add_argument('--patch_size', default=1, type=int,
                        help='number of tokens to group as a patch')

    # Generation parameters
    parser.add_argument('--num_iter', default=10, type=int,
                        help='number of autoregressive iterations to generate an image')
    parser.add_argument('--cfg', default=3.0, type=float, help="classifier-free guidance")
    parser.add_argument('--cfg_schedule', default="linear", type=str)
    parser.add_argument('--temperature', default=1.0, type=float, help='sampling temperature')
    parser.add_argument('--eval_bsz', type=int, default=8, help='generation batch size')

    # Model architecture parameters
    parser.add_argument('--mask_ratio_min', type=float, default=0.7,
                        help='Minimum mask ratio')
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--attn_dropout', type=float, default=0.1,
                        help='attention dropout')
    parser.add_argument('--proj_dropout', type=float, default=0.1,
                        help='projection dropout')
    parser.add_argument('--buffer_size', type=int, default=64)
    parser.add_argument('--class_num', default=1000, type=int)

    # Diffusion Loss params
    parser.add_argument('--diffloss_d', type=int, default=3)
    parser.add_argument('--diffloss_w', type=int, default=1024)
    parser.add_argument('--num_sampling_steps', type=str, default="100")
    parser.add_argument('--diffusion_batch_mul', type=int, default=1)

    # System parameters
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save results')
    parser.add_argument('--device', default='cuda',
                        help='device to use for evaluation')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--resume', default='', required=True,
                        help='path to model checkpoint')

    # Distributed parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    """Main evaluation function for ImageNet class-conditional generation"""
    
    # Initialize distributed training if needed
    misc.init_distributed_mode(args)

    print('Evaluation directory: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("Configuration:")
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Define the target ImageNet classes we want to evaluate
    # These represent a diverse set of objects, animals, and scenes
    target_classes = {
        'golden_retriever': 207,       # Animal - Dog breed
        'tabby_cat': 281,             # Animal - Cat breed  
        'red_fox': 277,               # Animal - Wild animal
        'monarch_butterfly': 323,      # Animal - Insect
        'daisy': 985,                 # Plant - Flower
        'rose': 973,                  # Plant - Flower
        'lighthouse': 437,            # Structure - Building
        'castle': 483,                # Structure - Historical building
        'cottage': 500,               # Structure - House
        'sports_car': 817,            # Vehicle - Car
        'steam_locomotive': 820,       # Vehicle - Train
        'sailboat': 554,              # Vehicle - Boat
        'aircraft_carrier': 403,       # Vehicle - Ship
        'mountain_bike': 671,         # Vehicle - Bicycle
        'pizza': 963,                 # Food - Prepared food
        'strawberry': 949,            # Food - Fruit
        'coffee_mug': 504,            # Object - Container
        'violin': 889,                # Object - Musical instrument
        'backpack': 414,              # Object - Bag
        'umbrella': 879               # Object - Tool
    }

    print(f"\nEvaluating {len(target_classes)} carefully selected ImageNet classes:")
    for class_name, class_id in target_classes.items():
        print(f"  {class_name}: {class_id}")

    # Initialize VAE model for decoding latent representations
    print("\nInitializing VAE...")
    vae = AutoencoderKL(
        embed_dim=args.vae_embed_dim, 
        ch_mult=(1, 1, 2, 2, 4), 
        ckpt_path=args.vae_path
    ).cuda().eval()
    
    # Freeze VAE parameters
    for param in vae.parameters():
        param.requires_grad = False
    
    # Initialize FAR model with specified architecture
    print(f"Initializing FAR model: {args.model}")
    model = far.__dict__[args.model](
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        mask=False,  # No masking during evaluation
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
    
    print("Model = %s" % str(model))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.2f}M".format(n_params / 1e6))

    # Handle distributed training if needed
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # Load model checkpoint
    if args.resume and os.path.exists(os.path.join(args.resume, "checkpoint-last.pth")):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(os.path.join(args.resume, "checkpoint-last.pth"), map_location='cpu')
        
        # Load model state dict
        model_without_ddp.load_state_dict(checkpoint['model'])
        model_params = list(model_without_ddp.parameters())
        
        # Load EMA parameters if available
        if 'model_ema' in checkpoint:
            ema_state_dict = checkpoint['model_ema']
            ema_params = [ema_state_dict[name].cuda() for name, _ in model_without_ddp.named_parameters()]
            print("EMA parameters loaded successfully")
        else:
            ema_params = copy.deepcopy(model_params)
            print("No EMA parameters found, using regular parameters")
            
        print("Checkpoint loaded successfully")
        del checkpoint
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.resume}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run the evaluation
    print("\nStarting ImageNet class-conditional evaluation...")
    torch.cuda.empty_cache()
    
    evaluate_imagenet_classes(
        model_without_ddp=model_without_ddp,
        vae=vae,
        ema_params=ema_params,
        args=args,
        epoch=0,  # Not relevant for evaluation
        target_classes=target_classes,
        batch_size=args.eval_bsz,
        log_writer=None,
        cfg=args.cfg,
        use_ema=True
    )
    
    print("Evaluation completed successfully!")

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)