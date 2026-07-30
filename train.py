#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import subprocess
import sys


def configure_cuda_device(argv):
    """Select the requested GPU before importing or initializing PyTorch."""
    requested_gpu = None
    for index, argument in enumerate(argv):
        if argument == "--gpu" and index + 1 < len(argv):
            requested_gpu = argv[index + 1]
        elif argument.startswith("--gpu="):
            requested_gpu = argument.split("=", maxsplit=1)[1]

    if requested_gpu and requested_gpu != "-1":
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_gpu
        return

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return

    try:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        memory_used = [
            int(value.strip())
            for value in query.stdout.splitlines()
            if value.strip()
        ]
        if memory_used:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                min(range(len(memory_used)), key=memory_used.__getitem__)
            )
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        # Let PyTorch use its default device selection and report any CUDA
        # problem normally when the training process starts.
        pass


configure_cuda_device(sys.argv[1:])

import numpy as np


import torch
import torchvision
import json
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
try:
    import wandb
except ImportError:
    wandb = None
try:
    import lpips
except ImportError:
    lpips = None
from random import randint
from utils.loss_utils import (
    l1_loss,
    scaling_volume_regularization,
    selective_gradient_loss,
    ssim,
)
from gaussian_renderer import prefilter_voxel, render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.training_budget import (
    TrainingBudget,
    should_checkpoint_iteration,
    validate_checkpoint_interval,
    validate_max_runtime_minutes,
)
from utils.runtime_utils import configure_torch_runtime
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    validate_positive_int,
)

# torch.set_num_threads(32)
# COMPATIBILITY: avoid constructing a GPU LPIPS model during --help/import.
# Evaluation initializes it lazily on the rendered image's device.
lpips_fn = None

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    
    print('Backup Finished!')


def _save_training_checkpoint(gaussians, iteration, model_path, logger):
    checkpoint_path = os.path.join(model_path, f"chkpnt{iteration}.pth")
    logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
    torch.save((gaussians.capture(), iteration), checkpoint_path)
    return checkpoint_path


def _log_cuda_peak_memory(device, logger):
    """Record allocator peaks without forcing a CUDA synchronization."""

    if logger is None or not torch.cuda.is_available():
        return
    logger.info(
        "CUDA peak memory: allocated={} MiB, reserved={} MiB".format(
            torch.cuda.max_memory_allocated(device) // (1024 * 1024),
            torch.cuda.max_memory_reserved(device) // (1024 * 1024),
        )
    )


def training(
    dataset,
    opt,
    pipe,
    dataset_name,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    wandb=None,
    logger=None,
    ply_path=None,
    runtime_budget=None,
    checkpoint_interval=0,
    log_interval=1,
    enable_gui=True,
    config_snapshot=None,
):
    first_iter = 0
    if runtime_budget is None:
        runtime_budget = TrainingBudget()
    checkpoint_interval = validate_checkpoint_interval(checkpoint_interval)
    log_interval = validate_positive_int(
        log_interval,
        name="log_interval",
    )
    # COMPATIBILITY: direct callers may still provide only ModelParams. The
    # CLI path records the complete resolved Namespace for reproducibility.
    tb_writer = prepare_output_and_logger(
        dataset if config_snapshot is None else config_snapshot
    )
    gaussians = GaussianModel(
        feat_dim=dataset.feat_dim,
        n_offsets=dataset.n_offsets,
        voxel_size=dataset.voxel_size,
        update_depth=dataset.update_depth,
        update_init_factor=dataset.update_init_factor,
        update_hierachy_factor=dataset.update_hierachy_factor,
        use_feat_bank=dataset.use_feat_bank,
        appearance_dim=dataset.appearance_dim,
        ratio=dataset.ratio,
        add_opacity_dist=dataset.add_opacity_dist,
        add_cov_dist=dataset.add_cov_dist,
        add_color_dist=dataset.add_color_dist,
        use_second_order=dataset.use_second_order,
        num_eigenvectors=dataset.num_eigenvectors,
        lambda_sgl=dataset.lambda_sgl,
        sogs_chunk_size=dataset.sogs_chunk_size,
        sogs_checkpointing=dataset.sogs_checkpointing,
        sogs_validate_numerics=dataset.sogs_validate_numerics,
        sogs_cache_render_features=dataset.sogs_cache_render_features,
    )
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    resolved_torch_runtime = configure_torch_runtime(
        tf32_mode=getattr(dataset, "tf32_mode", "default"),
        cudnn_benchmark=getattr(dataset, "cudnn_benchmark", False),
    )
    # COMPATIBILITY: persist the resolved optimizer/runtime implementation in
    # addition to cfg_args, where optimizer_backend may still be "auto".
    runtime_metadata = {
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "requested_optimizer_backend": str(
            getattr(opt, "optimizer_backend", "default")
        ),
        "resolved_optimizer_backend": str(gaussians.optimizer_backend),
        "tf32_mode": str(getattr(dataset, "tf32_mode", "default")),
        "cudnn_benchmark": bool(
            getattr(dataset, "cudnn_benchmark", False)
        ),
        "resolved_torch_runtime": resolved_torch_runtime,
        "sogs_checkpointing": bool(dataset.sogs_checkpointing),
        "sogs_validate_numerics": bool(dataset.sogs_validate_numerics),
        "sogs_cache_render_features": bool(
            dataset.sogs_cache_render_features
        ),
        "sogs_chunk_size": int(dataset.sogs_chunk_size),
        "densification_chunk_size": int(
            gaussians.densification_chunk_size
        ),
    }
    if torch.cuda.is_available():
        device_properties = torch.cuda.get_device_properties(
            gaussians.device
        )
        runtime_metadata["cuda_device"] = {
            "name": device_properties.name,
            "compute_capability": [
                int(device_properties.major),
                int(device_properties.minor),
            ],
            "total_memory_bytes": int(device_properties.total_memory),
        }
    with open(
        os.path.join(scene.model_path, "runtime_config.json"),
        "w",
    ) as runtime_file:
        json.dump(runtime_metadata, runtime_file, indent=2, sort_keys=True)
    if logger is not None:
        logger.info(f"resolved runtime: {runtime_metadata}")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(gaussians.device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    # COMPATIBILITY: the background is constant for a run. Reusing it avoids
    # one host allocation and host-to-device copy per iteration.
    background = torch.tensor(
        bg_color,
        dtype=torch.float32,
        device=gaussians.device,
    )

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    last_progress_iteration = first_iter - 1
    progress_interval = max(10, log_interval)
    for iteration in range(first_iter, opt.iterations + 1):        
        # network gui not available in scaffold-gs yet
        if enable_gui and network_gui.conn == None:
            network_gui.try_connect()
        while enable_gui and network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        log_scalars = (
            iteration % log_interval == 0
            or iteration == opt.iterations
        )
        measure_timing = bool(tb_writer is not None and log_scalars)
        if measure_timing:
            iter_start.record()

        gaussians.update_learning_rate(iteration)

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        gt_image = viewpoint_cam.original_image.to(
            device=image.device,
            non_blocking=True,
        )
        Ll1 = l1_loss(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling_volume_regularization(scaling)
        # COMPATIBILITY: retain the original Scaffold-GS objective exactly
        # when SOGS is disabled or lambda_sgl is zero.
        sgl_loss = image.new_zeros(())
        if dataset.use_second_order and dataset.lambda_sgl > 0:
            sgl_loss = selective_gradient_loss(
                image,
                gt_image,
                validate_finite=dataset.sogs_validate_numerics,
            )
        loss = (
            (1.0 - opt.lambda_dssim) * Ll1
            + opt.lambda_dssim * ssim_loss
            + 0.01 * scaling_reg
            + dataset.lambda_sgl * sgl_loss
        )

        loss.backward()
        
        if measure_timing:
            iter_end.record()

        with torch.no_grad():
            # Progress bar
            if (
                iteration % progress_interval == 0
                or iteration == opt.iterations
            ):
                ema_loss_for_log = (
                    0.4 * loss.item() + 0.6 * ema_loss_for_log
                )
                progress_bar.set_postfix(
                    {
                        "L1": f"{Ll1.item():.5f}",
                        "SSIM": f"{ssim_loss.item():.5f}",
                        "Vol": f"{scaling_reg.item():.5f}",
                        "SGL": f"{sgl_loss.item():.5f}",
                        "Total": f"{ema_loss_for_log:.7f}",
                    }
                )
                progress_bar.update(iteration - last_progress_iteration)
                last_progress_iteration = iteration
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            elapsed = (
                iter_start.elapsed_time(iter_end)
                if measure_timing
                else None
            )
            training_report(
                tb_writer,
                dataset_name,
                iteration,
                Ll1,
                loss,
                l1_loss,
                elapsed,
                testing_iterations,
                scene,
                render,
                (pipe, background),
                wandb,
                logger,
                ssim_loss=ssim_loss,
                scaling_reg=scaling_reg,
                sgl_loss=sgl_loss,
                log_scalars=log_scalars,
            )
            model_saved = iteration in saving_iterations
            if model_saved:
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            checkpoint_saved = should_checkpoint_iteration(
                iteration,
                checkpoint_iterations,
                checkpoint_interval,
            )
            if checkpoint_saved:
                _save_training_checkpoint(
                    gaussians,
                    iteration,
                    scene.model_path,
                    logger,
                )

            # INFERENCE: a wall-clock limit makes short competition runs
            # predictable. Finish the current optimizer step, then save both
            # renderable model files and a resumable optimizer checkpoint.
            if runtime_budget.expired() and iteration < opt.iterations:
                logger.info(
                    "\n[ITER {}] Runtime budget reached; saving a recoverable "
                    "model before stopping.".format(iteration)
                )
                if not model_saved:
                    scene.save(iteration)
                if not checkpoint_saved:
                    _save_training_checkpoint(
                        gaussians,
                        iteration,
                        scene.model_path,
                        logger,
                    )
                _log_cuda_peak_memory(gaussians.device, logger)
                progress_bar.close()
                return iteration, True

    _log_cuda_peak_memory(gaussians.device, logger)
    return opt.iterations, False

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
    tb_writer,
    dataset_name,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene : Scene,
    renderFunc,
    renderArgs,
    wandb=None,
    logger=None,
    *,
    ssim_loss=None,
    scaling_reg=None,
    sgl_loss=None,
    log_scalars=True,
):
    # COMPATIBILITY: retain the local Scaffold-GS positional interface while
    # accepting optional component tensors for SOGS logging.
    if ssim_loss is None:
        ssim_loss = loss.new_zeros(())
    if scaling_reg is None:
        scaling_reg = loss.new_zeros(())
    if sgl_loss is None:
        sgl_loss = loss.new_zeros(())
    if tb_writer and log_scalars:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(
            f'{dataset_name}/train_loss_patches/ssim_loss',
            ssim_loss.item(),
            iteration,
        )
        tb_writer.add_scalar(
            f'{dataset_name}/train_loss_patches/volume_regularization',
            scaling_reg.item(),
            iteration,
        )
        tb_writer.add_scalar(
            f'{dataset_name}/train_loss_patches/selective_gradient_loss',
            sgl_loss.item(),
            iteration,
        )
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        if elapsed is not None:
            tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None and log_scalars:
        wandb.log(
            {
                "train_l1_loss": Ll1,
                "train_ssim_loss": ssim_loss,
                "train_volume_regularization": scaling_reg,
                "train_selective_gradient_loss": sgl_loss,
                "train_total_loss": loss,
            }
        )
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
    
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        gaussians = GaussianModel(
            feat_dim=dataset.feat_dim,
            n_offsets=dataset.n_offsets,
            voxel_size=dataset.voxel_size,
            update_depth=dataset.update_depth,
            update_init_factor=dataset.update_init_factor,
            update_hierachy_factor=dataset.update_hierachy_factor,
            use_feat_bank=dataset.use_feat_bank,
            appearance_dim=dataset.appearance_dim,
            ratio=dataset.ratio,
            add_opacity_dist=dataset.add_opacity_dist,
            add_cov_dist=dataset.add_cov_dist,
            add_color_dist=dataset.add_color_dist,
            use_second_order=dataset.use_second_order,
            num_eigenvectors=dataset.num_eigenvectors,
            lambda_sgl=dataset.lambda_sgl,
            sogs_chunk_size=dataset.sogs_chunk_size,
            sogs_checkpointing=dataset.sogs_checkpointing,
            sogs_validate_numerics=dataset.sogs_validate_numerics,
            sogs_cache_render_features=dataset.sogs_cache_render_features,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)

        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    global lpips_fn

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):

        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        if lpips_fn is None:
            if lpips is None:
                raise RuntimeError(
                    "LPIPS is required for evaluation; install the lpips package"
                )
            lpips_fn = lpips.LPIPS(net='vgg').to(renders[0].device)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=0,
        help=(
            "Save a resumable checkpoint every N iterations; zero disables "
            "periodic checkpoints."
        ),
    )
    parser.add_argument(
        "--max_runtime_minutes",
        type=float,
        default=0.0,
        help=(
            "Stop training after this wall-clock budget and save the model "
            "plus a resumable checkpoint; zero disables the limit. "
            "Post-processing time is not included."
        ),
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=1,
        help=(
            "Write scalar logs every N iterations. Larger values reduce "
            "CPU/GPU synchronization; the original behavior is 1."
        ),
    )
    parser.add_argument(
        "--disable_gui",
        action="store_true",
        help="Disable the per-iteration network GUI connection check.",
    )
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument(
        "--skip_postprocess",
        action="store_true",
        help="Skip the built-in held-out rendering and metrics pass after training.",
    )
    args = parser.parse_args(sys.argv[1:])
    try:
        args.max_runtime_minutes = validate_max_runtime_minutes(
            args.max_runtime_minutes
        )
        args.checkpoint_interval = validate_checkpoint_interval(
            args.checkpoint_interval
        )
        args.log_interval = validate_positive_int(
            args.log_interval,
            name="log_interval",
        )
        runtime_settings = configure_torch_runtime(
            tf32_mode=args.tf32_mode,
            cudnn_benchmark=args.cudnn_benchmark,
        )
    except ValueError as error:
        parser.error(str(error))
    args.save_iterations.append(args.iterations)

    
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')
    logger.info(f"torch runtime: {runtime_settings}")

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

    

    try:
        saveRuntimeCode(os.path.join(args.model_path, 'backup'))
    except:
        logger.info(f'save code failed~')
        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError(
                "--use_wandb was requested but the wandb package is not installed"
            )
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"Scaffold-GS-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_gui:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # INFERENCE: one budget instance is shared by both passes so --warmup
    # cannot silently double a requested wall-clock limit.
    runtime_budget = TrainingBudget(args.max_runtime_minutes)
    completed_iteration, time_limited = training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        dataset,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        wandb,
        logger,
        runtime_budget=runtime_budget,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        enable_gui=not args.disable_gui,
        config_snapshot=args,
    )
    if args.warmup:
        if time_limited or runtime_budget.expired():
            time_limited = True
            logger.info(
                "\nSkipping the warmup pass because the runtime budget was "
                "exhausted."
            )
        else:
            logger.info("\n Warmup finished! Reboot from last checkpoints")
            new_ply_path = os.path.join(
                args.model_path,
                f"point_cloud/iteration_{completed_iteration}",
                "point_cloud.ply",
            )
            completed_iteration, time_limited = training(
                lp.extract(args),
                op.extract(args),
                pp.extract(args),
                dataset,
                args.test_iterations,
                args.save_iterations,
                args.checkpoint_iterations,
                args.start_checkpoint,
                args.debug_from,
                wandb=wandb,
                logger=logger,
                ply_path=new_ply_path,
                runtime_budget=runtime_budget,
                checkpoint_interval=args.checkpoint_interval,
                log_interval=args.log_interval,
                enable_gui=not args.disable_gui,
                config_snapshot=args,
            )

    # All done
    if time_limited:
        logger.info(
            "\nTime-budgeted training stopped safely at iteration {}.".format(
                completed_iteration
            )
        )
    else:
        logger.info("\nTraining complete.")

    if not args.skip_postprocess:
        # rendering
        logger.info(f'\nStarting Rendering~')
        visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger)
        logger.info("\nRendering complete.")

        # calc metrics
        logger.info("\n Starting evaluation...")
        evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
        logger.info("\nEvaluating complete.")
