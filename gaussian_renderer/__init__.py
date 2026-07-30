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
import torch
from utils.checkpoint_utils import activation_checkpoint

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel


def _run_attribute_mlp_chunked(
    module,
    features,
    view_direction,
    distance,
    *,
    include_distance,
    appearance=None,
    chunk_size=2048,
    checkpointed=False,
):
    """Run one attribute MLP in anchor chunks to cap retained activations."""

    n_rows = features.shape[0]
    if n_rows == 0:
        parts = [features, view_direction]
        if include_distance:
            parts.append(distance)
        if appearance is not None:
            parts.append(appearance)
        return module(torch.cat(parts, dim=1))

    # COMPATIBILITY: keep the original single-call path for small batches.
    # Larger batches honor the explicit activation budget even when an A100
    # profile retains activations instead of checkpointing them.
    if n_rows <= chunk_size:
        parts = [features, view_direction]
        if include_distance:
            parts.append(distance)
        if appearance is not None:
            parts.append(appearance)
        return module(torch.cat(parts, dim=1))

    outputs = []
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        parts = [features[start:end], view_direction[start:end]]
        if include_distance:
            parts.append(distance[start:end])
        if appearance is not None:
            parts.append(appearance[start:end])
        inputs = torch.cat(parts, dim=1)
        is_script_module = isinstance(module, torch.jit.ScriptModule)
        if checkpointed and inputs.requires_grad and not is_script_module:
            # INFERENCE: reentrant checkpointing is supported by the legacy
            # PyTorch 1.12 environment used by the competition machine.
            outputs.append(activation_checkpoint(module, inputs))
        else:
            outputs.append(module(inputs))
    return torch.cat(outputs, dim=0)


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    # COMPATIBILITY: every renderer consumer now shares the model-owned feature
    # path.  It returns first-order D features for Scaffold-GS and
    # D*(1+M) second-order features for SOGS.
    # COMPATIBILITY: statistics remain scene-global, but branch MLPs only
    # materialize features for anchors that survive the visibility filter.
    render_features = pc.get_render_features(visible_mask=visible_mask)
    expected_dim = (
        pc.feat_dim * (1 + pc.num_eigenvectors)
        if pc.use_second_order
        else pc.feat_dim
    )
    assert render_features.shape[-1] == expected_dim
    feat = render_features
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist.clamp_min(torch.finfo(ob_view.dtype).eps)

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feature_width = feat.shape[1]
        coarse_four = feat[:, ::4, :1]
        coarse_two = feat[:, ::2, :1]
        # COMPATIBILITY: preserve the original periodic feature-bank repeat for
        # widths divisible by four, and trim the repeated views for any valid
        # SOGS feature width that is not divisible by two/four.
        coarse_four = coarse_four.repeat(
            [1, math.ceil(feature_width / coarse_four.shape[1]), 1]
        )[:, :feature_width]
        coarse_two = coarse_two.repeat(
            [1, math.ceil(feature_width / coarse_two.shape[1]), 1]
        )[:, :feature_width]
        feat = (
            coarse_four * bank_weight[:, :, :1]
            + coarse_two * bank_weight[:, :, 1:2]
            + feat[:, ::1, :1] * bank_weight[:, :, 2:]
        )
        feat = feat.squeeze(dim=-1) # [n, c]


    needs_distance_features = (
        pc.add_opacity_dist or pc.add_cov_dist or pc.add_color_dist
    )
    # COMPATIBILITY: derive widths without allocating duplicate full input
    # tensors; the chunked helper builds only one anchor slice at a time.
    assert feat.shape[1] + ob_view.shape[1] == expected_dim + 3
    if needs_distance_features:
        assert feat.shape[1] + ob_view.shape[1] + ob_dist.shape[1] == expected_dim + 4
    chunk_size = getattr(pc, "sogs_chunk_size", 2048)
    checkpoint_attributes = bool(
        pc.use_second_order
        and is_training
        and pc.sogs_checkpointing
    )
    if pc.appearance_dim > 0:
        # COMPATIBILITY: appearance IDs only need the visible-row count; do
        # not force construction of the optional distance feature tensor.
        camera_indicies = torch.ones_like(
            feat[:, 0],
            dtype=torch.long,
            device=ob_dist.device,
        ) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)
    else:
        appearance = None

    # get offset's opacity
    neural_opacity = _run_attribute_mlp_chunked(
        pc.get_opacity_mlp,
        feat,
        ob_view,
        ob_dist,
        include_distance=pc.add_opacity_dist,
        appearance=None,
        chunk_size=chunk_size,
        checkpointed=checkpoint_attributes,
    ) # [N, k]

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    color = _run_attribute_mlp_chunked(
        pc.get_color_mlp,
        feat,
        ob_view,
        ob_dist,
        include_distance=pc.add_color_dist,
        appearance=appearance,
        chunk_size=chunk_size,
        checkpointed=checkpoint_attributes,
    )
    color = color.reshape([anchor.shape[0], pc.n_offsets, 3])

    # get offset's cov
    scale_rot = _run_attribute_mlp_chunked(
        pc.get_cov_mlp,
        feat,
        ob_view,
        ob_dist,
        include_distance=pc.add_cov_dist,
        appearance=None,
        chunk_size=chunk_size,
        checkpointed=checkpoint_attributes,
    )
    scale_rot = scale_rot.reshape([anchor.shape[0], pc.n_offsets, 7])

    # INFERENCE: mask each attribute tensor independently instead of building
    # and repeating every anchor K times. This removes substantial transient
    # VRAM at high anchor counts while preserving flattened row-major order.
    anchor_indices, offset_indices = mask.view(
        anchor.shape[0], pc.n_offsets
    ).nonzero(as_tuple=True)
    scaling_repeat = grid_scaling[anchor_indices]
    repeat_anchor = anchor[anchor_indices]
    color = color[anchor_indices, offset_indices]
    scale_rot = scale_rot[anchor_indices, offset_indices]
    offsets = grid_offsets[anchor_indices, offset_indices]
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        xyz,
        dtype=pc.get_anchor.dtype,
        requires_grad=True,
        device=xyz.device,
    ) + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        pc.get_anchor,
        dtype=pc.get_anchor.dtype,
        requires_grad=True,
        device=pc.get_anchor.device,
    ) + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
