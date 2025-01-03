from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

import torch

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(['/home/lab/Documents/scsv/thesis/thesis_2025/_data/campus_medium/hierarchy/inputs/images/cam01/001004_b.jpg', '/home/lab/Documents/scsv/thesis/thesis_2025/_data/campus_medium/hierarchy/inputs/images/cam01/001006_b.jpg'], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

    # add known things in the scene
    # scene.

    # intrinsics
    # I removed 

    old_focal = 1415.246433
    old_width = 2704
    new_width = 512

    scale = new_width/old_width


    new_focal = old_focal * scale
    print(new_focal)

    known_focals = [torch.tensor([321], dtype=torch.float32), torch.tensor([321], dtype=torch.float32)]
    # known_focals = [torch.tensor([new_focal], dtype=torch.float32), torch.tensor([new_focal], dtype=torch.float32)]
    # known_pp = [torch.tensor([1352.000000*scale, 1014.000000*scale], dtype=torch.float32, requires_grad=True)]

    # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr

    from scipy.spatial.transform import Rotation as R
    import numpy as np

    # Input for image1
    quaternion1 = [0.79420396,  0.00774901, -0.58312268, -0.17072771]  # qw, qx, qy, qz
    translation1 = [ 1.19137389,  0.70184634, -2.72513559]

    # Input for image2
    quaternion2 = [ 0.77840913,  0.00546752, -0.60347569, -0.17281906]  # qw, qx, qy, qz
    translation2 = [  1.0982092 ,  0.70460378, -2.70156784]

    # Convert quaternion to rotation matrix (scipy uses qx, qy, qz, qw format)
    rotation_matrix1 = R.from_quat([quaternion1[1], quaternion1[2], quaternion1[3], quaternion1[0]]).as_matrix()
    rotation_matrix2 = R.from_quat([quaternion2[1], quaternion2[2], quaternion2[3], quaternion2[0]]).as_matrix()

    # Print results
    print("Image1 Rotation Matrix:")
    print(rotation_matrix1)

    print("Image2 Rotation Matrix:")
    print(rotation_matrix2)

    def construct_camera_to_world(rotation_matrix, translation):
        # Create a 4x4 identity matrix
        camera_to_world = np.eye(4)
        camera_to_world[:3, :3] = rotation_matrix  # Set rotation
        camera_to_world[:3, 3] = translation       # Set translation
        return camera_to_world

    # Construct camera-to-world matrices
    camera_to_world1 = construct_camera_to_world(rotation_matrix1, translation1)
    camera_to_world2 = construct_camera_to_world(rotation_matrix2, translation2)

    # Print matrices
    print("Camera-to-World Matrix for Image1:")
    print(camera_to_world1)

    print("Camera-to-World Matrix for Image2:")
    print(camera_to_world2)

    known_poses = [
        torch.tensor(camera_to_world1, dtype=torch.float32, requires_grad=False),
        torch.tensor(camera_to_world2, dtype=torch.float32, requires_grad=False)
    ]
    print(known_poses)

    scene.preset_pose(known_poses=known_poses)
    scene.preset_focal(known_focals=known_focals)
    # scene.preset_principal_point(known_pp=known_pp)

    loss = scene.compute_global_alignment(init="known_poses", niter=niter, schedule=schedule, lr=lr)
    # loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)


    # # retrieve useful values from scene:
    # imgs = scene.imgs
    focals = scene.get_focals()
    print(focals)
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    # scene
    # confidence_masks = scene.get_masks()

    # visualize reconstruction
    scene.show()

    # # find 2D-2D matches between the two images
    # from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    # pts2d_list, pts3d_list = [], []
    # for i in range(2):
    #     conf_i = confidence_masks[i].cpu().numpy()
    #     pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
    #     pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    # reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    # print(f'found {num_matches} matches')
    # matches_im1 = pts2d_list[1][reciprocal_in_P2]
    # matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # # visualize a few matches
    # import numpy as np
    # from matplotlib import pyplot as pl
    # n_viz = 10
    # match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    # viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    # H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    # img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    # img = np.concatenate((img0, img1), axis=1)
    # pl.figure()
    # pl.imshow(img)
    # cmap = pl.get_cmap('jet')
    # for i in range(n_viz):
    #     (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
    #     pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    # pl.show(block=True)