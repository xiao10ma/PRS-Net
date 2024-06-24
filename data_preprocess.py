import os
import numpy as np
import torch
import open3d as o3d

# 列出ShapeNet数据集中的类别
cates = os.listdir('./data/ShapeNet')

# 定义一个函数来找到最近的点
def find_nearest_points(points, mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    return scene.compute_closest_points(points)['points']

models = None
points_set = None
pretreatment_set = None
save_path = "./data/processed_shapenet"

num_file = 0

# 遍历类别
for cate in cates:
    try:
        model_list = os.listdir('./data/ShapeNet/' + cate)
    except Exception as e:
        print(f"Error listing models in category {cate}: {e}")
        continue
    print(cate, end=' ')
    for model_id in model_list:
        path = './data/ShapeNet/{}/{}/models/'.format(cate, model_id)
        if len(model_id) != 32:
            continue

        if models is not None:
            print(num_file + 1, models.size(0))

        # 采样点云
        try:
            mesh_path = os.path.join(path, 'model_normalized.obj')
            print(f"Reading mesh from: {mesh_path}")
            mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=False)
            if mesh.is_empty():
                print(f"Mesh is empty for model {model_id}")
                continue
            pcd = mesh.sample_points_uniformly(number_of_points=1000)
            points = torch.from_numpy(np.asarray(pcd.points))
        except Exception as e:
            print(f"Error reading mesh for model {model_id}: {e}")
            continue

        # 预训练最近点
        try:
            pretreatment = np.zeros([32, 32, 32, 3], dtype=np.float32)
            for i in range(32):
                for j in range(32):
                    for k in range(32):
                        pretreatment[i, j, k] = np.array([i, j, k])
            pretreatment = pretreatment / 32 + 1 / 64 - 0.5
            pretreatment = find_nearest_points(pretreatment, mesh)
            pretreatment = torch.from_numpy(pretreatment.numpy())
        except Exception as e:
            print(f"Error in pretreatment for model {model_id}: {e}")
            continue

        # 提取体素
        try:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=0.03125)
            voxel_cells = torch.from_numpy(np.stack(list(vx.grid_index for vx in voxel_grid.get_voxels())))
            voxel_cells[:, 0] += ((points[:, 0].min() + 0.5) * 32).int()
            voxel_cells[:, 1] += ((points[:, 1].min() + 0.5) * 32).int()
            voxel_cells[:, 2] += ((points[:, 2].min() + 0.5) * 32).int()
            model = torch.zeros([32, 32, 32], dtype=torch.float32)
            for v in voxel_cells:
                try:
                    model[v[0], v[1], v[2]] = 1
                except:
                    continue
            model = model.unsqueeze(0)
        except Exception as e:
            print(f"Error extracting voxels for model {model_id}: {e}")
            continue

        # 合并数据
        if pretreatment_set is None:
            # pretreatment_set = pretreatment.unsqueeze(0)
            pretreatment_set = pretreatment
        else:
            pretreatment_set = torch.cat([pretreatment_set, pretreatment.unsqueeze(0)], dim=0)

        if models is None:
            # models = model.unsqueeze(0)
            models = model
        else:
            models = torch.cat([models, model.unsqueeze(0)], 0)

        if points_set is None:
            # points_set = points.unsqueeze(0)
            points_set = points
        else:
            points_set = torch.cat([points_set, points.unsqueeze(0)], dim=0)

        size = models.size(0)
        if size == 1:
            num_file += 1
            np.save(os.path.join(save_path, str(num_file)) + '_voxel', models.numpy())
            np.save(os.path.join(save_path, str(num_file)) + '_points', points_set.numpy())
            np.save(os.path.join(save_path, str(num_file)) + '_pre', pretreatment_set.numpy())
            models = None
            points_set = None
            pretreatment_set = None

print('Processing complete!')
