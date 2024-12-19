import numpy as np
import trimesh
import _pickle as pickle
import os


def save_meshes(predictions, save_path_Meshes, n_meshes):
    tri = trimesh.load( './template/template/template.obj', process=False)
    triangles=tri.faces
    for i in range(n_meshes):
        tri_mesh = trimesh.Trimesh(np.asarray(np.squeeze(predictions[i])), np.asarray(triangles), process=False)
        tri_mesh.export(os.path.join(save_path_Meshes, "tst{0:03}.ply".format(i)))






