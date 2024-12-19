# python Data_processing.py --data_path="../../dataset/CoMA/" --Split="Id"  --test_fold=1
from scipy.io import savemat, loadmat
import time
import numpy as np
import trimesh
import os, argparse
from get_landmarks import get_landmarks



parser = argparse.ArgumentParser(description='Arguments for dataset split')
parser.add_argument('--test_fold', type=int)
parser.add_argument('--target', type=bool, default=False,
            help='target=True -> save target data, to generate neutral data don\'t change this argument it is false by default')
parser.add_argument('--data_path', type=str,
            help='path to COMA dataset',default="../../dataset/CoMA/")
parser.add_argument('--Split', type=str,
            help='Split according to Expressions (Expr) or Identities (Id)', default='Id')

args = parser.parse_args()

Split=args.Split
test_fold=args.test_fold
target = args.target
data_path= args.data_path
save_path='./Data/'+ Split +'Split/fold_' + str(test_fold)



if test_fold==1:
    fold=[11, 10,9]
elif test_fold==2:
    fold=[8, 7,6]
elif test_fold==3:
    fold=[5, 4,3]
elif test_fold == 4:
    fold = [2, 1, 0]


points_neutral=[]
points_target=[]
landmarks_target=[]
landmarks_neutral=[]
count=0
neutral_lm=[]
netutral_mesh=[]
for (i_subj, subjdir) in enumerate(os.listdir(data_path)):
    print(i_subj)
    for (i_expr, expr_dir) in enumerate(os.listdir(os.path.join(data_path, subjdir))):
        data_neutral = trimesh.load(os.path.join(data_path, subjdir, 'bareteeth', 'bareteeth.000001.ply'), process=False)
        lands_neutral = get_landmarks(data_neutral.vertices, template='./template/template/template.obj')
                      
        if Split == 'Id':
            helper = i_subj
        elif Split == 'Expr':
            helper = i_expr
        else:
            print('undefined split!  Please chose Id or Expr split')
        if helper in fold:
           cc=0
           for mesh in os.listdir(os.path.join(data_path, subjdir, expr_dir)):
                   
                      
                   data_target = trimesh.load(os.path.join(data_path, subjdir, expr_dir, mesh), process=False)
                   lands_target = get_landmarks(data_target.vertices, template='./template/template/template.obj')
                   points_neutral.append(data_neutral.vertices)
                   points_target.append(data_target.vertices)
                   landmarks_target.append(lands_target)
                   landmarks_neutral.append(lands_neutral)



if not os.path.exists(os.path.join(save_path, 'points_neutral')):
    os.makedirs(os.path.join(save_path, 'points_neutral'))

if not os.path.exists(os.path.join(save_path, 'points_target')):
    os.makedirs(os.path.join(save_path, 'points_target'))

if not os.path.exists(os.path.join(save_path, 'landmarks_target')):
    os.makedirs(os.path.join(save_path, 'landmarks_target'))

if not os.path.exists(os.path.join(save_path, 'landmarks_neutral')):
    os.makedirs(os.path.join(save_path, 'landmarks_neutral'))

for j in range(len(points_neutral)):
            np.save(os.path.join(save_path, 'points_neutral', '{0:08}_frame'.format(j)), points_neutral[j])
            np.save(os.path.join(save_path, 'points_target', '{0:08}_frame'.format(j)), points_target[j])
            np.save(os.path.join(save_path, 'landmarks_target', '{0:08}_frame'.format(j)), landmarks_target[j])
            np.save(os.path.join(save_path, 'landmarks_neutral', '{0:08}_frame'.format(j)), landmarks_neutral[j])
files = []
for r, d, f in os.walk(os.path.join(save_path, 'points_target')):
            for file in f:
                if '.npy' in file:
                    files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'vertices.npy'), files)

files = []
for r, d, f in os.walk(os.path.join(save_path, 'landmarks_target')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'landmarks.npy'), files)

files = []
for r, d, f in os.walk(os.path.join(save_path, 'points_neutral')):
            for file in f:
                if '.npy' in file:
                    files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'points_neutral.npy'), files)

files = []
for r, d, f in os.walk(os.path.join(save_path, 'landmarks_neutral')):
    for file in f:
        if '.npy' in file:
            files.append(os.path.splitext(file)[0])
np.save(os.path.join(save_path, 'landmarks_neutral.npy'), files)





