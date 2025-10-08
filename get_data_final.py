import numpy as np
import nibabel as nib
import math
from sklearn.preprocessing import normalize
from dipy.denoise.noise_estimate import estimate_sigma

def NormalizeVector(v):
    d = np.vdot(v, v)
    return v/d

def calculate_angle_adj(bval, bvec, threshold_angle=30):
    size = len(bval)
    adj = np.zeros((size, size))
    angle_adj = np.zeros((size, size))
    for x in range(0, size):
        normal1 = NormalizeVector(bvec[x, :])  #标准化方向
        b1 = bval[x]
        for y in range(0, size):
            normal2 = NormalizeVector(bvec[y, :])
            b2 = bval[y]
            if(b1 != b2):
                angle_adj[x, y] = 0 #不同shell之间角度为0
                continue
            data_M = np.sqrt(np.sum(normal1 * normal1, axis=0))
            data_N = np.sqrt(np.sum(normal2 * normal2, axis=0))
            cos_theta = np.sum(normal1 * normal2, axis=0) / (data_M * data_N)  #计算反余弦值
            if(cos_theta >= 1):
                theta = np.degrees(0)
            elif(cos_theta <= -1):
                theta = np.degrees(math.pi)
            elif -1 < cos_theta < 0:
                theta = 180 - np.degrees(np.arccos(cos_theta))
            else:
                theta = np.degrees(np.arccos(cos_theta)) #根据反余弦值算出角度
            if(theta < threshold_angle):
                adj[x, y] = 1
            else:
                adj[x, y] = 0
            angle_adj[x, y] = theta
    return angle_adj, adj

def normalized_laplacian(adj):
    """
    :math:`I - D^(-1/2)AD^(-1/2)`
    """
    N = adj.shape[0]
    I = np.eye(N)
    d = adj.sum(axis=-1) ** (-0.5)
    d[np.isinf(d)] = 0.
    return I - d.reshape(-1, 1) * adj * d

def sub_qspace(bvals, bvecs, angle_threshold):
    #标准化
    bvecs = normalize(bvecs, axis=1, norm='l2')

    #生成角度矩阵
    angle_adj, _ = calculate_angle_adj(bvals, bvecs)

    index_sim = []
    index_all = []
    sim = []
    for i in range(0, angle_adj.shape[0]):
        index = np.where(angle_adj[i, :] < angle_threshold)
        if len(index[0]) == 1:
            index_all.append(i)
            sim.append([i, i])
        else:
            min_angle = 180
            min_index = 0
            np.sort(index[0])
            for j in range(0, len(index[0])):
                if i != index[0][j]:
                    if angle_adj[i, index[0][j]] < min_angle:
                        min_angle = angle_adj[i, index[0][j]]
                        min_index = index[0][j]
            sim.append([i, min_index])
    return index_sim, index_all, sim

if __name__ == '__main__':
    patch_size = 16
    offset = 8
    b0_threshold = 15
    snr = '225'
    angle_threshold = 15
    calc_dtype = np.float32
    data_path = ''
    gt_path = ''
    mask_path = ''
    bval_path = ''
    bvec_path = ''

    data = nib.load(data_path).get_fdata()
    data = np.array(data, dtype=np.float32)

    sigma = estimate_sigma(data)
    print('sigma {}'.format(sigma))
    np.save('./sigma_' + str(snr) + '_' + str(angle_threshold) + '.npy', sigma)

    dmri_data = nib.load(gt_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()


    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)

    if bvecs.shape[1] != 3:
        bvecs = np.transpose(bvecs, [1, 0])
    bvals_shell = np.unique(bvals)

    """
    q空间下采样
    """
    sub_bval_0, sub_bval_1 = [], []
    alternative_qsub = {}
    for i in range(0, len(bvals_shell)):
        if i == 0 and bvals_shell[i] < b0_threshold:
            idx = np.argwhere(bvals <= b0_threshold).reshape(-1)
            sub_bval_0 = idx
            sub_bval_1 = idx
            sim = np.concatenate([np.expand_dims(idx, axis=1), np.expand_dims(idx, axis=1)], axis=1)
        else:
            shells = np.argwhere(bvals == bvals_shell[i]).reshape(-1)
            index_sim, index_all, sim_index = sub_qspace(bvals[shells], bvecs[shells, :], angle_threshold)
            sim_index_all = []
            for j in range(0, len(shells)):
                sim_index_all.append(shells[sim_index[j]])
            sim_index_all = np.array(sim_index_all)
            sim = np.concatenate([sim, sim_index_all], axis=0)

    x_size = dmri_data.shape[0]
    y_size = dmri_data.shape[1]
    z_size = dmri_data.shape[2]

    section_all_set = []
    section_sim_set = []
    fa_set = []
    val_set = []

    for i in range(0, x_size, offset):
        for j in range(0, y_size, offset):
            for k in range(0, z_size, offset):
                if np.count_nonzero(mask[i:i + patch_size, j:j + patch_size,
                                    k:k + patch_size]) >= 1:
                    select_data = data[i:i + patch_size, j:j + patch_size, k:k + patch_size, :]
                    if select_data.shape[0] * select_data.shape[1] * select_data.shape[2] == patch_size * patch_size * patch_size:
                        section_all_set.append(select_data[:, :, :, sim[:, 0]])
                        section_sim_set.append(select_data[:, :, :, sim[:, 1]])
                    else:
                        H, W, L, _ = select_data.shape
                        L_size = max(patch_size, L)
                        W_size = max(patch_size, W)
                        H_size = max(patch_size, H)
                        select_data = np.pad(select_data,((0, H_size - H),  (0, W_size - W), (0, L_size - L), (0, 0)))

                        section_all_set.append(select_data[:, :, :, sim[:, 0]])
                        section_sim_set.append(select_data[:, :, :, sim[:, 1]])


    section_all_set = np.array(section_all_set)
    section_sim_set = np.array(section_sim_set)

    print(section_all_set.shape)
    print(section_sim_set.shape)


    np.save('./val_'+str(snr)+'_'+str(angle_threshold)+'.npy', data)
    np.save('./data_'+str(snr)+'_'+str(angle_threshold)+'_all.npy', section_all_set)
    np.save('./data_'+str(snr)+'_'+str(angle_threshold)+'_sim.npy', section_sim_set)
    np.save('./gt_data.npy', dmri_data)









