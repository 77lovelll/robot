import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.makedirs("solutions", exist_ok=True)
#参数
alpha = np.pi / 12
w = 2 * np.pi * 0.5

#末端相对机体系的旋转矩阵 R_BD(t)
def R_BD(t):
    ct, st = np.cos(w * t), np.sin(w * t)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa],
        [st,  ct * ca, -ct * sa],
        [ 0,      sa,      ca]
    ])

#四元数 -> 旋转矩阵
def quat2rot(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * qy ** 2 - 2 * qz ** 2,
         2 * qx * qy - 2 * qz * qw,
         2 * qx * qz + 2 * qy * qw],
        [2 * qx * qy + 2 * qz * qw,
         1 - 2 * qx ** 2 - 2 * qz ** 2,
         2 * qy * qz - 2 * qx * qw],
        [2 * qx * qz - 2 * qy * qw,
         2 * qy * qz + 2 * qx * qw,
         1 - 2 * qx ** 2 - 2 * qy ** 2]
    ])

#旋转矩阵 -> 四元数，并保证 qw >= 0
def rot2quat(R):
    trace = np.trace(R)
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        idx = np.argmax(np.diag(R))
        if idx == 0:  #i = x
            S  = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif idx == 1:  #i = y
            S  = np.sqrt(1 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:  #i = z
            S  = np.sqrt(1 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)

tracking = pd.read_csv('tracking.csv')
t = tracking['t'].values
q_WB = tracking[['qw', 'qx', 'qy', 'qz']].values

#计算
results = []
for ti, qi in zip(t, q_WB):
    R_WB = quat2rot(qi)
    R_WD = R_WB @ R_BD(ti)      #世界 -> 末端
    q_WD = rot2quat(R_WD)
    results.append([ti, *q_WD])

out = pd.DataFrame(results, columns=['t', 'qw', 'qx', 'qy', 'qz'])
out.to_csv('solutions/endEffector_q.csv', index=False, float_format='%.7f')

#画图
plt.figure(figsize=(6, 4))
for lbl in ['qw', 'qx', 'qy', 'qz']:
    plt.plot(out['t'], out[lbl], label=lbl)
plt.xlabel('t [s]')
plt.ylabel('quaternion')
plt.legend()
plt.tight_layout()
plt.savefig('solutions/endEffector_q.png', dpi=300)
print('endEffector_q.csv & .png 已生成')