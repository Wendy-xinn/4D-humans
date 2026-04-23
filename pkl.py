import pickle

# 假设你的文件名是 data.pkl
with open("/home/zhanghongwen/.cache/4DHumans/data/SMPL_to_J19.pkl", "rb") as f:
    joints = pickle.load(f)

print(type(joints))
print(joints.shape)