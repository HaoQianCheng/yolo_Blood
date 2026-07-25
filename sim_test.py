import numpy as np, cv2, sys
from rknn.api import RKNN

onnx = '/mnt/g/WSL/model/best.onnx'
dataset = '/mnt/g/WSL/calib_images/calib.txt'
test_img = '/root/blood_label/test.jpg' if False else '/mnt/g/WSL/calib_images/' + sorted(__import__('os').listdir('/mnt/g/WSL/calib_images'))[0]

def preprocess(img, imgsz=640):
    h, w = img.shape[:2]
    scale = imgsz / max(h, w)
    nw, nh = int(w*scale), int(h*scale)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    c = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    px, py = (imgsz-nw)//2, (imgsz-nh)//2
    c[py:py+nh, px:px+nw] = r
    return np.expand_dims(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), axis=0)

print(f'测试图: {test_img}')
img = cv2.imread(test_img)
inp = preprocess(img)

rknn = RKNN(verbose=False)
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3576',
            disable_rules=['reduce_reshape_op_around_concat'])
rknn.load_onnx(model=onnx)
rknn.build(do_quantization=True, dataset=dataset)
rknn.init_runtime()  # PC simulator

out = rknn.inference(inputs=[inp])
o = out[0][0].T  # (8400, 6)
names = ['cx','cy','w','h','conf','angle']
print('\n=== PC simulator int8 输出 ===')
for i, n in enumerate(names):
    ch = o[:, i]
    print(f'  {n:5s}: min={ch.min():8.3f} max={ch.max():8.3f} mean={ch.mean():8.3f}')
conf = o[:, 4]
print(f'  conf>0.25: {(conf>0.25).sum()}, top1={conf.max():.4f}')
rknn.release()
