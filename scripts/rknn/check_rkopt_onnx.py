import onnx
m = onnx.load('/mnt/c/Users/qianc/Projects/yolo_Blood/runs/obb/runs/obb/blood_label_n-2/weights/best.onnx')
print('opset:', [o.version for o in m.opset_import])
print('inputs:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print('outputs:')
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print('  ', o.name, dims)
