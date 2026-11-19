"""Build the browser inference graph from the released PCam checkpoint (no retraining)."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import io
import json
import pickle
import struct
import sys
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

root = Path(__file__).resolve().parents[1] / "pcam-assets"
bundle = Path(sys.argv[1]).read_bytes()
digest = hashlib.sha256(bundle).hexdigest()
if digest != "8b6a73b9c3bd445c16e0c50116524c13fc32ce59fad9dd1b5efbe32cbc9255aa":
    raise ValueError("Expected the released PCam deployment bundle")
archive = zipfile.ZipFile(io.BytesIO(bundle))
prefix = archive.namelist()[0].split("/")[0] + "/"

class StorageReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {("collections","OrderedDict"):OrderedDict,
                   ("torch","FloatStorage"):"float", ("torch","LongStorage"):"long",
                   ("torch._utils","_rebuild_tensor_v2"):lambda storage, offset, shape, *_:(storage,offset,shape)}
        return allowed[(module,name)]
    def persistent_load(self, pid):
        return pid[1:3]

checkpoint = StorageReader(io.BytesIO(archive.read(prefix+"data.pkl"))).load()
state = checkpoint.get("model_state_dict",checkpoint.get("state_dict"))
nodes, initializers, raw = [], [], {}
for key, ((kind, storage), element_offset, shape) in state.items():
    for tag in ("backbone.","model.","module."):
        key = key.removeprefix(tag)
    if kind != "float": continue
    info = archive.getinfo(prefix+"data/"+storage)
    assert info.compress_type == zipfile.ZIP_STORED
    n,extra = struct.unpack_from("<HH",bundle,info.header_offset+26)
    offset = info.header_offset+30+n+extra+element_offset*4
    count = int(np.prod(shape))
    raw[key] = np.frombuffer(bundle,"<f4",count=count,offset=offset).reshape(shape)
    tensor = TensorProto(name=key,data_type=TensorProto.FLOAT,dims=shape,data_location=TensorProto.EXTERNAL)
    for k,v in (("location","pcam.pt"),("offset",offset),("length",count*4)):
        tensor.external_data.add(key=k,value=str(v))
    initializers.append(tensor)

def op(kind, inputs, **attrs):
    output = "v"+str(len(nodes))
    nodes.append(helper.make_node(kind,inputs,[output],**attrs))
    return output

def const(name, array):
    initializers.append(numpy_helper.from_array(np.asarray(array),name))
    return name

def conv(x, prefix, stride=1, pad=1):
    return op("Conv",[x,prefix+".weight"],strides=[stride,stride],pads=[pad]*4)

def bn(x,prefix):
    return op("BatchNormalization",[x]+[prefix+"."+k for k in ["weight","bias","running_mean","running_var"]],epsilon=1e-5)

x = op("Relu",[bn(conv("image","stem.0"),"stem.1")])
for stage,blocks in enumerate([3,4,6,3],1):
    for block in range(blocks):
        p = f"layer{stage}.{block}."
        stride = 2 if stage>1 and block==0 else 1
        residual = bn(conv(x,p+"downsample.0",2,0),p+"downsample.1") if stride==2 else x
        first = op("Relu",[bn(conv(x,p+"conv1",stride),p+"bn1")])
        x = op("Relu",[op("Add",[residual,bn(conv(first,p+"conv2"),p+"bn2")])])
feature = x
crop = op("Slice",[x,const("starts",[4,4]),const("ends",[8,8]),const("axes",[2,3])])
def pool(x): return op("Flatten",[op("GlobalAveragePool",[x])],axis=1)
pooled = op("Concat",[pool(x),pool(crop)],axis=1)
head = op("Gemm",[pooled,"head.0.weight","head.0.bias"],transB=1)
active = op("Relu",[bn(head,"head.1")])
logit = op("Gemm",[active,"head.4.weight","head.4.bias"],transB=1)
nodes.append(helper.make_node("Sigmoid",[logit],["score"]))
# For dual mean pooling, the spatial mean of each channel's gradient is
# (global derivative + central derivative) / 144, including the central crop.
gain = raw["head.4.weight"].flatten()*raw["head.1.weight"]/np.sqrt(raw["head.1.running_var"]+1e-5)
left = op("Slice",["head.0.weight",const("left_start",[0]),const("left_end",[512]),const("head_axis",[1])])
right = op("Slice",["head.0.weight",const("right_start",[512]),const("right_end",[1024]),"head_axis"])
jac = op("Mul",[op("Add",[left,right]),const("head_gain",(gain[:,None]/144).astype("float32"))])
mask = op("Cast",[op("Greater",[active,const("zero",np.float32(0))])],to=TensorProto.FLOAT)
grad = op("Reshape",[op("MatMul",[mask,jac]),const("grad_shape",[1,512,1,1])])
weighted = op("Mul",[feature,grad])
summed = op("ReduceSum",[weighted,const("channel_axis",[1])],keepdims=0)
nodes.append(helper.make_node("Relu",[summed],["cam"]))
def vi(name,shape): return helper.make_tensor_value_info(name,TensorProto.FLOAT,shape)
graph = helper.make_graph(nodes,"PCam",[vi("image",[1,3,96,96])],[vi("score",[1,1]),vi("cam",[1,12,12])],initializers)
model = helper.make_model(graph,opset_imports=[helper.make_opsetid("",13)],ir_version=8)
(root/"network.onnx").write_bytes(model.SerializeToString())
(root/"model.json").write_text(json.dumps({"sha256":digest,"threshold":float(checkpoint.get("optimal_threshold",checkpoint.get("threshold",0.126))),"size":len(bundle),"samples":sorted(p.name for p in root.glob("*patch*.png"))},indent=2))
