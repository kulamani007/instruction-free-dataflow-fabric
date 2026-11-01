"""
Extract exact layer tables and operator sets of reference CNNs (one-off tool).

Writes model/workloads/<model>.csv (one row per conv / linear layer) and
model/workloads/ops.json (operators each model needs, plus totals).  The fit-map
model reads only these files, so it needs numpy alone; this extractor needs
torch, torchvision and ultralytics (YOLOv8 is built from its yaml, no weights).

  pip install torch torchvision ultralytics
  python model/workloads/extract_layers.py

Batch-norm is ignored (folded into the preceding conv at inference).  YOLOv8's
box decoding (DFL, anchors, NMS) is post-processing and is excluded by running
the Detect head in its raw-output mode.
"""
import csv, json, os, operator
import torch, torch.nn as nn, torch.nn.functional as F
from torch.overrides import TorchFunctionMode
import torchvision.models as tvm

OUT = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------- models torchvision lacks ----
def bn_relu(c): return [nn.BatchNorm2d(c), nn.ReLU(inplace=True)]

def mobilenet_v1():
    cfg = [(32,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1),(256,512,2)] + \
          [(512,512,1)]*5 + [(512,1024,2),(1024,1024,1)]
    L = [nn.Conv2d(3, 32, 3, 2, 1, bias=False), *bn_relu(32)]
    for ci, co, s in cfg:
        L += [nn.Conv2d(ci, ci, 3, s, 1, groups=ci, bias=False), *bn_relu(ci),
              nn.Conv2d(ci, co, 1, bias=False), *bn_relu(co)]
    L += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 1000)]
    return nn.Sequential(*L)

def ds_cnn():                        # MLPerf Tiny KWS reference (DS-CNN, 64 ch)
    L = [nn.Conv2d(1, 64, (10, 4), 2, (5, 1), bias=False), *bn_relu(64)]
    for _ in range(4):
        L += [nn.Conv2d(64, 64, 3, 1, 1, groups=64, bias=False), *bn_relu(64),
              nn.Conv2d(64, 64, 1, bias=False), *bn_relu(64)]
    L += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 12)]
    return nn.Sequential(*L)

def yolov8(scale):
    from ultralytics.nn.tasks import DetectionModel
    return DetectionModel(f"yolov8{scale}.yaml", verbose=False)

MODELS = {   # name: (constructor, input shape, task)
    "DS-CNN":            (ds_cnn,                       (1, 1, 49, 10),   "keyword spotting"),
    "MobileNetV1":       (mobilenet_v1,                 (1, 3, 224, 224), "classification"),
    "MobileNetV2":       (lambda: tvm.mobilenet_v2(),   (1, 3, 224, 224), "classification"),
    "MobileNetV3-Small": (lambda: tvm.mobilenet_v3_small(), (1, 3, 224, 224), "classification"),
    "EfficientNet-B0":   (lambda: tvm.efficientnet_b0(), (1, 3, 224, 224), "classification"),
    "SqueezeNet1.1":     (lambda: tvm.squeezenet1_1(),  (1, 3, 224, 224), "classification"),
    "ResNet-18":         (lambda: tvm.resnet18(),       (1, 3, 224, 224), "classification"),
    "ResNet-50":         (lambda: tvm.resnet50(),       (1, 3, 224, 224), "classification"),
    "YOLOv8n":           (lambda: yolov8("n"),          (1, 3, 640, 640), "detection"),
    "YOLOv8s":           (lambda: yolov8("s"),          (1, 3, 640, 640), "detection"),
    "VGG-16":            (lambda: tvm.vgg16(),          (1, 3, 224, 224), "classification"),
}

# ------------------------------------------------------------ op tracing ----
ACT = {F.relu: "relu", F.relu6: "relu6", F.hardtanh: "relu6",F.silu: "silu", F.hardswish: "hardswish",
       F.hardsigmoid: "hardsigmoid", torch.sigmoid: "sigmoid", F.sigmoid: "sigmoid",
       F.gelu: "gelu", torch.relu: "relu", torch.Tensor.relu: "relu", torch.Tensor.sigmoid: "sigmoid"}
ADD = {operator.add, operator.iadd, torch.add, torch.Tensor.add, torch.Tensor.add_,
       torch.Tensor.__add__, torch.Tensor.__iadd__, torch.Tensor.__radd__}
MUL = {operator.mul, torch.mul, torch.Tensor.mul, torch.Tensor.__mul__, torch.Tensor.mul_,
       torch.Tensor.__rmul__}

class OpLog(TorchFunctionMode):
    def __init__(self):
        super().__init__(); self.ops = {}
    def note(self, k): self.ops[k] = self.ops.get(k, 0) + 1
    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        ts = [a for a in args if isinstance(a, torch.Tensor)]
        if func in ACT: self.note("act:" + ACT[func])
        elif func in ADD and len(ts) == 2: self.note("residual add")
        elif func in MUL and len(ts) == 2:
            self.note("channel scale (SE)" if ts[0].shape != ts[1].shape else "elementwise mul")
        elif func in (torch.cat, torch.concat): self.note("concat")
        elif func in (torch.Tensor.chunk, torch.Tensor.split, torch.chunk, torch.split): self.note("split")
        elif func is F.max_pool2d:
            k = args[1] if len(args) > 1 else kwargs.get("kernel_size")
            self.note(f"maxpool {k}x{k}" if isinstance(k, int) else f"maxpool {k}")
        elif func is F.avg_pool2d: self.note("avgpool")
        elif func is F.adaptive_avg_pool2d:
            out = args[1] if len(args) > 1 else kwargs.get("output_size")
            if tuple(args[0].shape[-2:]) != (tuple(out) if isinstance(out, (tuple, list)) else (out, out)):
                self.note("global avgpool" if out in (1, (1, 1), [1, 1]) else "adaptive avgpool")
        elif func is F.interpolate: self.note("upsample")
        elif func is F.dropout: pass
        return func(*args, **kwargs)

def extract(name, ctor, shape):
    model = ctor().eval()
    for m in model.modules():          # YOLO Detect: raw conv outputs, skip box decoding
        if type(m).__name__ == "Detect": m.training = True
    rows = []
    def hook(mod, inp, out):
        x = inp[0]
        if isinstance(mod, nn.Conv2d):
            g, ci, co = mod.groups, mod.in_channels, mod.out_channels
            kh, kw = mod.kernel_size
            kind = "dwconv" if g == ci == co else ("gconv" if g > 1 else "conv")
            ho, wo = out.shape[2], out.shape[3]
            rows.append(dict(kind=kind, kh=kh, kw=kw, stride=mod.stride[0], groups=g, cin=ci, cout=co,
                             hin=x.shape[2], win=x.shape[3], hout=ho, wout=wo,
                             params=sum(p.numel() for p in mod.parameters(recurse=False)),
                             macs=ho * wo * co * (ci // g) * kh * kw))
        elif isinstance(mod, nn.Linear):
            n = x.numel() // x.shape[-1]
            rows.append(dict(kind="linear", kh=1, kw=1, stride=1, groups=1, cin=mod.in_features,
                             cout=mod.out_features, hin=1, win=1, hout=1, wout=1,
                             params=sum(p.numel() for p in mod.parameters(recurse=False)),
                             macs=n * mod.in_features * mod.out_features))
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    log = OpLog()
    with torch.no_grad(), log:
        model(torch.zeros(shape))
    for h in hs: h.remove()
    for i, r in enumerate(rows): r["idx"] = i
    total_params = sum(p.numel() for p in model.parameters())
    return rows, log.ops, total_params

if __name__ == "__main__":
    summary = {}
    fields = ["idx", "kind", "kh", "kw", "stride", "groups", "cin", "cout", "hin", "win",
              "hout", "wout", "params", "macs"]
    for name, (ctor, shape, task) in MODELS.items():
        rows, ops, total = extract(name, ctor, shape)
        kinds = sorted({f"{r['kind']} {r['kh']}x{r['kw']}" for r in rows})
        with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        p = sum(r["params"] for r in rows); m = sum(r["macs"] for r in rows)
        summary[name] = dict(task=task, input=list(shape[1:]), layers=len(rows),
                             params=p, params_incl_bn=total, macs=m,
                             layer_kinds=kinds, ops=dict(sorted(ops.items())))
        print(f"{name:<18} {len(rows):>3} layers  {p/1e6:8.3f} M params (model total {total/1e6:8.3f})"
              f"  {m/1e9:7.3f} GMAC  ops: {', '.join(sorted(ops))}")
    with open(os.path.join(OUT, "ops.json"), "w") as f:
        json.dump(summary, f, indent=1)
