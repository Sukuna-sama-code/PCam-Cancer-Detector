import json
import os
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

from server import (
    BIO_FEATURES,
    SAMPLE_DIR,
    bio_clf,
    biomarker_init,
    checkpoint_metrics,
    default_threshold,
    list_reference_samples,
    pcam_model,
    test_auc,
    val_auc,
)


class CenterAwareResNet34ONNX(nn.Module):
    """Exports logit and exact analytical Layer-4 Grad-CAM in one ONNX forward pass."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.base.eval()

        w1 = self.base.head[0].weight.detach()
        bn = self.base.head[1]
        bn_scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
        w_eff = w1 * bn_scale.unsqueeze(1)
        w_sum = w_eff[:, :512] + w_eff[:, 512:]
        w2 = self.base.head[4].weight.detach().squeeze(0)

        self.register_buffer("w_sum", w_sum)
        self.register_buffer("w2", w2)

    def forward(self, x: torch.Tensor):
        b = self.base
        feat = b.layer4(b.layer3(b.layer2(b.layer1(b.stem(x)))))
        global_pool = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        center_pool = F.adaptive_avg_pool2d(feat[:, :, 4:8, 4:8], (1, 1)).flatten(1)

        fused = torch.cat([global_pool, center_pool], dim=1)
        h1 = b.head[2](b.head[1](b.head[0](fused)))
        logit = b.head[4](h1)

        mask = (h1 > 0).to(feat.dtype)
        weights = torch.matmul(mask * self.w2, self.w_sum) / 144.0
        cam_raw = torch.relu((feat * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1))
        cam_max = cam_raw.amax(dim=(1, 2), keepdim=True) + 1e-8
        cam = cam_raw / cam_max
        return logit, cam


def compress_initializers_int8(model: onnx.ModelProto) -> onnx.ModelProto:
    """Compress large weight tensors to per-axis-0 symmetric INT8 + Cast/Mul dequant."""
    graph = model.graph
    new_initializers = []
    dequant_nodes = []
    compressed_count = 0

    for idx, init in enumerate(list(graph.initializer)):
        arr = numpy_helper.to_array(init)
        if init.data_type == TensorProto.FLOAT and arr.ndim >= 2 and arr.size > 4096:
            d0 = arr.shape[0]
            flat = arr.reshape(d0, -1)
            scale_1d = np.maximum(np.max(np.abs(flat), axis=1), 1e-8) / 127.0
            scale_shape = (d0,) + (1,) * (arr.ndim - 1)
            scale = scale_1d.reshape(scale_shape).astype(np.float32)
            q = np.clip(np.round(arr / scale), -127, 127).astype(np.int8)

            q_name = f"{init.name}__q8"
            s_name = f"{init.name}__scale"
            f32_name = f"{init.name}__f32"

            new_initializers.append(numpy_helper.from_array(q, name=q_name))
            new_initializers.append(numpy_helper.from_array(scale, name=s_name))

            cast_node = helper.make_node(
                "Cast",
                inputs=[q_name],
                outputs=[f32_name],
                to=TensorProto.FLOAT,
                name=f"dequant_cast_{idx}",
            )
            mul_node = helper.make_node(
                "Mul",
                inputs=[f32_name, s_name],
                outputs=[init.name],
                name=f"dequant_mul_{idx}",
            )
            dequant_nodes.extend([cast_node, mul_node])
            compressed_count += 1
        else:
            new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    orig_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(dequant_nodes + orig_nodes)

    print(f"Compressed {compressed_count} weight initializers to symmetric per-channel INT8.")
    return model


def eval_trees_py(trees, row_values):
    row32 = [float(np.float32(v)) for v in row_values]
    total = 0.0
    for tr in trees:
        node = 0
        f_arr, t_arr, l_arr, r_arr, p_arr = tr["f"], tr["t"], tr["l"], tr["r"], tr["p"]
        while f_arr[node] >= 0:
            if row32[f_arr[node]] <= t_arr[node]:
                node = l_arr[node]
            else:
                node = r_arr[node]
        total += p_arr[node]
    return (total / len(trees)) * 100.0


def main():
    if pcam_model is None:
        raise RuntimeError("pcam_deployment_bundle.pt failed to load from server.py")

    models_dir = os.path.join("static", "models")
    samples_out_dir = os.path.join("static", "samples")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(samples_out_dir, exist_ok=True)

    fp32_tmp_path = os.path.join(models_dir, "_tmp_fp32.onnx")
    q8_path = os.path.join(models_dir, "pcam_resnet34_q8.onnx")

    wrapper = CenterAwareResNet34ONNX(pcam_model).cpu().eval()
    dummy = torch.randn(1, 3, 96, 96, dtype=torch.float32)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            fp32_tmp_path,
            input_names=["input"],
            output_names=["logit", "cam"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    onnx_model = onnx.load(fp32_tmp_path, load_external_data=True)
    onnx_model = compress_initializers_int8(onnx_model)
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, q8_path, save_as_external_data=False)

    for cleanup_name in (fp32_tmp_path, fp32_tmp_path + ".data", q8_path + ".data"):
        if os.path.exists(cleanup_name):
            os.remove(cleanup_name)

    model_size = os.path.getsize(q8_path)
    print(f"Saved {q8_path}: {model_size:,} bytes ({model_size / (1024 * 1024):.2f} MB)")

    # Copy all reference patches into static/samples/
    samples = []
    for item in list_reference_samples():
        src = os.path.join(SAMPLE_DIR, item["filename"])
        dst = os.path.join(samples_out_dir, item["filename"])
        shutil.copy2(src, dst)
        samples.append({**item, "thumb_url": f"./samples/{item['filename']}"})

    manifest = {
        "default_threshold": round(float(default_threshold), 4),
        "val_auc": round(float(val_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "checkpoint_metrics": checkpoint_metrics,
        "model_url": "./models/pcam_resnet34_q8.onnx",
        "model_size_bytes": model_size,
        "samples": samples,
    }
    manifest_path = os.path.join(models_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # Export biomarker metadata + 180 Random Forest trees
    bio_data = biomarker_init()
    imps = sorted(
        [
            {"feature": f.title(), "value": round(float(v * 100.0), 2)}
            for f, v in zip(BIO_FEATURES, bio_clf.feature_importances_)
        ],
        key=lambda x: x["value"],
    )
    trees = []
    for est in bio_clf.estimators_:
        t = est.tree_
        probs = [round(float(v[0][1] / max(1.0, v[0].sum())), 6) for v in t.value]
        trees.append(
            {
                "f": t.feature.tolist(),
                "t": [float(x) for x in t.threshold.tolist()],
                "l": t.children_left.tolist(),
                "r": t.children_right.tolist(),
                "p": probs,
            }
        )

    # Verify exported RF trees reproduce sklearn predictions within < 0.05 percentage points
    import pandas as pd

    for preset_name, preset_vals in bio_data["presets"].items():
        row = [float(preset_vals[f]) for f in BIO_FEATURES]
        sk_prob = float(bio_clf.predict_proba(pd.DataFrame([row], columns=BIO_FEATURES))[0][1]) * 100.0
        py_prob = eval_trees_py(trees, row)
        diff = abs(sk_prob - py_prob)
        print(f"RF preset '{preset_name}': sklearn={sk_prob:.3f}%, exported={py_prob:.3f}%, diff={diff:.4f}%")
        assert diff < 0.05, f"RF export drift {diff} exceeds 0.05 on preset {preset_name}"

    bio_data["importances"] = imps
    bio_data["trees"] = trees
    bio_path = os.path.join(models_dir, "biomarker_rf.json")
    with open(bio_path, "w", encoding="utf-8") as fh:
        json.dump(bio_data, fh, separators=(",", ":"))

    for path in (q8_path, manifest_path, bio_path):
        print(f"  -> {path}: {os.path.getsize(path):,} bytes")
    print(f"  -> Copied {len(samples)} sample images to {samples_out_dir}")


if __name__ == "__main__":
    main()
