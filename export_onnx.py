import torch
from model.tak_net import TakNet


MODEL_PATH = 'model_checkpoints/version_14_62.pt'


info = torch.load(MODEL_PATH)
model = TakNet(**info["params"])
model.load_state_dict(info["weights"])
model.eval()

dummy = torch.randn(256, 25, 6, 6)

torch.onnx.export(
    model,
    dummy,
    "model_releases/release_2_2.onnx",
    input_names=["input"],
    output_names=["policy", "value"],   # same order as the tuple your forward returns
    dynamic_axes={
        "input":  {0: "batch"},
        "policy": {0: "batch"},
        "value":  {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
    dynamo=False,
)