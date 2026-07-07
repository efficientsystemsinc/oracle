"""flax judge_v2 -> pytorch -> ONNX (CPU-deployable artifact).

usage: python3 export_judge_onnx.py   (run in a dir with judge_v2.msgpack; writes judge_v2_flax/ + judge_v2_onnx/ incl. model.onnx + tokenizer)
"""
import numpy as np, torch
from flax import serialization
from transformers import AutoTokenizer, FlaxBertForSequenceClassification, BertForSequenceClassification
fx = FlaxBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3, from_pt=True)
fx.params = serialization.from_bytes(fx.params, open("judge_v2.msgpack","rb").read())
fx.save_pretrained("judge_v2_flax")
pt = BertForSequenceClassification.from_pretrained("judge_v2_flax", from_flax=True)
pt.eval()
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tok.save_pretrained("judge_v2_onnx")
enc = tok("OLD [2026-06-01] [verified] [status] prod is X", "NEW (+14d) [2026-06-15] [asserted] [status] prod is Y",
          return_tensors="pt", padding="max_length", max_length=256, truncation=True)
torch.onnx.export(pt, (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"]),
    "judge_v2_onnx/model.onnx", input_names=["input_ids","attention_mask","token_type_ids"],
    output_names=["logits"], dynamic_axes={k:{0:"b"} for k in ["input_ids","attention_mask","token_type_ids","logits"]},
    opset_version=17)
# sanity: flax vs onnx logits agree
import onnxruntime as ort
sess = ort.InferenceSession("judge_v2_onnx/model.onnx")
o = sess.run(None, {k: enc[k].numpy() for k in ["input_ids","attention_mask","token_type_ids"]})[0]
with torch.no_grad(): p = pt(**enc).logits.numpy()
print("max logit diff", float(np.abs(o-p).max()))
print("ONNX_EXPORT_DONE")
