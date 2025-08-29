from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True


def llava_infer(model, processor, image_paths, prompts, batch_size=4, max_new_tokens=200):
    assert len(image_paths) == len(prompts)
    model.eval()
    device = next(model.parameters()).device
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id or eos_id
    out = []
    with torch.inference_mode():
        for s in range(0, len(image_paths), batch_size):
            paths = image_paths[s:s+batch_size]
            pr = prompts[s:s+batch_size]
            imgs = [Image.open(p).convert("RGB") for p in paths]
            conversations = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}] for t in pr]
            chat_prompts = [processor.apply_chat_template(c, add_generation_prompt=True) for c in conversations]
            inputs = processor(images=imgs, text=chat_prompts, padding=True, return_tensors="pt")
            to_fp16 = {"pixel_values"}
            to_long = {"input_ids", "attention_mask", "position_ids", "pixel_attention_mask"}
            proc = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if k in to_fp16:
                        proc[k] = v.to(device, dtype=torch.float16)
                    elif k in to_long:
                        proc[k] = v.to(device, dtype=torch.long)
                    else:
                        proc[k] = v
                else:
                    proc[k] = v
            gen = model.generate(**proc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id, eos_token_id=eos_id, repetition_penalty=1.1, use_cache=True)
            input_lens = (proc["input_ids"] != pad_id).sum(dim=1)
            for i in range(gen.size(0)):
                cont = gen[i, input_lens[i]:]
                text = processor.decode(cont, skip_special_tokens=True).strip()
                out.append({"image_path": paths[i], "text": text})
    # return out
    return text
    
def load_llava(model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"):
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def main():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    results = llava_infer(
        model, processor,
        image_paths=[
            "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21496_s180.jpg",
            "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21497_s150.jpg",
            "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21497_s150.jpg",
        ],
        prompts=[
            "Please briefly describe the image.",
            "why am i so handsome",
            "Please briefly describe the image.",
        ],
        batch_size=3
    )
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
