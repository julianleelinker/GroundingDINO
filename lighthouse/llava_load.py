from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

def infer(model, processor, image_path_list, prompt_list, batch_size=8, max_new_tokens=200):
    """
    Args:
        model: LlavaOnevisionForConditionalGeneration (already loaded)
        processor: matching AutoProcessor (already loaded)
        image_path_list: list[str]  -> paths to images
        prompt_list:     list[str]  -> user prompts (same length as image_path_list)
        batch_size:      int
        max_new_tokens:  int

    Returns:
        list[{"image_path": str, "text": str}]
    """
    assert len(image_path_list) == len(prompt_list), "image_path_list and prompt_list must align"
    model.eval()
    device = next(model.parameters()).device
    eos_id = processor.tokenizer.eos_token_id

    out = []
    with torch.inference_mode():
        for s in range(0, len(image_path_list), batch_size):
            paths = image_path_list[s:s+batch_size]
            prompts = prompt_list[s:s+batch_size]

            # Load images (RGB)
            imgs = [Image.open(p).convert("RGB") for p in paths]

            # Build one-image chat turns and apply template
            conversations = [[
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": pr}
                ]}
            ] for pr in prompts]
            chat_prompts = [processor.apply_chat_template(c, add_generation_prompt=True) for c in conversations]

            # Tokenize + generate
            inputs = processor(images=imgs, text=chat_prompts, padding=True, return_tensors="pt")
            inputs = {k: (v.to(device, torch.float16) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


            processed_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    if k in ['input_ids', 'attention_mask']:
                        # Text-related tensors must stay as long/int for embeddings
                        processed_inputs[k] = v.to(device, dtype=torch.long)
                    else:
                        # Image-related tensors can be float16
                        processed_inputs[k] = v.to(device, dtype=torch.float16)
                else:
                    processed_inputs[k] = v

            gen = model.generate(
                **processed_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
                repetition_penalty=1.3
            )

            # Decode; trim everything before the assistant’s reply if present
            decoded = processor.batch_decode(gen, skip_special_tokens=True)
            cleaned = []
            for sdec in decoded:
                # Try to cut at the last occurrence of an assistant marker
                cut = sdec
                for key in ["assistant", "image.assistant", "ASSISTANT", "Assistant"]:
                    if key in cut:
                        cut = cut.split(key)[-1]
                cleaned.append(cut.strip())

            out.extend({"image_path": p, "text": t} for p, t in zip(paths, cleaned))
    return out


# --- Minimal usage example (same model as your script) ---
# model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

results = infer(
    model, processor,
    image_path_list=[
        "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21496_s180.jpg",
        "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21497_s150.jpg",
    ],
    prompt_list=["Please briefly describe the image.", "Describe the scene."]
)
print(results)
import ipdb; ipdb.set_trace()
