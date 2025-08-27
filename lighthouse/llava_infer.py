from PIL import Image, ImageFile
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import traceback
ImageFile.LOAD_TRUNCATED_IMAGES = True


def safe_infer(model, processor, image_path_list, prompt_list, batch_size=4, max_new_tokens=200):
    """
    Safe inference function with comprehensive error handling
    """
    assert len(image_path_list) == len(prompt_list), "image_path_list and prompt_list must align"
    
    model.eval()
    device = next(model.parameters()).device
    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id else eos_id
    
    print(f"Using device: {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    out = []
    
    with torch.inference_mode():
        for s in range(0, len(image_path_list), batch_size):
            paths = image_path_list[s:s+batch_size]
            prompts = prompt_list[s:s+batch_size]
            
            print(f"\nProcessing batch {s//batch_size + 1}: {len(paths)} images")
            
            try:
                # Load images with error handling
                imgs = []
                for i, path in enumerate(paths):
                    try:
                        img = Image.open(path).convert("RGB")
                        print(f"  Image {i+1}: {img.size}")
                        imgs.append(img)
                    except Exception as e:
                        print(f"  Error loading {path}: {e}")
                        # Create dummy image
                        imgs.append(Image.new('RGB', (224, 224), color='black'))
                
                # Build conversations with simpler format
                conversations = []
                for pr in prompts:
                    conv = [
                        {
                            "role": "user", 
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": pr}
                            ]
                        }
                    ]
                    conversations.append(conv)
                
                # Apply chat template
                try:
                    chat_prompts = []
                    for conv in conversations:
                        prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
                        chat_prompts.append(prompt)
                    print(f"  Generated {len(chat_prompts)} chat prompts")
                except Exception as e:
                    print(f"  Error applying chat template: {e}")
                    # Fallback to simple format
                    chat_prompts = [f"USER: {prompt}\nASSISTANT:" for prompt in prompts]
                
                # Process inputs
                try:
                    inputs = processor(
                        images=imgs, 
                        text=chat_prompts, 
                        padding=True, 
                        return_tensors="pt"
                    )
                    print(f"  Processor output keys: {list(inputs.keys())}")
                except Exception as e:
                    print(f"  Error in processor: {e}")
                    traceback.print_exc()
                    continue
                
                # Move to device with careful dtype handling
                # processed_inputs = {}
                # for k, v in inputs.items():
                #     if isinstance(v, torch.Tensor):
                #         print(f"  Tensor '{k}': shape={v.shape}, dtype={v.dtype}")
                        
                #         # Handle different tensor types
                #         if k in ['input_ids', 'attention_mask']:
                #             # Token indices and attention must be long
                #             processed_inputs[k] = v.to(device, dtype=torch.long)
                #         elif 'pixel' in k.lower() or 'image' in k.lower():
                #             # Image data can be float16
                #             processed_inputs[k] = v.to(device, dtype=torch.float16)
                #         elif 'position' in k.lower():
                #             # Position indices should be long
                #             processed_inputs[k] = v.to(device, dtype=torch.long)
                #         else:
                #             # For unknown tensors, try to preserve dtype
                #             if v.dtype in [torch.int32, torch.int64, torch.long]:
                #                 processed_inputs[k] = v.to(device, dtype=torch.long)
                #             else:
                #                 processed_inputs[k] = v.to(device, dtype=torch.float16)
                #     else:
                #         processed_inputs[k] = v


                to_device_fp16 = {"pixel_values"}                 # vision tensors
                to_device_long = {"input_ids", "attention_mask", "position_ids"}  # text tensors
                optional_long  = {"pixel_attention_mask"}         # if present

                processed_inputs = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if k in to_device_fp16:
                            processed_inputs[k] = v.to(device, dtype=torch.float16)
                        elif k in to_device_long or k in optional_long:
                            processed_inputs[k] = v.to(device, dtype=torch.long)
                        else:
                            # Leave ALL other tensors (e.g., batch_num_images, image_counts, sizes) on CPU
                            processed_inputs[k] = v
                    else:
                        processed_inputs[k] = v

                
                print("  Final input summary:")
                for k, v in processed_inputs.items():
                    if isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.shape} {v.dtype}")
                
                # Generate with comprehensive error handling
                try:
                    gen = model.generate(
                        **processed_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_id,
                        eos_token_id=eos_id,
                        repetition_penalty=1.1,
                        use_cache=True,
                    )
                    print(f"  Generated tokens shape: {gen.shape}")
                    
                except Exception as e:
                    print(f"  Generation error: {e}")
                    traceback.print_exc()
                    # Add dummy results for failed batch
                    for path in paths:
                        out.append({"image_path": path, "text": f"Error during generation: {str(e)}"})
                    continue

                # After `gen = model.generate(...)`, do this instead of batch_decode + splitting:
                pad_id = pad_id  # already defined above
                input_ids = processed_inputs["input_ids"]
                # tokens actually belonging to the prompt (per example)
                input_lens = (input_ids != pad_id).sum(dim=1)  # shape: [B]

                cleaned = []
                for i in range(gen.size(0)):
                    # slice out only the generated continuation
                    cont = gen[i, input_lens[i]:]
                    text = processor.decode(cont, skip_special_tokens=True).strip()
                    cleaned.append(text)



                
                ## Decode results
                # try:
                #     decoded = processor.batch_decode(gen, skip_special_tokens=True)
                #     print(f"  Decoded {len(decoded)} responses")
                    
                #     # Clean up responses
                #     cleaned = []
                #     for sdec in decoded:
                #         # Try to cut at the last occurrence of an assistant marker
                #         cut = sdec
                #         for key in ["assistant", "Assistant", "ASSISTANT", "image.assistant"]:
                #             if key in cut:
                #                 cut = cut.split(key)[-1]
                #         cleaned_text = cut.strip()
                        
                #         # Remove common prefixes that might remain
                #         for prefix in [":", ":\n", " :"]:
                #             if cleaned_text.startswith(prefix):
                #                 cleaned_text = cleaned_text[len(prefix):].strip()
                        
                #         cleaned.append(cleaned_text)
                    
                # Add results
                for path, text in zip(paths, cleaned):
                    out.append({"image_path": path, "text": text})
                        
                # except Exception as e:
                #     print(f"  Decoding error: {e}")
                #     for path in paths:
                #         out.append({"image_path": path, "text": f"Error during decoding: {str(e)}"})
                
            except Exception as e:
                print(f"  Batch error: {e}")
                traceback.print_exc()
                # Add error results for the whole batch
                for path in paths:
                    out.append({"image_path": path, "text": f"Batch processing error: {str(e)}"})
        import ipdb; ipdb.set_trace()
    
    return out

# --- Usage with better error handling ---
def main():
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
    try:
        print("Loading model...")
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded successfully")
        
        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("Processor loaded successfully")
        
        # Test inference
        results = safe_infer(
            model, processor,
            image_path_list=[
                "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21496_s180.jpg",
                # "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21496_s180.jpg",
                "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21497_s150.jpg",
                "/home/julian/work/GroundingDINO/demo/CCTV_T10_1223雨天順行_車廂_camip7_sq21497_s150.jpg",
            ],
            prompt_list=[
                "Please briefly describe the image.",
                # "Please briefly describe the image.",
                "why am i so handsome",
                "Please briefly describe the image.",
            ],
            batch_size=3  # Start with batch_size=1 for debugging
        )
        
        print("\n=== RESULTS ===")
        for result in results:
            print(f"Image: {result['image_path'].split('/')[-1]}")
            print(f"Text: {result['text']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
    import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()