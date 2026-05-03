import torch
import gradio as gr
from diffusers import StableDiffusionPipeline

# ---------------------------------------------------------------------------
# Configuration — edit these two values to match your LoRA training setup
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_MODEL_PATH = "samburwood23/my-painting-style-lora"  # your HF model repo
TRIGGER_KEYWORD = "samburwood_style"                      # word used during LoRA training
# ---------------------------------------------------------------------------

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, photographic, photo, realistic, "
    "watermark, signature, text, cropped, out of frame, worst quality, "
    "low resolution, jpeg artifacts, ugly, duplicate, morbid, mutilated"
)

pipe = None


def load_pipeline():
    global pipe
    if pipe is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    )

    try:
        pipe.load_lora_weights(LORA_MODEL_PATH)
    except Exception as exc:
        print(f"Warning: could not load LoRA weights from {LORA_MODEL_PATH}: {exc}")

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()


def generate_painting(prompt, steps, guidance, seed):
    load_pipeline()

    full_prompt = f"{TRIGGER_KEYWORD}, {prompt}" if TRIGGER_KEYWORD not in prompt else prompt

    generator = torch.Generator().manual_seed(int(seed)) if seed >= 0 else None

    result = pipe(
        prompt=full_prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Painting Space — Sam Burwood") as demo:
    gr.Markdown(
        "# Painting Space\n"
        "Generate paintings in Sam Burwood's style. "
        f"The trigger keyword **`{TRIGGER_KEYWORD}`** is automatically prepended to every prompt."
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt_box = gr.Textbox(
                label="Prompt",
                placeholder="A misty Scottish loch at dawn, rolling hills, dramatic sky",
                lines=3,
            )

            with gr.Accordion("Advanced Settings", open=False):
                steps_slider = gr.Slider(
                    minimum=20,
                    maximum=50,
                    value=30,
                    step=1,
                    label="Inference Steps",
                )
                guidance_slider = gr.Slider(
                    minimum=7.0,
                    maximum=12.0,
                    value=8.5,
                    step=0.5,
                    label="Guidance Scale",
                )
                seed_number = gr.Number(
                    value=42,
                    label="Seed (-1 for random)",
                    precision=0,
                )

            generate_btn = gr.Button("Generate Painting", variant="primary")

        with gr.Column(scale=3):
            output_image = gr.Image(label="Infinite Blueprint", type="pil")

    generate_btn.click(
        fn=generate_painting,
        inputs=[prompt_box, steps_slider, guidance_slider, seed_number],
        outputs=output_image,
    )

if __name__ == "__main__":
    demo.launch()
