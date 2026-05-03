import gradio as gr
import torch
from diffusers import DiffusionPipeline

# Load the Base Model
model_id = "black-forest-labs/FLUX.1-schnell" # Schnell is faster for testing
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")

# Load YOUR specific Art Style
# This points directly to the repo you just sent me!
pipe.load_lora_weights("Samburwood22312/artGenerator")

def generate(prompt, trigger_word="TOK"):
    # We combine the trigger word with the user's prompt
    full_prompt = f"{trigger_word} style, {prompt}"
    
    # Generate the image
    image = pipe(
        full_prompt, 
        num_inference_steps=4, # Schnell only needs 4 steps
        guidance_scale=3.5
    ).images[0]
    return image

# Gradio Interface Setup
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Sam Burwood's Infinite Blueprint Generator")
    gr.Markdown("Enter a concept to see it rendered in Sam's architectural abstract style.")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Prompt", 
                placeholder="A futuristic city skyline..."
            )
            trigger_input = gr.Textbox(
                label="Trigger Word", 
                value="TOK", 
                info="Use the word you used in training (default is TOK)"
            )
            btn = gr.Button("Generate Blueprint")
        with gr.Column():
            output_img = gr.Image(label="Generated Art")
    
    btn.click(fn=generate, inputs=[prompt_input, trigger_input], outputs=output_img)

demo.launch()



