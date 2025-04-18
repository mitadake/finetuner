import gradio as gr
import pandas as pd
import json
import os
from datasets import Dataset
from unsloth import FastLanguageModel
from typing import Tuple, Dict

# Configuration
SUPPORTED_MODELS = {
    "llama3-8b": "unsloth/llama-3-8b-bnb-4bit",
    "gemma-7b": "unsloth/gemma-7b-bnb-4bit",
}
DEFAULT_HYPERPARAMS = {
    "learning_rate": 2e-5,
    "batch_size": 2,
    "num_epochs": 3,
    "max_seq_length": 2048,
    "lora_r": 16,
}

def validate_and_preprocess_data(file_path: str, file_type: str, input_col: str, output_col: str) -> Dataset:
    """Validate and convert uploaded data to Hugging Face dataset"""
    try:
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "jsonl":
            df = pd.read_json(file_path, lines=True)
        elif file_type == "text":
            with open(file_path) as f:
                texts = f.readlines()
            df = pd.DataFrame({input_col: texts, output_col: texts})
        else:
            raise ValueError("Unsupported file type")
        
        # Basic validation
        assert input_col in df.columns, f"Input column '{input_col}' not found"
        assert output_col in df.columns, f"Output column '{output_col}' not found"
        
        return Dataset.from_pandas(df)
    except Exception as e:
        raise gr.Error(f"Data validation failed: {str(e)}")

def training_job(
    model_name: str,
    dataset: Dataset,
    hyperparams: Dict,
    input_col: str,
    output_col: str,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """Handle the fine-tuning process"""
    try:
        progress(0, desc="Loading Model...")
        
        # Load 4bit model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = SUPPORTED_MODELS[model_name],
            max_seq_length = hyperparams["max_seq_length"],
            dtype = None,
            load_in_4bit = True,
        )
        
        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r = hyperparams["lora_r"],
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_alpha = hyperparams["lora_r"] * 2,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = True,
            random_state = 42,
        )

        # Formatting function for dataset
        def formatting_prompts_func(examples):
            return {
                "text": f"{examples[input_col]}\n\n### Response:\n{examples[output_col]}"
            }

        progress(0.2, desc="Preprocessing Data...")
        dataset = dataset.map(formatting_prompts_func, batched=False)

        # Trainer arguments
        from trl import SFTTrainer
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = hyperparams["max_seq_length"],
            dataset_num_proc = 2,
            packing = False,
            args = {
                "learning_rate": hyperparams["learning_rate"],
                "num_train_epochs": hyperparams["num_epochs"],
                "per_device_train_batch_size": hyperparams["batch_size"],
                "logging_steps": 1,
                "output_dir": "./outputs",
                "optim": "adamw_8bit",
                "seed": 42,
                "fp16": not torch.cuda.is_bf16_supported(),
                "bf16": torch.cuda.is_bf16_supported(),
            },
        )

        progress(0.3, desc="Training...")
        trainer.train()

        # Save model
        progress(0.9, desc="Saving Model...")
        output_dir = f"./outputs/{model_name}_finetuned"
        model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
        
        return f"Training complete! Model saved to {output_dir}", output_dir
    except Exception as e:
        raise gr.Error(f"Training failed: {str(e)}")

# Gradio UI
with gr.Blocks(title="LLaMA/Gemma Fine-Tuning UI") as demo:
    gr.Markdown("# 🦙 LLaMA/Gemma Fine-Tuning UI")
    
    with gr.Tab("Dataset Configuration"):
        dataset_upload = gr.File(label="Upload Dataset", file_types=[".csv", ".jsonl", ".txt"])
        file_type = gr.Dropdown(["csv", "jsonl", "text"], label="File Type")
        input_col = gr.Textbox(label="Input Column Name")
        output_col = gr.Textbox(label="Output Column Name")
        data_preview = gr.Dataframe()
    
    with gr.Tab("Training Configuration"):
        model_choice = gr.Dropdown(list(SUPPORTED_MODELS.keys()), label="Model Selection")
        lr = gr.Number(label="Learning Rate", value=DEFAULT_HYPERPARAMS["learning_rate"])
        batch_size = gr.Slider(1, 16, value=DEFAULT_HYPERPARAMS["batch_size"], step=1, label="Batch Size")
        epochs = gr.Slider(1, 10, value=DEFAULT_HYPERPARAMS["num_epochs"], step=1, label="Epochs")
        lora_r = gr.Slider(8, 64, value=DEFAULT_HYPERPARAMS["lora_r"], step=8, label="LoRA Rank (r)")
    
    with gr.Tab("Training Progress"):
        status = gr.Textbox(label="Training Status")
        loss_plot = gr.LinePlot(label="Training Loss")
        output_dir = gr.Textbox(label="Saved Model Path", visible=False)
        download_btn = gr.Button("Download Model", visible=False)
    
    # Event handlers
    demo.load(
        fn=lambda: DEFAULT_HYPERPARAMS,
        outputs=[lr, batch_size, epochs, lora_r]
    )
    
    start_btn = gr.Button("Start Training", variant="primary")
    start_btn.click(
        fn=training_job,
        inputs=[
            model_choice,
            dataset_upload,
            gr.JSON({
                "learning_rate": lr,
                "batch_size": batch_size,
                "num_epochs": epochs,
                "lora_r": lora_r,
                "max_seq_length": 2048,
            }),
            input_col,
            output_col,
        ],
        outputs=[status, output_dir]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)
