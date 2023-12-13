from datasets import load_dataset
dataset_name = "timdettmers/openassistant-guanaco"
dataset = load_dataset(dataset_name)
train_dataset, test_dataset = dataset["train"], dataset["test"]
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
class args:
    model_name_or_path = "EleutherAI/gpt-neo-125m"
    cache_dir = "./cache/"
    model_revision = "main"
    use_fast_tokenizer = True
config = AutoConfig.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=None,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    cache_dir=args.cache_dir,
    use_fast=args.use_fast_tokenizer,
    revision=args.model_revision,
    use_auth_token=None,
)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
    cache_dir=args.cache_dir,
    revision=args.model_revision,
    use_auth_token=None,
)
from transformers import TrainingArguments
from trl import SFTTrainer
# Customize training arguments
output_dir = "./results"
per_device_train_batch_size = 4
gradient_accumulation_steps = 4
optim = "adamw_torch"
save_steps = 10
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    #fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    gradient_checkpointing=True,
)
# Initialize trainer
max_seq_length = 512
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
import gradio as gr
model = model.to("cuda")
def response(message, history):
    ''' Write your own generate function using the fine-tuned model '''
    input_text = f"User: {message}\nBot: {history[-1][1] if history else ''}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response
trainer.train()
gr.ChatInterface(response).launch()
