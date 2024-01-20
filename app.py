from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os
import gradio as gr



model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


with open("normans_wikipedia.txt", "r", encoding="utf-8") as file:
    data = file.read()

output_dir = "./normans_fine-tuned"
os.makedirs(output_dir, exist_ok=True)


input_ids = tokenizer.encode(data, return_tensors="pt")
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="normans_wikipedia.txt",
    block_size=512,  
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir=output_dir,
    logging_steps=100,
    report_to=[],
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)


try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted by user.")


model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
fine_tuned_model = GPT2LMHeadModel.from_pretrained(output_dir)


def generate_response(user_input):
    user_input_ids = tokenizer.encode(user_input, return_tensors="pt")

    generated_output = fine_tuned_model.generate(
        user_input_ids,
        max_length=100,
        num_beams=5,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.90,
        temperature=0.9
    )

    chatbot_response = tokenizer.decode(
        generated_output[0], skip_special_tokens=True)
    return "Chatbot: " + chatbot_response


iface = gr.Interface(
    fn=generate_response,
    inputs="text",
    outputs="text",
    live=True
)

iface.launch()
