
from transformers import (T5Tokenizer, T5ForConditionalGeneration,
                          Trainer, TrainingArguments)
from datasets import load_dataset
# 用 T5-small 在本地 csv 上微调，生成邮件主题

data_file = '../data/enron_clean.csv'

dataset = load_dataset('csv', data_files=str(data_file))['train']

tokenizer = T5Tokenizer.from_pretrained('t5-small') # 需要联网 tokenizer作用 文字——ID

prefix = "generate subject: "

def preprocess(batch):
    inputs = [prefix + msg for msg in batch['message']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
    labels = tokenizer(batch['subject'], max_length=64, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

model = T5ForConditionalGeneration.from_pretrained('t5-small')

args = TrainingArguments(
    output_dir='../saved_model',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=50,
    save_steps=500,
    eval_strategy='no',
    fp16=True,
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
model.save_pretrained('../saved_model')
tokenizer.save_pretrained('../saved_model')
print(">>> 训练完成，模型已保存到 saved_model")
