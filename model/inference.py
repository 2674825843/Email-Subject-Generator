from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('../saved_model', local_files_only=True)
model     = T5ForConditionalGeneration.from_pretrained('../saved_model', local_files_only=True)

def generate_subject(email_body):
    input_text = "generate subject: " + email_body
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_length=20, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(generate_subject("Hi Alice, I hope you're doing well. I wanted to follow up on our meeting last week regarding the new product launch timeline."))