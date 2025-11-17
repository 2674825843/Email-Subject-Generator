from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('saved_model', local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained('saved_model', local_files_only=True)

def generate_subject(body):
    inputs = tokenizer("generate subject: " + body, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=20, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 示例
print(generate_subject(
    "Alas, how could this be? I should have been a 17-year-old student from Hengshui, Hebei, living in a dormitory with six others, waking up at 5 a.m. every morning to run exercises under the diligent guidance of my teachers, encountering stern school principals inspecting our uniforms on the way to class; spending every minute of the day at school memorizing texts and solving problems; and not returning to my dorm until 10:30 p.m. after evening self-study, rushing to shower. Instead, I’m now this idle DK in Japan, with nothing to do every afternoon at 3 p.m., and on weekend evenings, wearing a yukata to watch fireworks with my handsome and charming lover. Who stole my life as a high school student in Hebei? Please return it to me as soon as possible, okay?"
))