# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# model_name = "google/flan-t5-small"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# def generate_answer(prompt):

#     inputs = tokenizer(
#         prompt,
#         return_tensors="pt",
#         truncation=True
#     )

#     outputs = model.generate(
#         **inputs,
#         max_length=100
#     )

#     return tokenizer.decode(
#         outputs[0],
#         skip_special_tokens=True
#     )