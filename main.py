from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


# tokenization

# text = "Machine learning is"
# input_ids = tokenizer.encode(text, return_tensors="pt")
# print(f"Input text: {text}")
# print(f"Tokens: {input_ids[0].tolist()}")
# print(f"Tokenized Text: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
#
# # model interface
# import torch
#
# with torch.no_grad():
#     output = model(input_ids)
# predicted_token_id = torch.argmax(output.logits[:, -1, :]).item()
# predicted_token = tokenizer.decode([predicted_token_id])
# print(f"Predicted token: {predicted_token}")

# all together
def generate_text(text, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, early_stopping=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


prompt = "Machine learning is"
example_generated_text = generate_text(prompt, max_length=50)

print(example_generated_text)
