import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

# load pre-trained model tokenizer
tokenizerGPT2 = GPT2Tokenizer.from_pretrained('gpt2')
tokenizerGPT2.pad_token = tokenizerGPT2.eos_token

# custom dataset class
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length

        # load and preprocess the dataset
        self.examples = []
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                tokens = self.tokenizer.tokenize(line, truncation=True, max_length=max_length, padding='max_length')
                self.examples.append(tokens)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return { key: torch.tensor(val[item]) for key, val in self.examples[item].items() }

# load pre-trained model
modelGPT2 = GPT2LMHeadModel.from_pretrained('gpt2')

# hyperparameters
batch_size = 2
learning_rate = 5e-5
num_epochs = 3
warmup_steps = 30

# dataloader
dataset = TextDataset("your_dataset.txt", tokenizerGPT2)
dataloader = DataLoader(dataset, batch_size=batch_size)

# optimization
optimizer = AdamW(modelGPT2.parameters(), lr=learning_rate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=1)

# fine-tuning loop
modelGPT2.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        outputs = modelGPT2(**batch, labels=batch['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    print(f'Epoch {epoch + 1} Loss: {loss.item()} ')

print("Fine-tuning finished")