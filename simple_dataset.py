from torch.utils.data import Dataset
from simple_tokenizer import build_vocab, SimpleTokenizerV1


class SimpleDataset(Dataset):

    def __init__(self, text: str, tokenizer, max_length: int = 128, stride: int = 128):
        self.tokenizer = tokenizer
        self.data = []
        self.labels = []

        # tokenize the entire text
        _tokens, token_ids = self.tokenizer.encode(text)

        # create data and labels
        for i in range(0, len(token_ids) - max_length, stride):
            data = token_ids[i:i + max_length]
            label = token_ids[i + 1:i + max_length + 1]
            self.data.append(data)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


if __name__ == "__main__":
    with open("the-verdict.txt", "r") as f:
        text = f.read()

    vocab = build_vocab(text, special_tokens=['<unk>'])
    tokenizer = SimpleTokenizerV1(vocab)

    dataset = SimpleDataset(text, tokenizer, max_length=32, stride=32)
    x, y = dataset[0]
    print(f"x: {x}")
    print(f"y: {y}")
