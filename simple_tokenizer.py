import re
import os
import urllib.request

SPLIT_PATTERN = r'([,.:;?_!"()\']|--|\s)'


def build_vocab(text: str, special_tokens: list[str]) -> dict[str, int]:
    preprocessed = re.split(SPLIT_PATTERN, text)
    tokens = [item.strip() for item in preprocessed if item.strip()]
    tokens = sorted(list(set(tokens)))
    tokens.extend(special_tokens)
    vocab = {token: idx for idx, token in enumerate(tokens)}
    return vocab


class SimpleTokenizerV1:

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.split_pattern = re.compile(SPLIT_PATTERN)
        self.unk_token = "<unk>"

    def encode(self, text):
        preprocessed = self.split_pattern.split(text)
        tokens = []
        for item in preprocessed:
            if item.strip():
                if item.strip() in self.str_to_int:
                    tokens.append(item.strip())
                else:
                    tokens.append(self.unk_token)
        ids = [self.str_to_int[s] for s in tokens]
        return tokens, ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        # text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        sub_pattern = r"\s+" + SPLIT_PATTERN
        text = re.sub(sub_pattern, r'\1', text)

        # remove spaces after the specified punctuations "' s "
        sub_pattern = r"' s "
        text = re.sub(sub_pattern, "'s ", text)

        return text


if __name__ == "__main__":
    # Load text
    import os
    import urllib.request

    text = ""
    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        file_path = "the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r") as f:
        text = f.read()

    vocab = build_vocab(text, special_tokens=['<unk>'])
    print(f"Vocab size: {len(vocab)}")
    for token, idx in vocab.items():
        print(f"{token}: {idx}")

    tokenizer = SimpleTokenizerV1(vocab)

    sentence = "The verdict, it's there. I am Quang"
    tokens, token_ids = tokenizer.encode(sentence)
    print(f"{sentence}\n Tokens: {tokens}\n Token IDs: {token_ids}")

    # decode
    print(f"{sentence}\n Decoded: {tokenizer.decode(token_ids)}")
