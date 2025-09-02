from llama_cpp import Llama
import tiktoken
import argparse

llm = Llama.from_pretrained(
	repo_id="sugatoray/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M-GGUF",
	filename="deepseek-coder-v2-lite-instruct-q4_k_m.gguf",#lightweight model i can test on my machine.
)


def chunker(text, chunk_size=500, overlap=50):# we dont want to truncate words.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def preprocess(file_path, token_limit=400):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    needed = tokens[:token_limit]
    result = enc.decode(needed)
    return result

def main():
    parser = argparse.ArgumentParser(description="Ask a question using local LLM.")
    parser.add_argument("--query", type=str, required=True, help="Your question to the LLM")
    parser.add_argument("--data", type=str, default="/content/data.txt", help="Path to the input text file")
    args = parser.parse_args()

    result = preprocess(args.data)

    prompt = f"Context:\n{result}\n\nQuestion: {args.query}\nAnswer: <verbose>:"

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant who quotes "\
            "the requested data from context.Do not use <think> or internal"\
            " reasoning markers."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,#for our use case we don't need creativity and tangents.
		max_tokens = 2000
    )


    print("Prompt tokens:", response['usage']['prompt_tokens'])
    print((response['choices'][0]['message']['content']).strip())

    if name == "__main__":
        main()
