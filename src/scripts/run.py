from llmkit import Client

def main():
    client = Client()

    res = client.create(
        model="claude-sonnet-5",
        input="What should I do in order to get rid of dandruff?",
        max_output_tokens=1000
    )

    print(res.output, res.input_tokens, res.output_tokens, sep="\n")

if __name__ == "__main__":
    main()