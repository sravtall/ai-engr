from llmkit import Client

def main():
    client = Client()

    res = client.create(
        model="gpt-5.4-mini",
        input="What should I do in order to get rid of dandruff?",
        max_output_tokens=500
    )

    print(res.output, res.input_tokens, res.output_tokens, sep="\n")

if __name__ == "__main__":
    main()