"""CLI with the chatbot for testing purposes"""
import inference as inference


def main():
    query = ""
    while query != "exit":
        query = input("Ask a question: ")
        print(inference.generate_response(query))


if __name__ == "__main__":
    main()
