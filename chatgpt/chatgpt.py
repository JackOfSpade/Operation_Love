import openai
import os
from dotenv import load_dotenv, find_dotenv

# Look for .env file in the current directory and its parent
load_dotenv(find_dotenv())
openai.api_key = os.environ.get("OPENAI_API_KEY")

def get_gpt_response(messages):
    """Send a sequence of messages to ChatGPT and retrieve a response."""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content']

def main():
    # Get the directory of the current script
    script_directory = os.getcwd()

    # Check the directory from which the script is being run
    if script_directory.endswith("chatgpt"):
        directory = os.path.join(script_directory, '..', 'bulk_images_ocr_text.txt')
    else:
        directory = os.path.join(script_directory, 'bulk_images_ocr_text.txt')

    # Read the file
    with open("bulk_images_ocr_text.txt", 'r') as file:
        file_content = file.read()

    # Specify a prefix/suffix for the request
    prefix = "The following text is from a dating profile for a woman: \n"
    suffix = "\n====== End of dating profile ======= Generate a question that's less than or equal to 33 characters based on what you think is the most interesting detail from her profile. The question should not be an invitation to do anything or already answered on her profile."

    messages = prefix + file_content + suffix

    # Start the conversation with the initial message
    messages = [{"role": "user", "content": messages}]
    response = get_gpt_response(messages)

    # Store the assistant's response in the messages list
    messages.append({"role": "assistant", "content": response})

    # Ask a follow-up question
    follow_up_question = "Make sure it's less than or equal to 33 characters."
    messages.append({"role": "user", "content": follow_up_question})
    response = get_gpt_response(messages).strip('"').strip("'").rstrip('.')

    print(response)

    # Write the final response to a file
    with open("chatgpt_response.txt", "w") as output_file:
        output_file.write(response)

if __name__ == "__main__":
    main()
