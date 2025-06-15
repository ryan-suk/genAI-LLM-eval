import os
import pandas as pd
import openai
from dotenv import load_dotenv
import logging
import time
from tqdm import tqdm

# ---------------------------- Configure Logging ---------------------------- #

logging.basicConfig(
    filename='platform_prompt_type_summarization.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ---------------------------- Load Environment Variables ---------------------------- #

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Verify API key presence
if not openai_api_key:
    logging.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' in the .env file.")
    raise ValueError("OpenAI API key not found. Please set the 'OPENAI_API_KEY' in the .env file.")

# Set OpenAI API key
openai.api_key = openai_api_key

logging.info("Successfully loaded OpenAI API key.")

# ---------------------------- Define File Paths ---------------------------- #

# Path to the output file from the previous summarization step
input_file_path = r'C:\Users\ryann\OneDrive\A_2024 Manuscripts\AI chatbot\Asian American\ai_response_summaries_with_prompt_type.xlsx'

# Path to save the new prompt type summaries
output_file_path = r'C:\Users\ryann\OneDrive\A_2024 Manuscripts\AI chatbot\Asian American\prompt_type_ai_platform_summaries_v11.xlsx'

# ---------------------------- Load the Excel Data ---------------------------- #

try:
    df = pd.read_excel(input_file_path)
    logging.info(f"Successfully loaded data from {input_file_path}.")
except FileNotFoundError:
    logging.error(f"Excel file not found at path: {input_file_path}.")
    raise FileNotFoundError(f"Excel file not found at path: {input_file_path}.")
except Exception as e:
    logging.error(f"An error occurred while loading the Excel file: {e}")
    raise e

# ---------------------------- Data Validation ---------------------------- #

# Ensure that 'prompt_type' and 'ai_platform' exist in the dataframe
if 'prompt_type' not in df.columns or 'ai_platform' not in df.columns:
    logging.error("'prompt_type' or 'ai_platform' column missing from the dataset.")
    raise ValueError("'prompt_type' or 'ai_platform' column missing from the dataset.")

# ---------------------------- Group the Data by Prompt Type and AI Platform ---------------------------- #

grouped = df.groupby(['prompt_type', 'ai_platform'])

# ---------------------------- Helper Functions to Split Text ---------------------------- #

def split_text(text, max_tokens=3000):
    """
    Split text into smaller chunks that fit within the max token limit.
    Args:
        text (str): The text to split.
        max_tokens (int): The maximum number of tokens per chunk.
    Returns:
        List of text chunks.
    """
    sentences = text.split('.')
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        token_count = len(sentence.split())
        if current_length + token_count > max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += token_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# ---------------------------- Define OpenAI Summarization Function ---------------------------- #

def generate_summary(texts, prompt_type, ai_platform, model="gpt-3.5-turbo", retries=3, backoff=5):
    """
    Generates a high-level summary for a group of analysis texts based on prompt_type and ai_platform using OpenAI API.
    Focuses on extracting key themes, patterns, strengths, and differences.

    Args:
        texts (list): List of analysis summaries to concatenate and summarize.
        prompt_type (str): The type of prompt (Ethnic-Specific or Race-Neutral).
        ai_platform (str): The AI platform being analyzed.
        model (str): The OpenAI model to use for summarization.
        retries (int): Number of retry attempts for API calls.
        backoff (int): Seconds to wait before retrying after a failed attempt.

    Returns:
        str: Generated summary or error message.
    """
    combined_text = " ".join(texts)
    text_chunks = split_text(combined_text, max_tokens=3000)  # Split text into chunks

    overall_summary = ""

    for chunk in text_chunks:
        prompt = (
            f"Provide a high-level summary of the cancer screening recommendations by prompt type and AI platform, focusing on potential bias and compliance with cancer screening guidelines. "
            f"Focus on common themes within the prompt type '{prompt_type}', list the types of cancer screening that were not common in race-neutral prompts, "
            f"and the differences across AI platform '{ai_platform}'."
        )

        for attempt in range(1, retries + 1):
            try:
                logging.info(f"Attempt {attempt}: Generating summary for Prompt Type='{prompt_type}', AI Platform='{ai_platform}'.")
                response = openai.ChatCompletion.create(
                    model=model,  # 'gpt-4' or 'gpt-3.5-turbo'
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant with qualitative analysis expertise."},
                        {"role": "user", "content": prompt + "\n\n" + chunk}
                    ],
                    temperature=0.0,  # Adjust for more factual responses
                    max_tokens=500,  # Adjust based on desired summary length
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                summary = response.choices[0].message['content'].strip()
                logging.info(f"Successfully generated summary for chunk in Prompt Type='{prompt_type}', AI Platform='{ai_platform}'.")
                overall_summary += " " + summary
                break  # If successful, break the retry loop
            except openai.error.RateLimitError:
                wait_time = backoff * attempt
                logging.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds... (Attempt {attempt}/{retries})")
                time.sleep(wait_time)
            except openai.error.InvalidRequestError as e:
                logging.error(f"Invalid request on attempt {attempt} for Prompt Type='{prompt_type}', AI Platform='{ai_platform}': {e}")
                return "Summary generation failed due to invalid request."
            except openai.error.OpenAIError as e:
                logging.error(f"OpenAI API error on attempt {attempt} for Prompt Type='{prompt_type}', AI Platform='{ai_platform}': {e}")
                wait_time = backoff * attempt
                logging.info(f"Waiting for {wait_time} seconds before retrying.")
                time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Unexpected error on attempt {attempt} for Prompt Type='{prompt_type}', AI Platform='{ai_platform}': {e}")
                wait_time = backoff * attempt
                logging.info(f"Waiting for {wait_time} seconds before retrying.")
                time.sleep(wait_time)
        else:
            logging.error(f"Failed to generate summary after multiple attempts for chunk in Prompt Type='{prompt_type}', AI Platform='{ai_platform}'.")

    return overall_summary.strip()

# ---------------------------- Summarize by Prompt Type and AI Platform ---------------------------- #

# Initialize an empty DataFrame for the cross-tab table
prompt_types = df['prompt_type'].unique()
ai_platforms = df['ai_platform'].unique()
cross_tab_df = pd.DataFrame(index=prompt_types, columns=ai_platforms)

# Iterate through each group and generate summaries
for (prompt_type, ai_platform), group_df in tqdm(grouped, desc="Processing Groups"):
    all_summaries = group_df['summary'].dropna().astype(str).tolist()

    # Generate summary for each combination of prompt_type and ai_platform
    try:
        summary = generate_summary(all_summaries, prompt_type, ai_platform, model="gpt-3.5-turbo")
    except openai.error.InvalidRequestError:
        logging.warning(f"gpt-4 may not be available. Falling back to gpt-3.5-turbo for Prompt Type='{prompt_type}', AI Platform='{ai_platform}'.")
        summary = generate_summary(all_summaries, prompt_type, ai_platform, model="gpt-3.5-turbo")

    # Fill the cross-tab table
    cross_tab_df.loc[prompt_type, ai_platform] = summary

# ---------------------------- Save the Cross-Tab Table ---------------------------- #

try:
    # Save the cross-tab DataFrame to an Excel file
    cross_tab_df.to_excel(output_file_path)
    logging.info(f"Successfully saved cross-tab summaries to {output_file_path}.")
    print(f"Summarization complete. Cross-tab table saved to '{output_file_path}'.")
except Exception as e:
    logging.error(f"Failed to save cross-tab summaries to {output_file_path}: {e}")
    print(f"An error occurred while saving the cross-tab summaries: {e}")
