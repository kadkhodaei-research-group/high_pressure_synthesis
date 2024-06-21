import openai
import requests
import PyPDF2
import re
import csv
import os
import requests
import pandas as pd
import tiktoken
import time
from io import StringIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np
import ast
from matplotlib import pyplot as plt

api_key=""

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def remove_ref(pdf_text):
    """This function removes reference section from a given PDF text. It uses regular expressions to find the index of the words to be filtered out."""
    # Regular expression pattern for the words to be filtered out
    pattern = r'(REFERENCES|Acknowledgment|ACKNOWLEDGMENT)'
    match = re.search(pattern, pdf_text)

    if match:
        # If a match is found, remove everything after the match
        start_index = match.start()
        clean_text = pdf_text[:start_index].strip()
    else:
        # Define a list of regular expression patterns for references
        reference_patterns = [
            '\[[\d\w]{1,3}\].+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5};','\([\d\w]{1,3}\).+?[\d]{3,5}\.','\[[\d\w]{1,3}\].+?[\d]{3,5},',
            '\([\d\w]{1,3}\).+?[\d]{3,5},','\[[\d\w]{1,3}\].+?[\d]{3,5}','[\d\w]{1,3}\).+?[\d]{3,5}\.','[\d\w]{1,3}\).+?[\d]{3,5}',
            '\([\d\w]{1,3}\).+?[\d]{3,5}','^[\w\d,\.– ;)-]+$',
        ]

        # Find and remove matches with the first eight patterns
        for pattern in reference_patterns[:8]:
            matches = re.findall(pattern, pdf_text, flags=re.S)
            pdf_text = re.sub(pattern, '', pdf_text) if len(matches) > 500 and matches.count('.') < 2 and matches.count(',') < 2 and not matches[-1].isdigit() else pdf_text

        # Split the text into lines
        lines = pdf_text.split('\n')

        # Strip each line and remove matches with the last two patterns
        for i, line in enumerate(lines):
            lines[i] = line.strip()
            for pattern in reference_patterns[7:]:
                matches = re.findall(pattern, lines[i])
                lines[i] = re.sub(pattern, '', lines[i]) if len(matches) > 500 and len(re.findall('\d', matches)) < 8 and len(set(matches)) > 10 and matches.count(',') < 2 and len(matches) > 20 else lines[i]

        # Join the lines back together, excluding any empty lines
        clean_text = '\n'.join([line for line in lines if line])

    return clean_text

def combine_section(df):
    """Merge sections, page numbers, add up content, and tokens based on the pdf name."""
    aggregated_df = df.groupby('file name').agg({
        'content': aggregate_content,
        'tokens': aggregate_tokens
    }).reset_index()

    return aggregated_df


def aggregate_content(series):
    """Join all elements in the series with a space separator. """
    return ' '.join(series)


def aggregate_tokens(series):
    """Sum all elements in the series."""
    return series.sum()


def extract_title(file_name):
    """Extract the main part of the file name. """
    title = file_name.split('_')[0]
    return title.rstrip('.pdf')


def combine_main_SI(df):
    """Create a new column with the main part of the file name, group the DataFrame by the new column,
    and aggregate the content and tokens."""
    df['main_part'] = df['file name'].apply(extract_title)
    merged_df = df.groupby('main_part').agg({
        'content': ''.join,
        'tokens': sum
    }).reset_index()

    return merged_df.rename(columns={'main_part': 'file name'})


def df_to_csv(df, file_name):
    """Write a DataFrame to a CSV file."""
    df.to_csv(file_name, index=False, escapechar='\\')


def csv_to_df(file_name):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(file_name)

def tabulate_condition(df,column_name):
    """This function converts the text from a ChatGPT conversation into a DataFrame.
    It also cleans the DataFrame by dropping additional headers and empty lines.    """

    table_text = df[column_name].str.cat(sep='\n')

    # Remove leading and trailing whitespace
    table_text = table_text.strip()

    # Split the table into rows
    rows = table_text.split('\n')

    # Extract the header row and the divider row
    header_row, divider_row, *data_rows = rows

    # Extract column names from the header row

    column_names = [
            'substrate type', 'gas flow rate', 'annealing temperature', 'CH4 flow rate',
            'H2 flow rate', 'chamber pressure', 'substrate planar index', 'annealing time',
            'growth time', 'Ar/H2 ratio', 'H2/CH4 ratio', 'domain size', 'domain density',
            'No. of domains', 'nucleation density', 'edge orientation', 'CVD reactor type',
            'temperature gradient', 'no. of layers']

    # Create a list of dictionaries to store the table data
    data = []

    # Process each data row
    for row in data_rows:

        # Split the row into columns
        columns = [col.strip() for col in row.split('|') if col.strip()]

        # Create a dictionary to store the row data
        row_data = {col_name: col_value for col_name, col_value in zip(column_names, columns)}

        # Append the dictionary to the data list
        data.append(row_data)

    df = pd.DataFrame(data)

    return df


def split_content(input_string, tokens):
    """Splits a string into chunks based on a maximum token count. """

    MAX_TOKENS = tokens
    split_strings = []
    current_string = ""
    tokens_so_far = 0

    for word in input_string.split():
        # Check if adding the next word would exceed the max token limit
        if tokens_so_far + count_tokens(word) > MAX_TOKENS:
            # If we've reached the max tokens, look for the last dot or newline in the current string
            last_dot = current_string.rfind(".")
            last_newline = current_string.rfind("\n")

            # Find the index to cut the current string
            cut_index = max(last_dot, last_newline)

            # If there's no dot or newline, we'll just cut at the max tokens
            if cut_index == -1:
                cut_index = MAX_TOKENS

            # Add the substring to the result list and reset the current string and tokens_so_far
            split_strings.append(current_string[:cut_index + 1].strip())
            current_string = current_string[cut_index + 1:].strip()
            tokens_so_far = count_tokens(current_string)

        # Add the current word to the current string and update the token count
        current_string += " " + word
        tokens_so_far += count_tokens(word)

    # Add the remaining current string to the result list
    split_strings.append(current_string.strip())

    return split_strings


def table_text_clean(text):
    """Cleans the table string and splits it into lines."""

    # Pattern to find table starts
    pattern = r"\|\s*substrate type\s*.*"

    # Use re.finditer() to find all instances of the pattern in the string and their starting indexes
    matches = [match.start() for match in re.finditer(pattern, text, flags=re.IGNORECASE)]

    # Count the number of matches
    num_matches = len(matches)

    # Base table string
    table_string = """| substrate type | gas flow rate | annealing temperature | CH4 flow rate | H2 flow rate |
| chamber pressure | substrate planar index | annealing time | growth time | Ar/H2 ratio |
| H2/CH4 ratio | domain size | domain density | No. of domains | nucleation density |
| edge orientation | CVD reactor type | temperature gradient | no. of layers | experiment no. |
|---------------|---------------|------------------------|--------------|--------------|
|------------------|-----------------------|------------------|----------------|-------------|
|----------------|----------------|------------------|----------------|-------------------|
|-------------------|-----------------|--------------------|-------------------|"""

    if num_matches == 0:  # No table in the answer
        print("No table found in the text: " + text)
        splited_text = ''

    else:  # Split the text based on header
        splited_text = ''
        for i in range(num_matches):
            # Get the relevant table slice
            splited = text[matches[i]:matches[i + 1]] if i != (num_matches - 1) else text[matches[i]:]

            # Remove the text after last '|'
            last_pipe_index = splited.rfind('|')
            splited = splited[:last_pipe_index + 1]

            # Remove the header and \------\
            pattern_dash = r"-(\s*)\|"
            match = max(re.finditer(pattern_dash, splited), default=None, key=lambda x: x.start())

            if not match:
                print("'-|' pattern not found.")
            else:
                first_pipe_index = match.start()
                splited = '\n' + splited[(first_pipe_index + len('-|\n|') - 1):]  # Start from "\"

            splited_text += splited

    table_string = table_string + splited_text
    return table_string

def add_similarity(df, given_embedding):
    """Adds a 'similarity' column to a dataframe based on cosine similarity with a given embedding."""
    def calculate_similarity(embedding):
        # Check if embedding is a string and convert it to a list of floats if necessary
        if isinstance(embedding, str):
            embedding = [float(x) for x in embedding.strip('[]').split(',')]
        return cosine_similarity([embedding], [given_embedding])[0][0]

    df['similarity'] = df['embedding'].apply(calculate_similarity)
    return df


def select_top_neighbors(df):
    """
    Selects top neighbors based on the median similarity score from the dataframe.
    Also shows a plot of similarity scores.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['similarity'], bins=30, alpha=0.75)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.show()
    
    # Calculating the median similarity score
    median_similarity = df['similarity'].median()
    print(f"Median similarity score: {median_similarity}")

    # Select rows where the similarity score is above the median
    above_median = df[df['similarity'] > median_similarity]

    # Sort dataframe by 'file name' and 'similarity' in descending order
    above_median.sort_values(['file name', 'similarity'], ascending=[True, False], inplace=True)

    # Group dataframe by 'file name' and add neighboring rows (one above and one below) to the selection
    neighbors = []
    for name, group in above_median.groupby('file name'):
        indices = group.index
        extended_indices = [i for idx in indices for i in (idx - 1, idx + 1) if 0 <= i < df.shape[0]]
        neighbors.extend(extended_indices)

    # Create a new dataframe with only the selected rows
    selected_indices = set(above_median.index).union(set(neighbors))
    selected_df = df.loc[list(selected_indices)]  # Convert set to list here

    return selected_df


def add_emb(df):
    """Adds an 'embedding' column to a dataframe using OpenAI API."""
    openai.api_key = api_key
    if 'embedding' in df.columns:
        print('The dataframe already has embeddings. Please double check.')
        return df

    embed_msgs = []
    for _, row in df.iterrows():
        context = row['content']
        context_emb = openai.Embedding.create(model="text-embedding-3-large", input=context)
        embed_msgs.append(context_emb['data'][0]['embedding'])

    df = df.copy()
    df.loc[:, 'embedding'] = embed_msgs

    return df

#To be used in the runtime later
def check_system(syn_df, paper_df, paper_df_emb):
    """Check if the data is correctly loaded"""
    # check if openai.api_key is not placeholder
    if openai.api_key  == "Add Your OpenAI API KEY Here.":
        print("Error: Please replace openai.api_key with your actual key.")
        return False

    # check if 'content' column exists in syn_df
    if 'content' not in syn_df.columns:
        print("Error: 'content' column is missing in syn_df.")
        return False

    # check if 'paper_df' has at least four columns
    expected_columns = ['file name', 'page number', 'page section', 'content']
    if not all(col in paper_df.columns for col in expected_columns):
        print("Error: 'paper_df' should have these columns: 'file name', 'page number', 'page section', 'content'.")
        return False

    # check if 'embedding' column exists in paper_df_emb
    if 'embedding' not in paper_df_emb.columns:
        print("Error: 'embedding' column is missing in paper_df_emb.")
        return False

    print("All checks passed.")
    return True