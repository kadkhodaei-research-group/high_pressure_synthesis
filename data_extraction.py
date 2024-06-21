import PyPDF2
import pandas as pd
import utils
import openai
import time

def identify_headings_with_chatgpt(text):
    """
    Function to interact with ChatGPT to strictly identify headings and their metadata in the provided text.
    Returns a list of tuples containing (heading, start_index, end_index, last_word_before, next_word_after).
    If no clear headings are identified, the response indicates this.
    """
    prompt = f"""
    Here is the text from a PDF page:

    {text}

    Q: Strictly identify the section headings in the text above. For each clearly identified heading, provide the following details in a structured format:
    - Heading title
    - Start index of the heading
    - End index of the heading
    - Last word before the heading and its start index
    - First word after the heading and its start index

    If no clear headings are identifiable, please respond with 'No headings found.'

    Format your response as:
    Heading title, Start index, End index, Last word, Last word index, First word, First word index

    Ensure each entry is on a new line and only includes the requested information. Do not include any additional explanations or text.
    A: 
    """

    attempts = 3
    while attempts > 0:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": "Identify the section headings as clearly and strictly as requested."},
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response['choices'][0]['message']['content']
            if "No headings found" in response_text:
                print("No clear headings identified in the text.")
                return []
            headings = parse_headings_from_response(response_text)
            return headings
        except Exception as e:
            attempts -= 1
            print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining")
            time.sleep(60)

def parse_headings_from_response(response_text):
    """
    Parse the response text to extract headings and the words immediately before and after them.
    Assumes the format of the response is structured for parsing.
    """
    headings = []
    heading_metadata = response_text.strip().split('\n')
    for metadata in heading_metadata:
        parts = metadata.split(',')
        if len(parts) == 3:
            heading = parts[0].strip()
            last_word_before = parts[1].strip()
            next_word_after = parts[2].strip()
            headings.append((heading, last_word_before, next_word_after))
        else:
            print(f"Skipping due to format error: {metadata}")
    return headings


def get_txt_from_pdf(pdf_files, filter_ref=False, combine=False):
    """Convert pdf files to dataframe"""
    data = []

    for pdf in pdf_files:
        with open(pdf, 'rb') as pdf_content:
            pdf_reader = PyPDF2.PdfReader(pdf_content)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()

                if filter_ref:
                    page_text = utils.remove_ref(page_text)

                if not page_text:
                    print(f"No text extracted from page {page_num} of {pdf}. Skipping this page.")
                    continue

                headings = identify_headings_with_chatgpt(page_text)
                sections = []
                if not headings:  # If no headings were identified
                    print(f"No headings identified on page {page_num} of {pdf}. Applying default sectioning.")
                    # Divide the page text into 4 equal sections
                    page_len = len(page_text)
                    part_len = page_len // 4
                    sections = [page_text[i*part_len:(i+1)*part_len] for i in range(4)]
                    sections[-1] += page_text[4*part_len:]  # Adjust the last section to capture any remainder
                else:
                    # Sectioning using the specific sequence of words around headings
                    last_position = 0
                    for heading, last_word_before, next_word_after in headings:
                        # Search for the sequence last_word_before, heading, next_word_after
                        sequence = f"{last_word_before} {heading} {next_word_after}"
                        start_index = page_text.find(sequence, last_position)
                        if start_index == -1:
                            print(f"Sequence not found for heading: {heading}")
                            continue
                        end_index = start_index + len(sequence)

                        sections.append(page_text[last_position:start_index].strip())
                        last_position = end_index

                    # Append the remaining part of the text as the last section
                    sections.append(page_text[last_position:].strip())

                for i, section in enumerate(sections):
                    if utils.count_tokens(section) > 40:  # Ensure section has more than 40 tokens
                        data.append({
                            'file name': pdf,
                            'page number': page_num + 1,
                            'page section': i + 1,
                            'content': section,
                            'tokens': utils.count_tokens(section)
                        })

    df = pd.DataFrame(data)
    if combine:
        df = utils.combine_section(df)
    return df