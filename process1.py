import openai
import pandas as pd
import utils
import time

class Process1:

    def __init__(self, api_key):
        self.api_key = api_key

    def model_1(self, df):
        """Model 1 will turn text in dataframe to a summarized reaction condition table.The dataframe should have a column "file name" and a column "content"."""
        response_msgs = []
        openai.api_key = self.api_key
    
        for index, row in df.iterrows():
            column1_value = row[df.columns[0]]
            column2_value = row['content']
    
            max_tokens = 3000
            if utils.count_tokens(column2_value) > max_tokens:
                context_list = utils.split_content(column2_value, max_tokens)
            else:
                context_list = [column2_value]
    
            answers = ''  # Collect answers from chatGPT
            for context in context_list:
                print("Start to analyze paper " + str(column1_value) )
                user_heading = f"This is an experimental section on graphene CVD synthesis from paper {column1_value}\n\nContext:\n{context}"
                user_ending = """Q: Can you summarize the following details in a table:
                        substrate type, gas flow rate, annealing temperature, CH4 flow rate,
                        H2 flow rate, chamber pressure, substrate planar index, annealing time,
                        growth time, Ar/H2 ratio, H2/CH4 ratio, domain size, domain density, No. of domains,
                        nucleation density, edge orientation, CVD reactor type, temperature gradient, and no. of layers?
                        If any information is not provided or you are unsure, use "N/A".
                        Please focus on extracting experimental conditions specifically related to the graphene CVD synthesis and ignore other unrelated details.
                        If multiple conditions are provided for the same parameter, use multiple rows to represent them.
                        If multiple experiments are detailed, for the same paragraph, list the experiment number, or else specify 0 for a single experiment.
                        If multiple units or components are provided for the same factor (e.g., ml/min and sccm for gas flow rate, multiple temperatures and times), include them in the same cell and separate by a comma.
                        The table should have the following columns, all in lowercase:
                        | substrate type | gas flow rate | annealing temperature | CH4 flow rate |
                        | H2 flow rate | chamber pressure | substrate planar index | annealing time |
                        | growth time | Ar/H2 ratio | H2/CH4 ratio | domain size | domain density | No. of domains |
                        | nucleation density | edge orientation | CVD reactor type | temperature gradient | no. of layers | experiment no. |
    
                        A:"""
    
                attempts = 3
                while attempts > 0:
                    try:
                        response = openai.ChatCompletion.create(
                            model='gpt-4',
                            messages=[{
                                "role": "system",
                                "content": """Answer the question as truthfully as possible using the provided context,
                                            and if the answer is not contained within the text below, say "N/A" """
                            },
                                {"role": "user", "content": user_heading + user_ending}]
                        )
                        answer_str = response.choices[0].message.content
                        if not answer_str.lower().startswith("n/a"):
                            answers += '\n' + answer_str
                        break
                    except Exception as e:
                        attempts -= 1
                        if attempts <= 0:
                            print(f"Error: Failed to process paper {column1_value}. Skipping. (model 1)")
                            break
                        print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 1)")
                        time.sleep(60)
    
            response_msgs.append(answers)
        df = df.copy()
        df.loc[:, 'summarized'] = response_msgs
        return df
