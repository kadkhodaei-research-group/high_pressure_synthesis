import openai
import pandas as pd
import utils
from process2 import Process2

class Process3:
    def __init__(self, api_key, prompt_choice, classification):
        self.api_key = api_key
        self.prompt_choice=prompt_choice
        self.classification=classification  

    def model_3(self, df):
        """Filter the dataframe based on embedding similarity to a prompt."""
        openai.api_key = self.api_key

        prompts = {
            "synthesis": "Provide a detailed description of the experimental section or synthesis method used in this research. This section should cover essential substrate type, gas flow rate, annealing temperature, CH4 flow rate, H2 flow rate, chamber pressure, substrate planar index, annealing time, growth time, Ar/H2 ratio, H2/CH4 ratio, domain size, domain density, No. of domains, nucleation density, edge orientation, CVD reactor type, temperature gradient, no.of layers,"
        }
        prompt_choice=self.prompt_choice

        prompt = prompts.get(prompt_choice)
        
        prompt_result = openai.Embedding.create(model="text-embedding-3-large", input=prompt)
        prompt_emb = prompt_result['data'][0]['embedding']

        '''If the DataFrame does not already have an embedding column'''
        if 'embedding' not in df.columns:
            df = utils.add_emb(df)

        #Plot the similarity scores- make it less arbitrary- for each paper- median (eg)
        '''Add a 'similarity' column to the dataframe by comparing the embeddings'''
        df = utils.add_similarity(df, prompt_emb) 
        '''Filter the dataframe to only include rows with top similarity and their neighbors'''
        df = utils.select_top_neighbors(df)
        classification=self.classification

        '''If the classification parameter is True, pass the dataframe to model_2 for further processing'''
        if classification:
            process2_object= Process2(api_key=self.api_key)
            return process2_object.model_2(df)
        
        return df
        