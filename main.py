import utils
import process3
import data_extraction
import pandas as pd
import os
import csv

def load_paper(filename):
    """Create a dataframe"""
    if os.path.exists(filename):
        dataframe = pd.read_csv(filename)
        return dataframe
    else:
        #load pdf names

        with open('pdf_pool.csv', 'r') as file:
            reader = csv.reader(file)
            pdf_pool = [row[0] for row in reader]
        dataframe = data_extraction.get_txt_from_pdf(pdf_pool,combine = False, filter_ref = True)

        #store the dataframe
        utils.df_to_csv(dataframe, filename)
        return dataframe

api_key=""
paper_df=load_paper("5paper_parsed.csv")
paper_df_emb= utils.add_emb(paper_df)

process3_object = process3.Process3(api_key, prompt_choice='synthesis', classification=True)

final_table = utils.tabulate_condition(process3_object.model_3(paper_df_emb),"summarized")

#SEctioning
#Threshold for similairty filtering
#Prompt engg. on combined DF- one unique row for each expt