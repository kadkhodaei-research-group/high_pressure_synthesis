import openai
import pandas as pd
import utils
from process1 import Process1
import time

class Process2:
    
    def __init__(self, api_key):
        self.api_key = api_key
        
    def model_2(self, df):

        '''The job of model 2 is to classify a given section as synthesis or not'''
        openai.api_key = self.api_key
    
        response_msgs = []
    
        prev_paper_name = None  
        total_pages = df.groupby(df.columns[0])[df.columns[1]].max() #  For message printing purpose
    
        for _, row in df.iterrows():
            paper_name = row[df.columns[0]]
            page_number = row[df.columns[1]]
            
            # Only print the message when the paper name changes
            if paper_name != prev_paper_name:
                print(f'Processing paper: {paper_name}. Total pages: {total_pages[paper_name]}')
                prev_paper_name = paper_name
    
            context = row['content']
    
            user_msg1 = """
            Context: Single-crystalline MgO(111) and (100) substrates were cleaned by sonicating in acetone and isopropyl alcohol. A 500 nm thick Cu film was deposited on these substrates with a power of 300W in Ar atmosphere (0.6 Pa) by RF magnetron sputtering(Shibaura Mechatoronics Corp., CFS-4ES). During sputtering,the substrate temperature was kept at 500 °C to promote epitaxial growth of the Cu film.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: Yes.
    
            Context:This is closely related to the polycrystalline structure of the Cu foil. Moreover, the widely used Cu foil has a (100) square lattice, which is not suitable for epitaxial graphene growth due to mismatch of the lattice symmetry.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: No.
    
            Context: To begin with, h-BN crystal flakes with thicknesses from 20 to 120 nm were exfoliated onto a 300-nm-thick SiO2/Si substrate by mechanical cleaving1. Epitaxial growth of graphene on h-BN was realized at a low temperature T = ~500 °C through a remote plasma-enhanced CVD (R-PECVD) process.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: Yes.
    
            Context: Graphene domains were synthesized on copper foils by atmospheric pressure chemical vapor deposition (APCVD).Copper foils were first annealed at 1045 °C for 3 h under 300 standard cubic centimeters per minute (sccm) Ar and 50 sccmH 2, and then a mixture of 0.5 sccm CH4 and 500 sccm H2 was applied for graphene growth. After 15.5 min, CH4 and H2 were shut off and 1000 sccm Ar was introduced for 30 s. Finally, the substrates were cooled down from 1045 to 500 °C at a rate of 0.1 °C/s under 300 sccm Ar and 4 sccm H2.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: Yes.
    
            Context: Using this system, the synthesis of large-size (∼4.5 mm2 ) single-crystal monolayer graphene on Cu was achieved as follows. Cu surfaces were ﬁrst cleaned using the electrochemical polishing method (see Supporting Information) and then annealed at 1500 Torr and 1077 °C for 7 h with a H 2 ﬂow rate of 500 sccm. After annealing, the ﬂow rate of H2 was decreased to 70 sccm and the chamber pressure was adjusted to ∼108 Torr by the metering valve.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: Yes.
    
            Context: A recent study has demonstrated an apparent change in growth kinetics when the deposition pressure is raised to atmospheric pressure with few-layer graphene formed, 12 and a further study yielded bilayer graphene growth on copper. 13 Both of these studies demonstrate that the previously observed self-limiting growth of graphene on copper does not apply under all synthesis conditions.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: No.
    
            Context: The growth of graphene by low-pressure CVD starts with the nucleation of a large number of domains across the copper surface. These domains form a 4-fold symmetry lobe structure and continue to grow with time until they merge together to form asingle sheet of polycrystalline graphene.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: No.
    
            Context: Raman spectroscopy is a frequently used technique to characterize graphene. 18,19 Identiﬁable peaks are the G peak at∼1580 cm-1, a radial C-C stretching mode associated with sp 2 bonded carbon; the D peak at ∼1350 cm-1 , a ﬁrst order zone boundary phonon mode associated with defects in the graphene or graphene edges; and the 2D peak at ∼2700 cm-1 , a second order zone boundary phonon mode associated with graphene and graphite.
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: No.
    
            Context: i, X.; Cai, W.; An, J.; Kim, S.; Nah, J.; Yang, D.; Piner, R.;Velamakanni, A.; Jung, I.; Tutuc, E.; Banerjee, S. K.; Colombo, L.; Ruoﬀ,R. S. Large-area synthesis of high-quality and uniform graphene ﬁlms oncopper foils. Science 2009, 324, 1312–1314
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer: No.
    
            Context:
              """
    
            user_msg2 = """
            Question: Does the section contain a comprehensive graphene CVD synthesis with explicit reaction conditions and steps?
            Answer:
            """
    
            attempts = 3
            while attempts > 0:
                try:
                    response = openai.ChatCompletion.create(
                        model='gpt-4',
                        messages=[
                            {"role": "system", "content": "Determine if the section comes from an experimental section for graphene synthesis, which contains information on at least one of the following: substrate type, reaction gas, flow rate, temperature, growth time, no.of layers and no.of domains. Answer will be either Yes or No."},
                            {"role": "user", "content": user_msg1 + context + user_msg2}
                        ]
                    )
                    answers = response.choices[0].message.content
                    break
    
                except Exception as e:
                    attempts -= 1
                    if attempts > 0:
                        print(f"Error: {str(e)}. Retrying in 60 seconds. {attempts} attempts remaining. (model 2)")
                        time.sleep(60)
                    else:
                        print(f"Error: Failed to process paper {paper_name}. Skipping. (model 2)")
                        answers = "No"
                        break
    
            response_msgs.append(answers)
        df = df.copy()
        df.loc[:,'classification'] = response_msgs
    
    
        # The following section creates a new dataframe after applying some transformations to the old dataframe
        # Create a boolean mask for rows where 'results' starts with 'No'
        mask_no = df["classification"].str.startswith("No")

        '''THIS NEXT STEP IS VERY IMPORTANT'''
        '''for a given row- if classification = no - we have the option to either remove it or not 
        we check the row above and we check the row below
        
        if both are no, that means there's a continunity in classification as no- we only remove  that no which is surrounded by nos, as something which not surrounded might be related to a previous synthesis paragraph'''
        
        # Create a boolean mask for rows where both the row above and below have 'No' in the 'results' column
        mask_surrounded_by_no = mask_no.shift(1, fill_value=False) & mask_no.shift(-1, fill_value=False)
        # Combine the two masks with an AND operation
        mask_to_remove = mask_no & mask_surrounded_by_no
        # Invert the mask and filter the DataFrame
        filtered_df = df[~mask_to_remove]
        #combined

        #WE can also ask how many experiments were done - then extract for each expt
        combined_df= utils.combine_main_SI(utils.combine_section(filtered_df))
        #call model 1 to summarized results
        process1_object= Process1(api_key=self.api_key)
        add_table_df = process1_object.model_1(combined_df)
        return add_table_df
        