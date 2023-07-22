import os
from dotenv import dotenv_values

from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate

config = dotenv_values(".env")
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']


########## Extractor Prompt##########

prompt_extract = """As a hypothetical doctor, The below text is related to drug labels. extract the main name of the drug put it in 'drug' key. \
DO NOT MENTION THE NAME OF THE PRODUCER  OR THE BRAND NAME. only the obvious main name of the drug. make sure that the name is a valid drug name. DO NOT CHOOSE A RANDOM NAME FOR DRUG NAME.
if the drug is a vitamin supplement only say the name of the vitamin in the drug name field.
extract he dosage and unit of it and put it in 'dose' key. 
extract usage_type choose only between these: (oral,rectal,inject,topical) and put it in 'usage_type' key. like the sample bellow. 
the output must be in JSON format.

Don't explain extra.
example output:

"drug":...,
"dose":
    "value":...,
    "unit":...
"usage_type":...


Text: {text}"""

########## Tasks ##########

prompt_task1 = """
you are given with a dictionary of some drugs interactions. as a medical assisstant, rewrite the description part in a way that the patient could understand. \
make it simple. 

use your expert medical knowledge to choose a value for severity between (minor, moderate, major) to identify how dangerouse is that if the patient take \
those drugs together. IT IS IMPORTANT TO CHOOSE WISELY AND CORRECT. THE INTENSITY OF INTERACTIONS EFFECTS IS THE MOST IMPORTANT.


the output must be in JSON forma. same is input.

input: {interactions}

"""

prompt_task2 = """
you are given a dictionary of name and usage type of some drugs.as a medical assisstant, for each drug, based on usage_type, name at most three most common \
side efects that might happen after taking that drug.

the output must be in JSON format same as input.

never mention 'output' before your answer. only return the json. don't explain extra.
input: {side_effects}

"""

prompt_task3 = """
you are given with a dictionary of some drug names. as a medical assisstant, use your expert knowledge on drug interactions with foods, \
for each drug, NAME at least one and at most three food interactions and FILL food_interactions key. only name keywords. dont explain extra. 

the output must be in JSON format same as input.

input: {food_interactions}

"""

prompt_task4 = """
you are given with a dictionary of some drugs name with dosage_frequency and a list of drug interactions. \
for each drug you have to estimate one or more times of usage as timing key. the number of values in timing is depend on dosage_frequency. \
IT IS IMPORTANT THAT FOR DRUGS IN drug_interactions THE TIMING MUST BE DIFFERENT FROM MAIN DRUG AS MUCH AS POSSIBLE. \
the timing values must be on clock times. all timings for all drugs must differ at least 3 hours from each other.

the output must be in JSON format same as input.
dont mentions "output" name in your answer. only write the json format.

input: {timing} 

"""

#####################

class ChatGPT:
    """
    this class is a wrapper for the OpenAI API
    """

    def __init__(self,
                 model_name="gpt-3.5-turbo-0613",
                 temperature=0,
                 ) -> None:
        
        self.model_name = model_name
        
        self.language_model = OpenAI(
            temperature=temperature, model_name=self.model_name
        )
        
        self.prompt_template_extract = PromptTemplate(
            input_variables=["text"], template=prompt_extract
            )
        
        self.prompt_template_task1 = PromptTemplate(
            input_variables=["interactions"], template=prompt_task1
            )
        
        self.prompt_template_task2 = PromptTemplate(
            input_variables=["side_effects"], template=prompt_task2
            )
        
        self.prompt_template_task3 = PromptTemplate(
            input_variables=["food_interactions"], template=prompt_task3
            )
        
        self.prompt_template_task4 = PromptTemplate(
            input_variables=["timing"], template=prompt_task4
            )


    def drug_extract(self, drug_labels):
        """
        this function extracts the name, unit and dosage of the drug
        """
        return self.language_model(self.prompt_template_extract.format(text=str(drug_labels)))

    def task1(self, interactions):
        """
        this function rewrites interaction description and severity
        """
        return self.language_model(self.prompt_template_task1.format(interactions=interactions))

    def task2(self, side_effects):
        """
        this function generates side effects, based on drugs informations
        """
        return self.language_model(self.prompt_template_task2.format(side_effects=side_effects))

    def task3(self, food_interactions):
        """
        this function generates food interactions, based on drugs informations
        """
        return self.language_model(self.prompt_template_task3.format(food_interactions=food_interactions))
    
    def task4(self, timing):
        """
        this function generates an estimation of drug usage time, based on drugs informations
        """
        return self.language_model(self.prompt_template_task4.format(timing=timing))
