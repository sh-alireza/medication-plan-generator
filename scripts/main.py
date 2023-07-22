import json
import logging
import sys
from ast import literal_eval
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from paddleocr import PaddleOCR

from .prompt import ChatGPT


# Attempt to load an OCR model
try:

    logging.info("Loading OCR Model . . . ")
    ocr_model = PaddleOCR(lang='en', use_gpu=False, use_angle_cls=True, det_db_box_thresh=0.5, det_db_thresh=0.5, 
                            show_log=False, use_onnx=False,det_model_dir="models/detection",rec_model_dir="models/recognition",
                            cls_model_dir="models/direction")
except Exception as e:
    
    logging.error(f'Can\'t load object from model ocr, {e}')
    sys.exit()

# constents
MODEL_NAME = "gpt-3.5-turbo-0613"
TEMPERATURE = 0

# Create an instance of the ChatGPT class
chatgpt = ChatGPT(model_name=MODEL_NAME, temperature=TEMPERATURE)

# Define a FastAPI app
app = FastAPI()

@app.post("/drug-extraction")
async def drug_extraction(image: UploadFile = File(...)):

    approx_check_uri = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json"
    user_data = {
            "result":{}
        }
    
    try:
        # Read the image file and convert it to a numpy array
        image_bytes = await image.read()
        image = np.array(Image.open(BytesIO(image_bytes)).convert('RGB'))
        
        # get the text from the given image
        resultocr = ocr_model.ocr(image)
        texts = [res[1][0] for res in resultocr[0] if len(res[1][0]) > 1]
        
        # extract drug name and other useful info from the texts
        result = chatgpt.drug_extract(texts)
        result = literal_eval(result)

        user_data['result']['drug_name'] = " ".join(result['drug'].split(" ")[:2]).lower()
        user_data['result']['value'] = str(result['dose']['value'])
        user_data['result']['unit'] = result['dose']['unit'].lower()
        user_data['result']['usage_type'] = result['usage_type'].lower()

        # check approximate matches in the database
        approx_check_req = requests.get(approx_check_uri,{"term":user_data['result']['drug_name'].strip().lower(),"maxEntries":20})
        
        rxcui_list = list()
        if "candidate" in approx_check_req.json()['approximateGroup']:
            for item in approx_check_req.json()['approximateGroup']['candidate'] :
                rxcui_list.append(item['rxcui'])

        rxcui_list = list(dict.fromkeys(rxcui_list))
        names_list = {}
        
        # get the rxcuid of each drug name
        for rxcuid in rxcui_list:
            name_uri = f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcuid}.json"
            response = requests.get(name_uri)
            try:
                name = response.json()['idGroup']['name'] 
                names_list[str(len(names_list))] ={"id":rxcuid,"name":name}
            except:
                pass

        if len(names_list)==0:
            return {"result":{"status":0,"error":f"couldn't find {user_data['result']['drug_name']} in the database"}}

        user_data['result']['drug_name'] = names_list
        user_data['result']["status"] = 1
        user_data['result']["error"] = ""
        
        return user_data

    except Exception as e:
        logging.error(f'/drug_extraction HTTP:/500, {e}')
        return {"result":{"status":0,"error":e}}

@app.post("/med-plan")
async def med_plan(drugs: list):
    
    uri = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"
    rxcui_ids = []
    side_effects_vars = {}
    food_interactions_vars = {}
    timing_vars = {}
    final_result = {}
    
    try:
        # loop on given list of drugs and make default variables
        for i,drug in enumerate(drugs):
            drug_id = drug['drug_id']
            rxcui_ids.append(drug_id)
            drug_name = drug['drug_name']
            dosage_freq = drug['dosage_frequency']
            dosage = drug['value'] + " " + drug['unit']
            usage_type = drug['usage_type']
            side_effects_vars[drug_name] = {
                "usage_type":usage_type,
                "side_effects":[]
            }
            
            food_interactions_vars[drug_name] = {
                "food_interactions":[]
            }
            
            timing_vars[drug_name] = {
                "dosage_frequency":dosage_freq,
                "usage_type":usage_type,
                "drug_interactions":[],
                "timing":[]
            }
            
            final_result[drug_name]={
                "dosage":dosage,
                "dosage_frequency":dosage_freq,
                "timing":[],
                "usage_type":usage_type,
                "drug_interactions":[],
                "food_interactions":[],
                "side_effects":[]
            }
        
        # get the interaction data from database
        request = requests.get(uri, params={'rxcuis': rxcui_ids})
        
        out = request.json()
        if request.status_code != 200:
            return {"result":{"status":0,"error":f"The drug library API has returned status code {request.status_code}"}}
        
        drug_interactions_vars = {}
        if out['fullInteractionTypeGroup']:
            for i in range(len(out['fullInteractionTypeGroup'][0]['fullInteractionType'])):
                drug_name1 = out['fullInteractionTypeGroup'][0]['fullInteractionType'][i]['minConcept'][0]['name']
                drug_name2 = out['fullInteractionTypeGroup'][0]['fullInteractionType'][i]['minConcept'][1]['name']
                description = out['fullInteractionTypeGroup'][0]['fullInteractionType'][i]['interactionPair'][0]['description']
                timing_vars[drug_name1]['drug_interactions'].append(drug_name2)
                timing_vars[drug_name2]['drug_interactions'].append(drug_name1)
                drug_interactions_vars[str(i)] = {}
                drug_interactions_vars[str(i)]['drugs'] = [drug_name1,drug_name2]
                drug_interactions_vars[str(i)]['severity'] = ""
                drug_interactions_vars[str(i)]['description'] = description
                
        if len(drug_interactions_vars) == 0:
            return {"result":{"status":0,"error":f"Input problems"}}
        
        # chatgpt tasks
        result1 = chatgpt.task1(drug_interactions_vars)
        result2 = chatgpt.task2(side_effects_vars)
        result3 = chatgpt.task3(food_interactions_vars)
        result4 = chatgpt.task4(timing_vars)
        
        # conver to dict format
        try:
            new_result1 = json.loads(result1)
            new_result2 = json.loads(result2)
            new_result3 = json.loads(result3)
            new_result4 = json.loads(result4)
            
        except Exception as e:
            logging.error(f'/med-plan HTTP:/500, {e}')
            return {"result":{"status":0,"error":f"Chat gpt output is not dict"}}
        
        # make a template for the answers
        for i in range(len(new_result1)):
            final_result[new_result1[str(i)]["drugs"][0]]['drug_interactions'].append({
                "name":new_result1[str(i)]["drugs"][1],
                "severity":new_result1[str(i)]["severity"],
                "description":new_result1[str(i)]["description"]
            })
            final_result[new_result1[str(i)]["drugs"][1]]['drug_interactions'].append({
                "name":new_result1[str(i)]["drugs"][0],
                "severity":new_result1[str(i)]["severity"],
                "description":new_result1[str(i)]["description"]
            })
            
        for key in list(new_result2.keys()):
            final_result[key]["side_effects"] = new_result2[key]["side_effects"]
            final_result[key]["food_interactions"] = new_result3[key]["food_interactions"]
            try:
                final_result[key]["food_interactions"].remove("none")
            except:
                pass
            final_result[key]["timing"] = new_result4[key]["timing"]

        return final_result

    except Exception as e:
        logging.error(f'/med-plan HTTP:/500, {e}')
        return {'error': f'{e}', 'status_code': 500}

