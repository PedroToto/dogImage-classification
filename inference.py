import os
from PIL import Image
import io
import json
import pickle
import numpy as np
import torchvision.models as models
import torch.nn as nn
from sagemaker_inference import content_types, decoder
import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from PIL import ImageFile
import requests

import sys
import logging
import argparse
import os
from io import BytesIO

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def model_fn(model_dir):

    logger.info("In model_fn. Model directory is -")
    
    model = models.resnet152(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(
                nn.Linear(num_features, 133),
                nn.LogSoftmax(dim=1)
    )
    
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    return model


def input_fn(request_body, content_type='application/json'):
    
    if content_type == 'application/json':
        logger.info(f'The request body {request_body}')
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        
        image_data = Image.open(requests.get(url, stream=True).raw)
        test_valid_transform = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        logger.info('Process the image')
        img_tensor = test_valid_transform(image_data)
        return img_tensor
    
    raise Exception(f'ContentType is not supported {content_type}')



def predict_fn(input_data, model):
    
    input_data = input_data.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)
        
    return ps


def output_fn(prediction_output, accept='application/json'):
    
    logger.info('List of classes')
    
    classes = {
        0: '103.Mastiff', 
        1: '019.Bedlington_terrier', 
        2: '077.Gordon_setter', 
        3: '089.Irish_wolfhound', 
        4: '110.Norwegian_lundehund', 
        5: '120.Pharaoh_hound', 
        6: '116.Parson_russell_terrier', 
        7: '059.Doberman_pinscher', 
        8: '107.Norfolk_terrier', 
        9: '008.American_staffordshire_terrier', 
        10: '036.Briard', 
        11: '130.Welsh_springer_spaniel', 
        12: '034.Boxer', 
        13: '100.Lowchen', 
        14: '111.Norwich_terrier', 
        15: '062.English_setter', 
        16: '012.Australian_shepherd', 
        17: '016.Beagle', 
        18: '007.American_foxhound', 
        19: '091.Japanese_chin', 
        20: '088.Irish_water_spaniel', 
        21: '046.Cavalier_king_charles_spaniel', 
        22: '069.French_bulldog', 
        23: '129.Tibetan_mastiff', 
        24: '021.Belgian_sheepdog', 
        25: '128.Smooth_fox_terrier', 
        26: '071.German_shepherd_dog', 
        27: '072.German_shorthaired_pointer', 
        28: '028.Bluetick_coonhound', 
        29: '063.English_springer_spaniel', 
        30: '050.Chinese_shar-pei', 
        31: '030.Border_terrier', 
        32: '029.Border_collie', 
        33: '085.Irish_red_and_white_setter', 
        34: '090.Italian_greyhound', 
        35: '064.English_toy_spaniel', 
        36: '014.Basenji', 
        37: '115.Papillon', 
        38: '067.Finnish_spitz', 
        39: '018.Beauceron', 
        40: '076.Golden_retriever',
        41: '052.Clumber_spaniel', 
        42: '124.Poodle', 
        43: '081.Greyhound', 
        44: '114.Otterhound', 
        45: '095.Kuvasz', 
        46: '058.Dandie_dinmont_terrier', 
        47: '093.Kerry_blue_terrier', 
        48: '056.Dachshund', 
        49: '032.Boston_terrier', 
        50: '075.Glen_of_imaal_terrier', 
        51: '023.Bernese_mountain_dog', 
        52: '031.Borzoi', 
        53: '074.Giant_schnauzer', 
        54: '042.Cairn_terrier', 
        55: '092.Keeshond', 
        56: '098.Leonberger', 
        57: '121.Plott', 
        58: '055.Curly-coated_retriever', 
        59: '037.Brittany', 
        60: '006.American_eskimo_dog', 
        61: '053.Cocker_spaniel', 
        62: '044.Cane_corso', 
        63: '096.Labrador_retriever', 
        64: '060.Dogue_de_bordeaux', 
        65: '033.Bouvier_des_flandres', 
        66: '024.Bichon_frise', 
        67: '027.Bloodhound', 
        68: '047.Chesapeake_bay_retriever', 
        69: '043.Canaan_dog', 
        70: '099.Lhasa_apso', 
        71: '112.Nova_scotia_duck_tolling_retriever', 
        72: '127.Silky_terrier', 
        73: '022.Belgian_tervuren', 
        74: '066.Field_spaniel', 
        75: '068.Flat-coated_retriever', 
        76: '039.Bull_terrier', 
        77: '061.English_cocker_spaniel', 
        78: '049.Chinese_crested', 
        79: '011.Australian_cattle_dog', 
        80: '109.Norwegian_elkhound', 
        81: '086.Irish_setter', 
        82: '035.Boykin_spaniel', 
        83: '087.Irish_terrier', 
        84: '108.Norwegian_buhund', 
        85: '026.Black_russian_terrier', 
        86: '118.Pembroke_welsh_corgi', 
        87: '048.Chihuahua', 
        88: '078.Great_dane', 
        89: '126.Saint_bernard', 
        90: '131.Wirehaired_pointing_griffon', 
        91: '084.Icelandic_sheepdog', 
        92: '106.Newfoundland', 
        93: '045.Cardigan_welsh_corgi', 
        94: '102.Manchester_terrier', 
        95: '041.Bullmastiff', 
        96: '025.Black_and_tan_coonhound', 
        97: '054.Collie', 
        98: '057.Dalmatian', 
        99: '082.Havanese', 
        100: '104.Miniature_schnauzer', 
        101: '051.Chow_chow', 
        102: '073.German_wirehaired_pointer', 
        103: '079.Great_pyrenees', 
        104: '015.Basset_hound', 
        105: '125.Portuguese_water_dog', 
        106: '101.Maltese', 
        107: '070.German_pinscher', 
        108: '065.Entlebucher_mountain_dog', 
        109: '113.Old_english_sheepdog', 
        110: '123.Pomeranian', 
        111: '017.Bearded_collie', 
        112: '020.Belgian_malinois', 
        113: '094.Komondor', 
        114:'117.Pekingese', 
        115: '013.Australian_terrier', 
        116: '009.American_water_spaniel', 
        117: '038.Brussels_griffon', 
        118: '005.Alaskan_malamute', 
        119: '097.Lakeland_terrier', 
        120: '133.Yorkshire_terrier', 
        121: '003.Airedale_terrier', 
        122: '040.Bulldog', 
        123: '132.Xoloitzcuintli', 
        124: '105.Neapolitan_mastiff', 
        125: '010.Anatolian_shepherd_dog', 
        126: '119.Petit_basset_griffon_vendeen', 
        127: '001.Affenpinscher', 
        128: '122.Pointer', 
        129: '083.Ibizan_hound', 
        130: '004.Akita', 
        131: '002.Afghan_hound', 
        132: '080.Greater_swiss_mountain_dog'
        }
    

    topk, topclass = prediction_output.topk(3, dim=1)
    result = []
    
    for i in range(3):
        pred = {'Prediction': classes[topclass.numpy()[0][i]], 'Score': f'{topk.numpy()[0][i] * 100}%'}
        logger.info(f'Adding pediction: {pred}')
        result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    
    raise Exception(f'ContenteType is not supported:{accept}')

