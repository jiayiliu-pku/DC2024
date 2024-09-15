import os
from queue import Queue
import numpy as np

from typing import Tuple

from ultralytics import YOLO
from utils_img import projection_with_angles

import torch

import json

import threading

class CustomThread(threading.Thread):
    def __init__(self, queue, **kwargs):
        super(CustomThread, self).__init__(**kwargs)
        self.__queue = queue

    def run(self):
        while True:
            item = self.__queue.get()
            item[0](*item[1:])
            self.__queue.task_done()

def generate_mip(nii_path,
                 subj_list,
                 mip_path='', 
                 multithread=True, 
                 num_threads=10,
    ) -> None:
    """
    Generate maximum projection for trace classification.
    Args:
        nii_path   : nifti path
        subj_list  : subject list [case0, case1, case2, ...]
        mip_path   : save maximum projection image directory
        param multithread: use multithread or not
        param num_threads: number of threads
    """

    if mip_path != '':
        os.makedirs(mip_path, exist_ok=True)

    if multithread:
        assert mip_path != '', 'MIP path must to be given if using multithread.'
        q_list = [Queue() for _ in range(num_threads)]
        for i in range(num_threads):
            t = CustomThread(q_list[i], daemon=True)
            t.start()

    thread_count = 0
    for subj in subj_list:

        pet_file = os.path.join(nii_path, f'{subj}_0001.nii.gz')
    
        if multithread:
            thread_idx = thread_count % num_threads
            q_list[thread_idx].put((projection_with_angles, pet_file, mip_path))
            thread_count += 1
        else:
            projection_with_angles(pet_file, savepath=mip_path)

    if multithread:
        for q in q_list:
            q.join()


def trace_classification(model_file,
                         input_path,
                         json_file: str='./output/output_cls.json'
    ) -> Tuple[list, list]:
    """
    Trace classification.
    Args:
        model_file: model file
        input_path: the directory contains images to make predictions on
        json_file : save classification results
    Return: 
        cls_retuls: trace labels, 0-fdg, 1-psma.
    """

    # Load model file
    model_cls  = YOLO(model_file)
    device     = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_cls.to(device)

    # Model prediction
    results    = model_cls(input_path)
    cls_result = [[result.probs.top1, os.path.splitext(os.path.basename(result.path))[0]]
                  for result in results]

    # Write classification results
    info = {'fdg': [], 'psma': []}
    for res in cls_result:
        if res[0] == 0:
            info['fdg'].append(res[1])
        elif res[0] == 1:
            info['psma'].append(res[1])
    if os.path.exists(os.path.dirname(json_file)) is False:
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
    print(f'Save classification to {json_file}.')
    with open(f'{json_file}', 'w') as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    fdg_subj = [res[1] for res in cls_result if res[0] == 0]
    psma_subj = [res[1] for res in cls_result if res[0] == 1]


    return fdg_subj, psma_subj





