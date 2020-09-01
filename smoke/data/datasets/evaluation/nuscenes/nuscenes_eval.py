import os
import collections
import json
import logging
import subprocess

from smoke.utils.miscellaneous import mkdir

ID_TYPE_CONVERSION = {
    0: 'bicycle',
    1: 'bus',
    2: 'car',
    3: 'construction_vehicle',
    4: 'motorcycle',
    5: 'pedestrian',
    6: 'trailer',
    7: 'truck'
}


def nusc_evaluation(
        eval_type,
        dataset,
        predictions,
        output_folder,
):
    logger = logging.getLogger(__name__)
    if "detection" in eval_type:
        logger.info("performing NuScenes detection evaluation: ")
        do_nusc_detection_evaluation(
            eval_type=eval_type,
            dataset=dataset,
            predictions=predictions,
            output_folder=output_folder,
            logger=logger
        )


def do_nusc_detection_evaluation(eval_type,
                                 dataset,
                                 predictions,
                                 output_folder,
                                 logger
                                ):
    predict_folder = os.path.join(output_folder, 'data')  # only recognize data
    mkdir(predict_folder)

    meta: {
        'use_camera': False,
        'use_lidar': False,
        'use_radar': False,
        'use_map': False,
        'use_external': False
    }
    
    used_inputs = eval_type.split('_')[1:]

    for used_input in used_inputs:
        meta['use_{}'.format(used_input)] = True
    
    results = collections.defaultdict(list)

    for image_id, prediction in predictions.items():
        sample_token = image_id.split()[0]
        result = generate_nusc_3d_detection(sample_token, prediction)
        results[sample_token].extend(result)
    

def generate_nusc_3d_detection(sample_token, prediction):
    # TODO: finish generate detection
    return None