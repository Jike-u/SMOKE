import os
import collections
import json
import logging
import subprocess
from tqdm import tqdm

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
    logger.info('used inputs: {}'.format(used_inputs))
    for used_input in used_inputs:
        meta['use_{}'.format(used_input)] = True
    
    logger.info('start generating results:')
    results = collections.defaultdict(list)
    for image_id, prediction in tqdm(predictions.items()):
        sample_token = image_id.split()[0]
        result = generate_nusc_3d_detection(prediction, sample_token)
        results[sample_token].extend(result)
    
    logger.info('writing results to output folder ...')
    with open(os.path.join(predict_folder, 'eval_result.json'), 'w') as f:
        json.dump({'meta': meta, 'results': results}, f)
    logger.info('finished.')
    return

def generate_nusc_3d_detection(prediction, sample_token):
    # TODO: finish generate detection
    result = []
    if len(prediction) > 0:
        for p in prediction:
            p = p.numpy()
            p = p.round(4)

            detection_name = ID_TYPE_CONVERSION[int(p[0])]

            single_result  = {
                'sample_token': sample_token,
                'translation': (0.0, 0.0, 0.0),        # <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
                'size': (0.0, 0.0, 0.0),              # <float> [3]   -- Estimated bounding box size in m: width, length, height.
                'rotation': (0.0, 0.0, 0.0, 0.0),      # <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
                'velocity': (0.0, 0.0),          # <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
                'detection_name': detection_name,    # <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
                'detection_score': 0.0, # <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
                'attribute_name': ''   # <str>         -- Name of the predicted attribute or empty string for classes without attributes.
            }


    return result