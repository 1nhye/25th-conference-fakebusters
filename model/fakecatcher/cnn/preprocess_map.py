import os
import yaml
import json
import argparse
from tqdm import tqdm
from utils.roi import ROIProcessor, DetectionError
from utils.logging import setup_logging
from ppg.ppg_map import PPG_MAP
from data.fakeforensics import load_fakeforensics_data

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to the config file.")
    parser.add_argument("-l", "--log_path", type=str, required=True, help="Path to the log file.")
    parser.add_argument("-o", "--output_directory", type=str, required=True, help="Path to the output directory.")
    return parser.parse_args()

if __name__ == "__main__":
    
    ## Parse arguments and set up logger, config.
    args = parse_arguments()
    logger = setup_logging(args.log_path)
    
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Config file not found: {args.config_path}")
    
    if not os.path.exists(args.output_directory):
        os.makedirs(os.path.dirname(args.output_directory), exist_ok=True)
    output_path = os.path.join(args.output_directory, "ppg_maps.json")

    with open(args.config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        logger.info("config file was loaded succesfully.")

    ## Load dataset metadata.
    logger.info("Loading video paths and labels...")
    video_paths, true_labels = load_fakeforensics_data(config["meta_data_csv_path"])

    ## Process videos
    logger.info("Processing videos to generate PPG maps...")
    results = []
    
    for i, (video_path, true_label) in enumerate(tqdm(zip(video_paths, true_labels), total=len(video_paths), desc="Processing videos")):        
        logger.info(f"Processing {os.path.basename(video_path)}...")
        
        ## Exception Handling for ROIProcessor initialization.
        try:
            landmarker = ROIProcessor(video_path=video_path, config=config)
        except DetectionError:
            logger.error("     DETECTION ERROR: This Video is TRASH!")
            continue
        except Exception as e:
            logger.error(f"     UNEXPECTED EXCEPTION (ROI_PROCESSOR): {e}")
            continue
        
        ## Exception Handling for ROIProcessor.detect_with_map.
        try:
            transformed_frames, fps = landmarker.detect_with_map()
        except Exception as e:
            logger.error(f"     UNEXPECTED EXCEPTION (DETECT_WITH_MAP): {e}")
            continue
        
        for index, segment in enumerate(transformed_frames):
            
            ## Exception Handling for PPG_MAP.
            try:
                ppg_map = PPG_MAP(segment, fps, config).compute_map()
            except Exception as e:
                logger.error(f"     UNEXPECTED EXCEPTION (PPG_MAP): {e}")
                continue
            
            ## Show successful result summary and save results.
            logger.info(f"     Segment {index}: Success!")
            logger.debug(f"     Shape of ppg map for segment {index}: {ppg_map.shape}")
            results.append({
                'label': true_label,
                'ppg_map': ppg_map.tolist()
            })
            
        if i%100==0:                
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, ensure_ascii=False, indent=4)
            logger.info(f"Preliminary Results saved to {output_path} at iteration {i}")
            
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)
    logger.info(f"Results saved to {output_path}")