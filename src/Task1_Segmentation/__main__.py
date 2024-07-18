import argparse
from scripts.inference import run_inference_on_test_images, load_all_models

def main(args):
    # Load models
    models = load_all_models(args.model_dir)

    # Run inference on the test images
    run_inference_on_test_images(models, args.test_image_paths, args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on segmentation task")
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing the trained models")
    parser.add_argument('--test_image_paths', type=str, nargs='+', required=True, help="Paths to the test images")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output segmentations")

    args = parser.parse_args()
    main(args)
