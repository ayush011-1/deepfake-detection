#!/usr/bin/env python3
"""
Deepfake Detection CLI Tool with Explainable AI (Grad-CAM)
Usage:
    python detect.py --input sample_data/ffpp_fake_1.jpg --model ffpp --gradcam
    python detect.py --input sample_data/celebdf_real_1.jpg --model celebdf --gradcam
    python detect.py --input sample_data/verified_real_portrait.jpg --model ensemble
    python detect.py --input sample_data/ --model ensemble
    python detect.py --input path/to/video.mp4 --model ensemble
"""

import os
import sys
import argparse
import cv2
from infer import DeepfakeDetector

def format_verdict(label, confidence, prob_fake, logit):
    color = "\033[91m" if label == 'FAKE' else "\033[92m"
    reset = "\033[0m"
    return f"{color}{label}{reset} (Confidence: {confidence*100:.2f}%, P(Fake)={prob_fake:.4f}, Logit={logit:.2f})"

def process_single_image(detector, img_path, output_dir, save_gradcam=True, save_lime=False):
    print(f"\nAnalyzing: {img_path}")
    result = detector.predict_image(img_path, return_gradcam=save_gradcam, return_lime=save_lime)
    
    primary = result['primary']
    label = primary['label']
    confidence = primary['confidence']
    prob = primary['prob_fake']
    logit = primary['raw_logit']
    arch = primary.get('architecture', 'XceptionNet')
    details = primary.get('ensemble_details', {})

    print(f"  Architecture:   {arch}")
    print(f"  Faces Detected: {result['faces_count']}")
    print(f"  Verdict:        {format_verdict(label, confidence, prob, logit)}")
    if 'signature' in details:
        print(f"  Signature:      {details['signature']}")
    if 'celebdf_prob' in details:
        print(f"  Model Breakdown: Celeb-DF P(Fake)={details['celebdf_prob']:.4f} | FF++ P(Fake)={details['ffpp_prob']:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Save annotated face/image
    annotated_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
    cv2.imwrite(annotated_path, result['annotated_image'])
    print(f"  Saved Detection: {annotated_path}")

    # Save Grad-CAM if available
    if save_gradcam and primary.get('gradcam_overlay') is not None:
        gradcam_path = os.path.join(output_dir, f"{base_name}_gradcam.jpg")
        cv2.imwrite(gradcam_path, primary['gradcam_overlay'])
        print(f"  Saved Grad-CAM:  {gradcam_path}")

    # Save LIME if available
    if save_lime and primary.get('lime_overlay') is not None:
        lime_path = os.path.join(output_dir, f"{base_name}_lime.jpg")
        cv2.imwrite(lime_path, primary['lime_overlay'])
        print(f"  Saved LIME XAI:  {lime_path}")

def process_video(detector, video_path, output_dir):
    print(f"\nAnalyzing Video: {video_path}")
    res = detector.predict_video(video_path, frame_skip=15, max_frames=60)
    print(f"  Frames Analyzed: {res['total_analyzed_frames']}")
    print(f"  Overall Verdict: {res['overall_label']}")
    print(f"  Fake Frames:     {res['fake_percentage']:.1f}%")
    print(f"  Real Frames:     {res['real_percentage']:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection using Machine Learning & Explainable AI (Grad-CAM & LIME)")
    parser.add_argument('--input', '-i', type=str, required=True, help="Path to input image, video, or directory")
    parser.add_argument('--model', '-m', type=str, default='ensemble', 
                        choices=['ensemble', 'celebdf', 'ffpp', 'synthetic', 'svm', 'rf', 'logreg'], 
                        help="Model classifier to use: 'ensemble', 'synthetic', 'celebdf', 'ffpp', 'svm', 'rf', or 'logreg'")
    parser.add_argument('--gradcam', action='store_true', default=True, help="Generate and save Grad-CAM visual heatmaps")
    parser.add_argument('--lime', action='store_true', default=False, help="Generate and save LIME superpixel segment explanations")
    parser.add_argument('--output-dir', '-o', type=str, default='runs/detect', help="Directory to save detection outputs")

    args = parser.parse_args()

    print("=" * 70)
    print("  DEEPFAKE DETECTION: MACHINE LEARNING WITH EXPLAINABLE AI (XAI)")
    print(f"  Model Classifier: {args.model.upper()} | Output: {args.output_dir}")
    print("=" * 70)

    detector = DeepfakeDetector(model_type=args.model)

    if os.path.isdir(args.input):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith(valid_exts)]
        if not files:
            print(f"No valid image files found in {args.input}")
            return
        for f in sorted(files):
            process_single_image(detector, f, args.output_dir, save_gradcam=args.gradcam, save_lime=args.lime)
    elif args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        process_video(detector, args.input, args.output_dir)
    elif os.path.isfile(args.input):
        process_single_image(detector, args.input, args.output_dir, save_gradcam=args.gradcam, save_lime=args.lime)
    else:
        print(f"Error: Path {args.input} does not exist.")

if __name__ == '__main__':
    main()
