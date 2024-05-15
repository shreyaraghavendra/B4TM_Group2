#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import sys
# Start your coding

import pickle
import pandas as pd

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # Step 1: Load the trained model from the pickle file
    with open(args.model_file, 'rb') as file:
        trained_model = pickle.load(file)
        print("Model loaded successfully.")

    # Step 2: Apply the model to the input file to do the prediction
    # Assume input data is stored in a CSV file
    input_data = pd.read_csv(args.input_file)

    # Select features
    selected_features = [1902, 1956, 1973, 2026, 2058, 2183, 2184, 2207, 2211, 2213, 2547, 2593, 1672, 118, 192, 695, 772, 791, 854, 1061, 1091, 1559, 1643, 1656, 1678, 1900, 2017, 2021, 2024, 2210, 2218, 2750, 2776, 2791, 2817, 2825]
    X_new_selected_features = input_data.iloc[:, selected_features]

    X = X_new_selected_features.to_numpy()

    # Perform the prediction
    predictions = trained_model.predict(X)

    # Map predictions to category labels
    category_labels = ['HER2+', 'HR+', 'Triple Neg']
    predictions_labels = [category_labels[pred] for pred in predictions]

    # Prepare output DataFrame
    output_data = pd.DataFrame({
        'Sample': input_data.iloc[:, 0],  # Assuming the first column contains sample labels
        'Subgroup': predictions_labels
    })

    # Step 3: Write the predictions into the designated output file
    output_data.to_csv(args.output_file, sep='\t', index=False, quoting=1)
    print(f"Predictions written to {args.output_file}")

    # End your coding


if __name__ == '__main__':
    main()
