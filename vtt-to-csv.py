#! /usr/bin/env python3

import webvtt
import csv
import os
import sys
import re
import argparse


def convert_vtt_to_csv(vtt_file, csv_file):
    """
    Convert a WebVTT file to a CSV file.
    """
    # Check if the input file exists
    if not os.path.isfile(vtt_file):
        print(f"Error: The file {vtt_file} does not exist.")
        sys.exit(1)

    # Read the WebVTT file
    vtt = webvtt.read(vtt_file)

    # Open the CSV file for writing
    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        # Write the header
        csvwriter.writerow(["Text", "Speaker", "Start Time", "End Time"])

        # Write each caption to the CSV file
        for caption in vtt:
            print(caption.lines, caption.text)
            # The speaker is the first part of the text before the colon
            # If the text does not contain a colon, set speaker to 'Unknown'

            if ":" in caption.text:
                speaker, text = caption.text.split(":", 1)
                speaker = speaker.strip()
                text = text.strip()
            else:
                speaker = "Unknown"
                text = caption.text.strip()

            start_time = caption.start
            end_time = caption.end
            csvwriter.writerow([text, speaker, start_time, end_time])
    print(f"Converted {vtt_file} to {csv_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Convert WebVTT files to CSV format.")
    parser.add_argument("input", type=str, help="Input WebVTT file")
    parser.add_argument("output", type=str, help="Output CSV file")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the VTT file to CSV
    convert_vtt_to_csv(args.input, args.output)


if __name__ == "__main__":
    main()
