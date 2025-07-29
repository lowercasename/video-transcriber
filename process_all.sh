#!/bin/bash

for fullpath in Inputs/*.{mp4,wav}; do
    filename=$(basename -- "$fullpath")
    extension="${filename##*.}"
    filename="${filename%.*}"
    ./video_transcriber.py --input "$fullpath" --output "Outputs/${filename}_transcript.txt" --model base --diarize --csv --hf_token ihf_DWEXVgbOImUrhRHhHAoixHerCsvGMxAMNe
done