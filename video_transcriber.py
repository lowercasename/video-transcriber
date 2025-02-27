#!/usr/bin/env python3
import argparse
import datetime
import os
import tempfile
import json
from pathlib import Path
import torch
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import sys
import traceback
import json


def extract_audio(video_path, output_audio_path=None):
    """Extract audio from a video file using moviepy."""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        raise ImportError("moviepy is required. Install it with: pip install moviepy")

    print(f"Extracting audio from {video_path}...")

    if output_audio_path is None:
        # Create a temporary file with .wav extension (better for diarization)
        temp_dir = tempfile.gettempdir()
        video_filename = Path(video_path).stem
        output_audio_path = os.path.join(temp_dir, f"{video_filename}.wav")

    # Extract the audio
    with VideoFileClip(video_path) as video:
        audio = video.audio
        if audio is None:
            raise ValueError(f"No audio track found in {video_path}")
        audio.write_audiofile(output_audio_path, logger=None)

    print(f"Audio extracted to {output_audio_path}")
    return output_audio_path


def transcribe_with_whisper(audio_path, model_size="base", language=None):
    """
    Transcribe audio using the Whisper model with timestamps.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("whisper is required.")

    print(f"Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print("Transcribing with Whisper...")
    options = {
        "verbose": False,
        "fp16": False,
    }

    if language:
        options["language"] = language

    result = model.transcribe(audio_path, **options)

    return result


def perform_diarization(audio_path, num_speakers=None, hf_token=None):
    """
    Perform speaker diarization using pyannote.audio.
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        raise ImportError("pyannote.audio is required.")

    print("Performing speaker diarization...")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "HuggingFace token is required for pyannote.audio. "
                "Provide with --hf_token or set the HF_TOKEN environment variable."
            )

    # For Apple Silicon, we need to specify mps as device if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device} for diarization")

    # Initialize the pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    # Move to proper device (mps for Apple Silicon or cpu)
    pipeline = pipeline.to(torch.device(device))

    # Run diarization
    diarization_options = {}
    if num_speakers is not None:
        diarization_options["num_speakers"] = num_speakers

    diarization = pipeline(audio_path, **diarization_options)

    return diarization


def merge_whisper_and_diarization(whisper_result, diarization, tolerance=0.5):
    """
    Merge Whisper transcription with PyAnnote diarization.
    """
    print("Merging transcription with speaker diarization...")
    diarized_segments = []

    # Extract diarization results into a more accessible format
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append(
            {"start": turn.start, "end": turn.end, "speaker": speaker}
        )

    # For each whisper segment, find the corresponding speaker
    for segment in whisper_result["segments"]:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"]

        # Find speakers that overlap with this segment
        speakers_in_segment = []
        for speaker_segment in speaker_segments:
            # Check if speaker segment overlaps with the current whisper segment
            if (
                speaker_segment["start"] <= segment_end
                and speaker_segment["end"] >= segment_start
            ) and (
                speaker_segment["end"] - speaker_segment["start"]
            ) > tolerance:  # Filter out very short segments
                speakers_in_segment.append(speaker_segment["speaker"])

        # Get the most common speaker for this segment (simple approach)
        if speakers_in_segment:
            from collections import Counter

            speaker = Counter(speakers_in_segment).most_common(1)[0][0]
        else:
            speaker = "UNKNOWN"

        diarized_segments.append(
            {
                "start": segment_start,
                "end": segment_end,
                "text": segment_text,
                "speaker": speaker,
            }
        )

    return diarized_segments


def whisper_to_segments(whisper_result):
    """
    Convert whisper result to a standard segment format (without speaker info)
    """
    segments = []
    for segment in whisper_result["segments"]:
        segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker": "UNKNOWN",
            }
        )
    return segments


def format_text_transcript(segments, include_timestamps=True, compact=False):
    """Format the segments into a readable transcript."""
    transcript = ""
    current_speaker = None

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        if compact:
            # Compact format - only print speaker when it changes
            if speaker != current_speaker:
                current_speaker = speaker
                speaker_line = f"\n[Speaker {speaker}]"
                if include_timestamps:
                    speaker_line += f" [{format_time(start)}]"
                transcript += f"{speaker_line}\n"

            if include_timestamps:
                transcript += f"[{format_time(start)} --> {format_time(end)}] {text}\n"
            else:
                transcript += f"{text}\n"
        else:
            # Standard format - print speaker for each segment
            if include_timestamps:
                transcript += f"[Speaker {speaker}] [{format_time(start)} --> {format_time(end)}] {text}\n\n"
            else:
                transcript += f"[Speaker {speaker}] {text}\n\n"

    return transcript


def format_csv_transcript(segments, include_timestamps=True):
    """Format the segments into a CSV format."""
    # Header
    if include_timestamps:
        transcript = "text,speaker,start,end\n"
    else:
        transcript = "text,speaker\n"

    for segment in segments:
        speaker = segment.get("speaker", "UNKNOWN")
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].replace(",", ";")  # Avoid CSV issues

        if include_timestamps:
            transcript += f"{text},{speaker},{format_time(start)},{format_time(end)}\n"
        else:
            transcript += f"{text},{speaker}\n"

    return transcript


def format_time(seconds):
    return str(datetime.timedelta(seconds=seconds))


class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def write(self, string):
        self.original_stdout.write(string)
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Auto-scroll

    def flush(self):
        self.original_stdout.flush()


class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Transcriber")
        self.root.geometry("800x700")

        # Settings file path
        self.settings_file = os.path.join(
            os.path.expanduser("~"), ".video_transcriber_settings.json"
        )

        # Create the main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add icon and title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=10)
        title_label = ttk.Label(
            title_frame, text="Video Transcriber", font=("Arial", 18)
        )
        title_label.pack()

        # Input section
        input_frame = ttk.LabelFrame(main_frame, text="Input Options", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        # Video input
        video_frame = ttk.Frame(input_frame)
        video_frame.pack(fill=tk.X)
        ttk.Label(video_frame, text="Video File:").pack(side=tk.LEFT, padx=5)
        self.video_path = tk.StringVar()
        ttk.Entry(video_frame, textvariable=self.video_path, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )
        ttk.Button(video_frame, text="Browse...", command=self.browse_video).pack(
            side=tk.RIGHT
        )

        # Output directory
        output_frame = ttk.Frame(input_frame)
        output_frame.pack(fill=tk.X, pady=5)
        ttk.Label(output_frame, text="Output File:").pack(side=tk.LEFT, padx=5)
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).pack(
            side=tk.RIGHT
        )

        # Options section
        option_frame = ttk.LabelFrame(
            main_frame, text="Transcription Options", padding="10"
        )
        option_frame.pack(fill=tk.X, pady=5)

        # Model size
        model_frame = ttk.Frame(option_frame)
        model_frame.pack(fill=tk.X, pady=5)
        ttk.Label(model_frame, text="Whisper Model:").pack(side=tk.LEFT, padx=5)
        self.model_size = tk.StringVar(value="base")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_size, width=15)
        model_combo["values"] = ("tiny", "base", "small", "medium", "large")
        model_combo.pack(side=tk.LEFT, padx=5)
        model_combo.bind("<<ComboboxSelected>>", lambda e: self.save_settings())

        # Language
        ttk.Label(model_frame, text="Language:").pack(side=tk.LEFT, padx=5)
        self.language = tk.StringVar()
        language_entry = ttk.Entry(model_frame, textvariable=self.language, width=10)
        language_entry.pack(side=tk.LEFT, padx=5)
        language_entry.bind("<FocusOut>", lambda e: self.save_settings())
        ttk.Label(model_frame, text="(optional, e.g. 'en', 'fr')").pack(side=tk.LEFT)

        # Speaker diarization options
        diarize_frame = ttk.Frame(option_frame)
        diarize_frame.pack(fill=tk.X, pady=5)

        self.diarize = tk.BooleanVar(value=False)
        diarize_check = ttk.Checkbutton(
            diarize_frame,
            text="Enable Speaker Diarization",
            variable=self.diarize,
            command=self.toggle_diarization,
        )
        diarize_check.pack(side=tk.LEFT, padx=5)

        # HF Token
        ttk.Label(diarize_frame, text="HuggingFace Token:").pack(side=tk.LEFT, padx=5)
        self.hf_token = tk.StringVar()
        self.hf_token_entry = ttk.Entry(
            diarize_frame, textvariable=self.hf_token, width=25, state="disabled"
        )
        self.hf_token_entry.pack(side=tk.LEFT, padx=5)
        self.hf_token_entry.bind("<FocusOut>", lambda e: self.save_settings())

        # Number of speakers
        ttk.Label(diarize_frame, text="Number of Speakers:").pack(side=tk.LEFT, padx=5)
        self.num_speakers = tk.StringVar()
        self.num_speakers_entry = ttk.Entry(
            diarize_frame, textvariable=self.num_speakers, width=5, state="disabled"
        )
        self.num_speakers_entry.pack(side=tk.LEFT, padx=5)
        self.num_speakers_entry.bind("<FocusOut>", lambda e: self.save_settings())
        ttk.Label(diarize_frame, text="(optional)").pack(side=tk.LEFT)

        # Output options
        output_options_frame = ttk.Frame(option_frame)
        output_options_frame.pack(fill=tk.X, pady=5)

        self.include_timestamps = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            output_options_frame,
            text="Include Timestamps",
            variable=self.include_timestamps,
            command=self.save_settings,
        ).pack(side=tk.LEFT, padx=5)

        self.compact_format = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            output_options_frame,
            text="Compact Format",
            variable=self.compact_format,
            command=self.save_settings,
        ).pack(side=tk.LEFT, padx=5)

        # File format options
        format_frame = ttk.Frame(option_frame)
        format_frame.pack(fill=tk.X, pady=5)

        ttk.Label(format_frame, text="Output Formats:").pack(side=tk.LEFT, padx=5)

        self.txt_output = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            format_frame, text="TXT", variable=self.txt_output, state="disabled"
        ).pack(side=tk.LEFT, padx=5)

        self.csv_output = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            format_frame,
            text="CSV",
            variable=self.csv_output,
            command=self.save_settings,
        ).pack(side=tk.LEFT, padx=5)

        self.json_output = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            format_frame,
            text="JSON",
            variable=self.json_output,
            command=self.save_settings,
        ).pack(side=tk.LEFT, padx=5)

        # Run button
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        self.progress = ttk.Progressbar(
            btn_frame, orient=tk.HORIZONTAL, length=300, mode="indeterminate"
        )
        self.progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.run_btn = ttk.Button(
            btn_frame, text="Start Transcription", command=self.run_transcription
        )
        self.run_btn.pack(side=tk.RIGHT, padx=5)

        # Log output
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Add scrollbar to log
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # For capturing print output
        self.text_redirect = RedirectText(self.log_text)

        # Bind window close event to save settings
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, pady=2)

        # Load saved settings
        self.load_settings()

    def browse_video(self):
        filetypes = [("Video files", "*.mp4 *.mov"), ("All files", "*.*")]

        # Use last directory if available
        initial_dir = ""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    if "last_video_dir" in settings and os.path.exists(
                        settings["last_video_dir"]
                    ):
                        initial_dir = settings["last_video_dir"]
            except:
                pass

        filename = filedialog.askopenfilename(
            title="Select a video file",
            filetypes=filetypes,
            initialdir=initial_dir if initial_dir else None,
        )

        if filename:
            self.video_path.set(filename)
            # Auto-suggest output path
            if not self.output_path.get():
                default_output = os.path.splitext(filename)[0] + "_transcript.txt"
                self.output_path.set(default_output)

            # Save settings after changing path
            self.save_settings()

    def browse_output(self):
        filetypes = [("Text files", "*.txt"), ("All files", "*.*")]

        # Use last directory if available
        initial_dir = ""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    if "last_output_dir" in settings and os.path.exists(
                        settings["last_output_dir"]
                    ):
                        initial_dir = settings["last_output_dir"]
            except:
                pass

        filename = filedialog.asksaveasfilename(
            title="Save transcript as",
            filetypes=filetypes,
            defaultextension=".txt",
            initialdir=initial_dir if initial_dir else None,
        )

        if filename:
            self.output_path.set(filename)
            # Save settings after changing path
            self.save_settings()

    def toggle_diarization(self):
        if self.diarize.get():
            self.hf_token_entry.config(state="normal")
            self.num_speakers_entry.config(state="normal")
        else:
            self.hf_token_entry.config(state="disabled")
            self.num_speakers_entry.config(state="disabled")

        # Save settings after changing diarization setting
        self.save_settings()

    def run_transcription(self):
        # Check for required fields
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file")
            return

        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output file")
            return

        if (
            self.diarize.get()
            and not self.hf_token.get()
            and not os.environ.get("HF_TOKEN")
        ):
            messagebox.showerror(
                "Error", "HuggingFace token is required for speaker diarization"
            )
            return

        # Clear log
        self.log_text.delete(1.0, tk.END)

        # Update UI
        self.run_btn.config(state="disabled")
        self.progress.start()
        self.status_var.set("Transcribing...")

        # Start processing in a separate thread
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()

    def process_video(self):
        # Redirect stdout to our log widget
        sys.stdout = self.text_redirect

        try:
            time_start = datetime.datetime.now()
            print(f"Starting transcription at {time_start}")

            # Get values from UI
            video_path = self.video_path.get()
            output_path = self.output_path.get()
            model_size = self.model_size.get()
            language = self.language.get() if self.language.get() else None
            diarize = self.diarize.get()
            hf_token = self.hf_token.get() if self.diarize.get() else None

            # Parse number of speakers
            num_speakers = None
            if self.num_speakers.get():
                try:
                    num_speakers = int(self.num_speakers.get())
                except ValueError:
                    print("Warning: Invalid number of speakers. Ignoring.")

            # Extract audio
            audio_path = extract_audio(video_path)

            # Transcribe with Whisper
            whisper_result = transcribe_with_whisper(
                audio_path, model_size=model_size, language=language
            )

            # Determine output base path (without extension)
            output_base = os.path.splitext(output_path)[0]

            # Process segments based on whether diarization is requested
            if diarize:
                diarization = perform_diarization(
                    audio_path, num_speakers=num_speakers, hf_token=hf_token
                )
                # Merge whisper and diarization results
                segments = merge_whisper_and_diarization(whisper_result, diarization)
            else:
                # Convert whisper results to our standard segment format
                segments = whisper_to_segments(whisper_result)

            # Generate and save outputs in all requested formats

            # Default TXT output (always included)
            txt_output = format_text_transcript(
                segments,
                include_timestamps=self.include_timestamps.get(),
                compact=self.compact_format.get(),
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(txt_output)
            print(f"Transcript saved to {output_path}")

            # CSV output if requested
            if self.csv_output.get():
                csv_path = f"{output_base}.csv"
                csv_output = format_csv_transcript(
                    segments, include_timestamps=self.include_timestamps.get()
                )
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(csv_output)
                print(f"CSV data saved to {csv_path}")

            # JSON output if requested
            if self.json_output.get():
                json_path = f"{output_base}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, indent=2)
                print(f"JSON data saved to {json_path}")

            # Clean up temp audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")

            time_end = datetime.datetime.now()
            elapsed_time = time_end - time_start
            print(f"Processing completed in {elapsed_time}")

            # Save settings after successful run
            self.save_settings()

            # Show completion message
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    "Complete",
                    f"Transcription completed!\nOutput saved to {output_path}",
                ),
            )

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))

        finally:
            # Reset UI
            sys.stdout = sys.__stdout__  # Restore original stdout
            self.root.after(0, self.reset_ui)

    def reset_ui(self):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status_var.set("Ready")

    def on_close(self):
        """Handle window close event"""
        self.save_settings()
        self.root.destroy()

    def save_settings(self):
        """Save current settings to a JSON file"""
        settings = {
            "model_size": self.model_size.get(),
            "language": self.language.get(),
            "diarize": self.diarize.get(),
            "hf_token": self.hf_token.get(),
            "num_speakers": self.num_speakers.get(),
            "include_timestamps": self.include_timestamps.get(),
            "compact_format": self.compact_format.get(),
            "csv_output": self.csv_output.get(),
            "json_output": self.json_output.get(),
            "last_video_dir": (
                os.path.dirname(self.video_path.get()) if self.video_path.get() else ""
            ),
            "last_output_dir": (
                os.path.dirname(self.output_path.get())
                if self.output_path.get()
                else ""
            ),
        }

        try:
            with open(self.settings_file, "w") as f:
                json.dump(settings, f)
            print(f"Settings saved to {self.settings_file}")
        except Exception as e:
            print(f"Error saving settings: {str(e)}")

    def load_settings(self):
        """Load settings from JSON file"""
        if not os.path.exists(self.settings_file):
            print("No saved settings found.")
            return

        try:
            with open(self.settings_file, "r") as f:
                settings = json.load(f)

            # Apply loaded settings
            if "model_size" in settings:
                self.model_size.set(settings["model_size"])
            if "language" in settings:
                self.language.set(settings["language"])
            if "diarize" in settings:
                self.diarize.set(settings["diarize"])
                self.toggle_diarization()  # Update UI based on diarization setting
            if "hf_token" in settings:
                self.hf_token.set(settings["hf_token"])
            if "num_speakers" in settings:
                self.num_speakers.set(settings["num_speakers"])
            if "include_timestamps" in settings:
                self.include_timestamps.set(settings["include_timestamps"])
            if "compact_format" in settings:
                self.compact_format.set(settings["compact_format"])
            if "csv_output" in settings:
                self.csv_output.set(settings["csv_output"])
            if "json_output" in settings:
                self.json_output.set(settings["json_output"])

            print("Settings loaded successfully.")
        except Exception as e:
            print(f"Error loading settings: {str(e)}")


def main():
    # Support both command line and GUI modes
    parser = argparse.ArgumentParser(
        description="Extract audio from video and transcribe it with optional speaker diarization"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Launch the graphical user interface"
    )
    parser.add_argument("--input", help="Path to the input video file (mp4 or mov)")

    # Parse just the gui flag first
    args, _ = parser.parse_known_args()

    # If GUI mode or no arguments provided, launch the GUI
    if args.gui or len(sys.argv) == 1:
        root = tk.Tk()
        app = TranscriptionApp(root)
        root.mainloop()
    else:
        # Full command line mode - parse all arguments
        parser.add_argument("--output", help="Path to save the transcript text file")
        parser.add_argument(
            "--audio_output", help="Path to save the extracted audio file (optional)"
        )
        parser.add_argument(
            "--language", help="Language code for transcription (e.g., 'en', 'fr')"
        )
        parser.add_argument(
            "--model",
            choices=["tiny", "base", "small", "medium", "large"],
            default="base",
            help="Whisper model size to use",
        )
        parser.add_argument(
            "--diarize",
            action="store_true",
            help="Perform speaker diarization",
        )
        parser.add_argument(
            "--num_speakers", type=int, help="Number of speakers to detect (if known)"
        )
        parser.add_argument("--hf_token", help="HuggingFace token for pyannote model")
        parser.add_argument(
            "--no_timestamps",
            action="store_true",
            help="Don't include timestamps in the output",
        )
        parser.add_argument(
            "--compact",
            action="store_true",
            help="Use compact format (group by speaker)",
        )
        parser.add_argument(
            "--csv", action="store_true", help="Also save the transcript as a CSV file"
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Also save the full segment data as JSON",
        )

        args = parser.parse_args()

        # Check required arguments for command line mode
        if not args.input:
            parser.error("--input is required in command line mode")
        if not args.output:
            parser.error("--output is required in command line mode")

        time_start = datetime.datetime.now()

        # Validate input file
        if not os.path.exists(args.input):
            print(f"Error: Input file '{args.input}' does not exist.")
            return

        if not args.input.lower().endswith((".mp4", ".mov")):
            print(
                f"Warning: Input file '{args.input}' does not have a .mp4 or .mov extension."
            )

        # Validate HF token if diarization is requested
        if args.diarize and not args.hf_token and not os.environ.get("HF_TOKEN"):
            print("Error: HuggingFace token is required for speaker diarization.")
            return

        try:
            # Extract audio
            audio_path = extract_audio(args.input, args.audio_output)

            # Transcribe with Whisper
            whisper_result = transcribe_with_whisper(
                audio_path, model_size=args.model, language=args.language
            )

            # Determine output base path (without extension)
            output_base = os.path.splitext(args.output)[0]

            # Process segments based on whether diarization is requested
            if args.diarize:
                diarization = perform_diarization(
                    audio_path, num_speakers=args.num_speakers, hf_token=args.hf_token
                )
                # Merge whisper and diarization results
                segments = merge_whisper_and_diarization(whisper_result, diarization)
            else:
                # Convert whisper results to our standard segment format
                segments = whisper_to_segments(whisper_result)

            # Generate and save outputs in all requested formats

            # Default TXT output
            txt_output = format_text_transcript(
                segments,
                include_timestamps=not args.no_timestamps,
                compact=args.compact,
            )
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(txt_output)
            print(f"Transcript saved to {args.output}")

            # CSV output if requested
            if args.csv:
                csv_path = f"{output_base}.csv"
                csv_output = format_csv_transcript(
                    segments, include_timestamps=not args.no_timestamps
                )
                with open(csv_path, "w", encoding="utf-8") as f:
                    f.write(csv_output)
                print(f"CSV data saved to {csv_path}")

            # JSON output if requested
            if args.json:
                json_path = f"{output_base}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(segments, f, indent=2)
                print(f"JSON data saved to {json_path}")

            # Clean up temp audio file if we created one
            if args.audio_output is None and os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"Cleaned up temporary audio file: {audio_path}")

            time_end = datetime.datetime.now()
            elapsed_time = time_end - time_start
            print(f"Processing completed in {elapsed_time}")

        except Exception as e:
            import traceback

            print(f"Error: {str(e)}")
            print(traceback.format_exc())


if __name__ == "__main__":
    main()
