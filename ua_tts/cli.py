"""
CLI application for Ukrainian Text-to-Speech conversion (`ua-tts`).
"""

import sys
import argparse
import os
from typing import Optional

from ua_tts import __version__
from ua_tts.voices import get_available_voices, DEFAULT_VOICE, is_valid_voice, UKRAINIAN_VOICES
from ua_tts.audio import DEFAULT_BITRATE

# Configure UTF-8 encoding on standard output for Windows consoles
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Optional Rich import with graceful standard console fallback
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
    from rich.table import Table
    from rich.panel import Panel
    console = Console(highlight=False)
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def print_voices_table():
    """Print a table of supported Ukrainian voices."""
    voices = get_available_voices()
    if HAS_RICH:
        table = Table(title="[bold cyan]Supported Ukrainian Voices[/bold cyan]", show_header=True, header_style="bold magenta")
        table.add_column("Voice ID (-v)", style="cyan", justify="left")
        table.add_column("Name", style="green")
        table.add_column("Gender", style="yellow")
        table.add_column("Description", style="white")

        for v in voices:
            table.add_row(v["key"], v["name"], v["gender"], v["description"])

        console.print(table)
    else:
        print("\n--- Supported Ukrainian Voices ---")
        print(f"{'Voice ID (-v)':<15} | {'Name':<10} | {'Gender':<8} | Description")
        print("-" * 65)
        for v in voices:
            print(f"{v['key']:<15} | {v['name']:<10} | {v['gender']:<8} | {v['description']}")
        print()


def print_banner(
    input_path: str,
    output_path: str,
    voice_info: dict,
    speed: float,
    bitrate: str,
    engine: str,
    device: str = "cpu"
):
    """Print execution banner."""
    format_label = "WAV" if output_path.lower().endswith(".wav") else "MP3"
    if HAS_RICH:
        banner_text = (
            f"[bold cyan]Ukrainian Text-to-Speech Converter (ua-tts)[/bold cyan]\n"
            f"* Input       : [yellow]{input_path}[/yellow]\n"
            f"* Output File : [yellow]{os.path.abspath(output_path)}[/yellow] ({format_label})\n"
            f"* Voice       : [green]{voice_info['name']} ({voice_info['gender']})[/green]\n"
            f"* Speed       : [magenta]{speed}x[/magenta]\n"
            f"* Device      : [blue]{device}[/blue]\n"
            f"* Engine      : [cyan]{engine}[/cyan]"
        )
        console.print(Panel(banner_text, border_style="cyan", expand=False))
    else:
        print("=" * 60)
        print(" Ukrainian Text-to-Speech Converter (ua-tts)")
        print("=" * 60)
        print(f" * Input       : {input_path}")
        print(f" * Output File : {os.path.abspath(output_path)} ({format_label})")
        print(f" * Voice       : {voice_info['name']} ({voice_info['gender']})")
        print(f" * Speed       : {speed}x")
        print(f" * Device      : {device}")
        print(f" * Engine      : {engine}")
        print("=" * 60)


def print_fallback_warning():
    """Warn user if algorithmic test mode was used instead of a neural model."""
    if HAS_RICH:
        warning_msg = (
            "[bold yellow][!] Notice: Synthetic Test Mode Active[/bold yellow]\n"
            "No neural Ukrainian TTS model backend is currently available.\n"
            "The output was generated using algorithmic test tones.\n\n"
            "[bold white]To synthesize natural Ukrainian speech:[/bold white]\n"
            "Install ukrainian-tts and PyTorch:\n"
            "  [cyan]pip install ukrainian-tts torch[/cyan]"
        )
        console.print(Panel(warning_msg, border_style="yellow", expand=False))
    else:
        print("\n" + "!" * 60)
        print(" NOTICE: Synthetic Test Mode Active")
        print(" No neural Ukrainian TTS backend is currently available.")
        print(" The output was generated using algorithmic test tones.")
        print(" To synthesize natural Ukrainian speech, install ukrainian-tts:")
        print("   pip install ukrainian-tts torch")
        print("!" * 60 + "\n")


def print_diagnostics(converter, chunks_count: int):
    """Print detailed audio and synthesis diagnostics."""
    diag = converter.last_conversion_stats.get("diagnostics", {})
    if HAS_RICH:
        table = Table(title="[bold magenta]Audio & Synthesis Diagnostics[/bold magenta]", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Active Engine", str(converter.engine_type))
        table.add_row("Is Fallback", str(converter.is_fallback))
        table.add_row("Chunks Synthesized", str(chunks_count))
        table.add_row("Audio Duration (s)", str(diag.get("duration_sec", 0.0)))
        table.add_row("Total Samples", str(diag.get("samples", 0)))
        table.add_row("RMS Energy", str(diag.get("rms", 0.0)))
        table.add_row("Peak Amplitude [Min, Max]", f"[{diag.get('min', 0.0)}, {diag.get('max', 0.0)}]")
        table.add_row("Is Silent / Inaudible", str(diag.get("is_silent", False)))
        table.add_row("Is Clipped", str(diag.get("is_clipped", False)))
        table.add_row("Has NaN/Inf", str(diag.get("has_nan_inf", False)))
        if converter.last_conversion_stats.get("accented_text"):
            table.add_row("Accented Text", str(converter.last_conversion_stats["accented_text"][:80]) + "...")

        if converter.engine_errors:
            errors_str = "\n".join([f"{k}: {v}" for k, v in converter.engine_errors.items()])
            table.add_row("Engine Error Details", errors_str)

        console.print(table)
    else:
        print("\n--- Audio & Synthesis Diagnostics ---")
        print(f" Active Engine           : {converter.engine_type}")
        print(f" Is Fallback             : {converter.is_fallback}")
        print(f" Chunks Synthesized      : {chunks_count}")
        print(f" Audio Duration (s)      : {diag.get('duration_sec', 0.0)}")
        print(f" Total Samples           : {diag.get('samples', 0)}")
        print(f" RMS Energy              : {diag.get('rms', 0.0)}")
        print(f" Peak Amplitude [Min,Max]: [{diag.get('min', 0.0)}, {diag.get('max', 0.0)}]")
        print(f" Is Silent / Inaudible   : {diag.get('is_silent', False)}")
        print(f" Is Clipped              : {diag.get('is_clipped', False)}")
        print(f" Has NaN/Inf             : {diag.get('has_nan_inf', False)}")
        if converter.last_conversion_stats.get("accented_text"):
            print(f" Accented Text           : {converter.last_conversion_stats['accented_text'][:80]}...")
        if converter.engine_errors:
            print(" Engine Error Details    :")
            for k, v in converter.engine_errors.items():
                print(f"   * {k}: {v}")
        print("-" * 37 + "\n")


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="ua-tts",
        description="Ukrainian Text-to-Speech CLI tool. Synthesizes text into high-quality WAV audio with real-time progress.",
        epilog="Example: ua-tts -t 'Привіт, як справи?' -o test.wav -v dmytro"
    )

    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to input Ukrainian text file (.txt)"
    )

    parser.add_argument(
        "-t", "--text",
        type=str,
        default=None,
        help="Direct Ukrainian text string to synthesize"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.wav",
        help="Path to destination audio file (default: output.wav)"
    )

    parser.add_argument(
        "-v", "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Ukrainian voice selection (default: {DEFAULT_VOICE}). Run --list-voices to see all options."
    )

    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed factor (e.g. 1.0, 1.25, 0.9)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "gpu", "mps"],
        help="Device to run TTS engine on (cpu, gpu, mps, cuda; default: cpu)"
    )

    parser.add_argument(
        "--stress",
        type=str,
        default="dictionary",
        choices=["dictionary", "model"],
        help="Stress prediction method (dictionary, model; default: dictionary)"
    )

    parser.add_argument(
        "--bitrate",
        type=str,
        default=DEFAULT_BITRATE,
        help=f"MP3 encoding bitrate if output is .mp3 (default: {DEFAULT_BITRATE})"
    )

    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=["auto", "ukrainian_tts", "silero", "synth"],
        help="TTS synthesis engine backend (auto, ukrainian_tts, silero, synth; default: auto)"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local path to model weights file or folder"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional directory path for cached models and resources"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed diagnostic logging for debugging audio and engine issues"
    )

    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Ukrainian voices and exit"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"ua-tts version {__version__}"
    )

    return parser


def main():
    """Main CLI execution logic."""
    parser = build_parser()
    args = parser.parse_args()

    if args.list_voices:
        print_voices_table()
        sys.exit(0)

    if not args.input and not args.text:
        if HAS_RICH:
            console.print("[bold red]Error:[/bold red] Missing input text. Please specify either `-i <file.txt>` or `-t 'текст'`.")
        else:
            print("Error: Missing input text. Please specify either `-i <file.txt>` or `-t 'текст'`.")
        parser.print_help()
        sys.exit(1)

    if args.input:
        input_path = os.path.abspath(args.input)
        if not os.path.exists(input_path):
            if HAS_RICH:
                console.print(f"[bold red]Error:[/bold red] Input file not found: [yellow]{input_path}[/yellow]")
            else:
                print(f"Error: Input file not found: {input_path}")
            sys.exit(1)
        input_display = input_path
    else:
        input_path = None
        input_display = f'"{args.text[:35]}..."' if len(args.text) > 35 else f'"{args.text}"'

    voice_key = args.voice.lower().strip()
    if not is_valid_voice(voice_key):
        if HAS_RICH:
            console.print(f"[bold red]Error:[/bold red] Invalid Ukrainian voice '[yellow]{args.voice}[/yellow]'.")
        else:
            print(f"Error: Invalid Ukrainian voice '{args.voice}'.")
        print_voices_table()
        sys.exit(1)

    voice_info = UKRAINIAN_VOICES[voice_key]

    # Read input and initialize converter
    try:
        from ua_tts.converter import UkrainianTTSConverter, split_ukrainian_text
    except ImportError as ie:
        if HAS_RICH:
            console.print(f"[bold red]Error:[/bold red] Missing required dependency: [yellow]{ie}[/yellow]")
            console.print("Please install the dependencies in your Python environment:\n  [cyan]pip install -r requirements.txt[/cyan]\n  or: [cyan]pip install numpy soundfile[/cyan]")
        else:
            print(f"Error: Missing required dependency: {ie}")
            print("Please install the dependencies in your Python environment:\n  pip install -r requirements.txt\n  or: pip install numpy soundfile")
        sys.exit(1)

    # Display execution banner
    print_banner(input_display, args.output, voice_info, args.speed, args.bitrate, args.engine, device=args.device)

    try:
        if input_path:
            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = args.text

        chunks = split_ukrainian_text(content)
        if not chunks:
            if HAS_RICH:
                console.print("[bold red]Error:[/bold red] Input text is empty.")
            else:
                print("Error: Input text is empty.")
            sys.exit(1)

        converter = UkrainianTTSConverter(
            voice=voice_key,
            speed=args.speed,
            backend=args.engine,
            model_path=args.model_path,
            cache_dir=args.cache_dir,
            device=args.device,
            stress=args.stress
        )

        if HAS_RICH:
            # Set up Rich progress bar
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task_id = progress.add_task(
                    f"[cyan]Synthesizing speech ({len(chunks)} chunk{'s' if len(chunks) > 1 else ''})...[/cyan]",
                    total=len(chunks)
                )

                def progress_callback(completed: int, total: int, chunk_text: str):
                    progress.update(
                        task_id,
                        completed=completed,
                        description=f"[cyan]Chunk {completed}/{total}:[/cyan] [dim]{chunk_text[:35]}...[/dim]"
                    )

                output_file, accented_text = converter.convert_text(
                    text=content,
                    output_file=args.output,
                    progress_callback=progress_callback,
                    bitrate=args.bitrate
                )

            console.print(f"\n[bold green]Success![/bold green] Speech synthesis complete.")
            console.print(f"Saved audio file to: [bold white]{output_file}[/bold white]")
            if accented_text:
                console.print(f"[bold cyan]Accented text:[/bold cyan] {accented_text}\n")

        else:
            def std_progress_callback(completed: int, total: int, chunk_text: str):
                percent = int((completed / total) * 100) if total > 0 else 0
                print(f" Progress: {completed}/{total} chunks ({percent}%) - {chunk_text[:30]}...", end="\r", flush=True)

            print(f"Synthesizing {len(chunks)} text chunks...")
            output_file, accented_text = converter.convert_text(
                text=content,
                output_file=args.output,
                progress_callback=std_progress_callback,
                bitrate=args.bitrate
            )
            print(f"\nSuccess! Speech synthesis complete.")
            print(f"Saved audio file to: {output_file}")
            if accented_text:
                print(f"Accented text: {accented_text}\n")

        # Warn user if fallback mode was used
        if converter.is_fallback:
            print_fallback_warning()

        # Show debug diagnostics if requested
        if args.debug:
            print_diagnostics(converter, len(chunks))

    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[bold red]Conversion Failed:[/bold red] {e}")
            if args.debug:
                console.print_exception()
        else:
            print(f"\nConversion Failed: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()
