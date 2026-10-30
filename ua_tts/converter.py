"""
Ukrainian TTS Converter engine with sentence chunking, progress bar callbacks, and WAV/MP3 output.
"""

import os
import re
from typing import List, Callable, Optional, Tuple, Generator, Dict, Any

try:
    import numpy as np
except ImportError:
    np = None

from ua_tts.voices import resolve_voice, DEFAULT_VOICE, UKRAINIAN_VOICES, Voices, Stress
from ua_tts.audio import (
    save_audio,
    save_as_wav,
    save_as_mp3,
    audio_diagnostics,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_BITRATE
)


def get_cache_dir() -> str:
    """Return persistent cache directory path for Ukrainian TTS models."""
    cache_dir = os.environ.get("UA_TTS_CACHE_DIR")
    if not cache_dir:
        if os.name == "nt":
            base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
            cache_dir = os.path.join(base, "ua_tts", "cache")
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ua_tts")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.abspath(cache_dir)


def split_ukrainian_text(text: str, max_chunk_chars: int = 250) -> List[str]:
    """
    Split long Ukrainian text into natural sentence/paragraph chunks for progress tracking.
    """
    text = text.strip()
    if not text:
        return []

    # Clean multiple spaces and blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    raw_text = " ".join(lines)

    # Split by sentence end punctuation (. ! ?)
    sentence_pattern = r'(?<=[.!?])\s+'
    raw_sentences = re.split(sentence_pattern, raw_text)

    chunks: List[str] = []
    current_chunk = ""

    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) + 1 <= max_chunk_chars:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text]


class UkrainianTTSConverter:
    """
    TTS Engine for converting Ukrainian text files into speech WAV (or MP3) files.
    """

    def __init__(
        self,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        backend: Optional[str] = None,
        model_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        stress: str = "dictionary"
    ):
        self.voice_key = resolve_voice(voice)
        self.voice_info = UKRAINIAN_VOICES[self.voice_key]
        self.speed = max(0.25, min(4.0, float(speed)))
        self.requested_backend = (backend or "auto").lower().strip()
        self.custom_model_path = model_path
        self.cache_dir = cache_dir
        self.device = (device or "cpu").lower().strip()
        if self.device == "gpu":
            self.device = "cuda"
        self.stress = stress
        self._engine = None
        self._engine_type = None
        self.engine_errors: Dict[str, str] = {}
        self.is_fallback: bool = False
        self.last_conversion_stats: Dict[str, Any] = {}
        self.last_accented_text: str = ""
        self.last_accented_chunks: List[str] = []

    @property
    def engine_type(self) -> str:
        """Return the active backend name."""
        if self._engine_type is None:
            self._init_engine()
        return self._engine_type or "unknown"

    def _try_init_ukrainian_tts(self) -> bool:
        """Attempt to initialize the dedicated ukrainian-tts engine using cached weights."""
        try:
            from ukrainian_tts.tts import TTS

            # Search order for model cache folder:
            # 1. Custom model path if it is a directory or file's parent directory
            # 2. Explicit cache_dir if specified
            # 3. Local workspace / package directory containing model.pth
            # 4. Current working directory containing model.pth
            # 5. Persistent user cache directory
            candidate_dirs = []
            if self.custom_model_path:
                if os.path.isdir(self.custom_model_path):
                    candidate_dirs.append(self.custom_model_path)
                else:
                    candidate_dirs.append(os.path.dirname(os.path.abspath(self.custom_model_path)))

            if self.cache_dir:
                candidate_dirs.append(self.cache_dir)

            pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate_dirs.extend([
                os.path.join(pkg_dir, "models", "ua_tts"),
                os.path.join(pkg_dir, "models"),
                os.path.join(os.getcwd(), "models", "ua_tts"),
                os.path.join(os.getcwd(), "models"),
                pkg_dir,
                os.getcwd(),
                get_cache_dir(),
                os.path.join(os.path.expanduser("~"), ".cache", "ua_tts"),
            ])

            chosen_cache_folder = None
            for d in candidate_dirs:
                if os.path.exists(os.path.join(d, "model.pth")):
                    chosen_cache_folder = os.path.abspath(d)
                    break

            if chosen_cache_folder is None:
                chosen_cache_folder = get_cache_dir()

            # Ensure config.yaml in cache folder has absolute path to feats_stats.npz
            if chosen_cache_folder and os.path.isdir(chosen_cache_folder):
                cfg_file = os.path.join(chosen_cache_folder, "config.yaml")
                stats_file = os.path.join(chosen_cache_folder, "feats_stats.npz")
                if os.path.isfile(cfg_file) and os.path.isfile(stats_file):
                    try:
                        import yaml
                        with open(cfg_file, "r", encoding="utf-8") as f:
                            c_data = yaml.safe_load(f)
                        if isinstance(c_data, dict):
                            norm_conf = c_data.get("normalize_conf", {})
                            current_stats = norm_conf.get("stats_file", "")
                            if not os.path.isabs(str(current_stats)) or not os.path.exists(str(current_stats)):
                                norm_conf["stats_file"] = os.path.abspath(stats_file).replace("\\", "/")
                                c_data["normalize_conf"] = norm_conf
                                with open(cfg_file, "w", encoding="utf-8") as f:
                                    yaml.dump(c_data, f)
                    except Exception:
                        pass

            try:
                self._engine = TTS(cache_folder=chosen_cache_folder, device=self.device)
            except TypeError:
                self._engine = TTS(device=self.device)
            self._engine_type = "ukrainian_tts"
            self.is_fallback = False
            return True
        except Exception as e:
            self.engine_errors["ukrainian_tts"] = str(e)
            return False

    def _try_init_silero(self) -> bool:
        """Attempt to initialize Silero Ukrainian neural TTS engine using cached weights."""
        try:
            import torch
            cache_dir = get_cache_dir()

            # Search order for cached v4_ua.pt
            candidate_paths = []
            if self.custom_model_path and os.path.isfile(self.custom_model_path):
                candidate_paths.append(self.custom_model_path)
            if os.environ.get("UA_TTS_MODEL_PATH"):
                candidate_paths.append(os.environ.get("UA_TTS_MODEL_PATH"))

            if self.cache_dir:
                candidate_paths.append(os.path.join(self.cache_dir, "v4_ua.pt"))

            pkg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate_paths.extend([
                os.path.join(pkg_dir, "models", "ua_tts", "v4_ua.pt"),
                os.path.join(pkg_dir, "models", "v4_ua.pt"),
                os.path.join(os.getcwd(), "models", "ua_tts", "v4_ua.pt"),
                os.path.join(os.getcwd(), "models", "v4_ua.pt"),
                os.path.join(cache_dir, "v4_ua.pt"),
                os.path.join(os.path.expanduser("~"), ".cache", "ua_tts", "v4_ua.pt"),
                os.path.join(pkg_dir, "v4_ua.pt"),
                os.path.join(os.getcwd(), "v4_ua.pt"),
            ])

            target_path = None
            for p in candidate_paths:
                if os.path.isfile(p) and os.path.getsize(p) > 0:
                    target_path = os.path.abspath(p)
                    break

            if target_path is None:
                target_path = os.path.join(cache_dir, "v4_ua.pt")
                temp_path = target_path + ".tmp"
                import urllib.request
                urls = [
                    "https://huggingface.co/Derur/silero-models/resolve/main/v4_ua.pt",
                    "https://models.silero.ai/models/tts/ua/v4_ua.pt",
                ]
                download_errs = []
                for url in urls:
                    try:
                        req = urllib.request.Request(
                            url,
                            headers={"User-Agent": "ua-tts/0.1.0"}
                        )
                        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_path, "wb") as out:
                            out.write(resp.read())
                        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                            if os.path.exists(target_path):
                                os.remove(target_path)
                            os.replace(temp_path, target_path)
                            break
                    except Exception as err:
                        if os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except OSError:
                                pass
                        download_errs.append(f"{url}: {err}")
                else:
                    raise RuntimeError("Download mirrors failed: " + "; ".join(download_errs))

            try:
                import torch.package
                importer = torch.package.PackageImporter(target_path)
                self._engine = importer.load_pickle("tts_models", "model")
            except Exception:
                self._engine = torch.jit.load(target_path)

            self._engine_type = "silero"
            self.is_fallback = False
            return True
        except Exception as e1:
            try:
                import torch
                hub_dir = os.path.join(get_cache_dir(), "torch_hub")
                os.makedirs(hub_dir, exist_ok=True)
                torch.hub.set_dir(hub_dir)
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ua",
                    speaker="v4_ua",
                    trust_repo=True
                )
                self._engine = model
                self._engine_type = "silero"
                self.is_fallback = False
                return True
            except Exception as e2:
                self.engine_errors["silero"] = f"{e1} | torch.hub: {e2}"
                return False

    def _init_engine(self):
        """Lazy initialization of underlying TTS synthesis backend."""
        if self._engine_type is not None:
            return

        backend = self.requested_backend

        # Explicit backend selection
        if backend == "ukrainian_tts":
            if self._try_init_ukrainian_tts():
                return
        elif backend == "silero":
            if self._try_init_silero():
                return
        elif backend == "synth":
            self._engine_type = "synth"
            self.is_fallback = True
            return

        # Auto detection order: ukrainian_tts -> silero -> synth
        if backend == "auto":
            if self._try_init_ukrainian_tts():
                return
            if self._try_init_silero():
                return

        # Fallback to algorithmic synthesis (for offline/test mode without downloaded weights)
        self._engine_type = "synth"
        self.is_fallback = True

    def synthesize_chunk(self, text_chunk: str) -> np.ndarray:
        """Synthesize a single text chunk into 24kHz float32 audio NumPy array."""
        self._init_engine()

        if self._engine_type == "ukrainian_tts":
            try:
                import io
                import soundfile as sf
                from ukrainian_tts.tts import Voices, Stress

                voice_enum_map = {
                    "tetiana": Voices.Tetiana.value,
                    "dmytro": Voices.Dmytro.value,
                    "mykyta": Voices.Mykyta.value,
                    "lada": Voices.Lada.value,
                    "oleksa": Voices.Oleksa.value,
                    "olena": getattr(Voices, "Olena", Voices.Tetiana).value,
                    "olga": getattr(Voices, "Olga", Voices.Tetiana).value,
                }
                v_val = voice_enum_map.get(self.voice_key, self.voice_key)

                # Determine stress value
                if hasattr(self.stress, "value"):
                    stress_val = self.stress.value
                elif hasattr(Stress, str(self.stress).capitalize()):
                    stress_val = getattr(Stress, str(self.stress).capitalize()).value
                elif str(self.stress).lower() in ["dictionary", "model"]:
                    stress_val = str(self.stress).lower()
                else:
                    stress_val = Stress.Dictionary.value

                buf = io.BytesIO()
                _, output_text = self._engine.tts(text_chunk, v_val, stress_val, buf)
                if output_text:
                    self.last_accented_chunks.append(str(output_text))
                buf.seek(0)
                audio_data, sr = sf.read(buf)
                if sr != DEFAULT_SAMPLE_RATE:
                    new_len = int(len(audio_data) * DEFAULT_SAMPLE_RATE / sr)
                    orig_x = np.linspace(0, 1, len(audio_data), endpoint=False)
                    new_x = np.linspace(0, 1, new_len, endpoint=False)
                    audio_data = np.interp(new_x, orig_x, audio_data)
                return audio_data.astype(np.float32)
            except Exception as e:
                self.engine_errors["ukrainian_tts_synthesis"] = str(e)

        if self._engine_type == "silero":
            try:
                self.last_accented_chunks.append(text_chunk)
                speaker_map = {
                    "tetiana": "tetiana",
                    "lada": "lada",
                    "mykyta": "mykyta",
                    "dmytro": "mykyta",
                    "oleksa": "mykyta",
                }
                speaker = speaker_map.get(self.voice_key, "tetiana")
                audio = self._engine.apply_tts(
                    text=text_chunk,
                    speaker=speaker,
                    sample_rate=DEFAULT_SAMPLE_RATE
                )
                if hasattr(audio, "numpy"):
                    return audio.numpy().astype(np.float32)
                return np.asarray(audio, dtype=np.float32)
            except Exception as e:
                self.engine_errors["silero_synthesis"] = str(e)

        # Pure synthetic waveform fallback (melodic speech-envelope tone for testing/offline mode)
        self.last_accented_chunks.append(text_chunk)
        duration_sec = max(0.5, len(text_chunk) * 0.05 / self.speed)
        num_samples = int(DEFAULT_SAMPLE_RATE * duration_sec)
        t = np.linspace(0, duration_sec, num_samples, endpoint=False)
        freq = 180.0 if self.voice_info["gender"] == "Female" else 130.0
        audio = 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-t / duration_sec)
        return audio.astype(np.float32)

    def convert_text(
        self,
        text: str,
        output_file: str = "output.wav",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        bitrate: str = DEFAULT_BITRATE
    ) -> Tuple[str, str]:
        """
        Synthesize text directly into a WAV (or MP3) speech audio file.

        Parameters:
            text: Ukrainian text to synthesize
            output_file: Destination audio path (default: "output.wav")
            progress_callback: Optional function(completed_chunks, total_chunks, current_chunk_text)
            bitrate: MP3 encoding bitrate if output is MP3 (default: "128k")

        Returns:
            Tuple of (saved_file_path, accented_text)
        """
        chunks = split_ukrainian_text(text)
        total_chunks = len(chunks)
        if total_chunks == 0:
            raise ValueError("Input text is empty.")

        self._init_engine()

        # If exactly 1 chunk, ukrainian_tts engine, and destination is WAV:
        # Write directly to file matching the example:
        # with open("test.wav", mode="wb") as file:
        #     _, output_text = tts.tts(text, voice, stress, file)
        if (
            self._engine_type == "ukrainian_tts"
            and total_chunks == 1
            and not output_file.lower().endswith(".mp3")
        ):
            from ukrainian_tts.tts import Voices, Stress
            if not output_file.lower().endswith(".wav"):
                output_file = output_file + ".wav"
            parent_dir = os.path.dirname(os.path.abspath(output_file))
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            voice_enum_map = {
                "tetiana": Voices.Tetiana.value,
                "dmytro": Voices.Dmytro.value,
                "mykyta": Voices.Mykyta.value,
                "lada": Voices.Lada.value,
                "oleksa": Voices.Oleksa.value,
                "olena": getattr(Voices, "Olena", Voices.Tetiana).value,
                "olga": getattr(Voices, "Olga", Voices.Tetiana).value,
            }
            v_val = voice_enum_map.get(self.voice_key, self.voice_key)

            if hasattr(self.stress, "value"):
                stress_val = self.stress.value
            elif hasattr(Stress, str(self.stress).capitalize()):
                stress_val = getattr(Stress, str(self.stress).capitalize()).value
            elif str(self.stress).lower() in ["dictionary", "model"]:
                stress_val = str(self.stress).lower()
            else:
                stress_val = Stress.Dictionary.value

            if progress_callback:
                progress_callback(0, 1, chunks[0])

            with open(output_file, mode="wb") as file:
                _, output_text = self._engine.tts(chunks[0], v_val, stress_val, file)

            if progress_callback:
                progress_callback(1, 1, chunks[0])

            self.last_accented_text = str(output_text)
            self.last_accented_chunks = [str(output_text)]
            self.last_conversion_stats = {
                "engine": self.engine_type,
                "is_fallback": self.is_fallback,
                "chunks_count": 1,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "accented_text": self.last_accented_text,
                "diagnostics": {"direct_wav_write": True}
            }
            return os.path.abspath(output_file), self.last_accented_text

        self.last_accented_chunks = []
        audio_segments: List[np.ndarray] = []

        for idx, chunk in enumerate(chunks, 1):
            if progress_callback:
                progress_callback(idx - 1, total_chunks, chunk)

            chunk_audio = self.synthesize_chunk(chunk)
            audio_segments.append(chunk_audio)

            if progress_callback:
                progress_callback(idx, total_chunks, chunk)

        full_audio = np.concatenate(audio_segments) if audio_segments else np.array([], dtype=np.float32)
        self.last_accented_text = " ".join(self.last_accented_chunks) if self.last_accented_chunks else text

        self.last_conversion_stats = {
            "engine": self.engine_type,
            "is_fallback": self.is_fallback,
            "chunks_count": total_chunks,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "accented_text": self.last_accented_text,
            "diagnostics": audio_diagnostics(full_audio, DEFAULT_SAMPLE_RATE)
        }

        saved_path = save_audio(
            audio_data=full_audio,
            output_path=output_file,
            sample_rate=DEFAULT_SAMPLE_RATE,
            bitrate=bitrate
        )

        return saved_path, self.last_accented_text

    def convert_file(
        self,
        input_file: str,
        output_file: str = "output.wav",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        bitrate: str = DEFAULT_BITRATE
    ) -> str:
        """
        Read input text file, synthesize speech in chunks with progress callback, and export to WAV (or MP3).

        Parameters:
            input_file: Path to Ukrainian input text file (.txt)
            output_file: Path to destination audio file (default: output.wav)
            progress_callback: Optional function(completed_chunks, total_chunks, current_chunk_text)
            bitrate: MP3 encoding bitrate if output is MP3 (default: "128k")

        Returns:
            Absolute path to output audio file.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input text file not found: {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            full_text = f.read()

        saved_path, _ = self.convert_text(
            text=full_text,
            output_file=output_file,
            progress_callback=progress_callback,
            bitrate=bitrate
        )
        return saved_path

