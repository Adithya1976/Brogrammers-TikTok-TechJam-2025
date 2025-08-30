import whisper
import whisperx
from presidio_analyzer import AnalyzerEngine
from typing import List, Tuple, Dict
from pydub import AudioSegment, generators
import traceback

class CensorIntervalExtractor:
    def __init__(
        self,
        whisper_model: str = "small",
        device: str = "cpu",
        pad_ms: int = 200,
        beep_freq: int = 1000,
        beep_gain_db: float = -5.0,
        merge_gap_ms: int = 30,
    ):
        """
        Create extractor.
        - whisper_model: model size for whisper
        - device: "cpu", "cuda", or "mps" if available
        - pad_ms: padding around detected word intervals (ms)
        - merge_gap_ms: merge intervals closer than this gap (ms)
        """
        self.whisper_model = whisper_model
        self.device = device
        self.pad_ms = pad_ms
        self.merge_gap_ms = merge_gap_ms

        # load whisper model lazily on first use
        self._whisper = None
        self._analyzer = AnalyzerEngine()

    def _load_whisper(self):
        if self._whisper is None:
            print(f"[CensorIntervalExtractor] Loading Whisper model '{self.whisper_model}' on device '{self.device}'...")
            self._whisper = whisper.load_model(self.whisper_model, device=self.device)
        return self._whisper

    def _transcribe(self, audio_path: str) -> dict:
        model = self._load_whisper()
        print("[CensorIntervalExtractor] Transcribing audio (Whisper)...")
        res = model.transcribe(audio_path, language="en")
        return res

    def _align_with_whisperx(self, audio_path: str, whisper_out: dict) -> List[dict]:
        """
        Attempt whisperx forced alignment. Returns list of word dicts:
        [{ "text": "...", "start_ms": float, "end_ms": float }, ...]
        May raise on failure.
        """
        print("[CensorIntervalExtractor] Attempting WhisperX forced alignment...")
        audio = whisperx.load_audio(audio_path)
        align_model, metadata = whisperx.load_align_model(language_code=whisper_out["language"], device=self.device)
        aligned = whisperx.align(whisper_out["segments"], align_model, metadata, audio, self.device)

        words = []
        for seg in aligned.get("segments", []):
            for w in seg.get("words", []):
                if w.get("start") is None or w.get("end") is None:
                    continue
                words.append({
                    "text": w.get("word", "").strip(),
                    "start_ms": float(w["start"]) * 1000.0,
                    "end_ms":   float(w["end"])   * 1000.0
                })
        if not words:
            raise RuntimeError("WhisperX alignment returned no words.")
        print(f"[CensorIntervalExtractor] WhisperX produced {len(words)} words.")
        return words

    def _fallback_segment_mapping(self, whisper_out: dict) -> Tuple[str, List[dict]]:
        """
        If whisperx fails, build segments meta and return full_text and a 'words' like list
        based on segment chunks (coarse timing).
        Each entry will have 'text','start_ms','end_ms' but not per-word resolution.
        """
        print("[CensorIntervalExtractor] WhisperX failed: falling back to segment-level mapping.")
        sep = " "
        seg_meta = []
        texts = []
        cursor = 0
        for s in whisper_out["segments"]:
            t = s["text"].strip()
            texts.append(t)
            seg_meta.append({
                "text": t,
                "start_ms": s["start"] * 1000.0,
                "end_ms": s["end"] * 1000.0,
                "char_start": cursor,
                "char_end": cursor + len(t)
            })
            cursor += len(t) + len(sep)
        full_text = sep.join(texts)
        # build coarse "words" entries from segments (each segment as a unit)
        words = []
        for seg in seg_meta:
            words.append({
                "text": seg["text"],
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"]
            })
        return full_text, words

    def _detect_pii(self, text: str):
        print("[CensorIntervalExtractor] Running Presidio analyzer on transcript...")
        results = self._analyzer.analyze(text=text, language="en")
        # sorted by start
        return sorted(results, key=lambda x: (x.start, x.end))

    def _map_entities_to_intervals(self, entities, words, full_text: str) -> List[Tuple[int,int]]:
        """
        Map Presidio entity character spans to word-aligned ms intervals.
        Returns list of (start_ms, end_ms) (ints).
        """
        PAD = self.pad_ms
        intervals = []

        # Reconstruct concatenated 'words' text (without spaces between words from whisperx)
        reconstructed = "".join([w["text"] for w in words])
        # Also a simple space-joined variant to compare
        reconstructed_spaced = " ".join([w["text"] for w in words])

        # Attempt to build char->word index on the 'reconstructed' stream (no spaces)
        char_to_word_index = {}
        cursor = 0
        for wi, w in enumerate(words):
            wtext = w["text"]
            for _ in range(len(wtext)):
                char_to_word_index[cursor] = wi
                cursor += 1

        # For each entity, try to find overlapping words
        for idx, ent in enumerate(entities):
            snippet = full_text[ent.start:ent.end]
            snippet_norm = snippet.replace(" ", "")
            matched_word_indices = set()

            # naive substring search on reconstructed (no spaces)
            start_pos = reconstructed.lower().find(snippet_norm.lower())
            if start_pos != -1:
                # map chars in reconstructed [start_pos, start_pos+len(snippet_norm)) to word indices
                for cpos in range(start_pos, start_pos + len(snippet_norm)):
                    wi = char_to_word_index.get(cpos)
                    if wi is not None:
                        matched_word_indices.add(wi)
            else:
                # fallback: fuzzy per-word match (substring)
                s_low = snippet.strip().lower()
                for wi, w in enumerate(words):
                    w_low = w["text"].strip().lower()
                    if not w_low:
                        continue
                    # match if any relation between snippet and word text
                    if s_low in w_low or w_low in s_low:
                        matched_word_indices.add(wi)

            if not matched_word_indices:
                # If still no match, continue (we can't map)
                print(f"[CensorIntervalExtractor] WARNING: could not map entity '{snippet}' to words.")
                continue

            matched = [words[i] for i in sorted(matched_word_indices)]
            start_ms = max(0, int(matched[0]["start_ms"] - PAD))
            end_ms = int(matched[-1]["end_ms"] + PAD)
            intervals.append((start_ms, end_ms))
            print(f"[CensorIntervalExtractor] Mapped entity '{snippet}' -> words {[w['text'] for w in matched]} -> [{start_ms},{end_ms}]ms")

        print(f"[CensorIntervalExtractor] Mapped {len(intervals)} entities to intervals (ms).")
        return intervals

    def _merge_intervals(self, intervals_ms: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
        if not intervals_ms:
            return []
        intervals_ms = sorted(intervals_ms, key=lambda x: x[0])
        merged = [list(intervals_ms[0])]
        for s,e in intervals_ms[1:]:
            if s <= merged[-1][1] + self.merge_gap_ms:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s,e])
        return [(a,b) for a,b in merged]

    def get_censor_intervals(self, input_audio_path: str) -> List[Tuple[float,float]]:
        """
        Main entry:
        - input_audio_path: path to audio (wav/m4a/mp3) or video file with audio
        Returns: list of (start_seconds, end_seconds) intervals to censor.
        """
        try:
            whisper_out = self._transcribe(input_audio_path)
            full_text = whisper_out.get("text", "").strip()
            if not full_text:
                print("[CensorIntervalExtractor] Empty transcript.")
                return []

            # Try WhisperX alignment
            try:
                words = self._align_with_whisperx(input_audio_path, whisper_out)
                aligned = True
            except Exception as ex:
                print("[CensorIntervalExtractor] WhisperX alignment failed:", ex)
                traceback.print_exc()
                # fallback: segment mapping
                full_text_fb, words = self._fallback_segment_mapping(whisper_out)
                # if fallback changes full_text, keep original for PII detection but fallback uses its own full_text_fb for mapping attempts
                aligned = False
                # we keep full_text as the transcript for Presidio
                # but mapping will use 'words' built from segments
                full_text = whisper_out.get("text", "").strip()

            # Detect PII on full transcript
            entities = self._detect_pii(full_text)

            if not entities:
                print("[CensorIntervalExtractor] No PII entities detected.")
                return []

            # Map entities to ms intervals (using words)
            intervals_ms = self._map_entities_to_intervals(entities, words, full_text)

            # Merge intervals and convert to seconds
            merged_ms = self._merge_intervals(intervals_ms)
            intervals_s = [(round(s/1000.0, 3), round(e/1000.0, 3)) for s,e in merged_ms]

            print(f"[CensorIntervalExtractor] Returning {len(intervals_s)} intervals (seconds): {intervals_s}")
            return intervals_s

        except Exception as e:
            print("[CensorIntervalExtractor] Fatal error in get_censor_intervals:", e)
            traceback.print_exc()
            return []

# inside your API handler
# extractor = CensorIntervalExtractor(whisper_model="small", device="cpu")
# intervals = extractor.get_censor_intervals("AudioTest.m4a")