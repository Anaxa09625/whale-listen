"""
whale-listen MCP server
Ears for AI — let Claude hear music as structured note data.

Tools:
  listen          — convert audio file → JSON notes
  analyze         — structural analysis of note data
  listen_and_analyze — both in one step
"""

import base64
import json
import pathlib
import sys
import tempfile
import traceback
from collections import Counter

from fastmcp import FastMCP

mcp = FastMCP("whale-listen")


# ── lazy imports ─────────────────────────────────────────────────────

_predict = None
_model_path = None
_pretty_midi = None


def _load_pitch_deps():
    """Load basic-pitch and pretty-midi on first use (not at startup)."""
    global _predict, _model_path, _pretty_midi

    if _predict is not None:
        return

    # Patch scipy before basic-pitch import
    import scipy.signal
    from scipy.signal.windows import gaussian as _gaussian
    if not hasattr(scipy.signal, "gaussian"):
        scipy.signal.gaussian = _gaussian

    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import pretty_midi

    _predict = predict
    _model_path = ICASSP_2022_MODEL_PATH
    _pretty_midi = pretty_midi


# ── helpers ──────────────────────────────────────────────────────────


def _find_model():
    """Find the ONNX model bundled with basic-pitch."""
    onnx = pathlib.Path(_model_path).parent / "nmp.onnx"
    if onnx.exists():
        return onnx
    return _model_path


def _convert(audio_path: str) -> dict:
    """Run pitch detection and return structured note data."""
    _load_pitch_deps()

    audio = pathlib.Path(audio_path)
    if not audio.exists():
        raise FileNotFoundError(f"Audio file not found: {audio}")

    model = _find_model()
    _model_output, midi_data, _note_events = _predict(str(audio), model)

    notes = []
    for inst in midi_data.instruments:
        for note in inst.notes:
            notes.append(
                {
                    "pitch": int(note.pitch),
                    "note_name": _pretty_midi.note_number_to_name(note.pitch),
                    "start": round(float(note.start), 3),
                    "end": round(float(note.end), 3),
                    "duration": round(float(note.end - note.start), 3),
                    "velocity": int(note.velocity),
                }
            )

    notes.sort(key=lambda n: n["start"])
    total_duration = round(notes[-1]["end"], 1) if notes else 0

    return {
        "source": audio.name,
        "duration_sec": total_duration,
        "total_notes": len(notes),
        "notes": notes,
    }


def _analyze(data: dict) -> str:
    """Return a structural analysis as text."""
    notes = data["notes"]
    if not notes:
        return "No notes found."

    total_dur = data.get("duration_sec", notes[-1]["end"])
    pitches = [n["pitch"] for n in notes]
    velocities = [n["velocity"] for n in notes]
    durations = [n["duration"] for n in notes]

    lines = []
    lines.append("=" * 50)
    lines.append(f"  {data.get('source', 'unknown')}")
    lines.append(f"  {len(notes)} notes over {total_dur}s ({total_dur / 60:.1f} min)")
    lines.append("=" * 50)

    # pitch range
    low = min(notes, key=lambda n: n["pitch"])
    high = max(notes, key=lambda n: n["pitch"])
    lines.append(
        f"\nPitch range: {low['note_name']} ({low['pitch']}) — "
        f"{high['note_name']} ({high['pitch']})"
    )

    # most common notes
    pitch_counts = Counter(n["note_name"] for n in notes)
    lines.append(
        "Most common: "
        + ", ".join(f"{name} ({count})" for name, count in pitch_counts.most_common(5))
    )

    # velocity
    lines.append(
        f"Velocity: {min(velocities)}–{max(velocities)}, "
        f"avg {sum(velocities) / len(velocities):.0f}"
    )

    # note duration
    lines.append(
        f"Note duration: {min(durations):.3f}s–{max(durations):.3f}s, "
        f"avg {sum(durations) / len(durations):.3f}s"
    )

    # density map
    lines.append(f"\nDensity (notes per 10s):")
    for t in range(0, int(total_dur) + 1, 10):
        count = len([n for n in notes if t <= n["start"] < t + 10])
        bar = "█" * count
        lines.append(f"  {t:3d}–{t + 10:3d}s: {count:3d} {bar}")

    # pitch map
    lines.append(f"\nPitch map (avg per 15s):")
    for t in range(0, int(total_dur) + 1, 15):
        segment = [n for n in notes if t <= n["start"] < t + 15]
        if segment:
            avg = sum(n["pitch"] for n in segment) / len(segment)
            lo = min(n["pitch"] for n in segment)
            hi = max(n["pitch"] for n in segment)
            lines.append(
                f"  {t:3d}–{t + 15:3d}s: avg={avg:.0f}  [{lo}–{hi}]  n={len(segment)}"
            )

    # chords
    chords = []
    i = 0
    while i < len(notes):
        cluster = [notes[i]]
        j = i + 1
        while j < len(notes) and abs(notes[j]["start"] - notes[i]["start"]) < 0.05:
            cluster.append(notes[j])
            j += 1
        if len(cluster) >= 2:
            names = [n["note_name"] for n in cluster]
            chords.append((cluster[0]["start"], names))
        i = j if j > i + 1 else i + 1

    if chords:
        lines.append(f"\nChords (simultaneous notes): {len(chords)}")
        for t, names in chords[:10]:
            lines.append(f"  {t:6.2f}s  {' + '.join(names)}")
        if len(chords) > 10:
            lines.append(f"  ... and {len(chords) - 10} more")

    # silences
    gaps = []
    for i in range(1, len(notes)):
        gap = notes[i]["start"] - notes[i - 1]["end"]
        if gap > 0.5:
            gaps.append((notes[i - 1]["end"], notes[i]["start"], gap))
    gaps.sort(key=lambda x: -x[2])

    if gaps:
        lines.append(f"\nLongest silences:")
        for end, start, gap in gaps[:5]:
            lines.append(f"  {end:.1f}s — {start:.1f}s  ({gap:.1f}s)")

    return "\n".join(lines)


# ── MCP tools ────────────────────────────────────────────────────────


@mcp.tool()
def listen_upload(audio_base64: str, filename: str = "upload.mp3") -> str:
    """Convert uploaded audio into structured JSON note data.

    Use this when the user uploads an audio file in chat.
    Claude should read the file and base64-encode it before calling this tool.

    Args:
        audio_base64: Base64-encoded audio data.
        filename: Original filename (used to detect format). Defaults to upload.mp3.

    Returns:
        JSON string with source, duration, total_notes, and a notes array,
        followed by structural analysis.
    """
    try:
        raw = base64.b64decode(audio_base64)
        suffix = pathlib.Path(filename).suffix or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        data = _convert(tmp_path)
        data["source"] = filename
        analysis = _analyze(data)

        # clean up
        pathlib.Path(tmp_path).unlink(missing_ok=True)

        result = json.dumps(data, indent=2, ensure_ascii=False)
        return f"{result}\n\n--- Analysis ---\n\n{analysis}"
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def listen(audio_path: str) -> str:
    """Convert an audio file (MP3, WAV, FLAC, etc.) into structured JSON note data.

    Each note includes pitch (MIDI number), note name (e.g. C4),
    start/end timestamps, duration, and velocity.

    Args:
        audio_path: Absolute path to the audio file on disk.

    Returns:
        JSON string with source, duration, total_notes, and a notes array.
    """
    try:
        data = _convert(audio_path)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


@mcp.tool()
def analyze(notes_json: str) -> str:
    """Analyze structured note data and return a human-readable breakdown.

    Includes: pitch range, most common notes, velocity stats,
    density map, pitch contour, chord detection, and silence structure.

    Args:
        notes_json: Either a JSON string of note data (as returned by `listen`),
                    or an absolute path to a .json file on disk.

    Returns:
        A text-formatted structural analysis.
    """
    try:
        # Try as file path first
        p = pathlib.Path(notes_json)
        if p.exists() and p.suffix == ".json":
            with open(p) as f:
                data = json.load(f)
        else:
            data = json.loads(notes_json)
        return _analyze(data)
    except Exception as e:
        return f"Error: {e}\n{traceback.format_exc()}"


@mcp.tool()
def listen_and_analyze(audio_path: str) -> str:
    """Convert audio to note data AND run structural analysis in one step.

    This is the recommended tool for a first listen — gives both the raw
    data and a high-level map of the music's shape.

    Args:
        audio_path: Absolute path to the audio file on disk.

    Returns:
        JSON note data followed by the structural analysis.
    """
    try:
        data = _convert(audio_path)
        analysis = _analyze(data)
        result = json.dumps(data, indent=2, ensure_ascii=False)
        return f"{result}\n\n--- Analysis ---\n\n{analysis}"
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


# ── entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    transport = os.environ.get("MCP_TRANSPORT", "stdio")
    if transport in ("sse", "streamable-http"):
        port = int(os.environ.get("PORT", 8000))
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        mcp.run()
