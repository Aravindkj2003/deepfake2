import argparse
import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


AudioAugFn = Callable[[np.ndarray, int, random.Random], np.ndarray]


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio.astype(np.float32)
    return (audio / peak).astype(np.float32)


def fix_length(audio: np.ndarray, target_length: int) -> np.ndarray:
    if len(audio) == target_length:
        return audio
    if len(audio) > target_length:
        return audio[:target_length]
    padded = np.zeros(target_length, dtype=np.float32)
    padded[: len(audio)] = audio
    return padded


def pitch_shift(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    steps = rng.uniform(-2.5, 2.5)
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)
    return normalize_audio(shifted)


def time_stretch(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    rate = rng.uniform(0.9, 1.1)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    return normalize_audio(stretched)


def add_background_noise(
    audio: np.ndarray,
    sr: int,
    rng: random.Random,
    noise_pool: List[np.ndarray],
) -> np.ndarray:
    snr_db = rng.uniform(8.0, 20.0)
    signal_power = np.mean(audio**2) + 1e-9

    if noise_pool:
        noise = noise_pool[rng.randrange(0, len(noise_pool))]
        if len(noise) < len(audio):
            reps = int(np.ceil(len(audio) / len(noise)))
            noise = np.tile(noise, reps)
        start = rng.randrange(0, max(1, len(noise) - len(audio) + 1))
        noise = noise[start : start + len(audio)]
    else:
        noise = np.random.normal(0, 1, len(audio)).astype(np.float32)

    noise_power = np.mean(noise**2) + 1e-9
    desired_noise_power = signal_power / (10 ** (snr_db / 10.0))
    scale = np.sqrt(desired_noise_power / noise_power)
    mixed = audio + noise * scale
    return normalize_audio(mixed)


def volume_change(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    gain_db = rng.uniform(-6.0, 6.0)
    gain = 10 ** (gain_db / 20.0)
    changed = audio * gain
    return normalize_audio(changed)


def time_shift(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    max_shift = int(0.15 * sr)
    shift = rng.randint(-max_shift, max_shift)
    shifted = np.roll(audio, shift)
    if shift > 0:
        shifted[:shift] = 0
    elif shift < 0:
        shifted[shift:] = 0
    return shifted.astype(np.float32)


def add_echo(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    delay_sec = rng.uniform(0.06, 0.18)
    decay = rng.uniform(0.25, 0.55)
    delay = max(1, int(sr * delay_sec))
    echoed = np.copy(audio)
    echoed[delay:] += audio[:-delay] * decay
    return normalize_audio(echoed)


def add_reverb(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    ir_len = rng.randint(int(0.02 * sr), int(0.08 * sr))
    t = np.linspace(0, 1, ir_len, dtype=np.float32)
    decay = rng.uniform(3.0, 8.0)
    impulse = np.exp(-decay * t)
    impulse *= rng.uniform(0.4, 0.8)
    wet = np.convolve(audio, impulse, mode="full")[: len(audio)]
    mixed = 0.75 * audio + 0.25 * wet
    return normalize_audio(mixed)


def effect_mix(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        return add_echo(audio, sr, rng)
    return add_reverb(audio, sr, rng)


def load_noise_pool(noise_dir: Path, sr: int, target_len: int) -> List[np.ndarray]:
    if not noise_dir.exists():
        return []
    files = sorted(noise_dir.rglob("*.wav"))
    pool = []
    for file in files:
        audio, _ = librosa.load(file, sr=sr, mono=True)
        audio = fix_length(normalize_audio(audio), target_len)
        pool.append(audio)
    return pool


def collect_labeled_files(input_dir: Path) -> List[Tuple[Path, str]]:
    pairs: List[Tuple[Path, str]] = []
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for file in class_dir.rglob("*.wav"):
            pairs.append((file, label))
    return pairs


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    sample_rate: int,
    duration_sec: float,
    copies_per_file: int,
    seed: int,
    noise_dir: Path,
) -> None:
    rng = random.Random(seed)
    np.random.seed(seed)

    target_len = int(sample_rate * duration_sec)
    labeled_files = collect_labeled_files(input_dir)
    if not labeled_files:
        raise ValueError(
            "No .wav files found. Expected structure: input_dir/<class_name>/*.wav"
        )

    noise_pool = load_noise_pool(noise_dir, sample_rate, target_len)

    augmentations: Dict[str, AudioAugFn] = {
        "pitch": pitch_shift,
        "stretch": time_stretch,
        "noise": lambda a, sr, r: add_background_noise(a, sr, r, noise_pool),
        "volume": volume_change,
        "shift": time_shift,
        "effect": effect_mix,
    }
    aug_names = list(augmentations.keys())

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["path", "label", "source", "augmentation"])

        for src_path, label in tqdm(labeled_files, desc="Augmenting", unit="file"):
            audio, _ = librosa.load(src_path, sr=sample_rate, mono=True)
            audio = fix_length(normalize_audio(audio), target_len)

            class_dir = output_dir / label
            class_dir.mkdir(parents=True, exist_ok=True)

            base_name = src_path.stem
            original_name = f"{base_name}_orig.wav"
            original_out = class_dir / original_name
            sf.write(original_out, audio, sample_rate)
            writer.writerow([str(original_out), label, str(src_path), "original"])

            for idx in range(copies_per_file):
                aug_name = aug_names[idx % len(aug_names)]
                aug_audio = augmentations[aug_name](audio, sample_rate, rng)

                if rng.random() < 0.35:
                    second_aug_name = rng.choice(aug_names)
                    aug_audio = augmentations[second_aug_name](aug_audio, sample_rate, rng)
                    aug_name = f"{aug_name}+{second_aug_name}"

                aug_audio = fix_length(normalize_audio(aug_audio), target_len)

                out_name = f"{base_name}_aug_{idx:02d}_{aug_name}.wav"
                out_path = class_dir / out_name
                sf.write(out_path, aug_audio, sample_rate)
                writer.writerow([str(out_path), label, str(src_path), aug_name])

    print(f"Done. Augmented dataset written to: {output_dir}")
    print(f"Manifest: {manifest_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Augment FoR-2sec style dataset with pitch, stretch, noise, volume, shift, and effects."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input dataset root: input_dir/<label>/*.wav",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for augmented files",
    )
    parser.add_argument(
        "--noise-dir",
        type=Path,
        default=Path("noise_bank"),
        help="Optional directory of background noise wav files",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument(
        "--copies-per-file",
        type=int,
        default=5,
        help="Number of augmented copies per input file. Example: 5 gives ~6x total including original.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        duration_sec=args.duration,
        copies_per_file=args.copies_per_file,
        seed=args.seed,
        noise_dir=args.noise_dir,
    )


if __name__ == "__main__":
    main()
