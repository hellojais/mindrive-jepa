"""
preprocess_data.py — batch tokenize all nuPlan mini SQLite files → processed tensors

Usage:
    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --data_dir data/raw/nuplan-v1.1_mini/data/cache/mini
    python scripts/preprocess_data.py --output_dir data/processed --toy_slice_n 5
"""
import argparse
from pathlib import Path

import yaml

from mindrive_jepa.data.nuplan_reader import NuPlanReader
from mindrive_jepa.data.tokenizer import SceneTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess nuPlan mini .db files into normalized tensors."
    )
    parser.add_argument(
        '--data_dir', type=str, default=None,
        help="Folder containing .db files (default: data.raw_data_dir from config)"
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help="Where to save processed tensors (default: data.processed_dir from config)"
    )
    parser.add_argument(
        '--toy_slice_n', type=int, default=5,
        help="How many scenarios to copy to toy_slice dir (default: 5)"
    )
    parser.add_argument(
        '--config', type=str, default='configs/default.yaml',
        help="Path to config YAML (default: configs/default.yaml)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = yaml.safe_load(open(args.config))
    data_cfg = config['data']

    # Resolve paths — CLI args override config defaults
    data_dir      = Path(args.data_dir)   if args.data_dir   else Path(data_cfg['raw_data_dir'])
    output_dir    = Path(args.output_dir) if args.output_dir else Path(data_cfg['processed_dir'])
    toy_slice_dir = Path(data_cfg['toy_slice_dir'])

    # Duration of each scenario window in seconds (sequence_len frames at 10 Hz)
    duration_sec = data_cfg['sequence_len'] / 10.0  # 50 / 10 = 5.0 seconds

    db_files = sorted(data_dir.glob('*.db'))
    if not db_files:
        print(f"ERROR: No .db files found in {data_dir}")
        print(f"  Check that --data_dir points to the folder containing .db files.")
        return

    print(f"Found {len(db_files)} .db files in {data_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Toy slice dir: {toy_slice_dir}  (first {args.toy_slice_n} scenarios)")
    print(f"Duration/scenario: {duration_sec}s  ({data_cfg['sequence_len']} frames @ 10Hz)")
    print()

    tokenizer = SceneTokenizer(data_cfg)
    all_tensors  = []
    all_metadata = []
    failed_files = 0

    for i, db_path in enumerate(db_files):
        print(f"[{i+1:02d}/{len(db_files)}] {db_path.name}")
        try:
            reader = NuPlanReader(str(db_path))
            scenarios = reader.get_all_scenarios(duration_sec=duration_sec)
            reader.close()
            tensors, metadata = tokenizer.tokenize_dataset(scenarios)
            all_tensors.extend(tensors)
            all_metadata.extend(metadata)
            print(f"         → {len(tensors)} scenarios")
        except Exception as e:
            print(f"         ERROR: {e}")
            failed_files += 1

    if not all_tensors:
        print("\nERROR: No tensors produced. Nothing saved.")
        return

    # Save full processed set
    tokenizer.save_processed(all_tensors, str(output_dir), metadata=all_metadata)

    # Save toy slice — first toy_slice_n tensors
    toy_tensors    = all_tensors[:args.toy_slice_n]
    toy_metadata   = all_metadata[:args.toy_slice_n]
    tokenizer.save_processed(toy_tensors, str(toy_slice_dir), metadata=toy_metadata)

    print()
    print("=" * 55)
    print("SUMMARY")
    print(f"  .db files processed : {len(db_files) - failed_files} / {len(db_files)}")
    if failed_files:
        print(f"  .db files failed    : {failed_files}")
    print(f"  Total scenarios     : {len(all_tensors)}")
    print(f"  Tensor shape        : {list(all_tensors[0].shape)}")
    print(f"  Processed dir       : {output_dir}")
    print(f"  Toy slice dir       : {toy_slice_dir}  ({len(toy_tensors)} scenarios)")
    print("=" * 55)


if __name__ == '__main__':
    main()
