import os
import sys
import json
import RNA
import numpy as np
from Bio import SeqIO
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _extract_vectors(bppm):
    """Extract compact per-position vectors from a full (L, L) BPPM matrix.

    Returns dict with: row_sum, entropy, top1_partner, top1_prob, cross_pair.
    Must stay in sync with src.data.data_loader._extract_bppm_vectors.
    """
    bppm = np.clip(bppm, 0.0, 1.0).astype(np.float32)
    L = bppm.shape[0]
    eps = 1e-8

    row_sum = bppm.sum(axis=-1)

    p_safe = np.clip(bppm, eps, None)
    entropy = -(p_safe * np.log(p_safe)).sum(axis=-1) / max(L, 1)

    np.fill_diagonal(bppm, 0.0)
    top1_partner = bppm.argmax(axis=-1).astype(np.int64)
    top1_prob = bppm.max(axis=-1)

    if L >= 2:
        S = bppm.cumsum(axis=0).cumsum(axis=1)
        S_i_last = S[:, L - 1]
        S_i_i = np.diag(S)
        cross_pair = (S_i_last - S_i_i)[:-1].astype(np.float32)
    else:
        cross_pair = np.zeros(0, dtype=np.float32)

    return {
        "row_sum": row_sum.astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "top1_partner": top1_partner,
        "top1_prob": top1_prob.astype(np.float32),
        "cross_pair": cross_pair,
    }


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--fasta_path", required=False, default="./data/pre_random/train_pre.fasta")
    p.add_argument("--output_dir", required=False, default="./data/pre_random/bppm_shards",
                   help="Output directory for sharded BPPM files (default: ./data/pre_random/bppm_shards)")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--truncate_overflow", dest="truncate_overflow", action="store_true")
    p.add_argument("--no_truncate_overflow", dest="truncate_overflow", action="store_false")
    p.set_defaults(truncate_overflow=True)
    p.add_argument("--num_workers", type=int, default=0,
                   help="Number of parallel workers (default: 0 = all CPU cores)")
    p.add_argument("--shard_size", type=int, default=5000,
                   help="Number of sequences per shard file (default: 5000)")
    p.add_argument("--merge", action="store_true",
                   help="After shard generation, merge all shards into a single output.pkl "
                        "(requires enough memory to hold all results)")
    return p


def _compute_batch_bppm(batch):
    """Compute BPPM for a batch of sequences and extract compact vectors.

    Args:
        batch: list of (idx, seq_id, seq_used) tuples

    Returns:
        list of (idx, vector_dict) tuples
    """
    results = []
    for idx, seq_id, seq_used in batch:
        seq_len = len(seq_used)
        fc = RNA.fold_compound(seq_used)
        fc.pf()
        bppm_raw = np.array(fc.bpp())
        bppm = bppm_raw[1:seq_len + 1, 1:seq_len + 1].astype(np.float32)
        bppm = np.maximum(bppm, bppm.T)

        vectors = _extract_vectors(bppm)
        results.append((idx, vectors))
    return results


def _merge_shards(output_dir: str, output_pkl: str):
    """Merge all shard files into a single pickle file."""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    all_results = []
    for shard_name in manifest["shards"]:
        shard_path = os.path.join(output_dir, shard_name)
        with open(shard_path, 'rb') as f:
            all_results.extend(pickle.load(f))

    print(f"正在保存合并文件至 {output_pkl}...")
    os.makedirs(os.path.dirname(output_pkl) or ".", exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"合并完成，共 {len(all_results)} 条序列，文件大小: {os.path.getsize(output_pkl) / 1024 / 1024:.2f} MB")


def process_rna_bppm(fasta_path: str, output_dir: str, max_len: int = 400,
                     truncate_overflow: bool = True, num_workers: int = 0,
                     shard_size: int = 5000):
    if max_len <= 0:
        raise ValueError(f"max_len 必须为正整数，当前为 {max_len}")

    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"找不到输入文件: {fasta_path}")

    if num_workers <= 0:
        num_workers = mp.cpu_count()

    print(f"正在读取 {fasta_path}...")
    with open(fasta_path, "r") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    if not records:
        raise ValueError("FASTA 文件中未找到任何序列")

    total = len(records)
    print(f"读取完成，共 {total} 条序列，使用 {num_workers} 个 worker 并行处理")

    overflows = [(r.id, len(r.seq)) for r in records if len(r.seq) > max_len]
    if overflows and not truncate_overflow:
        first_id, first_len = overflows[0]
        raise ValueError(f"存在长度超过 max_len 的序列但 truncate_overflow=False，例如 {first_id} 长度 {first_len}，max_len={max_len}")

    # Prepare work items: (idx, seq_id, seq_used)
    work_items = []
    for i, record in enumerate(records):
        seq_full = str(record.seq).upper().replace("T", "U")
        seq_used = seq_full[:max_len] if truncate_overflow else seq_full
        work_items.append((i, record.id, seq_used))

    # Release the original records to free memory
    del records

    # Batch work into groups of 100 to amortize IPC overhead
    batch_size = 100
    batches = []
    for i in range(0, total, batch_size):
        batches.append(work_items[i:i + batch_size])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # State for shard writing
    shard_buffer = []
    shard_files = []
    shard_start_idx = 0

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, desc="计算 BPPM", unit="seq")
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    def _flush_shard():
        """Write current shard buffer to disk and clear it."""
        nonlocal shard_start_idx
        if not shard_buffer:
            return
        shard_name = f"shard_{len(shard_files):05d}.pkl"
        shard_path = os.path.join(output_dir, shard_name)
        # Sort by original index to maintain FASTA order within the shard
        shard_buffer.sort(key=lambda x: x[0])
        with open(shard_path, 'wb') as f:
            pickle.dump([r for _, r in shard_buffer], f)
        shard_files.append(shard_name)
        shard_start_idx += len(shard_buffer)
        shard_buffer.clear()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use executor.map to preserve input order — results stream back in
        # the same order as batches, enabling sequential shard writing.
        for batch_results in executor.map(_compute_batch_bppm, batches):
            shard_buffer.extend(batch_results)

            if use_tqdm:
                pbar.update(len(batch_results))

            # Flush when the buffer reaches shard_size
            if len(shard_buffer) >= shard_size:
                _flush_shard()

    # Flush remaining results
    _flush_shard()

    if use_tqdm:
        pbar.close()

    # Verify all sequences were processed
    total_written = sum(
        len(pickle.load(open(os.path.join(output_dir, s), 'rb')))
        for s in shard_files
    )
    if total_written != total:
        raise RuntimeError(
            f"内部错误：期望写入 {total} 条但实际写入 {total_written} 条"
        )

    # Write manifest
    manifest = {
        "total": total,
        "shards": shard_files,
        "max_len": max_len,
        "truncate_overflow": truncate_overflow,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("处理完成！")
    print(f"- 处理序列总数: {total}")
    print(f"- 分片数: {len(shard_files)}（每片最多 {shard_size} 条）")
    print(f"- 输出目录: {output_dir}")
    print(f"- 总大小: {sum(os.path.getsize(os.path.join(output_dir, s)) for s in shard_files) / 1024 / 1024:.2f} MB")
    print(f"- BPPM 以实际序列长度存储（非填充至 max_len），由 collate_fn 在训练时动态对齐")
    print(f"- 训练时请将 pkl_path 指向此目录（data_loader 会自动识别 manifest.json）")


if __name__ == "__main__":
    mp.freeze_support()  # Windows 打包兼容
    args = build_argparser().parse_args()
    process_rna_bppm(
        fasta_path=args.fasta_path,
        output_dir=args.output_dir,
        max_len=args.max_len,
        truncate_overflow=args.truncate_overflow,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
    )

    if args.merge:
        merge_path = os.path.join(args.output_dir, "merged.pkl")
        _merge_shards(args.output_dir, merge_path)
