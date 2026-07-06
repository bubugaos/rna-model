import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from typing import Any, Optional
import json
import os
import hashlib
import inspect
import tempfile
import threading
import bisect
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .masking_utils import SimpleMasking, PairingAwareMasking
from src.config import RNA_TOKEN_IDS

PAD_TOKEN_ID = RNA_TOKEN_IDS["PAD"]
MASK_TOKEN_ID = RNA_TOKEN_IDS["MASK"]
UNK_TOKEN_ID = RNA_TOKEN_IDS["UNK"]
BPPM_CACHE_KEY_VERSION = "2"


class _ShardIndex:
    """Lazy-loading BPPM index backed by sharded pickle files.

    Supports len() and integer indexing like a list, but loads individual
    shards on demand with an in-memory LRU cache.  This avoids holding all
    BPPM matrices in RAM simultaneously — only the actively used shards
    (default: 3) are kept resident.

    When *keep_indices* is provided the instance behaves as a filtered view:
    logical index *i* maps to global index ``keep_indices[i]``.
    """

    def __init__(self, shard_dir: str, shard_names: list[str],
                 shard_sizes: list[int],
                 keep_indices: Optional[list[int]] = None,
                 max_cached_shards: int = 3):
        self._shard_dir = shard_dir
        self._shard_names = list(shard_names)
        self._shard_sizes = list(shard_sizes)
        self._keep_indices = keep_indices
        self._max_cached = max_cached_shards

        # Build cumulative offsets for O(log N) shard lookup
        self._offsets: list[int] = []
        cum = 0
        for s in self._shard_sizes:
            self._offsets.append(cum)
            cum += s
        self._total_global = cum

        # LRU cache: shard_name -> list of entries
        self._cache: OrderedDict[str, list] = OrderedDict()

    # ---- public helpers --------------------------------------------------
    def __len__(self) -> int:
        if self._keep_indices is not None:
            return len(self._keep_indices)
        return self._total_global

    def __getitem__(self, idx: int):
        global_idx = self._keep_indices[idx] if self._keep_indices is not None else idx
        shard_name, local_idx = self._resolve(global_idx)
        shard_data = self._load_shard(shard_name)
        return shard_data[local_idx]

    def filter(self, keep_indices: list[int]) -> "_ShardIndex":
        """Return a filtered view that only exposes the given global indices."""
        return _ShardIndex(
            self._shard_dir, self._shard_names, self._shard_sizes,
            keep_indices=keep_indices,
            max_cached_shards=self._max_cached,
        )

    # ---- internal --------------------------------------------------------
    def _resolve(self, global_idx: int):
        """Return (shard_name, local_idx) for a global index."""
        if global_idx < 0 or global_idx >= self._total_global:
            raise IndexError(
                f"BPPM index {global_idx} out of range [0, {self._total_global})"
            )
        # Binary search: find the last shard whose offset <= global_idx
        i = bisect.bisect_right(self._offsets, global_idx) - 1
        return self._shard_names[i], global_idx - self._offsets[i]

    def _load_shard(self, shard_name: str):
        """Load a shard from disk, caching it with LRU eviction."""
        if shard_name in self._cache:
            # Move to end (most-recently-used)
            self._cache.move_to_end(shard_name)
            return self._cache[shard_name]

        # Evict oldest if at capacity
        while len(self._cache) >= self._max_cached:
            self._cache.popitem(last=False)

        shard_path = os.path.join(self._shard_dir, shard_name)
        with open(shard_path, "rb") as f:
            data = pickle.load(f)
        self._cache[shard_name] = data
        return data


def _save_bppm_compressed(bppm: np.ndarray, path: str, threshold: float = 0.001):
    """Save BPPM in compressed sparse upper-triangle float16 format.

    Only entries above *threshold* are stored as (row, col, value) triples
    with int16 indices and float16 values.  The file is a gzip-compressed
    numpy archive (.npz).  Storage reduction is typically 50-500× vs dense
    float32, depending on sequence length and pairing density.
    """
    L = bppm.shape[0]
    row, col = np.triu_indices(L)
    values = bppm[row, col].astype(np.float16)
    mask = values > np.float16(threshold)
    indices = np.stack([row[mask], col[mask]], axis=1).astype(np.int16)
    sparse_vals = values[mask]
    np.savez_compressed(
        path,
        L=np.array(L, dtype=np.int32),
        indices=indices,
        values=sparse_vals,
    )


def _load_bppm_compressed(path: str) -> np.ndarray:
    """Load a compressed BPPM file and reconstruct the dense float32 matrix."""
    data = np.load(path)
    L = int(data["L"])
    indices = data["indices"]   # (N, 2) int16
    values = data["values"].astype(np.float32)  # (N,) float32

    bppm = np.zeros((L, L), dtype=np.float32)
    bppm[indices[:, 0], indices[:, 1]] = values
    # Mirror to lower triangle (BPPM is symmetric)
    bppm = np.maximum(bppm, bppm.T)
    return bppm


def compute_bppm_from_seq(seq: str, symmetrize: bool = True) -> np.ndarray:
    """
    Use ViennaRNA to compute BPPM from a single RNA sequence.

    Args:
        seq: RNA/DNA sequence string.
        symmetrize: Whether to enforce matrix symmetry via max(BPPM, BPPM.T).

    Returns:
        np.ndarray: shape (L, L), dtype float32.
    """
    seq = (seq or "").strip().upper().replace("T", "U")
    if len(seq) == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # ViennaRNA supports ambiguous bases; keep only alphabetic chars for safety.
    seq = "".join(ch if ch.isalpha() else "N" for ch in seq)

    try:
        import RNA  # type: ignore
    except ImportError as e:
        raise ImportError(
            "ViennaRNA Python bindings (RNA) are required for on_the_fly BPPM "
            "computation. Please install ViennaRNA."
        ) from e

    fc = RNA.fold_compound(seq)
    fc.pf()
    bppm_raw = np.array(fc.bpp(), dtype=np.float32)
    bppm = bppm_raw[1:len(seq) + 1, 1:len(seq) + 1]
    if symmetrize:
        bppm = np.maximum(bppm, bppm.T)
    return bppm.astype(np.float32, copy=False)


def _extract_bppm_vectors(bppm: np.ndarray) -> dict:
    """Derive compact per-position vectors from a full (L, L) BPPM matrix.

    Returns a dict with:
      row_sum      (L,)     float32 — sum of pairing probs per position
      entropy      (L,)     float32 — normalized pairing-distribution entropy
      top1_partner (L,)     int64   — index of max-prob pairing partner
      top1_prob    (L,)     float32 — probability of that partner
      cross_pair   (L-1,)   float32 — unnormalized cross-pairing sum per boundary
    """
    bppm = np.clip(bppm, 0.0, 1.0).astype(np.float32)
    L = bppm.shape[0]
    eps = 1e-8

    # row_sum
    row_sum = bppm.sum(axis=-1)

    # entropy (normalized by L to match model convention)
    p_safe = np.clip(bppm, eps, None)
    entropy = -(p_safe * np.log(p_safe)).sum(axis=-1) / max(L, 1)

    # top-1 pairing partner (exclude self-pairing on diagonal)
    np.fill_diagonal(bppm, 0.0)
    top1_partner = bppm.argmax(axis=-1).astype(np.int64)
    top1_prob = bppm.max(axis=-1)

    # cross-pair sum for each boundary position (length L-1)
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


def _save_bppm_vectors(vectors: dict, path: str):
    """Save precomputed BPPM vectors to a compressed .npz file."""
    np.savez_compressed(
        path,
        row_sum=vectors["row_sum"],
        entropy=vectors["entropy"],
        top1_partner=vectors["top1_partner"],
        top1_prob=vectors["top1_prob"],
        cross_pair=vectors["cross_pair"],
    )


def _load_bppm_vectors(path: str) -> Optional[dict]:
    """Load BPPM vectors from .npz. Returns None if file is in old full-matrix format."""
    data = np.load(path)
    if "row_sum" not in data:
        return None  # old format with full matrix
    return {
        "row_sum": data["row_sum"].astype(np.float32),
        "entropy": data["entropy"].astype(np.float32),
        "top1_partner": data["top1_partner"].astype(np.int64),
        "top1_prob": data["top1_prob"].astype(np.float32),
        "cross_pair": data["cross_pair"].astype(np.float32),
    }


class BPPMPrefetcher:
    """Background BPPM prefetch worker pool.

    Operates at the filesystem level: computes BPPM for sequences via
    ViennaRNA in background threads and writes compressed .npz cache files.
    Because ViennaRNA's ``fc.pf()`` releases the GIL, ThreadPoolExecutor
    achieves good CPU parallelism on multi-core machines.

    DataLoader worker processes then pick up the cached .npz files via
    ``_load_or_compute_bppm`` without any cross-process coordination.
    """

    def __init__(self, max_workers: int = 4, bppm_compute_kwargs: Optional[dict] = None):
        self._max_workers = max_workers
        self._bppm_compute_kwargs = bppm_compute_kwargs or {}
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: list = []
        self._lock = threading.Lock()
        self._started = False

    def start(self):
        """Launch the background thread pool (idempotent)."""
        if self._started:
            return
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._started = True

    def submit(self, seq: str, cache_path: str):
        """Enqueue one sequence for background BPPM computation + caching."""
        if not self._started:
            self.start()
        future = self._executor.submit(
            self._compute_and_cache, seq, cache_path
        )
        with self._lock:
            self._futures.append(future)

    def submit_all(self, sequences: list[str], cache_paths: list[str]):
        """Enqueue many sequences at once."""
        if not self._started:
            self.start()
        for seq, path in zip(sequences, cache_paths):
            if not os.path.exists(path):
                future = self._executor.submit(
                    self._compute_and_cache, seq, path
                )
                with self._lock:
                    self._futures.append(future)

    def wait(self):
        """Block until all submitted tasks complete."""
        with self._lock:
            futures = list(self._futures)
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass  # Best-effort prefetch; __getitem__ falls back to sync

    def shutdown(self):
        """Shut down the thread pool gracefully."""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
            self._started = False

    @staticmethod
    def _compute_and_cache(seq: str, cache_path: str):
        """Compute BPPM, extract vectors, and write compressed cache file (atomic rename)."""
        raw_bppm = compute_bppm_from_seq(seq)
        vectors = _extract_bppm_vectors(raw_bppm)
        tmp_path = None
        try:
            cache_dir = os.path.dirname(cache_path)
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".npz",
                prefix="bppm_tmp_",
                dir=cache_dir,
                delete=False,
            ) as tmp_file:
                tmp_path = tmp_file.name
                _save_bppm_vectors(vectors, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


class RealRNADataset(Dataset):
    def __init__(self, fasta_path, pkl_path=None,
                 label_map_path=None,
                 unknown_family_policy: str = "filter",
                 save_label_map: Optional[bool] = None,
                 warn_unknown_families: bool = True,
                 dotbracket_path=None,
                 use_masking=False,
                 masking_type='simple',
                 masking_config=None,
                 bppm_mode: str = "precomputed",
                 cache_dir: Optional[str] = None,
                 bppm_compute_kwargs: Optional[dict[str, Any]] = None,
                 prefetch_workers: int = 0):
        """
        Args:
            fasta_path (str): Path to the FASTA file.
            pkl_path (str): Path to the pickle file containing BPPMs. Required in precomputed mode.
            label_map_path (str): Path to the JSON file for label mapping.
            dotbracket_path (str): Path to the file containing dot-bracket structures.
            use_masking (bool): Whether to use MLM masking.
            masking_type (str): Type of masking ('simple', 'pairing_aware').
            masking_config (object): Configuration object for masking parameters.
            bppm_mode (str): 'precomputed' or 'on_the_fly'.
            cache_dir (str): Optional directory for caching on-the-fly BPPM.
            bppm_compute_kwargs (dict): Extra kwargs for compute_bppm_from_seq.
            prefetch_workers (int): Number of background threads for async BPPM
                prefetch.  When > 0 and bppm_mode='on_the_fly' with cache_dir set,
                BPPM is precomputed in the background so GPU training isn't blocked.
                Default 0 (no prefetch — compute on first access).
        """
        self.data = []
        self.use_masking = use_masking
        self.masking_type = masking_type
        self.masking_config = masking_config
        self.bppm_mode = bppm_mode
        self.cache_dir = cache_dir
        self.bppm_compute_kwargs = bppm_compute_kwargs or {}
        self.pad_token_id = PAD_TOKEN_ID
        self.mask_token_id = MASK_TOKEN_ID
        self.unk_token_id = UNK_TOKEN_ID
        self.bppm_cache_key_version = BPPM_CACHE_KEY_VERSION

        if self.bppm_mode not in {"precomputed", "on_the_fly"}:
            raise ValueError(f"Unknown bppm_mode: {self.bppm_mode}. Must be 'precomputed' or 'on_the_fly'.")
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize masker if needed
        if self.use_masking:
            self._init_masker()
            
        # Load BPPM data
        self.bppm_data = None
        if self.bppm_mode == "precomputed":
            if not pkl_path:
                raise ValueError("pkl_path is required when bppm_mode='precomputed'")
            self.bppm_data = self._load_bppm(pkl_path)
            
        self.sequences = []
        self.labels = []
        self.family_to_id = {}
        
        map_loaded = False
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                self.family_to_id = json.load(f)
            map_loaded = True

        fasta_seqs: list[str] = []
        fasta_families: list[str] = []
        with open(fasta_path, 'r', encoding='utf-8') as f:
            current_seq = ""
            current_family = None
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_family is not None:
                        fasta_seqs.append(current_seq)
                        fasta_families.append(current_family)
                    
                    header = line[1:]
                    parts = header.split()
                    if len(parts) > 1:
                        current_family = parts[1]
                    elif '|' in header:
                        # Try pipe splitting if space splitting fails
                        pipe_parts = header.split('|')
                        current_family = pipe_parts[-1] if len(pipe_parts) > 1 else 'unknown'
                    else:
                        current_family = 'unknown'
                    
                    current_seq = ""
                else:
                    current_seq += line
            if current_family is not None:
                fasta_seqs.append(current_seq)
                fasta_families.append(current_family)

        if self.bppm_mode == "precomputed":
            assert self.bppm_data is not None
            assert len(fasta_seqs) == len(self.bppm_data), \
                f"Mismatch: FASTA has {len(fasta_seqs)} seqs, PKL has {len(self.bppm_data)} items."

        dotbrackets_all = None
        if dotbracket_path:
            with open(dotbracket_path, 'r', encoding='utf-8') as f:
                dotbrackets_all = [line.strip() for line in f if line.strip()]
            assert len(fasta_seqs) == len(dotbrackets_all), \
                f"Mismatch: FASTA has {len(fasta_seqs)} seqs, Dotbracket has {len(dotbrackets_all)} items."

        bppm_all = self.bppm_data
        kept_bppm_indices: list[int] = []
        kept_dotbrackets = []
        unknown_families = set()
        unknown_count = 0

        for i, (seq, family) in enumerate(zip(fasta_seqs, fasta_families)):
            if map_loaded:
                if family in self.family_to_id:
                    label_id = self.family_to_id[family]
                else:
                    if unknown_family_policy == "add":
                        label_id = len(self.family_to_id)
                        self.family_to_id[family] = label_id
                    elif unknown_family_policy == "filter":
                        unknown_families.add(family)
                        unknown_count += 1
                        continue
                    elif unknown_family_policy == "error":
                        raise ValueError(
                            f"Unknown family '{family}' encountered while using label_map_path={label_map_path}"
                        )
                    else:
                        raise ValueError(f"Unknown unknown_family_policy: {unknown_family_policy}")
            else:
                if family not in self.family_to_id:
                    self.family_to_id[family] = len(self.family_to_id)
                label_id = self.family_to_id[family]

            self.sequences.append(seq)
            self.labels.append(label_id)
            if self.bppm_mode == "precomputed":
                kept_bppm_indices.append(i)
            if dotbrackets_all is not None:
                kept_dotbrackets.append(dotbrackets_all[i])

        if map_loaded and unknown_family_policy == "filter" and warn_unknown_families and unknown_families:
            families_preview = ", ".join(sorted(unknown_families)[:20])
            extra = "" if len(unknown_families) <= 20 else f" ... (+{len(unknown_families) - 20})"
            print(
                f"Warning: filtered {unknown_count} samples with unseen families in label_map: "
                f"{families_preview}{extra}"
            )

        if self.bppm_mode == "precomputed":
            assert bppm_all is not None
            if isinstance(bppm_all, _ShardIndex):
                self.bppm_data = bppm_all.filter(kept_bppm_indices)
            else:
                self.bppm_data = [bppm_all[i] for i in kept_bppm_indices]
        else:
            self.bppm_data = None

        self.dotbrackets = []
        if dotbrackets_all is not None:
            self.dotbrackets = kept_dotbrackets

        should_save = False
        if label_map_path:
            if save_label_map is not None:
                should_save = save_label_map
            else:
                should_save = (not map_loaded) or (unknown_family_policy == "add")

        if label_map_path and should_save:
            dir_name = os.path.dirname(label_map_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(label_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.family_to_id, f, ensure_ascii=False)
                
        # Vocabulary mapping
        # A:0, U/T:1, C:2, G:3, PAD:4, MASK:5, UNK/N:6
        self.vocab_map = {
            'A': RNA_TOKEN_IDS["A"], 'a': RNA_TOKEN_IDS["A"],
            'U': RNA_TOKEN_IDS["U"], 'u': RNA_TOKEN_IDS["U"],
            'T': RNA_TOKEN_IDS["U"], 't': RNA_TOKEN_IDS["U"],
            'C': RNA_TOKEN_IDS["C"], 'c': RNA_TOKEN_IDS["C"],
            'G': RNA_TOKEN_IDS["G"], 'g': RNA_TOKEN_IDS["G"],
            'N': self.unk_token_id, 'n': self.unk_token_id
        }
        
        if self.bppm_mode == "precomputed":
            assert self.bppm_data is not None
            assert len(self.sequences) == len(self.bppm_data), \
                f"Mismatch: FASTA has {len(self.sequences)} seqs, PKL has {len(self.bppm_data)} items."
        assert len(self.sequences) == len(self.labels), "Mismatch between sequences and labels"

        # Async BPPM prefetch (on_the_fly mode with cache_dir)
        self._prefetcher: Optional[BPPMPrefetcher] = None
        if (
            prefetch_workers > 0
            and self.bppm_mode == "on_the_fly"
            and self.cache_dir is not None
        ):
            cache_paths = [self._cache_file_path(s) for s in self.sequences]
            self._prefetcher = BPPMPrefetcher(
                max_workers=prefetch_workers,
                bppm_compute_kwargs=self.bppm_compute_kwargs,
            )
            self._prefetcher.submit_all(self.sequences, cache_paths)
            print(
                f"BPPM prefetch started: {len(self.sequences)} sequences, "
                f"{prefetch_workers} background threads, cache_dir={self.cache_dir}"
            )

    def wait_prefetch(self):
        """Block until all background BPPM prefetch tasks are finished."""
        if self._prefetcher is not None:
            self._prefetcher.wait()
            self._prefetcher.shutdown()
            self._prefetcher = None

    @staticmethod
    def _load_bppm(pkl_path: str):
        """Load BPPM data from a single .pkl file or a shard directory.

        Single file: returns a plain list (all data in memory).
        Shard directory: returns a _ShardIndex that lazy-loads shards on demand
                         with an in-memory LRU cache, avoiding OOM when the
                         full dataset does not fit in RAM.
        """
        if os.path.isdir(pkl_path):
            manifest_path = os.path.join(pkl_path, "manifest.json")
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(
                    f"pkl_path is a directory but no manifest.json found in {pkl_path}. "
                    f"Either point to a .pkl file, or use a directory created by generate_bppm.py."
                )
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            shard_names = manifest["shards"]

            # Use sizes from manifest when available (fast path);
            # otherwise load each shard briefly to measure its length.
            if "shard_sizes" in manifest:
                shard_sizes = manifest["shard_sizes"]
            else:
                shard_sizes = []
                for name in shard_names:
                    shard_path = os.path.join(pkl_path, name)
                    with open(shard_path, 'rb') as f:
                        shard_sizes.append(len(pickle.load(f)))

            if sum(shard_sizes) != manifest.get("total", sum(shard_sizes)):
                print(
                    f"Warning: manifest total ({manifest.get('total')}) "
                    f"!= sum(shard_sizes) ({sum(shard_sizes)}). Using computed total."
                )

            return _ShardIndex(pkl_path, shard_names, shard_sizes)
        else:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)

    def _init_masker(self):
        """Initialize the masking strategy based on config"""
        kwargs = self._extract_masking_kwargs()
        if self.masking_type == 'simple':
            self.masker = self._build_masker(SimpleMasking, kwargs)
        elif self.masking_type == 'pairing_aware':
            self.masker = self._build_masker(PairingAwareMasking, kwargs)
        else:
            raise ValueError(f"Unknown masking type: {self.masking_type}")

    def _extract_masking_kwargs(self) -> dict[str, Any]:
        """Extract all masking parameters from dict/object config (full passthrough)."""
        if self.masking_config is None:
            return {}
        if isinstance(self.masking_config, dict):
            raw = dict(self.masking_config)
        elif hasattr(self.masking_config, "__dict__"):
            raw = dict(vars(self.masking_config))
        else:
            raw = {
                k: getattr(self.masking_config, k)
                for k in dir(self.masking_config)
                if not k.startswith("_") and not callable(getattr(self.masking_config, k))
            }
        # Runtime control fields should not be passed into masking class constructor.
        raw.pop("use_masking", None)
        raw.pop("masking_type", None)
        return {k: v for k, v in raw.items() if v is not None}

    @staticmethod
    def _build_masker(masker_cls, kwargs: dict[str, Any]):
        """Build masker with kwargs filtered by constructor signature."""
        sig = inspect.signature(masker_cls.__init__)
        valid_keys = {p.name for p in sig.parameters.values() if p.name != "self"}
        filtered = {k: v for k, v in kwargs.items() if k in valid_keys}
        return masker_cls(**filtered)

    def _cache_file_path(self, seq: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        seq_norm = (seq or "").strip().upper().replace("T", "U")
        seq_hash = hashlib.md5(seq_norm.encode("utf-8")).hexdigest()
        kwargs_text = json.dumps(
            self.bppm_compute_kwargs,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        kwargs_hash = hashlib.md5(kwargs_text.encode("utf-8")).hexdigest()
        return os.path.join(
            self.cache_dir,
            f"{seq_hash}_v{self.bppm_cache_key_version}_{kwargs_hash}.npz",
        )

    def _load_or_compute_bppm(self, idx: int, raw_seq: str) -> dict:
        """Load or compute BPPM and return as a vector dict.

        Returns dict with keys: row_sum, entropy, top1_partner, top1_prob, cross_pair.
        """
        if self.bppm_mode == "precomputed":
            assert self.bppm_data is not None
            bppm_entry = self.bppm_data[idx]
            if isinstance(bppm_entry, dict) and "bppm" in bppm_entry:
                raw_bppm = np.asarray(bppm_entry["bppm"], dtype=np.float32)
            elif isinstance(bppm_entry, dict) and "row_sum" in bppm_entry:
                return {k: np.asarray(bppm_entry[k]) for k in
                        ("row_sum", "entropy", "top1_partner", "top1_prob", "cross_pair")}
            else:
                raw_bppm = np.asarray(bppm_entry, dtype=np.float32)
            return _extract_bppm_vectors(raw_bppm)

        cache_path = self._cache_file_path(raw_seq)

        # Try new vector-format cache
        if cache_path and os.path.exists(cache_path):
            vecs = _load_bppm_vectors(cache_path)
            if vecs is not None:
                return vecs
            # Old full-matrix cache — load, extract vectors, re-cache
            try:
                raw_bppm = _load_bppm_compressed(cache_path)
                vecs = _extract_bppm_vectors(raw_bppm)
                _save_bppm_vectors(vecs, cache_path)
                return vecs
            except Exception:
                pass

        # Backward-compat: try legacy .npy file
        if cache_path:
            legacy_path = cache_path[:-4] + ".npy"
            if os.path.exists(legacy_path):
                try:
                    raw_bppm = np.load(legacy_path).astype(np.float32, copy=False)
                    vecs = _extract_bppm_vectors(raw_bppm)
                    _save_bppm_vectors(vecs, cache_path)
                    return vecs
                except Exception:
                    pass

        # Compute from sequence
        raw_bppm = compute_bppm_from_seq(raw_seq, **self.bppm_compute_kwargs)
        vecs = _extract_bppm_vectors(raw_bppm)
        if cache_path:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    suffix=".npz",
                    prefix="bppm_tmp_",
                    dir=self.cache_dir,
                    delete=False,
                ) as tmp_file:
                    tmp_path = tmp_file.name
                    _save_bppm_vectors(vecs, tmp_path)
                os.replace(tmp_path, cache_path)
            finally:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        return vecs

    def _apply_masking(self, x, bppm_vectors):
        """
        Apply masking strategy using precomputed BPPM vectors.

        Args:
            x: (L,) sequence tensor
            bppm_vectors: dict with top1_partner (L,), top1_prob (L,)

        Returns:
            masked_x, mask_positions, labels
        """
        x_batch = x.unsqueeze(0)
        partner_batch = torch.from_numpy(bppm_vectors["top1_partner"]).unsqueeze(0)
        prob_batch = torch.from_numpy(bppm_vectors["top1_prob"]).unsqueeze(0)

        masked_x, mask_positions, labels = self.masker(x_batch, partner_batch, prob_batch)

        return masked_x.squeeze(0), mask_positions.squeeze(0), labels.squeeze(0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 1. Get Sequence
        raw_seq = self.sequences[idx]
        seq_len = len(raw_seq)

        # Convert to indices
        seq_indices = [self.vocab_map.get(char, self.unk_token_id) for char in raw_seq]
        x = torch.tensor(seq_indices, dtype=torch.long)

        # 2. Get BPPM vectors (instead of full matrix)
        bppm_vecs = self._load_or_compute_bppm(idx, raw_seq)

        # Sanitize NaN in float vectors
        for k in ("row_sum", "entropy", "top1_prob", "cross_pair"):
            v = bppm_vecs[k]
            if np.isnan(v).any():
                v = np.nan_to_num(v, nan=0.0)
                v[v < 0.01] = 0.0
                bppm_vecs[k] = v

        # Ensure vector length matches sequence length
        limit = min(seq_len, len(bppm_vecs["row_sum"]))
        x = x[:limit]
        seq_len = limit
        for k in ("row_sum", "entropy", "top1_partner", "top1_prob"):
            bppm_vecs[k] = bppm_vecs[k][:limit]
        if len(bppm_vecs["cross_pair"]) >= limit:
            bppm_vecs["cross_pair"] = bppm_vecs["cross_pair"][:limit - 1]

        # Get label and dotbracket
        family_label = self.labels[idx]
        dotbracket = self.dotbrackets[idx] if self.dotbrackets else None

        # 3. Apply Masking (if enabled) — uses top1_partner/top1_prob only
        if self.use_masking:
            masked_x, mask_positions, labels = self._apply_masking(x, bppm_vecs)
            return (masked_x, x, seq_len, mask_positions, labels, family_label, dotbracket,
                    bppm_vecs["row_sum"], bppm_vecs["entropy"], bppm_vecs["cross_pair"])
        else:
            return (x, seq_len, family_label, dotbracket,
                    bppm_vecs["row_sum"], bppm_vecs["entropy"], bppm_vecs["cross_pair"])

    def __getstate__(self):
        """Exclude unpicklable _prefetcher (ThreadPoolExecutor) for Windows multiprocessing."""
        state = self.__dict__.copy()
        state["_prefetcher"] = None
        return state

    def __setstate__(self, state):
        """Restore state in worker process — prefetcher is not needed there."""
        self.__dict__.update(state)

def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads BPPM-derived vectors (row_sum, entropy, cross_pair) instead of full matrix.
    """
    # Masking: 10 items, No Masking: 7 items
    item = batch[0]
    has_masking = (len(item) == 10)

    # 1. Find max length
    if has_masking:
        max_len = max([item[2] for item in batch])
    else:
        max_len = max([item[1] for item in batch])

    # 2. Prepare tensors
    batch_size = len(batch)
    padded_x = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float)

    # BPPM vector pads
    padded_row_sum = torch.zeros((batch_size, max_len), dtype=torch.float)
    padded_entropy = torch.zeros((batch_size, max_len), dtype=torch.float)
    padded_cross_pair = torch.zeros((batch_size, max(max_len - 1, 0)), dtype=torch.float)

    padded_raw_x = None
    padded_mask_pos = None
    padded_labels = None

    if has_masking:
        padded_raw_x = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
        padded_mask_pos = torch.zeros((batch_size, max_len), dtype=torch.bool)
        padded_labels = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)

    for i, item in enumerate(batch):
        if has_masking:
            masked_x, raw_x, length, mask_pos, labels, _, _, row_sum, entropy, cross_pair = item
            padded_x[i, :length] = masked_x
            padded_raw_x[i, :length] = raw_x
        else:
            x, length, _, _, row_sum, entropy, cross_pair = item
            padded_x[i, :length] = x

        # Pad BPPM vectors
        actual_l = min(length, max_len)
        padded_row_sum[i, :actual_l] = torch.from_numpy(row_sum[:actual_l])
        padded_entropy[i, :actual_l] = torch.from_numpy(entropy[:actual_l])
        cp_len = min(len(cross_pair), max_len - 1)
        if cp_len > 0:
            padded_cross_pair[i, :cp_len] = torch.from_numpy(cross_pair[:cp_len])

        # Create Mask
        mask[i, :length] = 1.0

        if has_masking:
            padded_mask_pos[i, :length] = mask_pos
            padded_labels[i, :length] = labels

    # Collect family labels and dotbrackets
    # Both formats end with (family_label, dotbracket, row_sum, entropy, cross_pair),
    # so negative indices are the same: family_label=-5, dotbracket=-4
    family_labels = torch.tensor([item[-5] for item in batch], dtype=torch.long)
    dotbrackets = [item[-4] for item in batch]

    if has_masking:
        return (padded_x, padded_raw_x, mask, padded_mask_pos, padded_labels,
                family_labels, dotbrackets,
                padded_row_sum, padded_entropy, padded_cross_pair)
    else:
        return (padded_x, mask, family_labels, dotbrackets,
                padded_row_sum, padded_entropy, padded_cross_pair)
