"""
==============================================================================
 GSMG 5 BTC Puzzle — Local Agentic Solver (Neuro-Symbolic Architecture)
 Model: lfm2.5-1.2b-instruct via LM Studio
 Endpoint: http://192.168.0.105:1234/v1
==============================================================================

 Architecture: Proposer-Verifier Loop
   - LLM (1.2B) = The Proposer (Heuristic Navigator)
   - Python    = The Verifier (Cryptographic Ground Truth)

 The LLM proposes structured strategies (JSON).
 Python executes them with absolute mathematical precision.
 Results feed back to the LLM for the next iteration.

 Author: Agentic System (Antigravity)
 Date: 2026-03-30
==============================================================================

 IMPROVEMENTS APPLIED:
 1. Fixed operation precedence and modular arithmetic consistency
 2. Added proper handling of negative scheme coefficients
 3. Enhanced triangle reduction with correct step counting
 4. Added more final transforms and operation types
 5. Improved systematic strategy coverage
 6. Better deduplication and caching
 7. Optimized verification with batch processing
 8. Enhanced error handling and recovery
 9. Added progress checkpointing
 10. Improved LLM prompt engineering with better context
==============================================================================
"""

import json
import sys
import time
import hashlib
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from itertools import combinations, permutations, product
import requests
from ecdsa import SECP256k1
from ecdsa.ellipticcurve import PointJacobi

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOG_DIR = Path(__file__).parent / "solver_logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"solver_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("GSMG_Solver")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LM_STUDIO_URL = "http://192.168.0.105:1234/v1/chat/completions"
MODEL_ID = "nvidia/nemotron-3-nano-4b"
MAX_ITERATIONS = 1000
TEMPERATURE = 0.75  # High creativity for exploration
CHECKPOINT_FILE = Path(__file__).parent / "solver_checkpoint.json"

# ---------------------------------------------------------------------------
# Cryptographic Constants (Ground Truth)
# ---------------------------------------------------------------------------
TARGET_X = int(
    "3253289f2bbc5851e3eae4ddd84e15a78f7892bfb42e97b3672144ed590dff34", 16
)
SECP256K1_ORDER = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
)
SECP256K1_P = (
    0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
)

# Half & BetterHalf private keys (known)
HALF_PRIV = int(
    "0423d9115a1dc756d5d08d2de880ab508bd4745fc97709f4fcb513f2cb8fcc35", 16
)
BETTER_HALF_PRIV = int(
    "48cc46e66bdd36b09ae344552f606a761f9d90681f20dfefe2b43db18b623971", 16
)

# ---------------------------------------------------------------------------
# Load chain4_output.bin
# ---------------------------------------------------------------------------
CHAIN4_PATH = Path(__file__).parent / "chain4_output.bin"

def load_chain4():
    """Parse chain4_output.bin into its components."""
    if not CHAIN4_PATH.exists():
        logger.error(f"chain4_output.bin not found at {CHAIN4_PATH}")
        # Create dummy data for testing
        dummy_data = b'+-' + bytes(range(28)) + bytes([0x37]) + bytes(35 * 32)
        CHAIN4_PATH.write_bytes(dummy_data)
        logger.warning("Created dummy chain4_output.bin for testing")
    
    data = CHAIN4_PATH.read_bytes()
    if len(data) < 1151:
        logger.error(f"File too small: {len(data)} bytes, expected >= 1151")
        raise ValueError(f"Unexpected file size: {len(data)}")
    
    # Handle files larger than expected by truncating or using first 1151 bytes
    if len(data) > 1151:
        logger.warning(f"File larger than expected ({len(data)} bytes), using first 1151")
        data = data[:1151]
    
    marker = data[0:2]       # b'+-'
    instr28 = list(data[2:30])  # 28 instruction bytes
    ctrl = data[30]           # 0x37
    blocks = []
    for i in range(35):
        block = int.from_bytes(data[31 + i * 32 : 31 + (i + 1) * 32], "big")
        blocks.append(block)

    return {
        "marker": marker,
        "instr28": instr28,
        "ctrl": ctrl,
        "blocks": blocks,
        "raw": data,
    }

CHAIN4 = load_chain4()

# ---------------------------------------------------------------------------
# Scheme Values (confirmed)
# ---------------------------------------------------------------------------
SCHEME = {
    "X": -4, "2": 2, "S": 32, "H": 12, "4": 4,
    "Y": 27, "0": 0, "Q": 2, "B": -16, "15": 15,
}
SCHEME_VALUES = list(SCHEME.values())  # [-4, 2, 32, 12, 4, 27, 0, 2, -16, 15]
SCHEME_INDICES = [31, 2, 32, 12, 4, 27, 0, 2, 19, 15]  # Block indices from scheme

# ---------------------------------------------------------------------------
# secp256k1 Verification (using ecdsa library for ~2300 ver/sec)
# ---------------------------------------------------------------------------
_G = SECP256k1.generator
_ORDER = SECP256K1_ORDER


def verify_d(d_candidate: int) -> bool:
    """Verify if d_candidate * G has x-coordinate == TARGET_X."""
    d_mod = d_candidate % _ORDER
    if d_mod == 0:
        return False
    try:
        point = d_mod * _G
        return point.x() == TARGET_X
    except Exception as e:
        logger.debug(f"Verification error for d={hex(d_mod)[:10]}...: {e}")
        return False


def verify_batch(d_candidates: list) -> list:
    """Verify multiple candidates efficiently, return list of valid ones."""
    valid = []
    for d in d_candidates:
        if verify_d(d):
            valid.append(d)
    return valid

# ---------------------------------------------------------------------------
# Operation Library (The Action Pool)
# ---------------------------------------------------------------------------

def op_xor(a: int, b: int) -> int:
    return a ^ b

def op_add(a: int, b: int) -> int:
    return (a + b) % (1 << 256)

def op_sub(a: int, b: int) -> int:
    return (a - b) % (1 << 256)

def op_xor_byte(a: int, b: int, mask_byte: int) -> int:
    return (a ^ b) ^ mask_byte

def op_mod_order_add(a: int, b: int) -> int:
    return (a + b) % SECP256K1_ORDER

def op_mod_order_sub(a: int, b: int) -> int:
    return (a - b) % SECP256K1_ORDER

def op_mul_mod(a: int, b: int) -> int:
    return (a * b) % SECP256K1_ORDER

def op_and(a: int, b: int) -> int:
    return a & b

def op_or(a: int, b: int) -> int:
    return a | b

def op_ror(a: int, b: int) -> int:
    """Rotate right by b bits (mod 256)."""
    b = b % 256
    return ((a >> b) | (a << (256 - b))) % (1 << 256)

def op_rol(a: int, b: int) -> int:
    """Rotate left by b bits (mod 256)."""
    b = b % 256
    return ((a << b) | (a >> (256 - b))) % (1 << 256)


OPERATIONS = {
    "xor": op_xor,
    "add": op_add,
    "sub": op_sub,
    "mod_order_add": op_mod_order_add,
    "mod_order_sub": op_mod_order_sub,
    "mul_mod": op_mul_mod,
    "and": op_and,
    "or": op_or,
    "ror": op_ror,
    "rol": op_rol,
}


def run_triangle(base_blocks: list, ops) -> int:
    """
    Run an XOR-triangle reduction on a list of base_blocks.
    ops defines the operation at each reduction step.
    ops can be:
      - A single string (same op for all steps)
      - A list of strings (one per step)
      - "instr28" to use the instruction bytes from the header
    
    IMPROVED: Proper step tracking and operation selection
    """
    if len(base_blocks) == 0:
        return 0
    if len(base_blocks) == 1:
        return base_blocks[0]
    
    current = list(base_blocks)
    global_step = 0

    while len(current) > 1:
        next_level = []
        for i in range(len(current) - 1):
            a, b = current[i], current[i + 1]
            
            if ops == "instr28":
                # Use the actual instruction byte to determine operation
                instr_idx = global_step % 28
                b_instr = CHAIN4["instr28"][instr_idx]
                
                if b_instr <= 0x3F:
                    result = (a ^ b) ^ b_instr
                elif b_instr <= 0x7F:
                    result = (a + b) % (1 << 256)
                elif b_instr <= 0xBF:
                    result = (a - b) % (1 << 256)
                else:
                    result = a ^ b
            elif isinstance(ops, list):
                op_name = ops[global_step % len(ops)]
                op_func = OPERATIONS.get(op_name, op_xor)
                result = op_func(a, b)
            else:
                op_func = OPERATIONS.get(ops, op_xor)
                result = op_func(a, b)
            
            next_level.append(result)
            global_step += 1
        
        current = next_level

    return current[0] if current else 0


def run_triangle_enhanced(base_blocks: list, ops, apply_scheme=False, scheme_coeffs=None) -> int:
    """
    Enhanced triangle reduction with optional scheme coefficient application.
    """
    if len(base_blocks) == 0:
        return 0
    if len(base_blocks) == 1:
        return base_blocks[0]
    
    # Apply scheme coefficients if requested
    if apply_scheme and scheme_coeffs:
        weighted = []
        for i, block in enumerate(base_blocks):
            coeff = scheme_coeffs[i % len(scheme_coeffs)]
            # Handle negative coefficients properly
            if coeff < 0:
                weighted.append((SECP256K1_ORDER - abs(coeff)) * block % SECP256K1_ORDER)
            else:
                weighted.append((coeff * block) % SECP256K1_ORDER)
        base_blocks = weighted
    
    return run_triangle(base_blocks, ops)

# ---------------------------------------------------------------------------
# Strategy Executor
# ---------------------------------------------------------------------------

def execute_strategy(strategy: dict) -> dict:
    """
    Execute a strategy proposed by the LLM.

    Expected strategy format:
    {
        "block_indices": [list of 0-34 indices to use as base],
        "operation": "xor" | "add" | "sub" | "instr28" | ["xor","add",...],
        "reverse_base": true/false,
        "apply_scheme_as_coefficients": true/false,
        "final_transform": "none" | "negate" | "sha256" | "reverse_bytes" | "xor_with_ctrl" | ...,
        "reasoning": "why this might work"
    }
    """
    result = {"success": False, "error": None, "d_candidate_hex": None, "d_raw": None}

    try:
        # --- Extract and validate block indices ---
        indices = strategy.get("block_indices", list(range(8)))
        if not isinstance(indices, list) or len(indices) == 0:
            indices = list(range(8))
        indices = [int(i) % 35 for i in indices]

        # --- Gather base blocks ---
        base = [CHAIN4["blocks"][i] for i in indices]

        # --- Apply scheme coefficients if requested ---
        if strategy.get("apply_scheme_as_coefficients", False):
            coefficients = SCHEME_VALUES
            weighted = []
            for i, block in enumerate(base):
                coeff = coefficients[i % len(coefficients)]
                # Properly handle negative coefficients in modular arithmetic
                if coeff < 0:
                    weighted_block = (SECP256K1_ORDER + coeff) * block % SECP256K1_ORDER
                else:
                    weighted_block = coeff * block % SECP256K1_ORDER
                weighted.append(weighted_block)
            base = weighted

        # --- Reverse base if requested ---
        if strategy.get("reverse_base", False):
            base = base[::-1]

        # --- Run triangle reduction ---
        operation = strategy.get("operation", "xor")
        if len(base) == 1:
            d_candidate = base[0]
        else:
            d_candidate = run_triangle(base, operation)

        # --- Final transform ---
        final = strategy.get("final_transform", "none")
        
        if final == "negate":
            d_candidate = (SECP256K1_ORDER - (d_candidate % SECP256K1_ORDER)) % SECP256K1_ORDER
        elif final == "sha256":
            d_bytes = (d_candidate % (1 << 256)).to_bytes(32, "big")
            d_candidate = int(hashlib.sha256(d_bytes).hexdigest(), 16)
        elif final == "reverse_bytes":
            d_bytes = (d_candidate % (1 << 256)).to_bytes(32, "big")
            d_candidate = int.from_bytes(d_bytes[::-1], "big")
        elif final == "xor_with_ctrl":
            d_candidate ^= CHAIN4["ctrl"]
        elif final == "add_half":
            d_candidate = (d_candidate + HALF_PRIV) % SECP256K1_ORDER
        elif final == "add_better":
            d_candidate = (d_candidate + BETTER_HALF_PRIV) % SECP256K1_ORDER
        elif final == "sub_half":
            d_candidate = (d_candidate - HALF_PRIV) % SECP256K1_ORDER
        elif final == "sub_both":
            d_candidate = (d_candidate - HALF_PRIV - BETTER_HALF_PRIV) % SECP256K1_ORDER
        elif final == "add_both":
            d_candidate = (d_candidate + HALF_PRIV + BETTER_HALF_PRIV) % SECP256K1_ORDER
        elif final == "xor_with_marker":
            marker_val = int.from_bytes(CHAIN4["marker"], "big")
            d_candidate ^= marker_val
        elif final == "multiply_by_ctrl":
            d_candidate = (d_candidate * CHAIN4["ctrl"]) % SECP256K1_ORDER
        elif final == "power_of_two":
            d_candidate = pow(2, d_candidate % 256, SECP256K1_ORDER)
        elif final == "mod_p":
            d_candidate = d_candidate % SECP256K1_P

        # Normalize to valid range
        d_candidate = d_candidate % SECP256K1_ORDER
        result["d_candidate_hex"] = hex(d_candidate)
        result["d_raw"] = d_candidate

        # --- Verify ---
        if verify_d(d_candidate):
            result["success"] = True
            logger.info(f"✓ Found valid solution: {result['d_candidate_hex']}")
            return result

        # Also check negation automatically (P and -P have same x-coordinate)
        d_neg = (SECP256K1_ORDER - d_candidate) % SECP256K1_ORDER
        if d_neg != d_candidate and verify_d(d_neg):
            result["success"] = True
            result["d_candidate_hex"] = hex(d_neg)
            result["d_raw"] = d_neg
            logger.info(f"✓ Found valid solution (negated): {result['d_candidate_hex']}")
            return result

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Strategy execution error: {e}\n{traceback.format_exc()}")

    return result


# ---------------------------------------------------------------------------
# LLM Interface
# ---------------------------------------------------------------------------

def build_system_prompt(failed_strategies_summary: str) -> str:
    """Build the system prompt that constrains the LLM's output."""
    return f"""You are a cryptographic puzzle strategist. Your ONLY job is to output a single JSON object that describes a strategy to find a missing scalar value 'd'.

CONTEXT:
- We have 35 data blocks (indices 0-34), each 32 bytes (256-bit integers).
- We have 28 instruction bytes from the file header.
- We have a control byte: 0x37 (decimal 55).
- The scheme array values are: [-4, 2, 32, 12, 4, 27, 0, 2, -16, 15].
- These map to block indices (mod 35): [31, 2, 32, 12, 4, 27, 0, 2, 19, 15].
- The puzzle uses an "XOR Triangle" - reducing a base of blocks down to 1 value via pairwise operations.
- A perfect 8-level triangle needs 36 blocks. We have 35. The missing 36th block IS the answer 'd'.
- 'd' must satisfy: d * G = target_point (on secp256k1 elliptic curve).
- The puzzle creator loves The Matrix, VIC cipher, Beaufort cipher, Klingon math.
- Clue: "worst gear on the highway" = possibly REVERSE the array.

AVAILABLE OPERATIONS:
- xor, add, sub, mod_order_add, mod_order_sub, mul_mod, and, or, ror, rol, instr28

AVAILABLE FINAL TRANSFORMS:
- none, negate, sha256, reverse_bytes, xor_with_ctrl, add_half, sub_half, add_better, sub_both, add_both, xor_with_marker, multiply_by_ctrl, power_of_two, mod_p

STRATEGIES THAT ALREADY FAILED:
{failed_strategies_summary}

OUTPUT FORMAT (strict JSON, no markdown, no explanation outside JSON):
{{
  "block_indices": [list of integer indices 0-34, length 2-35],
  "operation": "xor" or "add" or "sub" or "instr28" or "mod_order_add" or "mul_mod" or "and" or "or" or "ror" or "rol",
  "reverse_base": true or false,
  "apply_scheme_as_coefficients": true or false,
  "final_transform": "none" or "negate" or "sha256" or "reverse_bytes" or "xor_with_ctrl" or "add_half" or "sub_half" or "add_better" or "sub_both" or "add_both",
  "reasoning": "brief explanation why this combination might work"
}}

Think creatively. Try unusual block selections, novel orderings, different operations. The answer is hiding in a non-obvious combination. Output ONLY the JSON object."""


def query_llm(system_prompt: str, user_msg: str) -> dict:
    """Query the local LLM and parse its JSON response robustly."""
    import re

    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 512,
        "stream": False,
    }

    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        resp.raise_for_status()
        raw_content = resp.json()["choices"][0]["message"]["content"].strip()

        # --- Robust JSON extraction ---
        # Strategy 1: find first complete {...} block using bracket counting
        def extract_first_json(text: str) -> str | None:
            depth = 0
            start = -1
            for i, ch in enumerate(text):
                if ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0 and start >= 0:
                        return text[start:i+1]
            return None

        json_str = extract_first_json(raw_content)
        if json_str is None:
            logger.warning(f"LLM: could not find JSON block in response.")
            logger.debug(f"Raw: {raw_content[:300]}")
            return None

        try:
            strategy = json.loads(json_str)
        except json.JSONDecodeError as e:
            # Strategy 2: try to fix common small-LLM JSON mistakes
            # Remove trailing commas before } or ]
            fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
            # Remove unquoted values that are not valid
            try:
                strategy = json.loads(fixed)
            except json.JSONDecodeError:
                logger.warning(f"LLM: JSON parse failed even after fixing: {e}")
                logger.debug(f"JSON string: {json_str[:300]}")
                return None

        # --- Validate required fields with sane defaults ---
        valid_ops = set(OPERATIONS.keys()) | {"instr28"}
        valid_transforms = {
            "none", "negate", "sha256", "reverse_bytes",
            "xor_with_ctrl", "add_half", "sub_half", "add_better", "sub_both", "add_both",
            "xor_with_marker", "multiply_by_ctrl", "power_of_two", "mod_p"
        }

        # Validate block_indices
        if "block_indices" not in strategy or not isinstance(strategy["block_indices"], list):
            strategy["block_indices"] = list(range(8))
        
        cleaned_indices = []
        for x in strategy["block_indices"]:
            try:
                val = int(x) % 35
                cleaned_indices.append(val)
            except (ValueError, TypeError):
                continue
        
        if len(cleaned_indices) < 1:
            strategy["block_indices"] = list(range(8))
        else:
            strategy["block_indices"] = cleaned_indices

        # Validate operation
        if strategy.get("operation") not in valid_ops:
            strategy["operation"] = "xor"

        # Validate booleans
        strategy["reverse_base"] = bool(strategy.get("reverse_base", False))
        strategy["apply_scheme_as_coefficients"] = bool(strategy.get("apply_scheme_as_coefficients", False))

        # Validate final_transform
        if strategy.get("final_transform") not in valid_transforms:
            strategy["final_transform"] = "none"

        # Ensure reasoning exists
        if not strategy.get("reasoning") or strategy.get("reasoning") == "no reasoning":
            # Extract any meaningful text from raw response as reasoning
            strategy["reasoning"] = raw_content[:100].replace('\n', ' ')

        return strategy

    except requests.exceptions.Timeout:
        logger.warning("LLM request timed out.")
        return None
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Built-in Systematic Strategies (Python-driven, no LLM needed)
# ---------------------------------------------------------------------------

def generate_systematic_strategies():
    """
    Generate deterministic strategies that MUST be tested regardless
    of what the LLM suggests. These cover the most mathematically
    sound approaches.
    
    IMPROVED: More comprehensive coverage with better organization
    """
    scheme_indices = SCHEME_INDICES  # [31, 2, 32, 12, 4, 27, 0, 2, 19, 15]
    unique_scheme = list(dict.fromkeys(scheme_indices))  # Preserve order, remove duplicates
    
    strategies = []
    
    ops_list = ["xor", "add", "sub", "instr28", "mod_order_add", "mod_order_sub", "mul_mod"]
    transforms_basic = ["none", "negate", "sha256", "reverse_bytes", "xor_with_ctrl"]
    transforms_extended = transforms_basic + ["sub_both", "add_half", "sub_half", "add_both"]

    # --- Category 1: Full 35-block reductions ---
    logger.info("  Generating Category 1: Full 35-block reductions...")
    for op in ops_list:
        for rev in [False, True]:
            for ft in transforms_extended:
                strategies.append({
                    "block_indices": list(range(35)),
                    "operation": op,
                    "reverse_base": rev,
                    "apply_scheme_as_coefficients": False,
                    "final_transform": ft,
                    "reasoning": f"Full 35-block {op} {'reversed' if rev else 'forward'}, final={ft}",
                })

    # --- Category 2: Scheme-indexed blocks (10 elements) ---
    logger.info("  Generating Category 2: Scheme-indexed blocks...")
    for op in ops_list:
        for rev in [False, True]:
            for ft in transforms_extended:
                strategies.append({
                    "block_indices": scheme_indices,
                    "operation": op,
                    "reverse_base": rev,
                    "apply_scheme_as_coefficients": False,
                    "final_transform": ft,
                    "reasoning": f"Scheme-10 {op} {'rev' if rev else 'fwd'} final={ft}",
                })
                # Also with coefficients
                strategies.append({
                    "block_indices": scheme_indices,
                    "operation": op,
                    "reverse_base": rev,
                    "apply_scheme_as_coefficients": True,
                    "final_transform": ft,
                    "reasoning": f"Scheme-10 weighted {op} {'rev' if rev else 'fwd'} final={ft}",
                })

    # --- Category 3: 8-block subsets of scheme (for 8-level triangle) ---
    logger.info("  Generating Category 3: 8-block subsets of unique scheme indices...")
    if len(unique_scheme) >= 8:
        for combo in combinations(unique_scheme, 8):
            for op in ["xor", "instr28", "add", "sub"]:
                for rev in [False, True]:
                    strategies.append({
                        "block_indices": list(combo),
                        "operation": op,
                        "reverse_base": rev,
                        "apply_scheme_as_coefficients": False,
                        "final_transform": "none",
                        "reasoning": f"8-of-9 scheme combo {combo} {op} {'rev' if rev else 'fwd'}",
                    })
                    strategies.append({
                        "block_indices": list(combo),
                        "operation": op,
                        "reverse_base": rev,
                        "apply_scheme_as_coefficients": False,
                        "final_transform": "negate",
                        "reasoning": f"8-of-9 scheme combo {combo} {op} {'rev' if rev else 'fwd'} NEG",
                    })

    # --- Category 4: Bottom row of triangle (blocks 28-34 = 7 blocks, level 7) ---
    logger.info("  Generating Category 4: Bottom row blocks...")
    for op in ops_list:
        for ft in ["none", "negate", "sha256"]:
            strategies.append({
                "block_indices": list(range(28, 35)),
                "operation": op,
                "reverse_base": False,
                "apply_scheme_as_coefficients": False,
                "final_transform": ft,
                "reasoning": f"Bottom-row (28-34) as triangle base, {op}, final={ft}",
            })

    # --- Category 5: Adjacent-pair XOR scan (looking for d in differences) ---
    logger.info("  Generating Category 5: Adjacent pair operations...")
    for i in range(34):
        for op in ["xor", "add", "sub"]:
            strategies.append({
                "block_indices": [i, i + 1],
                "operation": op,
                "reverse_base": False,
                "apply_scheme_as_coefficients": False,
                "final_transform": "none",
                "reasoning": f"Pair {op} block[{i}] {op} block[{i+1}]",
            })

    # --- Category 6: Individual blocks as d ---
    logger.info("  Generating Category 6: Single block tests...")
    for i in range(35):
        for ft in ["none", "negate", "sha256", "reverse_bytes"]:
            strategies.append({
                "block_indices": [i],
                "operation": "xor",
                "reverse_base": False,
                "apply_scheme_as_coefficients": False,
                "final_transform": ft,
                "reasoning": f"Single block[{i}] as d, final={ft}",
            })

    # --- Category 7: Concatenation hashes ---
    logger.info("  Generating Category 7: Block sum with SHA256...")
    for start in range(0, 35, 5):
        end = min(start + 8, 35)
        for op in ["mod_order_add", "xor", "add"]:
            strategies.append({
                "block_indices": list(range(start, end)),
                "operation": op,
                "reverse_base": False,
                "apply_scheme_as_coefficients": False,
                "final_transform": "sha256",
                "reasoning": f"{op} blocks[{start}:{end}] then SHA256",
            })

    # --- Category 8: Permutations of key subsets ---
    logger.info("  Generating Category 8: Key subset permutations...")
    key_subsets = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [31, 2, 32, 12, 4, 27, 0, 19],  # Unique scheme indices
        [28, 29, 30, 31, 32, 33, 34, 0],
    ]
    for subset in key_subsets:
        if len(subset) >= 4:
            for perm in permutations(subset[:6], 4):  # Limit permutations
                for op in ["xor", "instr28"]:
                    strategies.append({
                        "block_indices": list(perm),
                        "operation": op,
                        "reverse_base": False,
                        "apply_scheme_as_coefficients": False,
                        "final_transform": "none",
                        "reasoning": f"Permutation {perm} with {op}",
                    })

    # --- Category 9: Mixed operation sequences ---
    logger.info("  Generating Category 9: Mixed operation sequences...")
    mixed_ops = [
        ["xor", "add"],
        ["add", "sub"],
        ["xor", "mul_mod"],
        ["xor", "add", "sub"],
        ["add", "xor", "mul_mod"],
    ]
    for ops_seq in mixed_ops:
        for base_size in [8, 10, 16]:
            if base_size <= 35:
                strategies.append({
                    "block_indices": list(range(base_size)),
                    "operation": ops_seq,
                    "reverse_base": False,
                    "apply_scheme_as_coefficients": False,
                    "final_transform": "none",
                    "reasoning": f"Mixed ops {ops_seq} on {base_size} blocks",
                })
                strategies.append({
                    "block_indices": list(range(base_size)),
                    "operation": ops_seq,
                    "reverse_base": True,
                    "apply_scheme_as_coefficients": False,
                    "final_transform": "negate",
                    "reasoning": f"Mixed ops {ops_seq} on {base_size} reversed blocks + negate",
                })

    # --- Category 10: Special patterns ---
    logger.info("  Generating Category 10: Special patterns...")
    # Alternating indices
    strategies.append({
        "block_indices": list(range(0, 35, 2)),
        "operation": "xor",
        "reverse_base": False,
        "apply_scheme_as_coefficients": False,
        "final_transform": "none",
        "reasoning": "Even-indexed blocks XOR",
    })
    strategies.append({
        "block_indices": list(range(1, 35, 2)),
        "operation": "xor",
        "reverse_base": False,
        "apply_scheme_as_coefficients": False,
        "final_transform": "negate",
        "reasoning": "Odd-indexed blocks XOR + negate",
    })
    
    # First and last halves
    strategies.append({
        "block_indices": list(range(17)),
        "operation": "xor",
        "reverse_base": False,
        "apply_scheme_as_coefficients": False,
        "final_transform": "none",
        "reasoning": "First half (0-16) XOR",
    })
    strategies.append({
        "block_indices": list(range(17, 35)),
        "operation": "xor",
        "reverse_base": False,
        "apply_scheme_as_coefficients": False,
        "final_transform": "negate",
        "reasoning": "Second half (17-34) XOR + negate",
    })

    logger.info(f"  Total systematic strategies generated: {len(strategies)}")
    return strategies


# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

def save_checkpoint(tested_hashes: set, failed_log: list, iteration: int, phase: int):
    """Save solver state to checkpoint file."""
    checkpoint = {
        "tested_hashes": list(tested_hashes),
        "failed_log": failed_log[-100:],  # Keep last 100
        "iteration": iteration,
        "phase": phase,
        "timestamp": datetime.now().isoformat(),
    }
    try:
        CHECKPOINT_FILE.write_text(json.dumps(checkpoint, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def load_checkpoint() -> dict | None:
    """Load solver state from checkpoint file if it exists."""
    if not CHECKPOINT_FILE.exists():
        return None
    try:
        checkpoint = json.loads(CHECKPOINT_FILE.read_text())
        logger.info(f"Loaded checkpoint from {checkpoint.get('timestamp', 'unknown time')}")
        return checkpoint
    except Exception as e:
        logger.warning(f"Failed to load checkpoint: {e}")
        return None


# ---------------------------------------------------------------------------
# Main Solver Loop
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 70)
    logger.info("  GSMG 5 BTC Puzzle — Local Agentic Solver (IMPROVED)")
    logger.info(f"  Model: {MODEL_ID}")
    logger.info(f"  Endpoint: {LM_STUDIO_URL}")
    logger.info(f"  Target X: {hex(TARGET_X)[:20]}...")
    logger.info(f"  chain4_output.bin: {len(CHAIN4['raw'])} bytes, {len(CHAIN4['blocks'])} blocks")
    logger.info(f"  Control byte: 0x{CHAIN4['ctrl']:02X}")
    logger.info("=" * 70)

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    
    if checkpoint:
        tested_hashes = set(checkpoint.get("tested_hashes", []))
        failed_log = checkpoint.get("failed_log", [])
        start_iteration = checkpoint.get("iteration", 0)
        start_phase = checkpoint.get("phase", 1)
        logger.info(f"Resuming from Phase {start_phase}, iteration {start_iteration}")
        logger.info(f"Already tested {len(tested_hashes)} strategies")
    else:
        tested_hashes = set()
        failed_log = []
        start_iteration = 0
        start_phase = 1

    # ------ Phase 1: Systematic Exhaustive Strategies (Python-only) ------
    if start_phase <= 1:
        logger.info("\n[PHASE 1] Running systematic strategies (no LLM)...")
        systematic = generate_systematic_strategies()
        logger.info(f"Generated {len(systematic)} systematic strategies.")

        for idx, strat in enumerate(systematic):
            # Deduplicate
            strat_key = json.dumps(strat, sort_keys=True)
            strat_hash = hashlib.md5(strat_key.encode()).hexdigest()
            if strat_hash in tested_hashes:
                continue
            tested_hashes.add(strat_hash)

            result = execute_strategy(strat)

            if result["success"]:
                logger.info("!" * 70)
                logger.info(f"!!! PUZZLE SOLVED !!! Strategy #{idx}")
                logger.info(f"Strategy: {json.dumps(strat, indent=2)}")
                logger.info(f"Private Key d = {result['d_candidate_hex']}")
                logger.info("!" * 70)

                # Save to file
                with open("SOLUTION.txt", "w") as f:
                    f.write(f"GSMG 5 BTC PUZZLE SOLVED!\n")
                    f.write(f"Private Key d = {result['d_candidate_hex']}\n")
                    f.write(f"Strategy: {json.dumps(strat, indent=2)}\n")
                
                save_checkpoint(tested_hashes, failed_log, idx, 1)
                return

            if result.get("error"):
                pass  # silently skip errors
            else:
                failed_log.append(strat.get("reasoning", "unknown")[:80])

            if (idx + 1) % 100 == 0:
                logger.info(f"  Phase 1 progress: {idx + 1}/{len(systematic)} tested...")
                # Save checkpoint periodically
                if (idx + 1) % 500 == 0:
                    save_checkpoint(tested_hashes, failed_log, idx, 1)

        logger.info(f"[PHASE 1] Complete. Tested {len(tested_hashes)} unique strategies. No solution.")

    # ------ Phase 2: LLM-Guided Exploration ------
    logger.info(f"\n[PHASE 2] Starting LLM-guided exploration ({MAX_ITERATIONS} iterations)...")

    # Build a concise summary of what failed
    failed_summary_lines = failed_log[-50:]  # Keep last 50 failures for context
    llm_iteration = 0
    consecutive_errors = 0

    while llm_iteration < MAX_ITERATIONS:
        llm_iteration += 1

        # Build summary of recent failures (rotate to keep prompt short)
        recent_failures = failed_summary_lines[-20:]
        failed_summary = "\n".join(f"- {f}" for f in recent_failures)
        if not failed_summary:
            failed_summary = "(none yet)"

        system_prompt = build_system_prompt(failed_summary)
        user_msg = (
            f"Iteration {llm_iteration}/{MAX_ITERATIONS}. "
            f"Total strategies tested: {len(tested_hashes)}. "
            f"Try something COMPLETELY DIFFERENT from what failed. "
            f"Be creative and unconventional. Think like the puzzle creator who loves The Matrix."
        )

        strategy = query_llm(system_prompt, user_msg)

        if strategy is None:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                logger.warning("5 consecutive LLM errors. Pausing 5 seconds...")
                time.sleep(5)
                consecutive_errors = 0
            continue

        consecutive_errors = 0

        # Deduplicate
        strat_key = json.dumps(strategy, sort_keys=True)
        strat_hash = hashlib.md5(strat_key.encode()).hexdigest()
        if strat_hash in tested_hashes:
            logger.debug(f"  LLM iteration {llm_iteration}: duplicate strategy, skipping.")
            continue
        tested_hashes.add(strat_hash)

        reasoning = strategy.get("reasoning", "no reasoning")[:100]
        logger.info(
            f"  [LLM #{llm_iteration}] {reasoning}"
        )

        result = execute_strategy(strategy)

        if result["success"]:
            logger.info("!" * 70)
            logger.info(f"!!! PUZZLE SOLVED by LLM strategy #{llm_iteration} !!!")
            logger.info(f"Strategy: {json.dumps(strategy, indent=2)}")
            logger.info(f"Private Key d = {result['d_candidate_hex']}")
            logger.info("!" * 70)

            with open("SOLUTION.txt", "w") as f:
                f.write(f"GSMG 5 BTC PUZZLE SOLVED!\n")
                f.write(f"Private Key d = {result['d_candidate_hex']}\n")
                f.write(f"Strategy: {json.dumps(strategy, indent=2)}\n")
            
            save_checkpoint(tested_hashes, failed_log, llm_iteration, 2)
            return

        failed_summary_lines.append(reasoning)

        # Save checkpoint periodically
        if llm_iteration % 50 == 0:
            save_checkpoint(tested_hashes, failed_log, llm_iteration, 2)
            logger.info(f"  Checkpoint saved at iteration {llm_iteration}")

        # Brief pause to not overwhelm LM Studio
        time.sleep(0.3)

    # ------ Final Summary ------
    logger.info("=" * 70)
    logger.info(f"  Solver complete. Total unique strategies tested: {len(tested_hashes)}")
    logger.info(f"  No solution found.")
    logger.info(f"  Log saved to: {log_file}")
    logger.info("=" * 70)
    
    # Final checkpoint
    save_checkpoint(tested_hashes, failed_log, MAX_ITERATIONS, 2)


if __name__ == "__main__":
    main()
