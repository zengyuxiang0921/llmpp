#!/usr/bin/env python3
from __future__ import annotations

import logging
import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent / 'gguf-py'))

import gguf  # noqa: E402

logger = logging.getLogger("convert-gguf")


def load_model_code(model_code_dir: Path) -> dict[str, str]:
    """Scan model-code directory recursively for C++ files and return
    a dict mapping 'llmpp.<relative_path_no_ext>' -> file content."""
    if not model_code_dir.is_dir():
        logger.warning(f'Model code directory not found: {model_code_dir}, skipping')
        return {}

    code_entries: dict[str, str] = {}
    extensions = {'.cu', '.cpp', '.c', '.h', '.cuh'}

    for fpath in sorted(model_code_dir.rglob('*')):
        if not fpath.is_file() or fpath.suffix not in extensions:
            continue
        rel = fpath.relative_to(model_code_dir)
        key = 'llmpp.' + str(rel.with_suffix('')).replace(os.sep, '/')
        content = fpath.read_text(encoding='utf-8')
        code_entries[key] = content
        logger.info(f'  Loaded model code: {rel} -> {key}')

    return code_entries


def copy_with_llmpp_metadata(
    reader: gguf.GGUFReader,
    writer: gguf.GGUFWriter,
    source_code_value: str,
    model_code_dir: Path,
) -> None:
    # Copy all existing metadata fields, skipping GGUF internals and architecture
    for field in reader.fields.values():
        if field.name.startswith('GGUF.'):
            logger.debug(f'Suppressing {field.name}')
            continue

        if field.name == gguf.Keys.General.ARCHITECTURE:
            logger.debug('Suppressing general.architecture (will be set to llmpp)')
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        val = field.contents()

        # Skip source_code and llmpp.* if they already exist (we will add fresh ones)
        if field.name == 'source_code' or field.name.startswith('llmpp.'):
            logger.debug(f'Replacing existing {field.name}')
            continue

        if val is not None:
            writer.add_key_value(field.name, val, val_type, sub_type=sub_type)
            logger.debug(f'Copying {field.name}')

    # Add the new source_code entry
    logger.info(f'Adding source_code: {source_code_value}')
    writer.add_string('source_code', source_code_value)

    # Embed model code from model-code/ directory
    model_code = load_model_code(model_code_dir)
    if model_code:
        logger.info(f'Embedding {len(model_code)} model code entries:')
        for key, content in model_code.items():
            # Strip leading blank lines for cleaner metadata
            content = content.lstrip('\n')
            writer.add_string(key, content)
            logger.info(f'  {key} ({len(content)} bytes)')

    # Copy tensor info
    total_bytes = 0
    for tensor in reader.tensors:
        total_bytes += tensor.n_bytes
        writer.add_tensor_info(
            tensor.name, tensor.data.shape,
            tensor.data.dtype, tensor.data.nbytes, tensor.tensor_type,
        )

    bar = tqdm(desc="Writing", total=total_bytes, unit="byte", unit_scale=True)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a GGUF file for llmpp: set architecture to llmpp and add source_code metadata",
    )
    parser.add_argument("model", type=Path, help="GGUF format model filename")
    parser.add_argument("source_code", type=str, help='Source code URL or identifier, e.g. "https://github.com/org/repo"')
    parser.add_argument("--output", "-o", type=Path, help="Output GGUF filename (default: overwrite input)")
    parser.add_argument("--model-code", type=Path, default=Path("./model-code"), help="Directory containing C++ model code to embed (default: ./model-code)")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")

    args = parser.parse_args(None if len(sys.argv) > 2 else ["--help"])

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    input_path: Path = args.model
    if not input_path.is_file():
        logger.error(f'Input file not found: {input_path}')
        sys.exit(1)

    output_path: Path = args.output if args.output is not None else input_path

    if input_path == output_path:
        logger.warning('*** Warning *** Warning *** Warning **')
        logger.warning('* Modifying a GGUF file in-place requires a full rewrite.')
        logger.warning('* The input file will be replaced. Proceed at your own risk.')
        logger.warning('* Enter exactly YES if you are positive you want to proceed:')
        response = input('YES, I am sure> ')
        if response != 'YES':
            logger.info("You didn't enter YES. Okay then, see ya!")
            sys.exit(0)

    logger.info(f'* Loading: {input_path}')
    reader = gguf.GGUFReader(str(input_path), 'r')

    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()

    logger.info(f'* Architecture: {arch} -> llmpp')

    # Write to a temporary file first for safety when overwriting
    if input_path == output_path:
        tmp_path = input_path.with_suffix('.gguf.tmp')
    else:
        tmp_path = output_path

    logger.info(f'* Writing: {tmp_path}')
    writer = gguf.GGUFWriter(str(tmp_path), arch='llmpp', endianess=reader.endianess)

    alignment_field = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if alignment_field is not None:
        alignment = alignment_field.contents()
        logger.debug(f'Setting custom alignment: {alignment}')
        writer.data_alignment = alignment

    copy_with_llmpp_metadata(reader, writer, args.source_code, args.model_code)

    # Replace input with output if in-place
    if input_path == output_path and tmp_path != output_path:
        tmp_path.replace(output_path)
        logger.info(f'* Replaced {output_path}')

    logger.info('* Done.')


if __name__ == '__main__':
    main()
