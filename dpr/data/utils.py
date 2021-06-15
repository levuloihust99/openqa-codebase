import gzip
import re
import os
import time
import json
import tensorflow as tf


def unpack(gzip_file: str, out_file: str, chunk_size=1024**3):
    """Iteratively unpack a gzip file that cannot fit into memory

    Args:
        gzip_file (str): Path to the input gzip file
        out_file (str): Path to the output uncompressed file
        chunk_size: Bytes read into memory at each step
    """

    reader = gzip.open(gzip_file)
    writer = open(out_file, "wb")

    while True:
        bytes_read = reader.read(chunk_size)
        if len(bytes_read) == 0:
            break
        print("Read {}".format(len(bytes_read)))
        writer.write(bytes_read)

    reader.close()
    writer.close()


def count_lines(file_path: str, chunk_size=1024**3):
    """Count number of lines of a file whose size does not fit into memory.

    Args:
        file_path (str): Input file that needs to count lines
        chunk_size: Bytes read into memory at each step

    Returns:
        Number of lines of the input file.
    """

    reader = open(file_path, "rb")

    count = 0
    while True:
        bytes_read = reader.read(chunk_size)
        if len(bytes_read) == 0:
            break
        
        matches = re.findall(b"\n", bytes_read)
        count += len(matches)
        print("Count {}".format(count))

    reader.close()
    return count


def json_to_jsonline(input_path: str, output_path: str, chunk_size=1024**3):
    """Convert .json file to .jsonl file.

    Args:
        input_path (str): Path to the input .json file
        output_path (str): Path to the output .jsonl file
        chunk_size ([type], optional): Number of bytes to be read into memory at each step. Defaults to 1024**3.
    """
    reader = open(input_path, "rb")
    writer = open(output_path, "w")

    bytes_read_accumulate = b""
    count = 0
    while True:
        bytes_read = reader.read(chunk_size)
        if len(bytes_read) == 0:
            break
        bytes_read_accumulate += bytes_read
        
        brace_start_matches = list(re.finditer(b"\n    {", bytes_read_accumulate))
        brace_end_matches = list(re.finditer(b"\n    }", bytes_read_accumulate))

        if len(brace_start_matches) == 0 or len(brace_end_matches) == 0:
            continue

        brace_start_positions = [match.span()[0] + 1 for match in brace_start_matches]
        brace_end_positions = [match.span()[1] for match in brace_end_matches]

        brace_pair_positions = list(zip(brace_start_positions, brace_end_positions))
        for start, end in brace_pair_positions:
            json_record = bytes_read_accumulate[start : end] 
            writer.write(json.dumps(eval(json_record), ensure_ascii=False))
            writer.write("\n")
        
        count += len(brace_pair_positions)
        print("Processed {} records".format(count))
        
        delta = len(bytes_read_accumulate) - end
        reader.seek(-delta, 1)
        bytes_read_accumulate = b""

    reader.close()
    writer.close()


def split_ctx_sources(
    file_path: str,
    out_dir: str,
    chunk_size=1024**2,
    lines_per_file=42031,
    skip_header_row=True,
):
    """Split a file into multiple smaller files. Currently apply to "psgs_w100.tsv"

    Args:
        file_path (str): The file that needs to be splitted
        out_dir (str): Directory of the output splitted files
        chunk_size: [description]. Defaults to 1024**2 (1MB).
        lines_per_file (int, optional): Number of lines in each output splitted file. Defaults to 420307.
        skip_header_row (bool, optional): Whether or not to skip the header row
    """

    reader = open(file_path, "rb")
    header_row = reader.readline() # skip the header row
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    idx = 0
    num_lines = 0
    marked_pos = len(header_row)
    flag = True

    while True:

        bytes_read = reader.read(chunk_size)
        if len(bytes_read) == 0:
            break

        if flag:
            writer = open(os.path.join(out_dir, "psgs_subset_{:02d}.tsv".format(idx)), "wb")
            if not skip_header_row:
                writer.write(header_row)
            flag = False

        matches = list(re.finditer(b"\n", bytes_read))
        num_lines += len(matches)

        delta = num_lines - lines_per_file

        if delta >= 0:
            num_lines = 0
            idx += 1

            last_match = matches[-1 - delta]
            relative_pos = last_match.span()[1]
            marked_pos += relative_pos

            reader.seek(marked_pos, 0)
            writer.write(bytes_read[:relative_pos])
            writer.close()
            flag = True

            print(os.path.abspath(writer.name))

        else:
            marked_pos += len(bytes_read)
            writer.write(bytes_read)

    if not writer.closed:
        writer.close()

    reader.close()


def benchmark(dataset: tf.data.Dataset):
    start_time = time.perf_counter()
    for _ in dataset:
        pass

    print("Execution time:", time.perf_counter() - start_time)


if __name__ == "__main__":
    # split_ctx_sources(
    #     file_path="data/wikipedia_split/psgs_subset.tsv",
    #     out_dir="data/wikipedia_split/shards-42031",
    #     chunk_size=1024**2,
    #     lines_per_file=42031, skip_header_row=True
    # )

    json_to_jsonline(
        input_path="data/retriever/vi-covid-train.json",
        output_path="data/retriever/vi-covid-train.jsonl"
    )