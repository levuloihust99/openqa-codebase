import gzip
import re
import os
import time
import json
import tensorflow as tf

from . import databuilder


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
            writer.write(json.dumps(eval(json_record)))
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
    if skip_header_row:
        header_row = reader.readline() # skip the header row

    idx = 0
    num_lines = 0
    marked_pos = 0
    flag = True

    while True:

        bytes_read = reader.read(chunk_size)
        if len(bytes_read) == 0:
            break

        if flag:
            writer = open(os.path.join(out_dir, "psgs_w100_{}.tsv".format(idx)), "wb")
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


def prepare_retriever_data(
    input_path: str,
    from_jsonl: bool=True,
    store_intermediate_text_record: bool=True,
    records_per_file: int = 5000,
    max_files: int = -1,
    bert_pretrained_model: str = 'bert-base-uncased'
):
    """Prepare tfrecord data from the original data (taken from DPR paper)

    Args:
        input_path (str): Path to the original data, i.e. data provided by DPR paper. Can be one-step preprocessed, i.e. already converted `.jsonl` extension
        out_dir (str): Output directory that will contain `.tfrecord` data ready for training retriever model
        from_jsonl (bool, optional): Whether the original data is in `.json` or `.jsonl` extension. Defaults to True.
        store_intermediate_text_record (bool, optional): Whether to store the intermediate `.tfrecord` data (each record contains text data). Defaults to True.
        records_per_file (int, optional): Number of records per `.tfrecord` file. Defaults to 5000.
        max_sequence_length (int, optional): Maximum length of the documents after tokenizing, i.e. already added [SEP] and [CLS] tokens. 
            Documents whose length exceed this max length will be truncated, but documents whose length is shorter is not padded (to save space when store in `.tfrecord` files).
            Defaults to 256.
    """
    if not from_jsonl:
        abspath = os.path.abspath(input_path)
        dir_name = os.path.dirname(abspath)
        json_line_path = os.path.join(dir_name, "nq-train.jsonl")

        json_to_jsonline(abspath, json_line_path)
        input_path = json_line_path

    dir_name = os.path.dirname(input_path)
    out_dir = os.path.join(dir_name, "N{}-INT".format(records_per_file))
    if store_intermediate_text_record:

        abspath = os.path.abspath(input_path)
        text_tfrecord_path = os.path.join(os.path.dirname(input_path), "N{}-TEXT".format(records_per_file))
        databuilder.main(
            called_as_module=True,
            input_path=input_path,
            build_type=0,
            max_files=max_files,
            output_dir=text_tfrecord_path,
            records_per_file=records_per_file,
        )

        databuilder.main(
            called_as_module=True,
            input_path=text_tfrecord_path + "/*",
            build_type=1,
            output_dir=out_dir,
            records_per_file=records_per_file,
            bert_pretrained_model=bert_pretrained_model,
        )
    
    else:
        databuilder.main(
            called_as_module=True,
            input_path=input_path,
            build_type=2,
            output_dir=out_dir,
            bert_pretrained_model=bert_pretrained_model,
            records_per_file=records_per_file,
        )

def benchmark(dataset: tf.data.Dataset):
    start_time = time.perf_counter()
    for _ in dataset:
        pass

    print("Execution time:", time.perf_counter() - start_time)


if __name__ == "__main__":
    prepare_retriever_data(
        input_path="data/retriever/nq-train.json",
        from_jsonl=False,
        store_intermediate_text_record=True,
        records_per_file=5000,
        max_files=-1,
        bert_pretrained_model='bert-base-uncased',
    )