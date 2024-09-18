import logging

logger = logging.getLogger(__name__)


def split_text_into_chunks(content=None, text_file_path="", chunk_size=250, overlap=25, print_output=False):
    """
    The function takes a text as input and breaks it down into smaller chunks based on a specified chunk size and overlap. It iterates over the text,
    creating chunks of the desired size while maintaining an overlap between adjacent chunks. The function returns a dictionary containing the chunked text and
    their corresponding indices.
    :param content:
    :param text_file_path:
    :param chunk_size:
    :param overlap:
    :param print_output:
    :return:
    """

    chunks = []
    indices = []

    prev_data = ""
    prev_chunk_index = 0
    prev_pg = 0

    if isinstance(content, str):
        content = [content]

    for pg, page in enumerate(content):

        string = f"{prev_data} {page}"
        words = string.split()
        total_words = len(words)
        start_index = 0
        skip = False
        chunk_index = 1

        if total_words < (chunk_size + 2 * overlap):
            prev_data = page
            prev_pg = pg
            prev_chunk_index = chunk_index
            continue

        while start_index < total_words:

            if start_index == 0:
                end_index = min(chunk_size + 2 * overlap, total_words)

                if chunk_size + 2 * overlap > total_words:
                    prev_data = ' '.join(words[start_index:end_index])
                    skip = True

            else:
                end_index = min(start_index + chunk_size + overlap, total_words)
                start_index = max(0, start_index - overlap)

                if start_index + chunk_size + overlap > total_words:
                    prev_data = ' '.join(words[start_index:end_index])
                    skip = True

            if not skip:

                if pg != prev_pg and chunk_index == 1:
                    chunk = ' '.join(words[start_index:end_index])
                    chunks.append(chunk)
                    indices.append({"page_no": prev_pg + 1,
                                    "chunk_no": prev_chunk_index})
                else:
                    chunk = ' '.join(words[start_index:end_index])
                    chunks.append(chunk)
                    indices.append({"page_no": pg + 1,
                                    "chunk_no": chunk_index})

                    chunk_index += 1
                prev_chunk_index = chunk_index

            if print_output:
                logger.info(f"{pg}: Iteration complete: start_index: {start_index}  end_index: {end_index}")

            start_index = end_index
            prev_pg = pg
            if skip:
                chunk_index = 1
                break

    chunks.append(prev_data)
    indices.append({"page_no": prev_pg + 1,
                    "chunk_no": prev_chunk_index + 1})

    return {"chunks": chunks, "indices": indices}
