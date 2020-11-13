import glob
import sys

DATA_DIR, CORPUS_DIR = sys.argv[1], sys.argv[2]
TRANSCRIPT_TEMPLATE = CORPUS_DIR + "/PART{}/DATA/CHANNEL0/SCRIPT/{}.TXT"

if __name__ == "__main__":
    data_fnames = glob.glob(f"{DATA_DIR}/*.data")
    utterances_seen = []
    for data_fname in data_fnames:
        file_basename = data_fname.split("/")[-1].replace(".data", "")
        part, transcript_basename = file_basename.split("-")
        transcript_filename = TRANSCRIPT_TEMPLATE.format(part, transcript_basename)
        with open(data_fname, "r") as data_file:
            with open(transcript_filename, "r", encoding='utf-8-sig') as transcript_file:
                with open(data_fname.replace(".data", ".dedup"), "w+") as out_file:
                    for transcript_line in transcript_file:
                        next(transcript_file)  # ignore next line
                        _, utterance = transcript_line.split("\t")
                        utterance = utterance.strip()
                        if utterance[-1] not in [".", "?", "!"]:  # transcript file doesn't contain punctuation!
                            print(utterance)
                            print(transcript_file, " lacks punctuation! Skipping.")
                            break
                        if utterance in utterances_seen:
                            utterances_seen.append(utterance)
                            seen = True
                            print(f"Duplicate sentence: {utterance}")
                        else:
                            seen = False
                            utterances_seen.append(utterance)
                        for word in utterance.split():
                            data_line = next(data_file)
                            old_word = data_line.split("\t")[0]  # old data files lack punctuation
                            if not seen:
                                out_file.write(data_line.replace(old_word, word))
