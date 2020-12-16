Sahil Jayaram (UNI saj2163)
12/19/2020

Automatic Punctuation Restoration Using Language Models and Prosodic Features

Abstract of paper: High-performing automatic speech recognition (ASR) systems can transcribe speech with impressive
accuracy, in some cases achieving lower word error rates than human scribes on benchmark corpora. Few of these systems,
however, are capable of producing output that contains punctuation, which has been shown to significantly increase the
readability of transcribed speech. State-of-the-art approaches to automatic punctuation restoration rely solely on text
input, ignoring the wealth of task-relevant information latent in the original audio. In this paper, I present a
solution for punctuation restoration that is based on a top-performing, transformer-based method, differing from its
precursor in that it can optionally utilize prosodic features in addition to text. My model achieved a combined F1 of
84.5 on the IWSLT 2012 TED Talk evaluation set, outperforming the current state-of-the-art model by 0.6. Furthermore, my
model consistently achieved its best performance when given prosodic features, demonstrating that prosody is a valuable
predictor of punctuation placement and is not rendered superfluous by text input.

See saj2163_final_paper.pdf to read the full report.

-----DATA-----

data/iwslt contains data from the IWSLT 2012 TED task (MT Track). The data is available at
http://hltc.cs.ust.hk/iwslt/index.php/evaluation-campaign/ted-task.html . The rest of the data used is from the
Singapore National Speech Corpus (obtained from Prof. Homayoon Beigi).


-----TOOLS & ENVIRONMENT-----

This project was constructing using Python 3.8. To create and activate the Python environment, enter the project root
directory and run

$ <PATH_TO_PYTHON_3.8> -m venv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt

Data extraction directly uses two additional open-source tools (besides Kaldi): sph2pipe and the Gentle forced-aligner.

For instructions on downloading and installing sph2pipe, see https://github.com/robd003/sph2pipe .

For instructions on downloading and installing Gentle, see https://github.com/lowerquality/gentle .

The above link provides instructions for running the Gentle Docker image. For instructions on downloading and installing
Docker, see https://docs.docker.com/get-docker/

Do not manually start the Gentle Docker image, as the script data_prep.sh will handle that.


-----PROJECT STRUCTURE-----

PROJECT ROOT/
--scripts/          - contains all the code used to preprocess the data (fully executed by data_prep.sh) as well as the
                    second main script, train_and_eval.sh
--data/
----demo/           - contains data used to demonstrate the functionality of the APR system
----iwslt/          - contains the raw IWSLT data (IWSLT12.TALK.train, en-fr-test/, tst2012_audio/*.sph) as well as the
                    processed IWSLT data (train.data, tst2012_audio/*.data)
----nsc/            - contains all the processed NSC data (all_data/*.data) and symlinks to these files in the
                    directories dev, test, and train
--src/              - contains all the Python code concerning the PyTorch model
--requirements.txt  - Python environment requirements (see above)


-----RUNNING THE SCRIPTS-----

Enter the scripts directory. Then run

$ sudo ./data_prep.sh

This script populates the data/ directory with .data files (the files used to train and evaluate the model), as well as
SCP files (intermediate files used by Kaldi).

Next, run

$ ./train_and_eval.sh > train_and_eval.log

This script produces 2 trained models: model-untuned.pt (trained only on NSC data) and model-tuned.pt (fine-tuned on
IWSLT text-only data). It also evaluates those models in various settings. After the process finishes running,
train_and_eval.log will contain the entire training log and all evaluation results.