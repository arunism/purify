# AI Engineer Assessment | MustBreak

This file includes the documentation to solutions implemented for each of the two quests.

> Due to lack of enough `GPU` on my device currently, all the experiments are implemented on a `CPU` device. So all the results are optimized automatically (realtive to time), when run in a system with enough `GPU` resource.


## Usage

To use the code you need to create `huggingface` token and export it using the following:

```shell
export HF_TOKEN="<YOUR-HUGGINGFACE-TOKEN>"
```

### Quest 1: Hallucination Problem

To run the script that demostrates stratigies to reduce hallucination problem in LLMs, run the following scipt:

```shell
python3 -m scripts.llm --mode <YOUR-MODE-CHOICE> --device <YOUR-DEVICE-CHOICE>
```

**mode:**
- basic
- advance (default)

**device:**
- cpu (default)
- cuda

For detailed documentation on Hellucination Mitigation Stratigies visit this [page](./docs/llm.md)


## Quest 2:  Ghost results in Speech-to-Text

To run the script that demostrates stratigies to reduce Ghost results in Speech-to-Text Systems, run the following scipt:

```shell
python3 -m scripts.stt --source <YOUR-AUDIO-SOURCE-PATH> --mode <YOUR-MODE-CHOICE> --device <YOUR-DEVICE-CHOICE>
```

**mode:**
- basic
- advance (default)

**device:**
- cpu (default)
- cuda

For detailed documentation on Stratigies to Mitigate Ghost results in Speech-to-Text systems visit this [page](./docs/stt.md)
