# OpenTab

This is the official repository of  [[ICLR2024] OpenTab: Advancing Large Language Models as Open-domain Table Reasoners](https://arxiv.org/abs/2402.14361).

## Dependencies
To establish the environment run this code in the shell:
```bash
conda env create -f py3.7opentab.yaml
```
That will create the environment `opentab` we used.

Activate the environment by running
``````shell
conda activate opentab
``````

## Dataset

Download the [Open-WikiTable](https://github.com/sean0042/Open_WikiTable/tree/main) dataset and put the extracted `data` folder under the `open_wikitable` folder.

Run the following to use BM25 for table retrieval.

``````shell
cd open_wikitable
python bm25_eval.py
``````

## Usage

### Add key
Apply and get `API keys` from [OpenAI API](https://openai.com/api/), save the key with `OPENAI_API_KEY` in local environment variable, make sure you have the rights to access the model (for the implementation of this repo, essentially `gpt-3.5-turbo-0613`) you need.

### Run
To reproduce the open-domain experiments, run

``````shell
cd scripts
python run_open.py
``````

And use `grsr.ipynb` for reference to realize the `GRSR` functionality.

To reproduce the closed-domain experiments, run

``````shell
cd scripts
python run_closed.py
``````

## Citation

```
@article{kong2024opentab,
  title={OpenTab: Advancing Large Language Models as Open-domain Table Reasoners},
  author={Kong, Kezhi and Zhang, Jiani and Shen, Zhengyuan and Srinivasan, Balasubramaniam and Lei, Chuan and Faloutsos, Christos and Rangwala, Huzefa and Karypis, George},
  journal={arXiv preprint arXiv:2402.14361},
  year={2024}
}
```

## Acknowledgments

OpenTab implementation was built based on https://github.com/xlang-ai/Binder. File datasets/wikitq.py is copied from https://github.com/xlang-ai/Binder/blob/main/datasets/wikitq.py. Folder utils/ is copied from https://github.com/xlang-ai/Binder/tree/main/utils, except utils/sql_utils.py and utils/bm25.py. File open_wikitable/dataloader.py is copied from https://github.com/sean0042/Open_WikiTable/blob/main/src/dataloader.py.