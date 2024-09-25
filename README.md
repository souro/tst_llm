# Leveraging Low-resource Parallel Data for Text Style Transfer

This repo contains the code and data of the paper: [Are Large Language Models Actually Good at Text Style Transfer?](https://aclanthology.org/2024.inlg-main.42/).

## Overview
This repository analyzes the performance of large language models (LLMs) on Text Style Transfer (TST), focusing on sentiment transfer and text detoxification in English, Hindi, and Bengali. We assess pre-trained LLMs using zero-shot and few-shot prompting, as well as parameter-efficient fine-tuning on publicly available datasets. 

Our evaluations, conducted with automatic metrics, GPT-4, and human assessments, show that while some LLMs excel in English, their performance in Hindi and Bengali is average. However, fine-tuning significantly enhances results, making them competitive with state-of-the-art methods, highlighting the need for specialized datasets and models for effective TST.


## Walkthrough
*Will add more information in this section soon.*

### Dependency

    pip install -r <requirements.txt>

## Citing
If you use this data or code please cite the following:
  
    @inproceedings{mukherjee-etal-2024-large-language,
    title = "Are Large Language Models Actually Good at Text Style Transfer?",
    author = "Mukherjee, Sourabrata  and
      Ojha, Atul Kr.  and
      Dusek, Ondrej",
    editor = "Mahamood, Saad  and
      Minh, Nguyen Le  and
      Ippolito, Daphne",
    booktitle = "Proceedings of the 17th International Natural Language Generation Conference",
    month = sep,
    year = "2024",
    address = "Tokyo, Japan",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.inlg-main.42",
    pages = "523--539",
    abstract = "",
}

## License

    Author: Sourabrata Mukherjee
    Copyright © 2023 Sourabrata Mukherjee.
    Licensed under the MIT License.

## Acknowledgements

This research was funded by the European Union (ERC, NG-NLG, 101039303) and Charles University project SVV 260 698. We acknowledge the use of resources provided by the LINDAT/CLARIAH-CZ Research Infrastructure (Czech Ministry of Education, Youth, and Sports project No. LM2018101). We also acknowledge Panlingua Language Processing LLP for collaborating on this research project. Atul Kr. Ojha would like to acknowledge the support of the Science Foundation Ireland (SFI) as part of Grant Number SFI/12/RC/2289_P2 Insight_2, Insight SFI Research Centre for Data Analytics.

We want to acknowledge the GitHub repository [LLaMA-Efficient-Tuning](https://github.com/andysdc/LLaMA-Efficient-Tuning), which we used for fine-tuning LLMs.

