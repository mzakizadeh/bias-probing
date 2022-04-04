# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""The Gab Hate Corpus."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets

_CITATION = """\

"""

_DESCRIPTION = """\

"""


class GabConfig(datasets.BuilderConfig):
    """BuilderConfig for Gab."""

    def __init__(self, **kwargs):
        """BuilderConfig for Gab.
            Args:
        .
              **kwargs: keyword arguments forwarded to super.
        """
        super(GabConfig, self).__init__(version=datasets.Version("0.2.0", ""), **kwargs)


class Gab(datasets.GeneratorBasedBuilder):
    # noinspection SpellCheckingInspection
    """Gab: A Hate Speech Corpus"""

    BUILDER_CONFIGS = [
        GabConfig(
            name="plain_text",
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "text": datasets.Value('string'),
                    "label": datasets.Value('bool'),
                    # "purity": datasets.Value('bool'),
                    # "harm": datasets.Value('bool'),
                    # "im": datasets.Value('bool'),
                    # "cv": datasets.Value('bool'),
                    # "ex": datasets.Value('bool'),
                    # "degradation": datasets.Value('bool'),
                    # "fairness": datasets.Value('bool'),
                    # "hd": datasets.Value('bool'),
                    # "mph": datasets.Value('bool'),
                    # "loyalty": datasets.Value('bool'),
                    # "care": datasets.Value('bool'),
                    # "betrayal": datasets.Value('bool'),
                    # "gen": datasets.Value('bool'),
                    # "cheating": datasets.Value('bool'),
                    # "subversion": datasets.Value('bool'),
                    # "rel": datasets.Value('bool'),
                    # "sxo": datasets.Value('bool'),
                    # "rae": datasets.Value('bool'),
                    # "nat": datasets.Value('bool'),
                    # "pol": datasets.Value('bool'),
                    # "authority": datasets.Value('bool'),
                    # "vo": datasets.Value('bool'),
                    # "idl": datasets.Value('bool'),
                }
            ),
            # No default supervised_keys (as we have to pass both premise
            # and hypothesis as input).
            supervised_keys=None,
            homepage="https://osf.io/edua3/",
            citation=_CITATION,
        )

    def _vocab_text_gen(self, filepath):
        for _, ex in self._generate_examples(filepath):
            yield " ".join([ex["evidence"], ex["claim"]])

    def _split_generators(self, dl_manager):
        train_path, validation_path, test_path = dl_manager.download([
            "https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations/raw/master/data/majority_gab_dataset_25k/train.jsonl",
            "https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations/raw/master/data/majority_gab_dataset_25k/dev.jsonl",
            "https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations/raw/master/data/majority_gab_dataset_25k/test.jsonl",
        ])
        # train_path = os.path.join(train_path, "train_fitems.jsonl")
        # validation_path = os.path.join(validation_path, "dev_fitems.jsonl")

        # Since the FEVER NLI dataset doesn't have labels for the dev set, we also download the original
        # FEVER dev set and match example CIDs to obtain labels.
        # orig_dev_path = dl_manager.download(
        #     "https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl"
        # )
        # id_to_label = {}
        # with open(orig_dev_path, 'rb') as f:
        #     for idx, line in enumerate(f):
        #         line = line.strip().decode('utf-8')
        #         json_obj = json.loads(line)
        #         if "id" not in json_obj:
        #             print("FEVER dev dataset is missing ID.")
        #             continue
        #         if "label" not in json_obj:
        #             print("FEVER dev dataset is missing label.")
        #             continue
        #         id_to_label[int(json_obj["id"])] = json_obj["label"]
        # self.id_to_label = id_to_label
        # print(len(self.id_to_label))

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name="validation", gen_kwargs={"filepath": validation_path}),
            datasets.SplitGenerator(name="test", gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate bias examples.
        Args:
          filepath: a string
        Yields:
          dictionaries containing "text" and "label" strings
        """
        with open(filepath, 'rb') as f:
            for idx, line in enumerate(f):
                line = line.strip().decode('utf-8')
                json_obj = json.loads(line)
                # if json_obj["label"] == "hidden":
                #     key = int(json_obj["cid"])
                #     if key not in self.id_to_label:
                #         continue
                #     json_obj["label"] = self.id_to_label[key]

                # Works for both splits even though dev has some extra human labels.
                yield idx, {
                    "id": idx,  # Some IDs show up as duplicates, but we need them to be unique
                    "text": json_obj["Text"],
                    "label": json_obj["hd"] or json_obj["cv"],
                }
