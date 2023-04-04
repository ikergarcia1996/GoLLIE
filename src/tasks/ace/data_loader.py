import math
from typing import Any, Dict, Tuple, Union
from jinja2 import Template

from src.tasks.ace.prompts import (
    ENTITY_DEFINITIONS,
    EVENT_DEFINITIONS,
    GPE,
    RELATION_DEFINITIONS,
    VALUE_DEFINITIONS,
    Acquit,
    Appeal,
    ArrestJail,
    Attack,
    BeBorn,
    Business,
    ChargeIndict,
    CitizenResidentReligionEthnicity,
    ContactInfo,
    Convict,
    Crime,
    DeclareBankruptcy,
    Demonstrate,
    Die,
    Divorce,
    Elect,
    Employment,
    EndOrg,
    EndPosition,
    Execute,
    Extradite,
    Facility,
    Family,
    Fine,
    Founder,
    Injure,
    InvestorShareholder,
    JobTitle,
    LastingPersonal,
    Located,
    Location,
    Marry,
    Meet,
    Membership,
    MergeOrg,
    Near,
    Nominate,
    Numeric,
    OrgLocationOrigin,
    Organization,
    Ownership,
    Pardon,
    Person,
    PhoneWrite,
    ReleaseParole,
    Sentence,
    SentenceAct,
    SportsAffiliation,
    StartOrg,
    StartPosition,
    StudentAlum,
    Subsidiary,
    Sue,
    Time,
    TransferMoney,
    TransferOwnership,
    Transport,
    TrialHearing,
    UserOwnerInventorManufacturer,
    Vehicle,
    Weapon,
    Geographical,
)
from ..utils_typing import DatasetLoader, Sampler

import json
import inspect
import random
import numpy as np


class ACEDatasetLoader(DatasetLoader):
    ENTITY_TO_CLASS_MAPPING = {
        "FAC": Facility,
        "GPE": GPE,
        "LOC": Location,
        "ORG": Organization,
        "PER": Person,
    }
    VALUE_TO_CLASS_MAPPING = {
        "Contact-Info": ContactInfo,
        "Crime": Crime,
        "Job-Title": JobTitle,
        "Numeric": Numeric,
        "Sentence": Sentence,
        "TIME": Time,
        "VEH": Vehicle,
        "WEA": Weapon,
    }
    RELATION_TO_CLASS_MAPPING = {
        "ART:User-Owner-Inventor-Manufacturer": UserOwnerInventorManufacturer,
        "GEN-AFF:Citizen-Resident-Religion-Ethnicity": CitizenResidentReligionEthnicity,
        "GEN-AFF:Org-Location": OrgLocationOrigin,
        "ORG-AFF:Employment": Employment,
        "ORG-AFF:Founder": Founder,
        "ORG-AFF:Investor-Shareholder": InvestorShareholder,
        "ORG-AFF:Membership": Membership,
        "ORG-AFF:Ownership": Ownership,
        "ORG-AFF:Sports-Affiliation": SportsAffiliation,
        "ORG-AFF:Student-Alum": StudentAlum,
        "PART-WHOLE:Artifact": (
            Geographical
        ),  # There is no definition for Artifact relation on the guidelines
        "PART-WHOLE:Geographical": Geographical,
        "PART-WHOLE:Subsidiary": Subsidiary,
        "PER-SOC:Business": Business,
        "PER-SOC:Family": Family,
        "PER-SOC:Lasting-Personal": LastingPersonal,
        "PHYS:Located": Located,
        "PHYS:Near": Near,
    }
    _EVENT_CONSTANTS_MAPPING = {
        "trigger": "mention",
        "Place": "place",
        "Time-After": "time",
        "Time-At-Begginning": "time",
        "Time-At-Beginning": "time",  # A bug on the data
        "Time-At-End": "time",
        "Time-Before": "time",
        "Time-Ending": "time",
        "Time-Holds": "time",
        "Time-Starting": "time",
        "Time-Within": "time",
    }
    EVENT_TO_CLASS_MAPPING = {
        "Business:Declare-Bankruptcy": {
            "class": DeclareBankruptcy,
            "Org": "org",
        },
        "Business:End-Org": {
            "class": EndOrg,
            "Org": "org",
        },
        "Business:Merge-Org": {
            "class": MergeOrg,
            "Org": "org",
        },
        "Business:Start-Org": {"class": StartOrg, "Agent": "agent", "Org": "org"},
        "Conflict:Attack": {
            "class": Attack,
            "Agent": "attacker",
            "Attacker": "attacker",
            "Instrument": "instrument",
            "Target": "target",
            "Victim": "target",
        },
        "Conflict:Demonstrate": {
            "class": Demonstrate,
            "Entity": "entity",
        },
        "Contact:Meet": {
            "class": Meet,
            "Entity": "entity",
        },
        "Contact:Phone-Write": {
            "class": PhoneWrite,
            "Entity": "entity",
        },
        "Justice:Acquit": {
            "class": Acquit,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Defendant": "defendant",
        },
        "Justice:Appeal": {
            "class": Appeal,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Plaintiff": "prosecutor",
        },
        "Justice:Arrest-Jail": {
            "class": ArrestJail,
            "Agent": "agent",
            "Crime": "crime",
            "Person": "person",
        },
        "Justice:Charge-Indict": {
            "class": ChargeIndict,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Defendant": "defendant",
            "Prosecutor": "prosecutor",
        },
        "Justice:Convict": {
            "class": Convict,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Defendant": "defendant",
        },
        "Justice:Execute": {
            "class": Execute,
            "Agent": "agent",
            "Crime": "crime",
            "Person": "person",
        },
        "Justice:Extradite": {
            "class": Extradite,
            "Agent": "agent",
            "Destination": "destination",
            "Origin": "origin",
            "Person": "person",
        },
        "Justice:Fine": {
            "class": Fine,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Entity": "entity",
            "Money": "money",
        },
        "Justice:Pardon": {
            "class": Pardon,
            "Adjudicator": "adjudicator",
            "Defendant": "defendant",
            "Crime": "crime",
        },
        "Justice:Release-Parole": {
            "class": ReleaseParole,
            "Crime": "crime",
            "Entity": "entity",
            "Person": "person",
        },
        "Justice:Sentence": {
            "class": SentenceAct,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Defendant": "defendant",
            "Sentence": "sentence",
        },
        "Justice:Sue": {
            "class": Sue,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Defendant": "defendant",
            "Plaintiff": "plaintiff",
        },
        "Justice:Trial-Hearing": {
            "class": TrialHearing,
            "Adjudicator": "adjudicator",
            "Crime": "crime",
            "Defendant": "defendant",
            "Prosecutor": "prosecutor",
        },
        "Life:Be-Born": {
            "class": BeBorn,
            "Person": "person",
        },
        "Life:Die": {
            "class": Die,
            "Agent": "agent",
            "Instrument": "instrument",
            "Person": "victim",
            "Victim": "victim",
        },
        "Life:Divorce": {
            "class": Divorce,
            "Person": "person",
        },
        "Life:Injure": {
            "class": Injure,
            "Agent": "agent",
            "Instrument": "instrument",
            "Victim": "victim",
        },
        "Life:Marry": {
            "class": Marry,
            "Person": "person",
        },
        "Movement:Transport": {
            "class": Transport,
            "Agent": "agent",
            "Artifact": "artifact",
            "Destination": "destination",
            "Origin": "origin",
            "Place": "destination",
            "Vehicle": "vehicle",
            "Victim": "artifact",  # MMMmmm WTF
            "Price": "price",
        },
        "Personnel:Elect": {
            "class": Elect,
            "Entity": "entity",
            "Person": "person",
            "Position": "position",
        },
        "Personnel:End-Position": {
            "class": EndPosition,
            "Entity": "entity",
            "Person": "person",
            "Position": "position",
        },
        "Personnel:Nominate": {
            "class": Nominate,
            "Agent": "agent",
            "Person": "person",
            "Position": "position",
        },
        "Personnel:Start-Position": {
            "class": StartPosition,
            "Entity": "entity",
            "Person": "person",
            "Position": "position",
        },
        "Transaction:Transfer-Money": {
            "class": TransferMoney,
            "Beneficiary": "beneficiary",
            "Giver": "giver",
            "Money": "money",
            "Recipient": "recipient",
        },
        "Transaction:Transfer-Ownership": {
            "class": TransferOwnership,
            "Artifact": "artifact",
            "Beneficiary": "beneficiary",
            "Buyer": "buyer",
            "Price": "price",
            "Seller": "seller",
        },
    }

    def __init__(self, path: str, group_by: str = "sentence", **kwargs) -> None:
        assert group_by in [
            "sentence",
            "document",
        ], "`group_by` must be either 'sentence' or 'document'."

        self.elements = {}

        with open(path, "rt") as in_f:
            for line in in_f:
                line = json.loads(line.strip())

                key = line["sent_id"] if group_by == "sentence" else line["doc_id"]
                if key not in self.elements:
                    self.elements[key] = {
                        "id": key,
                        "doc_id": line["doc_id"],
                        "text": "",
                        "entities": [],
                        "values": [],
                        "relations": [],
                        "events": [],
                    }

                entities = [
                    self.ENTITY_TO_CLASS_MAPPING[entity["entity_type"]](
                        span=entity["text"]
                    )
                    for entity in line["entity_mentions"]
                    if entity["entity_type"] in self.ENTITY_TO_CLASS_MAPPING
                ]
                values = [
                    self.VALUE_TO_CLASS_MAPPING[entity["entity_type"]](
                        span=entity["text"]
                    )
                    for entity in line["entity_mentions"]
                    if entity["entity_type"] in self.VALUE_TO_CLASS_MAPPING
                ]
                relations = [
                    self.RELATION_TO_CLASS_MAPPING[rel["relation_subtype"]](
                        arg1=rel["arguments"][0]["text"],
                        arg2=rel["arguments"][1]["text"],
                    )
                    for rel in line["relation_mentions"]
                    if rel["relation_subtype"] in self.RELATION_TO_CLASS_MAPPING
                ]
                events = []
                for event in line["event_mentions"]:
                    if event["event_type"] not in self.EVENT_TO_CLASS_MAPPING:
                        continue
                    info = self.EVENT_TO_CLASS_MAPPING[event["event_type"]]
                    _inst = {
                        param: []
                        for param in inspect.signature(info["class"]).parameters.keys()
                    }
                    _inst["mention"] = event["trigger"]["text"]
                    for argument in event["arguments"]:
                        if argument["role"] in info:
                            name = info[argument["role"]]
                            _inst[name].append(argument["text"])
                        elif argument["role"] in self._EVENT_CONSTANTS_MAPPING:
                            name = self._EVENT_CONSTANTS_MAPPING[argument["role"]]
                            if name not in _inst:
                                continue
                            _inst[name].append(argument["text"])
                        else:
                            raise ValueError(
                                f"Argument {event['event_type']}:{argument['role']} not"
                                " found!"
                            )

                    events.append(info["class"](**_inst))

                self.elements[key]["text"] += " " + line["sentence"].strip()
                self.elements[key]["entities"] += entities
                self.elements[key]["values"] += values
                self.elements[key]["relations"] += relations
                self.elements[key]["events"] += events

    def __iter__(self):
        for elem in self.elements.values():
            yield elem

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx: Union[int, str]) -> Dict[str, Any]:
        if isinstance(idx, int):
            return list(self.elements.values())[idx]  # Not very efficient
        else:
            return self.elements[idx]


class ACESampler(Sampler):
    def __init__(
        self,
        dataset_loader: ACEDatasetLoader,
        task: str = None,
        split: str = "train",
        parallel_instances: Union[int, Tuple[int, int]] = 1,
        max_guidelines: int = -1,
        guideline_dropout: float = 0.0,
        seed: float = 0,
        prompt_template: str = "templates/prompt.txt",
        ensure_positives_on_train: bool = False,
        group_by: str = "sentence",
        dataset_name: str = None,
        scorer: str = None,
        **kwargs,
    ) -> None:
        self.loader = dataset_loader
        assert task in [
            "NER",
            "VER",
            "RE",
            "EE",
        ], f"{task} must be either 'NER', 'VER', 'RE', 'EE'."
        self.task = task
        assert split in [
            "train",
            "dev",
            "test",
        ], f"{split} must be either 'train', 'dev' or 'test'."
        self.split = split
        parallel_instances = parallel_instances if group_by == "sentence" else (1, 1)
        if isinstance(parallel_instances, int):
            parallel_instances = (1, parallel_instances)
        self.parallel_instances = tuple(parallel_instances)
        self.guideline_dropout = guideline_dropout
        self.seed = seed
        self.task_definitions, self.task_target = {
            "NER": (ENTITY_DEFINITIONS, "entities"),
            "VER": (VALUE_DEFINITIONS, "values"),
            "RE": (RELATION_DEFINITIONS, "relations"),
            "EE": (EVENT_DEFINITIONS, "events"),
        }[self.task]
        if max_guidelines < 0 or max_guidelines > len(self.task_definitions):
            self.max_guidelines = len(self.task_definitions)
        else:
            self.max_guidelines = max_guidelines
        self.ensure_positives_on_train = ensure_positives_on_train

        with open(prompt_template, "rt") as f:
            self.template = Template(f.read())

        self.dataset_name = dataset_name
        self.scorer_cls = scorer

    def _sample(self, instances):
        if self.split == "train":
            positive_guidelines = {
                type(ann) for inst in instances for ann in inst[self.task_target]
            }
            # Assign a probability distribution that helps positive classes
            # if ensure_positives_on_train is True
            p = np.asarray(
                [
                    (
                        5.0
                        if _def in positive_guidelines and self.ensure_positives_on_train
                        else 0.0
                    )
                    for _def in self.task_definitions
                ]
            )
            p += 1.0 / p.shape[0]
            p /= p.sum()
            _guidelines = np.random.choice(
                np.asarray(self.task_definitions),
                size=(self.max_guidelines,),
                replace=False,
                p=p,
            ).tolist()
            # Apply guideline dropout
            _guidelines = [
                _def
                for _def in _guidelines
                if random.random() > self.guideline_dropout
                or (_def in positive_guidelines and self.ensure_positives_on_train)
            ]
            _ann = [
                ann
                for inst in instances
                for ann in inst[self.task_target]
                if type(ann) in _guidelines
            ]
            _text = " ".join([inst["text"] for inst in instances]).strip()
            yield {
                "ids": [inst["id"] for inst in instances],
                "task_id": f"{self.dataset_name}_{self.task}",
                "scorer_cls": self.scorer_cls,
                "labels": [ann.__repr__() for ann in _ann],
                "text": self.template.render(
                    guidelines=[
                        inspect.getsource(definition) for definition in _guidelines
                    ],
                    text=_text,
                    annotations=_ann,
                ),
            }
        else:
            guidelines = [definition for definition in self.task_definitions]
            random.shuffle(guidelines)
            splits = math.ceil(len(guidelines) / self.max_guidelines)
            for i in range(splits):
                _guidelines = guidelines[
                    i * self.max_guidelines : (i + 1) * self.max_guidelines
                ]
                _ann = [
                    ann
                    for inst in instances
                    for ann in inst[self.task_target]
                    if type(ann) in _guidelines
                ]
                _text = " ".join([inst["text"] for inst in instances]).strip()

                yield {
                    "ids": [inst["id"] for inst in instances],
                    "task_id": f"{self.dataset_name}_{self.task}",
                    "scorer_cls": self.scorer_cls,
                    "labels": [ann.__repr__() for ann in _ann],
                    "text": self.template.render(
                        guidelines=[
                            inspect.getsource(definition) for definition in _guidelines
                        ],
                        text=_text,
                        annotations=_ann,
                    ),
                }

    def __iter__(self):
        random.seed(self.seed)
        instances = []
        total_inst = random.randint(*self.parallel_instances)
        prev_id = None
        for elem in self.loader:
            # Prevent mixing sentences from different documents. TODO: generalize
            if (len(instances) == total_inst) or (
                prev_id is not None and elem["doc_id"] != prev_id
            ):
                for samp in self._sample(instances):
                    yield samp
                instances = []
                total_inst = random.randint(*self.parallel_instances)

            instances.append(elem)
            prev_id = elem["doc_id"]

        if len(instances):
            for samp in self._sample(instances):
                yield samp
