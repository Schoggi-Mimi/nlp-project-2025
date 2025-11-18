import json
import spacy
from spacy.tokens import DocBin
from spacy.util import minibatch, compounding
from spacy.training import Example
import random

def load_annotated_json(path):
    nlp = spacy.blank("en")
    db = DocBin()

    with open(path, "r", encoding="utf8") as f:
        data = json.load(f)

    anns = data["annotations"]

    for item in anns:
        if item is None:
            continue

        text, meta = item
        entities = meta["entities"]

        doc = nlp.make_doc(text)
        spans = []
        for start, end, label in entities:
            span = doc.char_span(start, end, label=label)
            if span is None:
                continue
            spans.append(span)
        doc.ents = spans
        db.add(doc)

    return db

def train_custom_ner(train_data_path, output_dir, labels, n_iter=20):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")
    for label in labels:
        ner.add_label(label)

    db = DocBin().from_disk(train_data_path)
    docs = list(db.get_docs(nlp.vocab))

    examples = []
    for doc in docs:
        ents = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        examples.append(Example.from_dict(doc, {"entities": ents}))

    optimizer = nlp.initialize()

    for i in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.2))
        for batch in batches:
            nlp.update(
                batch,
                drop=0.2,
                sgd=optimizer,
                losses=losses
            )
        print(f"Iteration {i+1}, Losses: {losses}")

    nlp.to_disk(output_dir)
    print(f"Saved model at: {output_dir}")

    return nlp

