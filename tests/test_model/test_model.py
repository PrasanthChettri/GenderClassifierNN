from web.main import classification_out
from classifier.classify import Classifier
import pytest

#TODO  : Make it more robust add stuff like torchtest

def test_model():
    names = ["Natasia",  "George", "Marie"]
    size = len(names)
    classifier = Classifier(classifications=size)
    classifications = classifier.classify(names)
    assert isinstance(classifications , list)