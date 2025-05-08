from dataclasses import dataclass
from typing import Dict, List, Union


@dataclass
class ModelConfig:
    name: str
    model_id: str
    type: str


class ModelConfigs:
    DISTILROBERTA = ModelConfig(
        name="DistilRoBERTa",
        model_id="j-hartmann/emotion-english-distilroberta-base",
        type="emotion"
    )

    VADER = ModelConfig(
        name="VADER",
        model_id="",
        type="sentiment"
    )

    @classmethod
    def get_all(cls) -> Dict[str, ModelConfig]:
        return {
            'distilroberta': cls.DISTILROBERTA,
            'vader': cls.VADER
        }


class EmotionLabels:
    JOY = "Joy"
    SADNESS = "Sadness"
    ANGER = "Anger"
    FEAR = "Fear"
    SURPRISE = "Surprise"
    NEUTRAL = "Neutral"
    LOVE = "Love"
    DISGUST = "Disgust"
    OPTIMISM = "Optimism"
    PESSIMISM = "Pessimism"
    TRUST = "Trust"
    ANTICIPATION = "Anticipation"

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        return {
            'joy': cls.JOY,
            'sadness': cls.SADNESS,
            'anger': cls.ANGER,
            'fear': cls.FEAR,
            'surprise': cls.SURPRISE,
            'neutral': cls.NEUTRAL,
            'love': cls.LOVE,
            'disgust': cls.DISGUST,
            'optimism': cls.OPTIMISM,
            'pessimism': cls.PESSIMISM,
            'trust': cls.TRUST,
            'anticipation': cls.ANTICIPATION
        }


EmotionScore = Union[float, Dict[str, float]]
EmotionResult = Dict[str, Union[str, List[Dict[str, Union[str, float]]], Dict[str, float]]]
