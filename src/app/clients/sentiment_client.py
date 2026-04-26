from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


class InferenceSentiment:
    def __init__(
        self,
        vectorizer_path: str | Path = "artifacts/models/tfidf.joblib",
        model_path: str | Path = "artifacts/models/sentiment_model.joblib",
    ) -> None:
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
    
    def __call__(self, texts: list[str]) -> list[dict[str, Any]]:
        clean_texts = [str(t or "") for t in texts]
        X = self.vectorizer.transform(clean_texts)

        labels = self.model.predict(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            classes = list(self.model.classes_)

            outputs = []
            for label, prob_row in zip(labels, probs):
                prob_map = {cls: float(p) for cls, p in zip(classes, prob_row)}
                confidence = max(prob_map.values()) if prob_map else 0.0

                outputs.append(
                    {
                        "sentiment_label": str(label),
                        "sentiment_confidence": float(confidence)
                    }
                )
            
            return outputs

        return [
            {
                "sentiment_label": str(label),
                "sentiment_confidence": 1.0,
            }
            for label in labels
        ]

def main() -> None:
    infer = InferenceSentiment()
    texts = ["Great quality Makes for easy clean up and avoids messy burners !,sleek stove ! great quality makes for easy clean up and avoids messy burners",
             "Horrible,You’d expect the water to taste clean and filtered for the purpose your refrigerator is made to do These filters were the opposite When needed to change the filter I put one of these and the taste of the water was horrible ! ! Not only the taste it continued to pop out unexpectedly and difficult to put back and get them to stay I ended up throwing them out ! I purchased the filters from the refrigerators maker and now all of great ! Personally I don’t advise getting these filters,horrible you’d expect the water to taste clean and filtered for the purpose your refrigerator is made to do these filters were the opposite when needed to change the filter i put one of these and the taste of the water was horrible ! ! not only the taste it continued to pop out unexpectedly and difficult to put back and get them to stay i ended up throwing them out ! i purchased the filters from the refrigerators maker and now all of great ! personally i don’t advise getting these filters"]\
    
    print(infer.predict(texts))

# if __name__ == "__main__":
#     main()