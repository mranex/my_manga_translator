## 2024-05-23 - Heavy Model Re-initialization
**Learning:** In Flask apps, heavy machine learning models (like NLLB or MangaOCR) should never be initialized inside request handlers or `__init__` methods of classes instantiated per request. This causes massive latency and memory spikes as the model is re-loaded from disk for every single user request.
**Action:** Always implement class-level caching or global singleton patterns for ML models. Use `_model_cache` class attributes or a global registry to ensure models are loaded once and shared.
