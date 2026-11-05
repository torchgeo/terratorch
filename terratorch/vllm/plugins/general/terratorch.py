from vllm.model_executor.models.terratorch import Terratorch

# This class is only needed for supporting vLLM >0.21.0 and <=0.24.0
# where models that do not define a language model fail loading.
# A proper fix in vLLM is under way.
class TerratorchFix(Terratorch):
    
    def get_language_model(self):
        return self
