from transformers import Wav2Vec2Model, Wav2Vec2Config, PreTrainedModel, PretrainedConfig, AutoModel, AutoConfig
import torch.nn as nn

DOWNSAMPLING_FACTOR = 320


class Wav2Vec2CommonPhoneConfig(PretrainedConfig):
    model_type="wav2vec2"

    def __init__(
        self,
        n_classes: int = 102,
        **kwargs,
    ):
        self.n_classes = n_classes
        super().__init__(**kwargs)


class Wav2Vec2(PreTrainedModel):

    config_class = Wav2Vec2CommonPhoneConfig

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        
        self.wav2vec = Wav2Vec2Model(Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53"))
        self.linear = nn.Linear(in_features=1024, out_features=config.n_classes)

    def get_trainable_parameters(self):
        params = []
        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
        return params

    def forward(self, x):
        # Output shape: (Batch, Time, Channels)
        x = self.wav2vec(x)
        y = self.linear(x.last_hidden_state)
        return y, x.last_hidden_state, x.extract_features
