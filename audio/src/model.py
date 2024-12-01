import torch
import torch.nn as nn


class Model1(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.proj=None
        self.fc=nn.Sequential(nn.Linear(768,num_classes)) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            x=x.mean(dim=1)
        x=self.fc(x)
        return x
    

class Model2(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.fc=nn.Sequential(nn.Linear(768,768//2), nn.PReLU() ,nn.Linear(768//2,num_classes)) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fc(x)
        return x   
    

class Model3(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.fc=nn.Sequential(nn.Linear(2*768,768), nn.PReLU() ,nn.Linear(768,768//2), nn.PReLU(), nn.Linear(768//2,num_classes)) 


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fc(x)
        return x   

class Model4(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        # Load model and tokenizer
        from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification
)
        model_name = "bhadresh-savani/albert-base-v2-emotion"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.fc=model.classifier
        self.map ={0:4, 1:2,3:0,4:1 }


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.fc(x)
        for key in self.map:
            x[...,key]
        return x   
    

class Wav2vecModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.projector = nn.Linear(1024, 768)
        self.classifier = nn.Linear(768, num_classes)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits 
    
class Em2vecModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.projector = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_classes)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits 
    
class Em2vecModel2linear(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.projector = nn.Linear(768, 768)
        self.classifier = nn.Sequential(nn.Linear(768,768//2), nn.PReLU() ,nn.Linear(768//2,num_classes)) 

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits 
    
class Em2vecModel2proj(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.projector = nn.Sequential(nn.Linear(768,768//2), nn.PReLU() ,nn.Linear(768//2,768//2)) 
        self.classifier = nn.Linear(768//2,num_classes)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits 
    
class Em2vecModel384(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()
        self.projector = nn.Linear(768, 768//2)
        self.classifier = nn.Linear(768//2, num_classes)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)
        logits = self.classifier(pooled_output)
        return logits 
    
class FullEm2vec384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768//2)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        x=x.mean(dim=1)
        x=self.classifier(x)
        return x
    

class FullEm2vecRNN384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.RNN(768//2, 768//2 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        x=self.rnn(x)[-1][-1]
        x=self.classifier(x)
        return x
    
class FullEm2vecGRU384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.backbone.blocks[-1:].parameters():
        #     param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.GRU(768, 768//2 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        # x=self.projector(x)
        x=self.rnn(x)[-1][-1]
        x=self.classifier(x)
        return x
    
class FullEm2vecBiGRU384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.backbone.blocks[-2:].parameters():
        #     param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.GRU(768, 768//2 ,num_layers=2, bidirectional=True ,batch_first=True)
        self.classifier = nn.Linear(768, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        # x=self.projector(x)

        o, h =self.rnn(x)
        last_hidden_forward = h[-2, :, :]  # Last hidden state from the forward direction
        last_hidden_backward = h[-1, :, :]  # Last hidden state from the backward direction
        all_h = torch.concat((last_hidden_forward,last_hidden_forward), dim=-1)
        o=self.classifier(all_h)
        return o




    
class FullEm2vecGRU768h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.backbone.blocks[-2:].parameters():
        #     param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.GRU(768, 768 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        # x=self.projector(x)
        x=self.rnn(x)[-1][-1]
        x=self.classifier(x)
        return x
    
class FullEm2vecGRU256h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.backbone.blocks[-2:].parameters():
        #     param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.GRU(768, 768//3 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768//3, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        # x=self.projector(x)
        x=self.rnn(x)[-1][-1]
        x=self.classifier(x)
        return x
    
class FullEm2vec1GRU384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.backbone.blocks[-2:].parameters():
        #     param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.GRU(768, 768//2 ,num_layers=1 ,batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        # x=self.projector(x)
        x=self.rnn(x)[-1][-1]
        x=self.classifier(x)
        return x
    

class FullEm2vecTanh384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768//2)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        x=torch.tanh(x.mean(dim=1))
        x=self.classifier(x)
        return x
    
    
class FullEm2vec768h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768)
        self.classifier = nn.Linear(768, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        x=x.mean(dim=1)
        x=self.classifier(x)
        return x
    
class FullEm2vec384hAProj(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        self.projector = nn.Sequential(nn.Linear(768, 384),nn.PReLU())
        self.attention = nn.Linear(384, 1)
        self.classifier = nn.Linear(384, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x = self.projector(x)
        weights = nn.functional.softmax(self.attention(x), dim=1)
        x = (x * weights).sum(dim=1)
        return self.classifier(x)
    

class FullEm2vecLSTM384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-6:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.LSTM(768//2, 768//2 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])

        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x
    

class FullEm2vecNPLSTM384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.LSTM(768, 768//2 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])

        x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        # x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x


class Wav2vec2LSTM512h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from transformers import  AutoModelForPreTraining

        model = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")

            
        self.backbone= model.wav2vec2
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.encoder.layers[-8:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        # self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.LSTM(1024, 1024//2 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(1024//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])

        x=self.backbone(x).last_hidden_state
            
        # x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x
    

class FullEm2vecLSTM768h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768)
        self.rnn = nn.LSTM(768, 768 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x
    

class FullEm2vec5LSTM384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.LSTM(768//2, 768//2 ,num_layers=5 ,batch_first=True)
        self.classifier = nn.Linear(768//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x
    

class FullEm2vecLargeLSTM384h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_large")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        # for param in self.backbone.blocks[-2:].parameters():
        #     param.requires_grad = True
        
        self.backbone.proj=None
        self.projector = nn.Linear(1024, 1024//2)
        self.rnn = nn.LSTM(1024//2, 1024//2 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(1024//2, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x
    


class FullEm2vecTest(nn.Module):
    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        from funasr import AutoModel

        self.emotion_mapping = {
            '生气/angry': 0,
            '开心/happy': 1,
            '中立/neutral': 2,
            '难过/sad': 3
        }
        self.full_emotions = ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', 
                            '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']
        
        model = AutoModel(model="iic/emotion2vec_plus_base")
        self.backbone = model.model
        for param in self.backbone.parameters():
            param.requires_grad = False
        

        
    def _map_emotions(self, logits: torch.Tensor) -> torch.Tensor:
        # Create mask of -inf for unwanted classes
        mask = torch.full_like(logits, float('-inf'))
        

        kept_values = []
        
        for idx, emotion in enumerate(self.full_emotions):
            if emotion in self.emotion_mapping:
                mask[..., idx] = logits[..., idx] 
                kept_values.append(logits[..., idx])
        
        # Create new tensor with only desired emotions
        new_logits = torch.stack(kept_values, dim=-1)
            
        return new_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            x = nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x = self.backbone.extract_features(x, padding_mask=None)['x']
            
        x = x.mean(dim=1)
        x =  self.backbone.proj(x)
        x = self._map_emotions(x)
        return x
    

class FullEm2vecLSTM192h(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.backbone= model.model
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.blocks[-2:].parameters():
            param.requires_grad = True
        
        self.backbone.proj=None
        self.projector = nn.Linear(768, 768//2)
        self.rnn = nn.LSTM(768//2, 768//4 ,num_layers=2 ,batch_first=True)
        self.classifier = nn.Linear(768//4, num_classes)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            x= nn.functional.layer_norm(x.squeeze(1), [x.shape[-1]])
            x=self.backbone.extract_features(x,padding_mask=None)['x']
            
        x=self.projector(x)
        o, (hn, cn)=self.rnn(x)
        x=self.classifier(hn[-1])
        return x