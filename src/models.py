import torch
import torch.nn as nn

from .models.audio import Model1
from .models.image import get_model_transfer_learning

class Modelv(nn.Module): #new
    def __init__(self, num_classes: int = 1000) -> None:

        super().__init__()


            
        self.im_model = get_model_transfer_learning('shufflenet_v2_x1_0', n_classes=80)
        self.audio = Model1(with_head=False)
        


        self.avgpool=nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, num_classes)
        )
        
        



    def forward(self, x) -> torch.Tensor:
        # (img, audio)
        # print(x[0].shape)
        x[0]=self.im_model(x[0])
        x[1]=self.avgpool(self.audio(x[1])).flatten(start_dim=1)
        im_au=torch.cat((x[0], x[1]), dim=1)
        return self.fc(im_au)
    

class AuViModel1(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        
        self.a_backbone.proj=torch.load('proj.pt')
        
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.f_backbone.classifier = nn.Identity()

        self.fc=nn.Linear(2*768,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            a_feats = self.a_backbone.proj(a_feats)
            a_feats=a_feats.mean(dim=1) # LSTM


            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            f_feats=f_feats.mean(dim=1) # LSTM
        
        feats = torch.concat([a_feats, f_feats], dim=-1)
        
        logits = self.fc(feats)
        return logits


class AuViRNNModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        
        self.a_backbone.proj=torch.load('proj.pt')
        
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.f_backbone.classifier = nn.Identity()
        self.f_rnn = nn.RNN(768,384, batch_first=True, num_layers=2)
        self.fc=nn.Linear(768+384,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            a_feats = self.a_backbone.proj(a_feats)
            # a_feats=a_feats.mean(dim=1) # LSTM
            a_feats = self.f_rnn(a_feats)[1][-1]


            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            f_feats=f_feats.mean(dim=1) # LSTM
        
        feats = torch.concat([a_feats, f_feats], dim=-1)
        
        logits = self.fc(feats)
        return logits


class ViModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()


        
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        for param in self.f_backbone.parameters():
            param.requires_grad = False

        for param in self.f_backbone.vit.encoder.layer[-2:].parameters():
            param.requires_grad = True
        

        self.f_backbone.classifier = nn.Identity()
        self.fc=nn.Linear(768,num_classes)


    def forward(self, samples) -> torch.Tensor:
        # audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            f_feats=f_feats.mean(dim=1) 
            
        logits = self.fc(f_feats)
        return logits
    
class ViLSTMModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()


        
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        for param in self.f_backbone.parameters():
            param.requires_grad = False

        

        self.f_backbone.classifier = nn.Identity()
        self.rnn = nn.LSTM(768, 768//2 ,num_layers=2 ,batch_first=True)
        self.fc=nn.Linear(768//2,num_classes)


    def forward(self, samples) -> torch.Tensor:
        # audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
        
        o, (hn, cn)=self.rnn(f_feats)
            
        logits = self.fc(hn[-1])
        return logits
    

class FullEm2vecLSTM384h(nn.Module): #new
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
    
    
class AuViLSTMModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        
        # self.a_backbone.proj=torch.load('proj.pt')
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.f_backbone.classifier = nn.Identity()
        # self.f_rnn = nn.LSTM(768//2, 768//2 ,num_layers=2 ,batch_first=True)
        self.a_rnn = nn.LSTM(768,384, batch_first=True, num_layers=2)
        self.fc=nn.Linear(768+384,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            # a_feats = self.a_backbone.proj(a_feats)
            # a_feats=a_feats.mean(dim=1) # LSTM
            
        o, (hn, cn)= self.a_rnn(a_feats)
        a_feats = hn[-1]
            

        with torch.inference_mode():
            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            # f_feats = self.f_rnn(f_feats)[1][-1]
            f_feats=f_feats.mean(dim=1) # LSTM
        
        feats = torch.concat([a_feats, f_feats], dim=-1)
        
        logits = self.fc(feats)
        return logits
    
class AuVi2LSTMModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        
        # self.a_backbone.proj=torch.load('proj.pt')
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.f_backbone.classifier = nn.Identity()
        self.f_rnn = torch.load('VLSTM.pt')
        # self.f_rnn = nn.LSTM(768, 768//2 ,num_layers=2 ,batch_first=True)
        # self.a_rnn = nn.LSTM(768,384, batch_first=True, num_layers=2)
        self.a_rnn = torch.load('ALSTM.pt')
        self.fc=nn.Linear(384+384,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            # a_feats = self.a_backbone.proj(a_feats)
            # a_feats=a_feats.mean(dim=1) # LSTM
            
            o, (hn, cn)= self.a_rnn(a_feats)
            a_feats = hn[-1]
            

        with torch.inference_mode():
            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            # f_feats = self.f_rnn(f_feats)[1][-1]
            # f_feats=f_feats.mean(dim=1) # LSTM
            o, (hn, cn)= self.f_rnn(f_feats)
            f_feats = hn[-1]
    
        feats = torch.concat([a_feats, f_feats], dim=-1)
        
        logits = self.fc(feats)
        return logits
    

class AuLSTMModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        

        self.a_rnn = nn.LSTM(768,384, batch_first=True, num_layers=2)
        self.fc=nn.Linear(384,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']

            
        o, (hn, cn)= self.a_rnn(a_feats)
        feats = hn[-1]

        
        logits = self.fc(feats)
        return logits
    

class AuViPLSTMModel(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        
        self.a_backbone.proj=torch.load('proj.pt')
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.f_backbone.classifier = nn.Identity()
        # self.f_rnn = nn.LSTM(768//2, 768//2 ,num_layers=2 ,batch_first=True)
        self.a_rnn = nn.LSTM(384,384, batch_first=True, num_layers=2)
        self.fc=nn.Linear(768+384,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            
            # a_feats=a_feats.mean(dim=1) # LSTM
        a_feats = self.a_backbone.proj(a_feats)    
        o, (hn, cn)= self.a_rnn(a_feats)
        a_feats = hn[-1]
            

        with torch.inference_mode():
            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            # f_feats = self.f_rnn(f_feats)[1][-1]
            f_feats=f_feats.mean(dim=1) # LSTM
        
        feats = torch.concat([a_feats, f_feats], dim=-1)
        
        logits = self.fc(feats)
        return logits



class AuViRNN1Model(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel

        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        
        self.a_backbone.proj=nn.Linear(768, 768//2)
        
        from transformers import AutoModelForImageClassification 


        
        self.f_backbone = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.f_backbone.classifier = nn.Identity()
        self.f_rnn = nn.RNN(768,384, batch_first=True, num_layers=1)
        self.a_rnn = nn.RNN(768,384, batch_first=True, num_layers=1)
        self.fc=nn.Linear(384+384,num_classes)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        frames =samples['frames']
        frames_shape =frames.shape
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            a_feats = self.a_backbone.proj(a_feats)
            # a_feats=a_feats.mean(dim=1) # LSTM
            a_feats = self.a_rnn(a_feats)[1][-1]
            


            frames =frames.view(-1,*frames_shape[-3:])
            f_feats = self.f_backbone(frames).logits
            f_feats = f_feats.view(*frames_shape[:-3], -1)
            f_feats = self.f_rnn(f_feats)[1][-1]
            # f_feats=f_feats.mean(dim=1) # LSTM
        
        feats = torch.concat([a_feats, f_feats], dim=-1)
        
        logits = self.fc(feats)
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

class AuModel1(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel
        from src.helpers import load_model

        model = AutoModel(model="iic/emotion2vec_plus_base", device='cuda')

            
        self.a_backbone= model

        
        # self.a_backbone.proj=None


        self.fc=Em2vecModel(num_classes=num_classes)
        load_model("Em2vecModelNoEYASEWithsampler0.5,0.3", self.fc)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio'].squeeze().cpu().numpy()
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            # audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            # a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']
            a_feats = torch.from_numpy(self.a_backbone.inference(audio, granularity="frame", extract_embedding=True,disable_pbar=True)[0]['feats'][None,:]).to('cuda')
            logits =self.fc(a_feats)

        return logits   


class AuModel2(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel
        from src.helpers import load_model
        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        


        

        self.fc=Em2vecModel(num_classes=num_classes)
        load_model("Em2vecModelNoEYASEWithsampler0.5,0.3", self.fc)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']

        # a_feats = self.a_backbone.proj(a_feats)
        # a_feats=a_feats.mean(dim=1) # LSTM
 
        logits = self.fc(a_feats)
        return logits
    

class AuModel3(nn.Module): #new
    def __init__(self,num_classes: int = 5) -> None:

        super().__init__()

        from funasr import AutoModel
        from src.helpers import load_model
        model = AutoModel(model="iic/emotion2vec_plus_base")

            
        self.a_backbone= model.model
        for param in self.a_backbone.parameters():
            param.requires_grad = False
        


        

        self.fc=Em2vecModel(num_classes=num_classes)
        load_model("Em2vecModelNoEYASEWithsampler0.5,0.3", self.fc)


    def forward(self, samples) -> torch.Tensor:
        audio = samples['audio']
        with torch.inference_mode():
            # layer normalization over just the last dimension (time dimension)
            audio= nn.functional.layer_norm(audio.squeeze(1), [audio.shape[-1]])
            a_feats =self.a_backbone.extract_features(audio,padding_mask=None)['x']

        # a_feats = self.a_backbone.proj(a_feats)
        # a_feats=a_feats.mean(dim=1) # LSTM
 
        logits = self.fc(a_feats)
        return logits