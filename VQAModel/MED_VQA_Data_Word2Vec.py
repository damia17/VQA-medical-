from torch.utils.data import Dataset
from FeatureExtractor import extract_image_features, get_sentence_embedding

# classe to load the datasets used informations for the dataloaders
class MED_VQA_Data(Dataset):
    def __init__(self, df ):
        self.df = df
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df["img_id"][idx]
        question = self.df['question'][idx]
        type_d = self.df['mode'][idx]
        type = self.df['answer_type'][idx]
        label = self.df['label'][idx]

        image_embedding = extract_image_features(image,type_d)#?[768]

        text_embedding = get_sentence_embedding(question)#?[768]        
        
        encoding = {"img_emb": image_embedding,"text_emb": text_embedding,"label": label,"type": type}
        return encoding