from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceHubEmbeddings
from secret import HUGGINGFACE_API_KEY

embedding = HuggingFaceHubEmbeddings(huggingfacehub_api_token=HUGGINGFACE_API_KEY, 
                                     repo_id="sentence-transformers/all-MiniLM-L12-v2")

text1 = 'simple text for FAISS testing'
text2 = 'another simple text for vectorDB testing'

vectorstore = FAISS.from_texts(
    [text1, text2], embedding=embedding
)

docs = vectorstore.similarity_search("hello")
print(docs)
docs = vectorstore.similarity_search("vectorDB")
print(docs)
docs = vectorstore.similarity_search("FAISS")
print(docs)

# -------------------------------------------------------------------

import torch 
import torch.nn as nn 
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io.image import ImageReadMode

import uuid
import faiss
from langchain.docstore.in_memory import InMemoryDocstore

class feat_extractor(nn.Module):
    def __init__(self, model): 
        super().__init__()
        self.model = model 

    def forward(self, x) : 
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x 

def load_images(image_paths, preprocess):
    images = [] 
    for path in image_paths: 
        img = read_image(path, mode=ImageReadMode.RGB) 
        images.append(preprocess(img))
    return torch.stack(images)

class CustomFAISS(): 
    def __init__(self, index, docstore):
        self.index = index
        self.docstore = docstore 

    def add(self, feats, image_paths): 
        self.index.add(feats)
        ids = [str(uuid.uuid4()) for _ in image_paths]
        self.docstore.add({id_: path for id_, path in zip(ids, image_paths)})
        self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(ids)}

    def _similiarity_search(self, feat, k=1):
        scores, indices = self.index.search(feat, k)
        docs = [] 
        for j, idx in enumerate(indices[0]): 
            if idx == -1 : continue 
            _id = self.index_to_docstore_id[idx]
            doc = self.docstore.search(_id)
            docs.append((doc, scores[0][j]))
        return docs[:k]


image_paths = ['dogs_image/chihuahua.jpg',
               'dogs_image/shiba.jpeg',
               'dogs_image/york.jpg',]

weights = ResNet50_Weights.DEFAULT
model = feat_extractor(resnet50(weights=weights))
model.eval()

preprocess = weights.transforms()

batch = load_images(image_paths, preprocess)
feats = model(batch)
feats = feats.detach().numpy()

index = faiss.IndexFlatIP(len(feats[0]))
docstore = InMemoryDocstore()

custom_faiss = CustomFAISS(index, docstore)
custom_faiss.add(feats, image_paths)

print('Chihuahua Test Image.. ', end='')
test_image_path = 'dogs_image/chihuahua_test.jpg'
batch = load_images([test_image_path], preprocess)
feat = model(batch).detach().numpy()
docs = custom_faiss._similiarity_search(feat, k=1)
print(f'most similiar images: {docs[0][0]} with score : {docs[0][1]:.3f}')


print('Shiba Test Image.. ', end='')
test_image_path = 'dogs_image/shiba_test.jpeg'
batch = load_images([test_image_path], preprocess)
feat = model(batch).detach().numpy()
docs = custom_faiss._similiarity_search(feat, k=1)
print(f'most similiar images: {docs[0][0]} with score : {docs[0][1]:.3f}')

print('York Test Image.. ', end='')
test_image_path = 'dogs_image/york_test.jpg'
batch = load_images([test_image_path], preprocess)
feat = model(batch).detach().numpy()
docs = custom_faiss._similiarity_search(feat, k=1)
print(f'most similiar images: {docs[0][0]} with score : {docs[0][1]:.3f}')



print()
print()
print()