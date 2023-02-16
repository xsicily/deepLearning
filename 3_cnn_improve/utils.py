import numpy as np
import matplotlib, time, copy, os, requests, zipfile, sys 
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt 

def download_file_from_google_drive(id, destination):
    #URL = "https://docs.google.com/uc?export=download"
    URL = "https://docs.google.com/uc?export=download&confirm=9iBg"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_fm(PATH):
    if not os.path.exists(PATH) or not os.path.exists(os.path.join(PATH, 'fashion_mnist_extracted')):
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        
        #file_id = '1MCBAt9JcbNSC6WbN1ns6r5IrL_RPG0oQ'
        file_id='16U6yu8AI1IAMnmwMQC2o5M3ioQXmjp98'
        #id=16U6yu8AI1IAMnmwMQC2o5M3ioQXmjp98&
        destination = os.path.join(PATH, 'fashion_mnist_extracted.zip')
        download_file_from_google_drive(file_id, destination)
        
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(PATH)
            
        print("Data downloaded and extracted!")
        
        os.remove(destination)
        
    else:
        print("Data was already downloaded and extracted!")

def get_preds_figure(net, images, labels):
    
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.cpu().numpy())
    probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]
    
    nrows, ncols = 2, 4
    fig, ax = plt.subplots(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            ax[i,j].imshow(images[i*ncols+j].mean(dim=0).cpu().numpy(), cmap="Greys")
            ax[i,j].set_title("{0}, {1:.1f}%\n(label: {2})".format(
                classes[preds[i*ncols+j]],
                probs[i*ncols+j] * 100.0,
                classes[labels[i*ncols+j]]), color=("green" if preds[i*ncols+j]==labels[i*ncols+j].item() else "red"))
            ax[i,j].axis('off')
              
    fig.tight_layout()
    return fig