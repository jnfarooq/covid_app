import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
from PIL import Image
import os
import glob
from fastai2.vision.all import *

n=512

label_dict = {0:'COVID-19',
                1:'NORMAL', 
                2:'Viral Pneumonia'}

st.title('Uploaded Image')
st.write('Hey, whats up')
# image = Image.open('12.png')
# st.image(image, caption='test image', use_column_width=True)
st.cache()
def load_model():
    return load_learner('export_18.pkl')

def hooked_backward(cat, m, xb):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g

def show_heatmap(hm, xb_im):
    _,ax = plt.subplots()
    xb_im.show(ctx=ax)
    ax.imshow(hm, alpha=0.6, extent=(0,n,n,0),
              interpolation='bilinear', cmap='magma')


def load_page():

    learn_inf = load_model()

    fname = st.sidebar.selectbox("Select Image", glob.glob("*.png"))

    # plt.imshow(Image.open(fname))
    # st.pyplot()

    st.image(Image.open(fname).resize((n,n)))
    # st.write(files)
    _, y, _ = learn_inf.predict(fname)
    # print(type(y.values().cpu()))


    if st.button('Classify'):
        pred = label_dict[int(y.numpy())]
        st.write(f'Preidction : {pred}')

    if st.button('Grad-CMA'):

        db = DataBlock(blocks=(ImageBlock), 
                         get_items=get_image_files, 
                         item_tfms=Resize(n),
                        ).dataloaders(Path(os.getcwd())).cpu()
        b = db.test_dl([fname]).one_batch()
        xb = b[0]
        xb_im = TensorImage(db.train.decode(b)[0][0])
        print(xb_im.shape)

        # xb.shape
        # st.write(f'{xb.shape}')

        m = learn_inf.model.eval().cpu()#cuda()

        hook_a,hook_g = hooked_backward(y, m, xb)

        acts  = hook_a.stored[0].cpu()
        # acts.shape

        avg_acts = acts.mean(0)
        # avg_acts.shape

        # ;
            
        grad = hook_g.stored[0][0].cpu()
        grad_chan = grad.mean(1).mean(1)
        # grad.shape,grad_chan.shape

        mult = (acts*grad_chan[...,None,None]).mean(0)
        show_heatmap(mult, xb_im)
        st.pyplot()

load_page()

