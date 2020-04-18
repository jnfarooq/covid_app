import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
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




        # file_buffer = get_image()
        # st.write(file_buffer)

        # image = np.array(Image.open(file_buffer))
        # st.write(image)
        # st.write('Waiting for the image')
        # st.image(image.astype(np.uint8),caption=['Resized Image'], width=150)
        # st.write('Done')

# def get_image():
#     '''
#      Displays radio options to upload images either from url or local machine
     
#      Returns:
#          Image numpy array
#     '''
#     options = ['local machine','url', ]
#     upload = st.radio("Upload image from", options, index=0)
#     if upload == options[0]:
#         url = st.text_input('Enter url')
#         if url:
#             image = utils.url_to_image(url)
#             return image, url
#     elif upload == options[1]:
#         file_buffer = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
#         if file_buffer:
#             # to PIL image obj and to np array for easy processing
#             image = np.array(PIL.Image.open(file_buffer))
#             return file_buffer

load_page()

# img_file_buffer = st.file_uploader('upload an image', type=["png", "jpg", "jpeg"])
# st.write(img_file_buffer)
# # img = Image.open(img_file_buffer)
# # image = np.array(img)
# if img_file_buffer is not None:
#     st.write('something is happenind')
#     image = Image.open(img_file_buffer)
#     img_array = np.array(image) # if you want to pass it to OpenCV
#     st.image(image, caption="The caption", use_column_width=True)
#     # plt.imshow(img_array)
#     # st.pyplot()
