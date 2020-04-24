import numpy as np 
import streamlit as st 
import matplotlib.pyplot as plt 
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor
from PIL import Image
import os
import glob
from fastai2.vision.all import load_learner, DataBlock, ImageBlock, get_image_files, Resize, Path, TensorImage, hook_output

n=512
label_dict = {0:'COVID-19',
                1:'NORMAL', 
                2:'Viral Pneumonia'}

# st.title('Uploaded Image')
# st.write('Hey, whats up')
# image = Image.open('12.png')
# st.image(image, caption='test image', use_column_width=True)

def display_results(learn_inf, file_name, y):
            '''
            shows the file results:

            '''
            db = DataBlock(blocks=(ImageBlock), 
                            get_items=get_image_files, 
                            item_tfms=Resize(n),
                            ).dataloaders(Path(os.getcwd())).cpu()
            b = db.test_dl([file_name]).one_batch()
            xb = b[0]
            xb_im = TensorImage(db.train.decode(b)[0][0])
            m = learn_inf.model.eval().cpu()#cuda()

            hook_a,hook_g = hooked_backward(y, m, xb)
            acts  = hook_a.stored[0].cpu()
            # acts.shape
            avg_acts = acts.mean(0)
            # avg_acts.shape
            grad = hook_g.stored[0][0].cpu()
            grad_chan = grad.mean(1).mean(1)
            # grad.shape,grad_chan.shape
            mult = (acts*grad_chan[...,None,None]).mean(0)

            return mult, xb_im

# st.cache()
def load_model():
    return load_learner('squeezenet.pkl')

def hooked_backward(cat, m, xb):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g

def show_heatmap(hm, xb_im, img):
    # st.write(xb_im.shape)
    # st.write(hm.shape)
    # _,ax = plt.subplots()

    _,(ax1, ax) = plt.subplots(1,2, figsize=(8,3))
    ax1.imshow(img)
    ax1.axis('off')
    xb_im.show(ctx=ax)
    ax.imshow(hm, alpha=0.6, extent=(0,n,n,0),
              interpolation='bilinear', cmap='OrRd')
    plt.subplots_adjust(bottom=0.01, top=0.99)
    st.pyplot()

class Hook():
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class HookBwd():
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)   
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

def show_heat_maps(cls, learn, x, x_dec):

    with HookBwd(learn.model[0]) as hookg:
        with Hook(learn.model[0]) as hook:
            output = learn.model.eval()(x)#.cpu())
            act = hook.stored
        output[0,cls].backward()
        grad = hookg.stored

    w = grad[0].mean(dim=[1,2], keepdim=True)
    cam_map = (w * act[0]).sum(0)

    # x_dec = TensorImage(dls.train.decode((x,))[0][0])
    _,ax = plt.subplots(figsize=(10,10))
    x_dec.show(ctx=ax)
    ax.imshow(cam_map.detach().cpu(), alpha=0.6, extent=(0,512,512,0),
                interpolation='bilinear', cmap='OrRd');

def load_page():

    learn_inf = load_model()
    st.title('COVID-19 Detection from chest X-rays')
    st.write('## How does it work?')
    st.write('Add a chest x-ray and the machine learning model will look at it and will try to predict if COVID-19 symptoms are present or not. It will also show a heatmap of why a decision is made.')

    # fname = st.sidebar.selectbox("Select Image", glob.glob("*.png"))
    plt.figure(figsize=(8,3))
    plt.imshow(Image.open('banner.png'))#.resize((500, 224)),)#, use_column_width=True,)
    plt.axis('off')
    plt.subplots_adjust(bottom=0.01, top=0.99)
    st.pyplot()

    # st.write('Upload your image')
    st.write('**Note:** The model has not been tested in clinical setting, **NOT** to be used for diagnosis.')

    uploaded_image = st.file_uploader("Choose a png or jpg image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        # st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)
        # st.image(img, caption="Uploaded Image", width=224, )#,use_column_width=True)
        img = img.convert('RGB')

        img = img.resize((n,n), Image.BILINEAR)
        img.save('test.png')
        file_name = Path("./test.png")

        img_np = np.array(img)
        # print(img_np.shape)
        pred, y, _ = learn_inf.predict(img_np)
        st.write("Click **Predict**")

        # _,(ax1, ax) = plt.subplots(1,2, figsize=(20,15))
        # ax1.imshow(img)
        # ax1.axis('off')

        if st.button('Predict',):

            st.write(f"Prediction from the model: **{pred}**",)

            mult, xb_im = display_results(learn_inf, file_name, y)
            show_heatmap(mult, xb_im, img)
 
load_page()

