import streamlit as st
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
import clip
from natsort import natsorted
from typing import List, Tuple
import json

# Lưu trạng thái vào tệp tin JSON
def save_session_state(session_state, filename):
    with open(filename, 'w') as file:
        json.dump(session_state, file)

# Đọc trạng thái từ tệp tin JSON
def load_session_state(filename):
    with open(filename, 'r') as file:
        session_state = json.load(file)
        return session_state

##Set up page
st.set_page_config(
    page_title="Kumo Search",
    page_icon=":🕷:",
    layout="wide",
)

#Set sidebar:
with st.sidebar:
    st.header(":blue[KUMO]:snow_cloud:", divider='rainbow')

    st.write("Member:\n")
    image = Image.open('DyThen.jpg')
    image = image.resize((200, 200))
    st.image(image, caption='Nguyễn Duy Thắng')

    st.write("🕸️ Contact us:\n")
    st.write("✉ Email: 22521333@gm.uit.edu.vn\n")
    st.write("󠁆󠁆ⓕ Facebook: https://www.facebook.com/dythen.kumo")

#Set logo
col = st.columns(4)
with col[0]:
    img = Image.open('AI-Challange.jpg')
    img = img.resize((100, 100))
    st.image(img)
with col[1]:
    img = Image.open('UIT.jpg')
    img = img.resize((100, 100))
    st.image(img)
with col[2]:
    img = Image.open('AI-Club.jpg')
    img = img.resize((100, 100))
    st.image(img)
with col[3]:
    img = Image.open('Kumo.jpg')
    img = img.resize((100, 100))
    st.image(img)

#Tạo model xử lý text
class TextEmbedding():
  def __init__(self):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model, _ = clip.load("ViT-B/32", device=self.device)

  def __call__(self, text: str) -> np.ndarray:
    text_inputs = clip.tokenize([text]).to(self.device)
    with torch.no_grad():
        text_feature = self.model.encode_text(text_inputs)[0]

    return text_feature.detach().cpu().numpy()

#Tạo model xử lý ảnh
class ImageEmbedding():
    def __init__(self):
        self.device = "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def __call__(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_feature = self.model.encode_image(image_input)[0]

        return image_feature.detach().cpu().numpy()

#Load clip-feature
@st.cache_data
def indexing_methods(features_path: str) -> pd.DataFrame:
    data = {'video_name': [], 'frame_index': [], 'features_vector': [], 'features_dimension': []}

    npy_files = natsorted([file for file in os.listdir(features_path) if file.endswith(".npy")])

    for feat_npy in tqdm(npy_files):
        video_name = feat_npy.split('.')[0]
        feats_arr = np.load(os.path.join(features_path, feat_npy))

        # Lặp qua từng dòng trong feats_arr, mỗi dòng là một frame
        for idx, feat in enumerate(feats_arr):
            data['video_name'].append(video_name)
            data['frame_index'].append(idx)
            data['features_vector'].append(feat)
            data['features_dimension'].append(feat.shape)

    df = pd.DataFrame(data)
    return df
#Load data
FEATURES_PATH = r"D:\CLB-AI\AI Challenge 2023\Data\clip-features-vit-b32"
visual_features_df = indexing_methods(FEATURES_PATH)

#Tính độ tương đồng giữa các vector
def similar(query_arr, feat_vec, measure_method):
    distance = 0
    if measure_method == "cosine_similarity":
        dot_product = query_arr @ feat_vec
        query_norm = np.linalg.norm(query_arr)
        feat_norm = np.linalg.norm(feat_vec)
        cosine_similarity = dot_product / (query_norm * feat_norm)
        distance = 1 - cosine_similarity
    elif measure_method == "euclid":
        distance = np.linalg.norm(query_arr - feat_vec)
    else:
        distance = np.linalg.norm(query_arr - feat_vec, ord=1)

    return distance

#Hàm tìm kiếm
def search_engine(query_arr: np.array,
                  db: list,
                  topk:int=10,
                  measure_method: str="cosine_similarity") -> List[dict]:
    measure = []
    for ins_id, instance in enumerate(db):
        video_name, idx, feat_vec, feat_dim = instance

        distance = similar(query_arr, feat_vec, measure_method)

        measure.append((ins_id, distance))

    measure = sorted(measure, key=lambda x:x[1])

    search_result = []
    for instance in measure[:topk]:
        ins_id, distance = instance
        video_name, idx = db[ins_id][0], db[ins_id][1]

        search_result.append({"video_name": video_name,
                              "keyframe_id": idx,
                              "score": distance})

    # Đảm bảo trả về đúng topk kết quả
    while len(search_result) < topk and len(measure) > len(search_result):
        ins_id, distance = measure[len(search_result)]
        video_name, idx = db[ins_id][0], db[ins_id][1]
        search_result.append({"video_name": video_name,
                              "keyframe_id": idx,
                              "score": distance})

    return search_result

#Hàm đọc ảnh tìm được
def read_image(results: List[dict]):
    images = []
    data = []
    IMAGE_KEYFRAME_PATH = r"D:\CLB-AI\AI Challenge 2023\Data\Keyframes"  # Đường dẫn đến thư mục chứa keyframes

    for res in results:
        data.append(res)
        video_name = res["video_name"]
        keyframe_id = res["keyframe_id"]
        temp = video_name.split('_')
        video_folder = os.path.join(IMAGE_KEYFRAME_PATH, 'Keyframes_' + temp[0], 'keyframes', video_name)

        if os.path.exists(video_folder):
            image_files = sorted(os.listdir(video_folder))

            if keyframe_id < len(image_files):
                image_file = image_files[keyframe_id]
                image_path = os.path.join(video_folder, image_file)
                image = Image.open(image_path)
                images.append(image)
            else:
                st.write(f"Keyframe id {keyframe_id} is out of range for video {video_name}.")

    return (images, data)

#Hàm tạo file nộp
def convert_df(lst):
    df = pd.DataFrame(lst)
    return df.to_csv(index=False, header=False).encode('utf-8')

MAP_KEYFRAME_PATH = r"D:\CLB-AI\AI Challenge 2023\Data\map-keyframes"
def submit(answer, name):
    if len(answer) > 0:
        last_answer = []
        #Tìm lại đúng keyframe_idx
        for ans in answer:
            namefile = str(ans[0]) + '.csv'
            path_name = os.path.join(MAP_KEYFRAME_PATH, namefile)
            df = pd.read_csv(path_name)
            
            temp = [ans[0], df['frame_idx'][ans[1]]]
            last_answer.append(temp)
            # #Thêm các keyframe lân cận
            # temp = [ans[0], df['frame_idx'][ans[1] - 1]]
            # last_answer.append(temp)
            # temp = [ans[0], df['frame_idx'][ans[1] + 1]]
            # last_answer.append(temp)

        #Tạo file để nộp
        filename = 'query-p3-' + str(name) + '.csv'
        file = convert_df(last_answer)
        st.download_button(label="Download data as CSV",
                        data=file,
                        file_name=filename,
                        mime='text/csv',)

#Hàm hiển thị ảnh 
def visualize(imgs,data):
    columns = st.columns(5)
    for i, img in enumerate(imgs):
        with columns[i%5]:
            name = "Keyframe-result number {}: ".format(i+1) + str(data[i]["video_name"]) + ' ' + str(data[i]["keyframe_id"])
            temp = img.resize((320, 180))
            st.image(temp, caption=name)    
#Kết thúc phần khởi tạo cho back-end

##Bắt đầu truy vấn
text_embedd = TextEmbedding()
file_name = 'Session.json'
# Thay đổi các giá trị dựa trên tùy chọn của người dùng
with st.form('Nhập các parameter: '):
    text_query = st.text_input('Mô tả (vui lòng viết bằng tiếng Anh): ')
    topk = st.slider('Số lượng kết quả tìm kiếm', 0, 50, 15)
    measure_method = st.selectbox('Chọn phương thức đo chính: ', ('cosine_similarity', 'mahatan', 'euclid'))
    submit_button = st.form_submit_button("OK")

st.info('Kumo có thể mất vài phút để làm việc', icon="🕷")
#Lưu lại phiên làm việc
if submit_button:
    session = {'run': True}
    save_session_state(session_state= session, filename=file_name)

# Bắt đầu chương trình
session = load_session_state(filename=file_name)
if session['run']:
    with st.spinner('Vui lòng chờ một chút: '):
        # Tạo vector biểu diễn cho câu truy vấn văn bản
        text_feat_arr = text_embedd(text_query)
        # Chuyển DataFrame thành danh sách tuples
        visual_features_db = visual_features_df.to_records(index=False).tolist()
        # Thực hiện tìm kiếm và hiển thị kết quả
        search_result = search_engine(text_feat_arr, visual_features_db, topk, measure_method)
        (images, data) = read_image(search_result)
        visualize(images, data)
        st.success('Hoàn tất truy vấn!', icon="✅")
    
    index = []
    for i in range(len(images)):
        index.append(i+1)
    with st.form('Submit parameter: '):
        idx = st.multiselect('Chọn keyframe để submit:', index)
        #Chọn tên querry để tạo file submit
        name = st.number_input('Chọn số thứ tự query:',1)
        sub = st.form_submit_button('Submit')

    answer = []
    if sub:
        for i in idx:
            temp = [data[i-1]["video_name"], data[i-1]["keyframe_id"]]
            answer.append(temp)

    submit(answer, name)
    #Kết thúc chương trình, đặt lại giá trị
    if st.button('Kết thúc truy vấn'):
        session = {'run': False}
        save_session_state(session_state= session, filename=file_name)